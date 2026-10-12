#!/usr/bin/env python3
"""Run GC-IDM evaluation for a frozen encoder.

The launcher uses only the standard library, so --dry-run needs no training
dependencies. Training and rollout stages use the original numerical code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
MODEL = ROOT / "src"
TASKS = {
    "reacher": ("dmc/reacher_random.h5", 2, 50, (42,)),
    "tworoom": ("tworoom.h5", 2, 50, (42,)),
    "cube": ("ogbench/cube_single_expert.h5", 5, 50, (42,)),
    "puzzle3x3": ("ogbench/puzzle3x3_train.h5", 5, 50, (42,)),
    "pusht": ("pusht_expert_train.h5", 2, 25, (42, 93111, 93112)),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_identity(path):
    stat = Path(path).stat()
    # Large prepared datasets are identified by location/stat, not content-hashed.
    return dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def source_identity():
    paths = {Path(__file__).resolve(), ROOT / "requirements.txt"}
    for folder in (EVAL, MODEL, ROOT / "controlled", ROOT / "third_party/moda_triton"):
        paths.update(folder.rglob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths) if path.is_file()}


def verify_artifacts(work, status, stage):
    record = status.get(stage, {})
    if record.get("status") != "completed" or not record.get("artifacts_sha256"):
        raise SystemExit(f"Stage {stage} has no verified completed artifacts; rerun it.")
    for name, digest in record["artifacts_sha256"].items():
        path = work / name
        if not path.is_file() or sha256(path) != digest:
            raise SystemExit(f"Stage {stage} artifact changed: {name}; use a fresh output directory.")


def bank_success_percent(work, task, bank):
    """Read one rollout bank's success rate in percent from its JSON record.

    The GC-IDM line in the stage log is only a fallback when the record is missing.
    """
    record = work / f"bank{bank}.json"
    if record.is_file():
        result = json.loads(record.read_text())
        if task == "pusht":
            return float(result["metrics"]["success_rate"])
        if task == "puzzle3x3":
            return 100.0 * result["successes"] / result["num_eval"]
        return float(result["success_rate"])
    log_path = work / f"rollout_bank{bank}.log"
    matches = re.findall(r"GC-IDM(?:-ENSEMBLE)?\s+([0-9.]+)%", log_path.read_text()) if log_path.is_file() else []
    if not matches:
        raise SystemExit(f"No success result for rollout bank {bank}: {record.name} is missing and "
                         f"{log_path.name} has no GC-IDM line; do not treat it as zero.")
    return float(matches[-1])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--model-family", choices=("full", "controlled", "official"), default="full",
                   help="Choose the matching original module namespace for object checkpoints")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Local trusted *_object.ckpt; preserve its encoder class/flags")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root containing the prepared stable-worldmodel datasets")
    p.add_argument("--output", type=Path, required=True,
                   help="A separate output directory for this one encoder checkpoint")
    p.add_argument("--stage", choices=("extract", "train", "rollout", "all"), default="all")
    p.add_argument("--gpu", type=int, default=0, help="One physical CUDA/EGL device")
    p.add_argument("--feature-stride", type=int, choices=(1, 4), default=None,
                   help="Defaults to 4 for controlled TwoRoom; otherwise 1")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running or writing")
    return p


def plan(a):
    checkpoint = a.checkpoint.expanduser().resolve()
    data = a.data_root.expanduser().resolve()
    work = a.output.expanduser().resolve()
    rel_h5, action_dim, horizon, banks = TASKS[a.task]
    py = sys.executable
    commands = []
    if a.stage in ("extract", "all"):
        commands.append(("extract", [py, "-u", str(EVAL / "train_gcidm_planner.py"), "extract",
            "--checkpoint", str(checkpoint), "--h5", str(data / rel_h5),
            "--output", str(work / "z.npz"), "--batch-size", "256", "--num-prefetch", "2",
            "--device", "cuda:0"]))
        if a.feature_stride == 4:
            commands.append(("feature_stride", [py, str(EVAL / "ensure_tworoom_stride.py"),
                str(work / "z.npz"), str(data / rel_h5)]))
    if a.stage in ("train", "all"):
        if a.task == "pusht":
            command = [py, "-u", str(EVAL / "train_single.py"), "--embeddings", str(work / "z.npz"),
                "--output", str(work / "idm.pt"), "--max-goal-horizon", "25", "--seed", "42",
                "--device", "cuda:0", "--embedding-normalization", "none"]
        else:
            command = [py, "-u", str(EVAL / "train_gcidm_planner.py"), "train",
                "--embeddings", str(work / "z.npz"), "--output", str(work / "idm.pt"),
                "--embed-dim", "192", "--action-dim", str(action_dim), "--frameskip", "3",
                "--hidden-dim", "1024", "--n-layers", "4", "--epochs", "300",
                "--batch-size", "8192", "--lr", "3e-4", "--seed", "42",
                "--feature-mode", "concat", "--train-split", "1.0", "--split-seed", "42",
                "--max-goal-horizon", str(horizon), "--embedding-normalization", "none", "--device", "cuda:0"]
        commands.append(("train", command))
    if a.stage in ("rollout", "all"):
        for bank in banks:
            common = ["--num-eval", "100", "--goal-offset", "25", "--eval-budget", "50",
                      "--seed", str(bank), "--device", "cuda:0"]
            if a.task == "pusht":
                command = [py, "-u", str(EVAL / "eval_gcidm_ensemble.py"),
                    "--checkpoint", str(checkpoint), "--idms", str(work / "idm.pt"),
                    "--aggregate", "mean", "--temporal", "none", "--horizon-mode", "goal",
                    "--action-clamp", "none", *common, "--label", "released_checkpoint",
                    "--result-json", str(work / f"bank{bank}.json")]
            elif a.task == "puzzle3x3":
                command = [py, "-u", str(EVAL / "eval_puzzle.py"), "--ckpt", str(checkpoint),
                    "--idm", str(work / "idm.pt"), "--aggregation", "mean",
                    "--npz", str(data / "ogbench/visual-puzzle-3x3-play-v0.npz"),
                    "--num-eval", "50", "--goal-offset", "25", "--eval-budget", "50",
                    "--seed", str(bank), "--device", "cuda:0",
                    "--output", str(work / f"bank{bank}.json"), "--tag", "released_checkpoint"]
            else:
                command = [py, "-u", str(EVAL / "eval_gcidm_planner.py"), "--dataset", a.task,
                    "--checkpoint", str(checkpoint), "--idm", str(work / "idm.pt"),
                    "--train-split", "1.0", "--split-seed", "42", *common,
                    "--result-json", str(work / f"bank{bank}.json")]
            commands.append((f"rollout_bank{bank}", command))
    return commands


def main():
    a = parser().parse_args()
    if a.feature_stride is None:
        a.feature_stride = 4 if a.model_family == "controlled" and a.task == "tworoom" else 1
    if a.gpu < 0:
        raise SystemExit("--gpu must be nonnegative")
    if a.feature_stride == 4 and (a.task != "tworoom" or a.model_family != "controlled"):
        raise SystemExit("The released stride-4 protocol is only defined for controlled TwoRoom.")
    commands = plan(a)
    for name, cmd in commands:
        print(f"[{name}] {shlex.join(cmd)}", flush=True)
    if a.dry_run:
        return

    work = a.output.expanduser().resolve()
    data = a.data_root.expanduser().resolve()
    checkpoint = a.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".ckpt":
        raise SystemExit("Expected a local trusted *_object.ckpt file, not a bare state_dict.")
    required = [data / TASKS[a.task][0]]
    if a.task == "puzzle3x3":
        required.append(data / "ogbench/visual-puzzle-3x3-play-v0.npz")
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Required prepared dataset is missing: {path}")
    all_args = argparse.Namespace(**vars(a)); all_args.stage = "all"
    versions = {}
    for name in ("torch", "torchvision", "numpy", "h5py", "transformers", "stable-worldmodel", "stable-pretraining", "ogbench", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = dict(schema_version=2, task=a.task, model_family=a.model_family, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    data_root=str(data), feature_stride=a.feature_stride, controller_seed=42,
                    controller_epochs=300, rollout_banks=list(TASKS[a.task][3]),
                    dataset_file_identity=[file_identity(path) for path in required],
                    dataset_identity_scope="resolved path, byte size and modification time; not a content hash",
                    source_sha256=source_identity(), dependency_versions=versions,
                    complete_command_plan=dict(plan(all_args)),
                    representation=dict(readout="cls", extract_stage="projector", embedding_normalization="none"))
    manifest_path = work / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("Output contains a different checkpoint/protocol; use a new --output directory.")
    if not manifest_path.exists() and work.exists() and any(work.iterdir()):
        raise SystemExit("Existing untracked artifacts cannot be reused. Choose a fresh output directory.")
    work.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    if a.stage in ("train", "rollout"):
        need = "z.npz" if a.stage == "train" else "idm.pt"
        if not (work / need).is_file():
            raise SystemExit(f"Missing {need}; run the preceding stage first.")
    env = dict(os.environ)
    controlled = ROOT / "controlled"
    if a.model_family == "controlled":
        model_paths = [controlled / "official", controlled, MODEL]
    elif a.model_family == "official":
        model_paths = [controlled / "official", MODEL]
    else:
        model_paths = [MODEL]
    paths = [*model_paths, EVAL, ROOT / "third_party/moda_triton"]
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), MUJOCO_EGL_DEVICE_ID=str(a.gpu),
               STABLEWM_HOME=str(data), GCIDM_READOUT="cls", GCIDM_EXTRACT_STAGE="projector",
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", WANDB_MODE="disabled",
               FLA_LIGHT_IMPORT="1", MODA_TRITON_ROOT=str(ROOT / "third_party/moda_triton"),
               OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join(str(x) for x in paths) + os.pathsep + env.get("PYTHONPATH", ""))
    for key in ("GCIDM_INTERFACE_SPEC", "LEWM_CODE_DIR", "LEWM_ROOT"):
        env.pop(key, None)
    stage_status = work / "stage_status.json"
    status = json.loads(stage_status.read_text()) if stage_status.exists() else {}
    for name, cmd in commands:
        if name == "train":
            required_stages = ["extract"] + (["feature_stride"] if a.feature_stride == 4 else [])
            if any(status.get(stage, {}).get("status") != "completed" for stage in required_stages):
                raise SystemExit("Feature extraction has not completed for this protocol.")
            verify_artifacts(work, status, "feature_stride" if a.feature_stride == 4 else "extract")
        if name.startswith("rollout_"):
            if status.get("train", {}).get("status") != "completed":
                raise SystemExit("Controller fitting has not completed; refusing a partial checkpoint.")
            verify_artifacts(work, status, "train")
            history = json.loads((work / "idm_history.json").read_text())
            if len(history["train_loss"]) != 300 or len(history["val_loss"]) != 300:
                raise SystemExit("Controller history does not contain the required 300 epochs.")
        # Recomputing any stage invalidates the previous score.
        (work / "summary.json").unlink(missing_ok=True)
        if name in ("extract", "train"):
            for bank in TASKS[a.task][3]:
                (work / f"bank{bank}.json").unlink(missing_ok=True)
        elif name.startswith("rollout_"):
            (work / (name.removeprefix("rollout_") + ".json")).unlink(missing_ok=True)
        # Mark downstream stages stale when their inputs are recomputed.
        if name == "extract":
            for stage in list(status):
                if stage != "extract":
                    status[stage]["status"] = "stale"
        elif name == "train":
            for stage in list(status):
                if stage.startswith("rollout_"):
                    status[stage]["status"] = "stale"
        log_path = work / f"{name}.log"
        # Explicit reruns overwrite only that stage's outputs and log.
        status[name] = {"status": "running", "command": cmd}
        write_json(stage_status, status)
        with log_path.open("w") as log:
            # Upstream renderers may create videos relative to their cwd.
            result = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        status[name]["returncode"] = result.returncode
        status[name]["status"] = "completed" if result.returncode == 0 else "failed"
        write_json(stage_status, status)
        if result.returncode:
            raise SystemExit(f"{name} failed ({result.returncode}); inspect {log_path}")
        artifacts = [log_path.name]
        if name in ("extract", "feature_stride"):
            artifacts += ["z.npz"]
        elif name == "train":
            artifacts += ["idm.pt", "idm_history.json"]
        elif name.startswith("rollout_"):
            artifacts += [name.removeprefix("rollout_") + ".json"]
        status[name]["artifacts_sha256"] = {name: sha256(work / name) for name in artifacts}
        write_json(stage_status, status)
        print(f"Completed {name}: {log_path}", flush=True)
    if a.stage in ("rollout", "all"):
        scores = []
        for bank in TASKS[a.task][3]:
            verify_artifacts(work, status, f"rollout_bank{bank}")
            scores.append(bank_success_percent(work, a.task, bank))
        summary = dict(task=a.task, run_manifest_sha256=sha256(manifest_path),
                       checkpoint_sha256=manifest["checkpoint_sha256"], controller_sha256=sha256(work / "idm.pt"),
                       success_percent=sum(scores) / len(scores),
                       rollout_banks=list(TASKS[a.task][3]), bank_success_percent=scores,
                       encoder_runs=1, controller_seeds=[42])
        write_json(work / "summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
