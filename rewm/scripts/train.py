#!/usr/bin/env python3
"""Train the complete encoder or a matched component ablation."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "moda_triton"))
ARMS = ("full", "full_no_moda_qknorm", "full_abspos", "full_gelu_mlp",
        "full_layernorm", "serial_multilevel_mean", "serial_multilevel_concat")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("reacher", "tworoom", "cube", "puzzle3x3", "pusht"), required=True)
    p.add_argument("--arm", choices=ARMS, default="full")
    p.add_argument("--data-root", type=Path, help="Local directory containing the HDF5 datasets.")
    p.add_argument("--dataset-name", help="Override the local HDF5 relative name, without .h5; data content must match the recipe.")
    p.add_argument("--output", type=Path, help="New, empty output directory.")
    p.add_argument("--seed", type=int, help="Encoder seed; default is the selected task recipe.")
    p.add_argument("--stop-after", type=int, help="Stop after this many updates without shortening the original scheduler.")
    p.add_argument("--save-steps", default="0,8,16,32,64,128,256,512", help="Comma-separated update checkpoints; the stop point is also saved.")
    p.add_argument("--anchor", type=Path, help="Optional trusted local Full step-0 object checkpoint for exact paired initialization.")
    p.add_argument("--config-only", action="store_true", help="Print the resolved run plan without loading data, CUDA, or running training.")
    a = p.parse_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / "configs" / "full" / f"{a.task}.yaml")
    protocol = json.loads((ROOT / "configs" / "full" / "protocols.json").read_text())[a.task]
    expected = {
        "trainer.devices": (cfg.trainer.devices, protocol["devices"]),
        "loader.batch_size": (cfg.loader.batch_size, protocol["batch_size"]),
        "trainer.max_epochs": (cfg.trainer.max_epochs, protocol["max_epochs"]),
        "optimizer.lr": (cfg.optimizer.lr, protocol["lr"]),
        "optimizer.weight_decay": (cfg.optimizer.weight_decay, protocol["wd"]),
        "gradient_accumulation": (cfg.trainer.get("accumulate_grad_batches", 1), 1),
    }
    for field, (actual, recorded) in expected.items():
        if actual != recorded:
            p.error(f"{field} differs from the recorded recipe: {actual} != {recorded}")
    if a.seed is not None:
        cfg.seed = a.seed
    if a.dataset_name:
        cfg.data.dataset.name = a.dataset_name
    stop = protocol["default_stop"] if a.stop_after is None else a.stop_after
    if not 1 <= stop <= protocol["steps_per_epoch"] * protocol["max_epochs"]:
        p.error("--stop-after must be positive and within the original training horizon")
    try:
        requested = [int(s.strip()) for s in a.save_steps.split(",") if s.strip()]
    except ValueError:
        p.error("--save-steps must contain comma-separated nonnegative integers")
    if any(s < 0 for s in requested):
        p.error("--save-steps cannot contain negative updates")
    save_steps = sorted({0, stop, *(s for s in requested if s <= stop)})
    plan = dict(task=a.task, arm=a.arm, seed=int(cfg.seed), protocol=protocol,
                stop_after=stop, save_steps=save_steps,
                dataset_name=str(cfg.data.dataset.name),
                data_root=str(a.data_root) if a.data_root else None,
                output=str(a.output) if a.output else None,
                anchor=str(a.anchor) if a.anchor else None)
    if a.config_only:
        print(json.dumps(plan, indent=2))
        return
    if a.data_root is None or a.output is None:
        p.error("training requires --data-root and --output")
    dataset_path = a.data_root.expanduser().resolve() / (cfg.data.dataset.name + ".h5")
    if not dataset_path.is_file():
        p.error(f"HDF5 dataset does not exist: {dataset_path}")
    if a.anchor is not None and not a.anchor.is_file():
        p.error(f"initialization checkpoint does not exist: {a.anchor}")
    output = a.output.expanduser().resolve()
    # Lightning re-executes this script for nonzero DDP ranks.
    if "LOCAL_RANK" not in os.environ and output.exists() and any(output.iterdir()):
        p.error("output is not empty; use a new directory (implicit resume is disabled)")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["STABLEWM_HOME"] = str(a.data_root.expanduser().resolve())
    os.environ["SAVE_STEP_GRID"] = ",".join(map(str, save_steps))
    os.environ["FLA_LIGHT_IMPORT"] = "1"
    os.environ["MODA_TRITON_ROOT"] = str(ROOT / "third_party" / "moda_triton")
    from training import train
    train(cfg, protocol, plan, output, a.anchor)


if __name__ == "__main__":
    main()
