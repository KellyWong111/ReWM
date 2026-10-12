# Frozen-encoder control evaluation

`python scripts/evaluate.py` runs feature extraction, GC-IDM fitting, and online
rollouts against **one supplied encoder checkpoint**. The launcher uses local
paths only; it does not submit cluster jobs or download private checkpoints.
Datasets, encoder checkpoints, NVIDIA EGL libraries, and trained controller
weights are not included in this archive.

## Prepared data

Set `--data-root` to the directory containing:

| Task | HDF5 path relative to data root | Raw action dimension |
|---|---|---:|
| Reacher | `dmc/reacher_random.h5` | 2 |
| TwoRoom | `tworoom.h5` | 2 |
| Cube | `ogbench/cube_single_expert.h5` | 5 |
| Puzzle 3×3 | `ogbench/puzzle3x3_train.h5` | 5 |
| PushT | `pusht_expert_train.h5` | 2 |

Puzzle also requires `ogbench/visual-puzzle-3x3-play-v0.npz` for state-restored
rollouts. The NPZ must contain `observations`, `qpos`, `qvel`, `terminals`, and
`button_states`. HDF5 extraction expects frame-aligned `pixels`, `action` (or
`actions`), `ep_len`, `ep_offset`, and optionally `ep_idx` / `episode_idx`.
The datasets used by the encoders contain prepared 224×224 RGB frames. The
extractor scales uint8 pixels and applies ImageNet image statistics; it does not
convert an arbitrary raw benchmark dataset into the paper's HDF5 dataset.
Environment evaluation additionally reads task state columns through
`stable_worldmodel.data.HDF5Dataset`; these columns must be preserved.

## Run the Full evaluation protocol

From the archive root, using the installed environment:

```bash
python scripts/evaluate.py --task reacher \
  --checkpoint /path/to/step8_object.ckpt \
  --data-root /path/to/data --output ../outputs/reacher/full_step8 --gpu 0
```

Run the same command with `--dry-run` to inspect all underlying commands without
loading torch, launching a GPU process, or creating output files.
Each GPU argument identifies one physical CUDA/EGL device. The renderer needs a
working system EGL driver; bundled private driver libraries are not required.
For headless use the launcher sets `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.

Use `--model-family full` (default), `controlled`, or `official` to resolve the
original model modules matching the checkpoint. These namespaces have files
with the same names and must not be interchanged. Controlled and official paths
prefer `controlled/official`; Full prefers `src`.

Use `--stage extract`, `--stage train`, or `--stage rollout` to run one stage.
Use the same task, checkpoint, output, and feature-stride settings for all stages.
The launcher binds the output directory to the checkpoint SHA-256 and protocol;
it rejects artifacts belonging to a different checkpoint, command plan, source
code, dependency version, or dataset file identity. Dataset identity here means
resolved path, size, and modification time; it is not a dataset content hash.
Stage outputs are SHA-256 checked before downstream reuse. A stage is marked
completed only after its subprocess exits successfully. Rerunning a stage
recomputes it, so use a fresh output directory for a new comparison. The launcher
does not choose a favorable checkpoint or rollout bank. Rerunning any stage
removes the previous summary; recomputing extraction or fitting also invalidates
downstream stage records and removes old per-bank JSON results.

### Settings retained from the experiments

| Setting | Reacher / TwoRoom / Cube | Puzzle | PushT |
|---|---:|---:|---:|
| Extracted representation dimension | 192 | 192 | 192 |
| GC-IDM hidden width / hidden layers | 1024 / 4 | 1024 / 4 | 1024 / 4 |
| Controller fitting epochs | 300 | 300 | 300 |
| Controller batch size / learning rate | 8192 / 3e-4 | 8192 / 3e-4 | 8192 / 3e-4 |
| Controller seed | 42 | 42 | 42 |
| Predicted action block length | 3 | 3 | 3 |
| Maximum training goal horizon | 50 | 50 | 25 |
| Rollout goal offset / action budget | 25 / 50 | 25 / 50 | 25 / 50 |
| Rollouts per bank | 100 | 50 | 100 |
| Rollout bank seeds | 42 | 42 | 42, 93111, 93112 |

The model is frozen and evaluated in eval mode during extraction and rollouts.
The readout is the encoder's **exported global token** at
`output.last_hidden_state[:, 0]`, passed through its learned projector. Full's
internal summary-token ordering is implemented by the encoder and is not changed
by this readout. The controller feature mode is `concat`, with the current
latent, goal latent, and remaining horizon.

Controller fitting retains AdamW weight decay 1e-4, cosine decay to LR/100,
dropout 0.1, gradient clipping at 1.0, and the original random 90/10 split of
eligible frame-pair indices for validation. `train_split=1.0` means that no extra
episode holdout is imposed by this controller script; it does **not** mean that
there is no validation split. The saved controller is selected by lowest
validation action MSE during its 300-epoch fit. The horizon-conditioned goal
sampling and validation behavior are retained from the original source. Validation
shares episodes with controller training and samples future goals during access;
it is not an episode-disjoint or fixed-goal validation bank. Rollout start/goal
states are sampled from the prepared training dataset with `train_split=1.0`,
so control results do not establish generalization to unseen dataset episodes.
This scope differs from the probe fit/validation/test split in the diagnostics.

Controller selection by validation action MSE does not determine the encoder
checkpoint. The launcher evaluates the checkpoint supplied by the caller. A peak
selected after inspecting rollout scores is a selected observed peak, not a
held-out validation-selected encoder result. Preserve the evaluated checkpoint
grid and raw per-bank values; do not replace them by a running maximum or splice
checkpoints from independent encoder runs. This archive alone cannot establish
that historical model/recipe selection never used downstream evaluation scores.

Three PushT rollout banks evaluate the same trained encoder and controller.
Their mean is a rollout-bank mean, **not** a mean over three training seeds.
PushT uses the dedicated goal-horizon policy: one controller, no temporal
ensembling, no additional action clamp, and goal offset 25. Puzzle retains its
state-restoration protocol, nearest-neighbor enlargement of 64×64 images followed
by edge padding to 224×224, and the all-buttons-success predicate.

### Keep the controlled experiment separate

Full TwoRoom uses all extracted frames (`--feature-stride 1`, the Full default).
The historical standard-block controlled TwoRoom comparison uses **global
stride 4** over extracted features, actions, and episode IDs:

```bash
python scripts/evaluate.py --task tworoom --model-family controlled --feature-stride 4 \
  --checkpoint /path/to/controlled_step32_object.ckpt \
  --data-root /path/to/data --output ../outputs/tworoom/controlled_step32 --gpu 0
```

This option exactly retains the recorded global `[::4]` slicing; it is not a new
per-episode resampling rule. Use it consistently for both arms of that controlled
comparison. `--model-family controlled --task tworoom` selects stride 4 by default.
Do not mix its scores with Full's full-frame protocol. Stride 4 is accepted only
with `--model-family controlled --task tworoom`.
The launcher does not infer that an arbitrary checkpoint is Full from its file
name. Supply a checkpoint trained with the intended released model/configuration.

## Low-level entry points

The launcher prints the complete low-level commands. To invoke these manually,
put the archived model modules and evaluator on the Python path:

```bash
export PYTHONPATH="$PWD/src:$PWD/evaluation:$PWD/third_party/moda_triton:$PYTHONPATH"
export FLA_LIGHT_IMPORT=1
export GCIDM_READOUT=cls
export GCIDM_EXTRACT_STAGE=projector
export STABLEWM_HOME=/path/to/data
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python evaluation/train_gcidm_planner.py extract --help
python evaluation/train_gcidm_planner.py train --help
python evaluation/eval_gcidm_planner.py --help
python evaluation/eval_puzzle.py --help
python evaluation/train_single.py --help
python evaluation/eval_gcidm_ensemble.py --help
```

The low-level utilities are retained for compatibility and contain options used
in other diagnostics. Their generic defaults are not all the paper defaults;
use the launcher or its printed commands for the released control protocol.
Checkpoint loading retains full Python-object compatibility because architectural
flags are not recoverable from a bare state dictionary alone. Only load
checkpoints you trust. Historical modules must resolve to their corresponding
source version; replacing those modules can change a loaded model's behavior.

Outputs include `z.npz`, `idm.pt`, `idm_history.json`, stage logs,
`run_manifest.json`, `stage_status.json`, `summary.json`, and one
`bank<seed>.json` record per rollout bank. A missing or failed rollout is an
error, never a zero success score.

## Validation scope

[validation.md](validation.md) lists the source, entry-point and runtime checks,
including real-checkpoint feature extraction and a short TwoRoom rollout.
Full training and the complete 300-epoch controller/rollout evaluations were
not repeated. Reproducing the reported results requires the original prepared
data and the corresponding checkpoints or a completed training run.
