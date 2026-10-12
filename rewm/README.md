# ReWorldModel (ReWM)

Code for the ReWM encoder, the five selected task configurations, ablations,
LeWM comparisons, GC-IDM evaluation, and frozen representation diagnostics.
Training and evaluation run locally with the supplied scripts.

[protocols.md](docs/protocols.md) lists the configurations and comparison settings.

## Setup

1. Install dependencies following [installation.md](docs/installation.md).
2. Prepare the original task data following [data.md](docs/data.md).
3. Choose a training or evaluation entry point below.
4. Check the configuration with `--config-only` or `--dry-run` before training.

| Experiment | Entry point | Configuration |
|---|---|---|
| Complete ReWM and component ablations | `scripts/train.py` | Selected task recipe; complete MoDA is removed as one module |
| Controlled standard-block comparison | `scripts/train_comparison.py --protocol controlled` | Reference LeWM and LeWM+MoDA share the controlled recipe |
| Official-recipe LeWM baseline | `scripts/train_comparison.py --protocol official` | Original encoder and default optimizer/batch recipe |
| Frozen-encoder control | `scripts/evaluate.py` | Explicit task, model family, checkpoint, controller and rollout protocol |
| Layerwise probes and prediction | `scripts/probe.py` | Frozen checkpoint and shared diagnostic frame indices |

The controlled comparison adds MoDA and RoPE to the standard LeWM block.
It uses its own training recipe, which differs from both Full ReWM and the
LeWM official defaults.
See [experiment_map.md](docs/experiment_map.md) for architecture and budget details.

## Preview and run

Run these commands from this directory after installation:

```bash
# Print the Full Reacher configuration.
python scripts/train.py --task reacher --arm full --config-only

# Train Full ReWM for 8 updates on 8 GPUs, keeping the original schedule.
python scripts/train.py --task reacher --arm full \
  --data-root /path/to/data --output /path/to/runs/reacher_full \
  --stop-after 8 --save-steps 0,8

# Preview both controlled arms; remove --dry-run to train.
python scripts/train_comparison.py --protocol controlled --arm reference \
  --task reacher --seed 78101 --data-root /path/to/data \
  --output /path/to/runs/reference_78101 --stop-after 512 --dry-run
python scripts/train_comparison.py --protocol controlled --arm lewm_moda \
  --task reacher --seed 78101 --data-root /path/to/data \
  --output /path/to/runs/insertion_78101 --stop-after 512 --dry-run

# Preview official LeWM (1 GPU, batch 128).
python scripts/train_comparison.py --protocol official --task reacher \
  --data-root /path/to/data --output /path/to/runs/official_reacher \
  --stop-after 512 --dry-run

python scripts/evaluate.py --help
python scripts/probe.py --help
```

`--stop-after` stops training at the requested update while keeping the original
scheduler horizon. Reducing `max_epochs` changes the schedule. Keep the specified
GPU count and per-device batch size to reproduce each training configuration.

## Layout

```text
src/                     Complete encoder, variants, objective and training
configs/full/            Five selected task recipes and their protocol metadata
controlled/              Controlled insertion code and original LeWM snapshot
scripts/                 Local training, evaluation and diagnostic launchers
evaluation/              Feature extraction, GC-IDM fitting and task rollouts
diagnostics/             Layerwise probes, native prediction, fixed index banks
third_party/             MoDA/FLA kernel snapshot and upstream license notices
docs/                    Installation, data, protocol and validation notes
```

Datasets and checkpoints must be obtained separately. The diagnostic `.npz`
files contain frame/window indices and action statistics for the original HDF5
exports. The runner checks their episode metadata before use. Images, cluster job
scripts, credentials, and private reports are excluded.

See [validation.md](docs/validation.md) for the implementation and interface
checks, and [training_core.md](docs/training_core.md) for initialization and
checkpoint replay instructions.

Third-party code retains its original licenses, including the GC-IDM
noncommercial license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the terms that apply to each component.
