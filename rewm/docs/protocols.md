# Released experiment protocols

The archive contains the complete ReWM implementation and its selected five-task
recipes, component and mean/concatenation ablations, the controlled LeWM+MoDA
comparison, the official-recipe LeWM baseline, GC-IDM evaluation, and original
FP32 eval-mode representation diagnostics.

## Fixed training configurations

The task YAML files and `configs/full/protocols.json` set the default seed,
optimizer, batch size, schedule and training budget. The full configuration
resolves before training with `python scripts/train.py --task TASK --config-only`.
The five Full recipes use eight GPUs:

| Task | Seed | Learning rate | Weight decay | Global batch | Default stop (updates) |
|---|---:|---:|---:|---:|---:|
| Reacher | 0 | 2e-4 | 1e-3 | 512 | 6398 |
| TwoRoom | 0 | 7e-4 | 1e-3 | 512 | 512 |
| Cube | 60106 | 5e-4 | 2e-3 | 896 | 512 |
| Puzzle 3×3 | 0 | 2e-4 | 1e-3 | 512 | 17260 |
| PushT | 3072 | 2.5e-4 | 1e-3 | 512 | 27864 |

Puzzle 3×3 uses a 6-block predictor with 3 heads and feed-forward width 512;
the other tasks use 16 heads and width 2048.

Warmup, scheduler horizons, checkpoint saving and paired initialization are
specified in [training_core.md](training_core.md). A prefix run retains its
original scheduler horizon. Data exports and any historical initialization
objects are supplied separately; see [data.md](data.md).

Each Full configuration is the recipe of one reported run, including its
subsequent measured checkpoints.
The encoder recipe/run selection is separate from GC-IDM controller selection,
which uses the lowest validation action MSE within a fixed 300-epoch fit.

## Comparison settings

Official-recipe LeWM defaults to seed 3072, global batch 128, LR 5e-5 and the
original 100-epoch scheduler horizon. Its Puzzle configuration adapts the
upstream OGBench recipe to the prepared Puzzle dataset. Full-versus-official
compares complete system recipes. The systems use different learning rates,
batches and schedules; their encoder window-split seeds match on PushT and
differ on the other four tasks.

The controlled LeWM/LeWM+MoDA pair uses global batch 896, LR 2e-4, weight decay
1e-3, 16 warmup updates and a 100-epoch scheduler horizon. The default seed is
78101; repeated runs explicitly pass the corresponding paired seed. The tuned
comparison in the paper appendix uses learning rate 3.5e-4 for both controlled
arms on PushT and Cube (`--lr 3.5e-4`). Complete MoDA and RoPE are inserted
into the standard LeWM block. Full component arms and mean/concatenation
readouts are defined in [experiment_map.md](experiment_map.md).

Processed sequences equal optimizer updates times actual global batch, including
repeated sequences. Threshold budgets refer to the first qualifying evaluated
checkpoint. Selected peaks, later checkpoints and across-seed statistics retain
these distinct meanings. The three PushT rollout banks evaluate one encoder and
one controller.

## Evaluation and diagnostics

GC-IDM uses a random 90/10 split of eligible frame-pair indices for validation.
Control start/goal states come from the prepared task dataset, with no extra
episode holdout. Full TwoRoom uses all frames; controlled TwoRoom uses global
stride four. See [evaluation.md](evaluation.md) for the controller and rollout
settings.

Probe fit/validation/test episode groups are disjoint. Encoder exclusion for the
supplied banks is at window level and uses the recorded original split seeds;
overlapping frames from the same episodes can occur in encoder training.
Diagnostics preserve the original FP32 eval mapping, frozen buffers and fixed
index banks. See [diagnostics.md](diagnostics.md) for the readouts, data statistics
and own-latent prediction definition.

## Source identity

`SOURCE_PROVENANCE.json` records original and released source hashes without
private filesystem paths. `SHA256SUMS` covers all released files. Adapted files
have separate original and release hashes; [validation.md](validation.md)
summarizes the implementation and interface checks.

Before running experiments, verify the extracted archive:

```bash
python scripts/verify_archive.py --strict
```

The comparison launcher records and fixes its child-process environment.
Legacy `TRAIN_EPOCH_TARGET` and `CANONICAL_INIT_*` variables from another run
do not override its budget or seeded initialization. The Full launcher accepts
an explicit `--anchor` for historical Full initialization replay.
