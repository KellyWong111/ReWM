# Frozen representation diagnostics

The runner reads raw summary states at blocks 3/6/9/12, the terminal transformed
summary, and the projected latent. Ridge fits use the fixed fit/validation/test
index bank; validation selects ridge strength. Native one-step prediction is
measured using each encoder's own targets and the original next-position
alignment. Parameters and persistent buffers are hashed before and after the
run to verify that the checkpoint stayed frozen.

```bash
python scripts/probe.py --task reacher --architecture full \
  --checkpoint /path/to/step_8_object.ckpt \
  --data-h5 /path/to/data/reacher.h5 --output /path/to/probes/full_step8 --smoke

python scripts/probe.py --task reacher --architecture official \
  --checkpoint /path/to/step_32_object.ckpt \
  --data-h5 /path/to/data/reacher.h5 --output /path/to/probes/official_step32 --smoke
```

`--smoke` uses a small subset and writes `SMOKE_METRICS.json`. It does not produce
a formal completion marker or reproduce a table entry. Remove `--smoke` for the
full fixed-bank diagnostic. `--dry-run` previews paths without reading datasets
or checkpoints. Always use a new output directory.

The matched budget in the example is 4,096 processed sequences: Full uses
batch 512, official LeWM uses batch 128. Other checkpoints must use their actual
recorded global batch. The native R2 targets differ across encoders and are not
physical targets shared between systems.

This portable runner preserves the original FP32 eval-mode mapping and frozen
buffers. It does not merge original-eval scores with train-mode measurements.
Model objects must come from the selected architecture, with their original
module definitions available. Serialized object checkpoints should only be
loaded from sources you trust.

The packaged banks were selected using the encoder split seeds recorded in each
bank's `metadata.json`: Reacher/TwoRoom use 0 and 3072; PushT uses 3072.
Using a checkpoint trained with a different seed or split does not preserve
the original held-out-window property. Rebuild the selections for a new split;
this release provides the original fixed banks, not a general bank generator.

The original bank first partitions episodes into 60/20/20 percent probe
fit/validation/test groups and then selects up to 2,048/512/1,024 windows.
The selected windows are outside the original encoder training-window splits;
other windows in the same episodes, and overlapping frames, may have appeared
in encoder training. Encoder holdout is therefore defined at window level.
The action mean/sample standard deviation were computed over the original full
dataset in its original dtype, then frozen. Ridge input/target statistics
are fitted only on the probe-fit rows, and ridge strength uses probe validation
rows, never probe-test rows.

A completed low-level diagnostic is reused only when the checkpoint, bank and
metadata, diagnostic/model/kernel source hashes, dependency versions, encoder
batch size, and dataset file
location/size/modification time agree and saved artifact hashes still match.
The file identity check is not a whole-dataset content hash. Smoke runs and
incomplete previous artifacts require a fresh directory, preventing a smoke run
from overwriting the arrays belonging to a formal completion marker.
