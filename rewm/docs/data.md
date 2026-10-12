# Data layout

The code uses prepared stable-worldmodel HDF5 files, not raw image folders.
The [upstream LeWM data collection](https://huggingface.co/collections/quentinll/lewm)
and [OGBench project](https://github.com/seohongpark/ogbench) are the original
public data/environment sources. Obtain and use them under their own terms.
Large datasets are not included in the supplement and no automated download is
triggered by the launchers.

Use this layout under the directory passed as `--data-root`:

```text
data/
  reacher.h5                 # alias of dmc/reacher_random.h5
  dmc/reacher_random.h5
  tworoom.h5
  pusht_expert_train.h5
  ogbench/
    cube_single_expert.h5
    puzzle3x3_train.h5
    visual-puzzle-3x3-play-v0.npz
```

The original training scripts refer to `reacher.h5`, while the controller
extractor refers to `dmc/reacher_random.h5`. These were verified to refer to the
same original dataset. Create a relative symlink `reacher.h5` pointing to
`dmc/reacher_random.h5`; do not silently substitute a different Reacher export.
Full training also accepts `--dataset-name` to change only a local alias.

`docs/dataset_schema.json` records original array names, shapes and dtypes for
all five tasks. A row count is a frame count, not an optimizer-update count.
All images are RGB uint8 arrays of shape `(frames, 224, 224, 3)`. Control
extraction also requires actions and episode metadata. Probe targets are
additional state columns; they are not inputs to the visual encoder.

## Puzzle data preparation

Convert the original `visual-puzzle-3x3-play-v0.npz` to the historical training
HDF5 layout with the included streaming converter:

```bash
python scripts/prepare_puzzle_data.py \
  --input /path/to/data/ogbench/visual-puzzle-3x3-play-v0.npz \
  --output /path/to/data/ogbench/puzzle3x3_train.h5
```

This preserves the original preprocessing: each 64-pixel RGB image is repeated
three times along each spatial axis, then edge-padded by 16 pixels per side to
224 pixels. It is not a direct resize to 224 pixels. The converter also preserves
action/state dtypes and terminal-based episode boundaries. It streams pixels,
writes uncompressed HDF5 (about 151 GB for the original dataset), and refuses to
overwrite an existing output. `--chunk-frames` controls memory use without
changing array values; its default is 2000.

The output contains `pixels`, `action`, `observation`, `ep_idx`, `ep_len`, and
`ep_offset`. The portable converter was compared with the historical converter
on small fixtures; sampled original pixels/actions/states and all original
episode lengths/offsets were also checked. The full image dataset was not
reconverted or hashed for this release. Keep the original NPZ for the Puzzle
rollout evaluator. Different NPZ contents or a newly rendered export represent
a different dataset and must not be substituted silently.

## Diagnostic index banks

`diagnostics/banks/` preserves the original frame/window selections for Reacher,
TwoRoom and PushT, plus action statistics. No image data is included. The probe
launcher supplies the local HDF5 path, while the evaluator checks the recorded
episode lengths, offsets, and bank hash. Use the original matching HDF5 export.
An episode mismatch aborts rather than silently reporting different probes.

Fit, validation and test probe episodes are disjoint. The encoder-training split
was window-level, so this does not assert that every frame of a probe episode
was held out from encoder training.
