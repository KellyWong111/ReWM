# Experiment families and reported quantities

## Complete ReWM

`src/moda_module.py` implements the complete visual encoder: complete MoDA
with QKNorm, RoPE, RMSNorm, and SwiGLU. Its output presents the summary token
at index zero for the world-model interface. `configs/full/` contains the selected per-task recipes.

The component arms replace one named component group at a time. The
`full_no_moda_qknorm` arm disables the complete MoDA module. This is the
`-MoDA` row, not a retrieval-only intervention. Mean/concat use block 3/6/9/12
summary tokens after the existing terminal transform on this same ablated
backbone. Concatenation adds 147,456 parameters. See the model constructors
for exact readout operations.

The selected Full runs demonstrate performance attained by individual runs.
A rollout bank is a set of evaluation episodes, not a new encoder seed.
Additional encoder seeds must be launched and reported separately.

## Controlled LeWM+MoDA

`controlled/component_controls.py` exposes `combo_base` and `combo_qrm`.
They retain the standard LeWM block, including LayerNorm, GELU FFN,
bidirectional spatial attention, and a first-position summary token.
The insertion uses the complete MoDA module and RoPE. This model is not Full.

Both controlled arms use global batch 896 (8 x 112), learning rate 2e-4,
weight decay 1e-3, AdamW beta2 .999, 16 warmup updates, and the original
100-epoch scheduling horizon. The paired seeds are 78101, 78102, and 78103.
The complete-system and controlled training recipes are different.
The controlled TwoRoom controller uses stride 4; the Full protocol uses all
frames. Specify the model family in the control evaluation launcher.

## Official-recipe LeWM

The original snapshot and its source manifest are under `controlled/official/`
and `controlled/upstream_manifest.json`. The baseline launcher uses the official
ViT encoder, learning rate 5e-5, global batch 128 on one GPU, and the original
100-epoch scheduler horizon. Checkpoint hooks only save requested update points
and optionally stop a prefix; they do not change the loss or optimizer.

The main Full-versus-official comparison compares complete systems, including
their recipes. Its sample-budget axis counts optimizer updates multiplied by
actual global batch. This count includes repeated sequences and does not measure
unique data, FLOPs, elapsed time, or GC-IDM fitting work.

## Success thresholds and diagnostics

A first-observed threshold is the first evaluated checkpoint reaching that
threshold. It is not a continuously observed convergence time. Do not interpolate
unmeasured checkpoints or replace an entire curve with its running maximum.
Control fitting is performed separately for each frozen checkpoint.

Layerwise position probes and other state readouts measure the specified target
variables. Native prediction R2 uses each encoder's own latent targets; it is
not common-target physical-state prediction accuracy. The diagnostic runner
keeps model parameters and persistent buffers fixed and verifies their hashes.

## Fixed recipes and reproduction

See [protocols.md](protocols.md) for default seeds, hyperparameters, selected-run
definitions, and the data-split settings of each comparison.
