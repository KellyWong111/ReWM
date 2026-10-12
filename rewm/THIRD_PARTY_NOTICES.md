# Third-party notices

This supplement combines code from multiple projects. Their original license
terms and attribution remain in force; the archive is not uniformly MIT-licensed.
Author names below identify upstream projects.

## LeWorldModel (LeWM)

- Source: https://github.com/lucas-maes/le-wm
- Copyright (c) 2026 Lucas Maes.
- License: MIT; reproduced at `third_party/lewm/LICENSE`.
- The supplied model/training utilities derive from LeWM. The ReWM encoder,
  component variants, budget handling, and portable experiment launchers are
  additions or adaptations. Original upstream files, where included, retain
  their own notices. This release does not claim authorship of LeWM.

## Mixture-of-Depths Attention and Flash Linear Attention

- MoDA: https://github.com/ByteDance-Seed/MoDA
- Paper: https://arxiv.org/abs/2603.15619
- Copyright (c) 2026 ByteDance-Seed; MIT.
- FLA: https://github.com/fla-org/flash-linear-attention
- Copyright (c) 2023–2025 Songlin Yang; MIT.
- Both license texts are included in `third_party/moda_triton/`.
- The code retains the experiment's MoDA kernel snapshot and the original
  implementation notices, including attribution in fused-normalization files.
  See that directory's README and source-hash manifest for its scope.

## Goal-Conditioned Inverse Dynamics Model (GC-IDM)

- Source: https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models
- Reference snapshot: `48c45b1cb2b34dd2c1c61d222c8309de567fde55`.
- Copyright (c) 2026 Hoang Nguyen, Xiaohao Xu, Xiaonan Huang,
  Department of Robotics, University of Michigan, Ann Arbor.
- License: Creative Commons Attribution–NonCommercial 4.0 International;
  the upstream notice is reproduced at `third_party/gcidm/LICENSE`.
- License text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
- The controller integration is adapted for frozen ReWM checkpoints,
  experiment-specific feature extraction, reproducible controller training,
  and rollout evaluation. It is not the unmodified upstream distribution.
  The upstream noncommercial condition applies to this included adaptation.

## Dependencies installed separately

PyTorch, Triton, Transformers, Lightning, stable-pretraining, stable-worldmodel,
OGBench, MuJoCo, and the other packages in `requirements.txt` are installed
separately under their respective licenses. OGBench is fixed to public commit
`1d4140997f60c52c6fb0702ec100dc988b18c548`; its package source matches the
experiment's environment snapshot. Dataset files, pretrained checkpoints,
GPU drivers, CUDA redistributables, and NVIDIA EGL binaries are not bundled.

No new blanket license is imposed on independently authored code in this
archive. Third-party permissions are not extended to
unrelated files by implication.
