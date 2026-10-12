# Implementation and interface checks

The release was checked in the existing Python 3.10/CUDA experiment environment.
The following numerical and runtime checks supplement source and interface
verification; reduced workloads are smoke checks, not paper-result estimates.

- Release-wide Python syntax and five-task Full configuration resolution, plus
  actual Hydra configuration parsing for all five official and controlled tasks.
- Seven Full-family arms: packaged/original initialization, forward outputs and
  parameter gradients agree bit-for-bit on the earlier 56-pixel GPU comparison;
  all seven also pass a 224-pixel forward/backward check with finite gradients.
- Full JEPA, SIGReg and predictor objective: two optimizer updates at 224 pixels
  with batch 2, preserving the original scheduler horizon. Encoder parameters
  change, and callback-saved state, embeddings and predictions match bit-for-bit
  after reload. All five task schedulers were instantiated and checked.
- Real TwoRoom Full and official ViT checkpoints: feature extraction on 64
  original frames and frozen diagnostic smoke checks with unchanged parameters
  and buffers. A small CPU GC-IDM fit, save and reload also passed.
- All five environments: actual reset, render and step checks. A reduced
  TwoRoom Full/controller rollout exercised their joint execution; its score
  is not a formal evaluation result. Relative video outputs stay in the chosen
  evaluation output directory.
- Historical Puzzle conversion: the portable converter matches the original
  on small compressed/uncompressed fixtures. Three real image/action/state
  rows and all original episode lengths/offsets match the prepared HDF5.
- Five HDF5 schemas and original diagnostic bank identity, including disjoint
  probe fit/validation/test episode groups.
- Public wheel comparison: the installed stable-pretraining and
  stable-worldmodel Python sources match their published pinned wheels.
  The 36 pinned PyPI packages passed dependency metadata resolution.
- Source manifests, privacy checks, archive integrity, relocated entrypoints,
  cache mismatch rejection, and isolation from legacy training environment
  overrides.

These checks do not constitute full training or complete rollout reproduction.
No complete eight-GPU training run or 300-epoch controller/evaluation suite was
repeated. A separate clean installation and full dataset downloads were not
tested. Runtime smoke checks used the existing environment, where user-site
packages supplied Pillow 12.3.0 and PyYAML 6.0.3 instead of the pinned 11.3.0 and
6.0.2. OGBench environment checks loaded the clean source snapshot at the fixed
public commit in `requirements.txt`; it was not installed as a distribution in
that environment. Follow [installation.md](installation.md) to create a separate
environment. Historical initialization objects and original data must be
provided separately; see [training_core.md](training_core.md) and [data.md](data.md).

Run `python scripts/verify_archive.py --strict` after extraction to check file
hashes and detect extra files. The default mode permits generated Python bytecode
inside `__pycache__`; all other unlisted files are rejected.
