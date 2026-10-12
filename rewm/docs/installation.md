# Installation

## Environment

The experiments used Linux, Python 3.10, PyTorch 2.7.1 with CUDA 12.8,
Torchvision 0.22.1, and Triton 3.3.1. CUDA is required for the supplied MoDA
kernels; a CPU-only run is not a replacement for the reported encoder.
The selected Full training recipes use eight GPUs. Evaluation uses one GPU per
controller/checkpoint. See the recipe documentation before changing batch size.

Create a clean Python 3.10 environment and, from the archive root, run:

```bash
export PYTHONNOUSERSITE=1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`PYTHONNOUSERSITE=1` prevents packages in a shared user directory from
overriding the versions installed in the new environment. Keep it set when
running experiments. A working NVIDIA driver compatible with the selected
PyTorch CUDA build is required. No private package index or cloud account is needed by these commands.
`requirements.txt` pins the direct packages from the recorded environment;
it is not a fully transitive lockfile or a claim that every new platform has
been tested. See the package validation report for checks actually performed.

The MoDA/FLA source snapshot is bundled. Supplied launchers add
`third_party/moda_triton` to Python's import path and enable `FLA_LIGHT_IMPORT=1`.
Do not replace it with a newer generic FLA release when reproducing the paper.
For manual imports, use:

```bash
export FLA_LIGHT_IMPORT=1
export PYTHONPATH="$PWD/third_party/moda_triton${PYTHONPATH:+:$PYTHONPATH}"
python -c "from fla.ops.moda import parallel_moda; from fla.modules import RMSNorm, RotaryEmbedding; print('MoDA imports OK')"
```

## Environment rendering

MuJoCo-based rollout evaluation needs a functioning rendering backend. For
headless GPU rendering, use the system's EGL installation:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

The archive intentionally does not redistribute NVIDIA driver or EGL shared
libraries. If EGL fails, check the driver and operating-system graphics setup
before starting long evaluations. Changing the renderer or simulator version
can change observations and should be recorded.

OGBench is installed from the public, fixed commit listed in `requirements.txt`.
Its Python code and environment assets were compared with the experiment
snapshot. No local path or privately patched OGBench copy is required.
Datasets must be obtained separately; follow `README.md` and the data-layout
instructions. Avoid substituting datasets silently: frame counts, episode
boundaries, and action dimensions affect both training and GC-IDM evaluation.

## License scope

Read `THIRD_PARTY_NOTICES.md` before redistributing this package. In particular,
GC-IDM code uses the upstream CC BY-NC 4.0 license; it must not be treated as
MIT solely because the encoder's upstream projects use MIT.
