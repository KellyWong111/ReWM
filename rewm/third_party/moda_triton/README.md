# MoDA Triton dependency snapshot

This directory contains the FLA Python source tree used by the experiments.
MoDA (Mixture-of-Depths Attention) is prior work by Lianghui Zhu et al., not a
new attention operator introduced by this repository.

- MoDA: https://github.com/ByteDance-Seed/MoDA
- FLA: https://github.com/fla-org/flash-linear-attention
- MoDA paper: https://arxiv.org/abs/2603.15619
- Licenses: `LICENSE.MoDA` and `LICENSE.FLA` (MIT)

`SOURCE_FILES.json` records the SHA-256 digest of every included Python source.
The complete `fla/` tree is retained because its package initializers import
shared operators. Tests, benchmarks, Git metadata, compiled caches, and binaries
are omitted. No global FLA installation is needed.

The recorded experiment snapshot contains a `FLA_LIGHT_IMPORT=1` switch in its
package initializer; the supplied launchers enable it. Other than removing
compiled caches, this source tree is copied without changing its code. The exact
MoDA upstream Git revision was not recorded, so this snapshot is identified by
its file hashes rather than an inferred commit.
