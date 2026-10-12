"""ReWM visual encoder.

A patch-token transformer whose attention also reads the keys and values
cached by earlier layers at the same token position (MoDA cross-depth
attention, run through the official Triton kernel). Blocks are pre-norm
RMSNorm residual blocks with RoPE, QK RMSNorm and a SwiGLU MLP; the depth
cache is scaled by a learned scalar. A learned summary token appended after
the patch tokens provides the global readout.

Depth cache layout follows the MoDA convention:
  cached_k : [B, T * L, H, D]
  cached_v : [B, T * L, H, D]
"""

import os
import sys
import math
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

# Locate the MoDA Triton kernel (MODA_TRITON_ROOT or the bundled third_party copy).
_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)
_MODA_ROOT_CANDIDATES = [
    os.environ.get("MODA_TRITON_ROOT"),
    os.path.abspath(os.path.join(_THIS_DIR, "..", "third_party", "moda_triton")),
]
for _moda_root in _MODA_ROOT_CANDIDATES:
    if _moda_root and _moda_root not in sys.path:
        sys.path.insert(0, _moda_root)

try:
    from fla.ops.moda import parallel_moda, parallel_moda_chunk_visible
    MODA_TRITON_AVAILABLE = True
except ImportError:
    MODA_TRITON_AVAILABLE = False

    # Model objects unpickled from checkpoints resolve these names at call time; without the
    # kernel they must fail rather than compute a different attention pattern.
    def _missing_moda_kernel(*args, **kwargs):
        raise RuntimeError("fla.ops.moda (the MoDA Triton kernel) is required; it is bundled under "
                           "third_party/moda_triton and needs an NVIDIA GPU.")

    parallel_moda = _missing_moda_kernel
    parallel_moda_chunk_visible = _missing_moda_kernel

try:
    from fla.modules import RMSNorm, RotaryEmbedding
except ImportError:
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return x * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, base=10000.0):
            super().__init__()
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        @staticmethod
        def _rotate_half(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        def forward(
            self,
            q,
            k,
            seqlen_offset=0,
            max_seqlen=None,
            cu_seqlens=None,
        ):
            del max_seqlen, cu_seqlens
            seq_len = q.shape[1]
            positions = torch.arange(
                seqlen_offset,
                seqlen_offset + seq_len,
                device=q.device,
                dtype=self.inv_freq.dtype,
            )
            freqs = torch.outer(positions, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, :, None, :]
            sin = emb.sin()[None, :, None, :]
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)
            return q, k


class MoDAAttention(nn.Module):
    """MoDA attention with RoPE and optional QK RMSNorm."""

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        qk_norm=True,
        rope_theta=10000.0,
        qkv_bias=False,
        norm_eps=1e-6,
        chunk_visible=False,
        attention_scale_multiplier=1.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.chunk_visible = bool(chunk_visible)
        self.attention_scale_multiplier = float(attention_scale_multiplier)

        if not MODA_TRITON_AVAILABLE:
            raise RuntimeError(
                "Official MoDA Triton kernel is unavailable. "
                "Install / import fla.ops.moda.parallel_moda before training."
            )

        self.q_proj = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(inner_dim, dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(dim_head, eps=norm_eps)
            self.k_norm = RMSNorm(dim_head, eps=norm_eps)

        self.rotary = RotaryEmbedding(dim=dim_head, base=rope_theta)

    def forward(self, x, cached_k=None, cached_v=None):
        """
        Args:
            x:        (B, T, D) — pre-modulated hidden states
            cached_k: (B, T*L, H, dim_head) or None — depth keys
                      **position-major**: [pos0_layer0, pos0_layer1, ...,
                                           pos1_layer0, pos1_layer1, ...]
            cached_v: (B, T*L, H, dim_head) or None — depth values
        Returns:
            out:   (B, T, D)
            cur_k: (B, T, H, dim_head)   — this layer's keys   (for cache)
            cur_v: (B, T, H, dim_head)   — this layer's values  (for cache)
        """
        q = rearrange(self.q_proj(x), "b t (h d) -> b t h d", h=self.heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b t h d", h=self.heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b t h d", h=self.heads)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.rotary(q, k, seqlen_offset=0, max_seqlen=x.shape[1])

        cur_k = k                   # (B, T, H, D) — store for depth cache
        cur_v = v                   # (B, T, H, D)

        out = self._parallel_moda(q, k, v, cached_k, cached_v)
        out = rearrange(out, "b t h d -> b t (h d)")
        return self.out_dropout(self.o_proj(out)), cur_k, cur_v

    def _parallel_moda(self, q, k, v, cached_k=None, cached_v=None):
        """Route attention through the official MoDA Triton kernel."""
        moda_kernel = parallel_moda_chunk_visible if self.chunk_visible else parallel_moda
        scale_multiplier = float(getattr(self, "attention_scale_multiplier", 1.0))
        return moda_kernel(
            q,
            k,
            v,
            cached_k=cached_k,
            cached_v=cached_v,
            scale=self.scale * scale_multiplier,
            moda_group_num=1,
            head_first=False,
            need_lse=False,
            warn_shape=False,
        )


class GatedMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, hidden_ratio=4, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * hidden_ratio)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class MoDABlock(nn.Module):
    """MoDA-style pre-norm residual block."""

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim=None,
        dropout=0.0,
        qk_norm=True,
        rope_theta=10000.0,
        qkv_bias=False,
        norm_eps=1e-6,
        hidden_ratio=4,
        chunk_visible=False,
        attention_scale_multiplier=1.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = MoDAAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
            qkv_bias=qkv_bias,
            norm_eps=norm_eps,
            chunk_visible=chunk_visible,
            attention_scale_multiplier=attention_scale_multiplier,
        )
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = GatedMLP(
            dim,
            hidden_dim=mlp_dim,
            hidden_ratio=hidden_ratio,
            dropout=dropout,
        )

    def forward(self, x, cached_k=None, cached_v=None):
        attn_out, cur_k, cur_v = self.attn(self.attn_norm(x), cached_k, cached_v)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, cur_k, cur_v


def _init_moda_module_weights(module, initializer_range):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)


def _rescale_prenorm_residual(module, depth, num_residuals_per_layer=2):
    scale = math.sqrt(num_residuals_per_layer * depth)
    for submodule in module.modules():
        p = None
        if hasattr(submodule, "o_proj") and hasattr(submodule.o_proj, "weight"):
            p = submodule.o_proj.weight
        elif hasattr(submodule, "down_proj") and hasattr(submodule.down_proj, "weight"):
            p = submodule.down_proj.weight
        if p is not None:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p /= scale


class MoDATransformer(nn.Module):
    """Transformer that accumulates depth K/V cache for official MoDA attention.

    With ``chunk_visible=False`` this matches the official v14 visible semantics:
    sequence attention is causal over token positions, while depth candidates for
    token position ``t`` come from all earlier layers at the same base position.

    With ``chunk_visible=True`` this switches to the official v16 chunk-visible
    kernel.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim=None,
        dropout=0.0,
        use_moda=True,
        depth_start_layer=1,
        qk_norm=True,
        rope_theta=10000.0,
        activation_checkpointing=False,
        initializer_range=0.02,
        qkv_bias=False,
        norm_eps=1e-6,
        hidden_ratio=4,
        chunk_visible=False,
        attention_scale_multiplier=1.0,
        depth_cache_gate_init=1.0,
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.depth = depth
        self.use_moda = use_moda
        self.depth_start_layer = depth_start_layer
        self.activation_checkpointing = activation_checkpointing
        self.initializer_range = initializer_range
        self.chunk_visible = bool(chunk_visible)
        self.attention_scale_multiplier = float(attention_scale_multiplier)
        self.depth_cache_gate = nn.Parameter(
            torch.tensor(float(depth_cache_gate_init))
        )

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [
                MoDABlock(
                    hidden_dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                    qk_norm=qk_norm,
                    rope_theta=rope_theta,
                    qkv_bias=qkv_bias,
                    norm_eps=norm_eps,
                    hidden_ratio=hidden_ratio,
                    chunk_visible=chunk_visible,
                    attention_scale_multiplier=attention_scale_multiplier,
                )
                for _ in range(depth)
            ]
        )
        self.apply(lambda module: _init_moda_module_weights(module, initializer_range))
        _rescale_prenorm_residual(self, depth=depth)

    def _build_depth_cache(self, k_cache, v_cache, T, layer_idx):
        """View cached K/V into MoDA format [B, T*L, H, D].

        MoDA expects **position-major** layout::

            [pos0_layer0, pos0_layer1, …, pos1_layer0, pos1_layer1, …]

        The backing storage is preallocated once as ``[B, T, depth, H, D]``.
        Each layer then takes a cheap prefix view instead of rebuilding the
        whole cache with ``torch.stack`` every iteration.
        """
        if layer_idx <= 0:
            return None, None
        stacked_k = k_cache[:, :, :layer_idx]
        stacked_v = v_cache[:, :, :layer_idx]
        B, _T, L, H, D = stacked_k.shape
        # Later layers append to the preallocated backing cache in-place. Clone
        # the prefix view so activation checkpointing does not see its version
        # counter change during backward replay.
        gated_k = stacked_k.reshape(B, T * L, H, D).clone()
        gated_v = stacked_v.reshape(B, T * L, H, stacked_v.shape[-1]).clone()
        gate = getattr(self, "depth_cache_gate", None)
        if gate is None:
            return gated_k, gated_v
        gate = gate.to(dtype=gated_k.dtype)
        return gated_k * gate, gated_v * gate

    def _run_block(self, block, x, cached_k, cached_v):
        if not self.training or not self.activation_checkpointing:
            return block(x, cached_k, cached_v)
        if cached_k is None or cached_v is None:
            return block(x, cached_k, cached_v)
        return checkpoint(block, x, cached_k, cached_v, use_reentrant=False)

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
            x = x + c

        T = x.shape[1]
        k_cache = None
        v_cache = None

        for layer_idx, block in enumerate(self.layers):
            if self.use_moda and layer_idx >= self.depth_start_layer and k_cache is not None:
                cached_k, cached_v = self._build_depth_cache(
                    k_cache, v_cache, T, layer_idx,
                )
            else:
                cached_k, cached_v = None, None

            x, cur_k, cur_v = self._run_block(block, x, cached_k, cached_v)
            if k_cache is None:
                B, _T, H, D = cur_k.shape
                k_cache = cur_k.new_empty(B, T, self.depth, H, D)
                v_cache = cur_v.new_empty(B, T, self.depth, H, cur_v.shape[-1])
            k_cache[:, :, layer_idx].copy_(cur_k)
            v_cache[:, :, layer_idx].copy_(cur_v)

        x = self.norm(x)
        return self.output_proj(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MoDAVisualEncoder(nn.Module):
    """Official-style MoDA visual encoder for LeWM image observations.

    The image is tokenized into raster-order patch tokens, then a learnable
    summary token is appended inside the causal/depth-aware MoDA sequence so it
    can aggregate preceding patch tokens. The returned ``last_hidden_state`` is
    reordered to expose that summary as token 0, making
    ``last_hidden_state[:, 0]`` the official LeWM/ViT-style global readout.
    When ``chunk_visible=False`` this is the encoder-side adapter for official
    v14 ``parallel_moda`` with visible depth semantics. When
    ``chunk_visible=True`` it switches to the official v16 chunk-visible path.
    """

    def __init__(
        self,
        *,
        img_size,
        patch_size,
        hidden_size,
        depth,
        heads,
        dim_head,
        mlp_dim=None,
        dropout=0.0,
        emb_dropout=0.0,
        in_chans=3,
        use_moda=True,
        depth_start_layer=1,
        qk_norm=True,
        rope_theta=10000.0,
        activation_checkpointing=False,
        initializer_range=0.02,
        qkv_bias=False,
        norm_eps=1e-6,
        hidden_ratio=4,
        chunk_visible=False,
        attention_scale_multiplier=1.0,
        depth_cache_gate_init=1.0,
        use_cls_token=True,
    ):
        super().__init__()
        self.use_cls_token = bool(use_cls_token)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=hidden_size,
        )
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.trunc_normal_(self.cls_token, std=initializer_range)
        else:
            self.register_parameter("cls_token", None)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = MoDATransformer(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_moda=use_moda,
            depth_start_layer=depth_start_layer,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
            activation_checkpointing=activation_checkpointing,
            initializer_range=initializer_range,
            qkv_bias=qkv_bias,
            norm_eps=norm_eps,
            hidden_ratio=hidden_ratio,
            chunk_visible=chunk_visible,
            attention_scale_multiplier=attention_scale_multiplier,
            depth_cache_gate_init=depth_cache_gate_init,
        )
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        x = self.patch_embed(pixels.float())
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([x, cls], dim=1)
        x = self.dropout(x)
        x = self.transformer(x, c=None)
        if self.cls_token is not None:
            x = torch.cat([x[:, -1:], x[:, :-1]], dim=1)
        return SimpleNamespace(last_hidden_state=x)
