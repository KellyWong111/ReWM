"""Standard LeWM ViT helper with optional cross-depth attention.

This helper retains the original LeWM block. The published controlled arms
are constructed by component_controls.ComboEncoder, which also configures
QKNorm and positional encoding. The helper alone is not the complete ReWM.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

import stable_pretraining as spt
UPSTREAM_VIT_FACTORY = spt.backbone.utils.vit_hf


_MODA_KERNEL_ERROR = None


def _load_moda_kernel():
    """Return the official Triton kernel, or None (the failure is kept for the error message)."""
    global _MODA_KERNEL_ERROR
    try:
        from fla.ops.moda import parallel_moda
    except Exception as exc:
        _MODA_KERNEL_ERROR = exc
        return None
    return parallel_moda


def _torch_moda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    depth_k: Optional[torch.Tensor],
    depth_v: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """Reference bidirectional MoDA for tests and non-Triton environments.

    Current-layer K/V are spatially global.  Historical K/V are restricted to
    the same base token position, matching the official vision MoDA row mask.
    Inputs and output use [batch, token, head, head_dim].
    """
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    spatial_scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale

    if depth_k is None or depth_v is None:
        probs = torch.softmax(spatial_scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(probs, vh).permute(0, 2, 1, 3).contiguous()

    batch, tokens, heads, head_dim = q.shape
    if depth_k.shape[1] % tokens:
        raise ValueError(
            f"depth cache length {depth_k.shape[1]} is not divisible by tokens {tokens}"
        )
    slots = depth_k.shape[1] // tokens
    dk = depth_k.view(batch, tokens, slots, heads, head_dim).permute(0, 3, 1, 2, 4)
    dv = depth_v.view(batch, tokens, slots, heads, head_dim).permute(0, 3, 1, 2, 4)
    depth_scores = torch.einsum("bhnd,bhnld->bhnl", qh, dk) * scale
    scores = torch.cat((spatial_scores, depth_scores), dim=-1)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    spatial_prob = probs[..., :tokens]
    depth_prob = probs[..., tokens:]
    out = torch.matmul(spatial_prob, vh)
    out = out + torch.einsum("bhnl,bhnld->bhnd", depth_prob, dv)
    return out.permute(0, 2, 1, 3).contiguous()


class CleanViTMoDAEncoder(nn.Module):
    """LeWM ViT with optional cross-layer MoDA."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        hidden_size: int = 192,
        depth: int = 12,
        heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.0,
        use_moda: bool = True,
        strict_moda_kernel: bool = True,
        **_ignored,
    ):
        super().__init__()
        if hidden_size != 192 or depth != 12 or heads != 3:
            raise ValueError(
                "The clean control is intentionally locked to upstream ViT-Tiny "
                f"(hidden=192, depth=12, heads=3); got {hidden_size}, {depth}, {heads}."
            )
        if mlp_dim != hidden_size * 4:
            raise ValueError(
                f"The clean control requires the upstream 4x GELU MLP ({hidden_size * 4}), "
                f"got {mlp_dim}."
            )
        if dropout != 0.0:
            raise ValueError("The upstream LeWM visual ViT attention dropout must remain zero.")

        self.vit = UPSTREAM_VIT_FACTORY(
            "tiny",
            patch_size=patch_size,
            image_size=img_size,
            pretrained=False,
            use_mask_token=False,
        )
        self.config = self.vit.config
        self.use_moda = bool(use_moda)
        self.strict_moda_kernel = bool(strict_moda_kernel)
        self._moda_kernel = _load_moda_kernel() if self.use_moda else None
        if self.use_moda and self.strict_moda_kernel and self._moda_kernel is None:
            raise RuntimeError(
                f"Official fla.ops.moda.parallel_moda is unavailable: {_MODA_KERNEL_ERROR!r}"
            ) from _MODA_KERNEL_ERROR

        if float(self.config.attention_probs_dropout_prob) != 0.0:
            raise ValueError("Clean MoDA requires zero ViT attention dropout")

    @property
    def _layers(self):
        if hasattr(self.vit, "encoder"):
            return self.vit.encoder.layer
        return self.vit.layers

    @staticmethod
    def _old_attention_parts(layer, normalized):
        attention = layer.attention.attention
        batch, tokens, _ = normalized.shape
        heads = attention.num_attention_heads
        head_dim = attention.attention_head_size
        shape = (batch, tokens, heads, head_dim)
        q = attention.query(normalized).view(shape)
        k = attention.key(normalized).view(shape)
        v = attention.value(normalized).view(shape)
        return attention, q, k, v

    @staticmethod
    def _new_attention_parts(layer, normalized):
        attention = layer.attention
        batch, tokens, _ = normalized.shape
        heads = attention.num_attention_heads
        head_dim = attention.head_dim
        shape = (batch, tokens, heads, head_dim)
        q = attention.q_proj(normalized).view(shape)
        k = attention.k_proj(normalized).view(shape)
        v = attention.v_proj(normalized).view(shape)
        return attention, q, k, v

    def _attention(self, q, k, v, cached_k, cached_v, scale):
        if self._moda_kernel is None:
            return _torch_moda(q, k, v, cached_k, cached_v, scale)
        return self._moda_kernel(
            q=q,
            k=k,
            v=v,
            g=None,
            scale=scale,
            cu_seqlens=None,
            cached_k=cached_k,
            cached_v=cached_v,
            moda_group_num=1,
            is_causal=False,
            head_first=False,
            warn_shape=False,
        )

    def _forward_old_layer(self, layer, hidden, cached_k, cached_v):
        normalized = layer.layernorm_before(hidden)
        attention, q, k, v = self._old_attention_parts(layer, normalized)
        context = self._attention(q, k, v, cached_k, cached_v, attention.scaling)
        context = context.reshape(hidden.shape[0], hidden.shape[1], -1)
        attention_out = layer.attention.output(context, normalized)
        hidden = hidden + attention_out
        mlp_out = layer.intermediate(layer.layernorm_after(hidden))
        hidden = layer.output(mlp_out, hidden)
        return hidden, k, v

    def _forward_new_layer(self, layer, hidden, cached_k, cached_v):
        normalized = layer.layernorm_before(hidden)
        attention, q, k, v = self._new_attention_parts(layer, normalized)
        context = self._attention(q, k, v, cached_k, cached_v, attention.scaling)
        context = context.reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = hidden + layer.dropout(attention.o_proj(context))
        residual = hidden
        hidden = layer.mlp(layer.layernorm_after(hidden))
        hidden = residual + layer.dropout(hidden)
        return hidden, k, v

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: bool = True,
        **_kwargs,
    ):
        # The serial arm uses the upstream implementation checked by the parity test.
        if not self.use_moda:
            return self.vit(
                pixel_values,
                bool_masked_pos=bool_masked_pos,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

        hidden = self.vit.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        keys = []
        values = []
        for layer in self._layers:
            if keys:
                # Position-major: token0-layer0..L, token1-layer0..L, ...
                cached_k = torch.stack(keys, dim=2).flatten(1, 2)
                cached_v = torch.stack(values, dim=2).flatten(1, 2)
            else:
                cached_k = cached_v = None

            if hasattr(layer.attention, "attention"):
                hidden, k, v = self._forward_old_layer(
                    layer, hidden, cached_k, cached_v
                )
            else:
                hidden, k, v = self._forward_new_layer(
                    layer, hidden, cached_k, cached_v
                )
            keys.append(k)
            values.append(v)

        hidden = self.vit.layernorm(hidden)
        return SimpleNamespace(last_hidden_state=hidden, pooled_state=hidden[:, 0])

