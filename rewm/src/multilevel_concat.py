"""Learned concatenation projection on the whole-MoDA-disabled Full backbone."""
from types import SimpleNamespace
import torch
from torch import nn


class MultiLevelConcatEncoder(nn.Module):
    def __init__(self, base, readout_layers=(3, 6, 9, 12)):
        super().__init__()
        assert not base.transformer.use_moda
        assert all(not b.attn.qk_norm for b in base.transformer.layers)
        assert base.cls_token is not None
        self.base = base
        self.config = base.config
        self.readout_layers = tuple(readout_layers)
        d = base.config.hidden_size
        with torch.random.fork_rng(devices=[]):
            self.fusion = nn.Linear(d * len(self.readout_layers), d, bias=False)
        # Begin at equal averaging; each depth gets a separately trainable matrix.
        with torch.no_grad():
            self.fusion.weight.copy_(torch.cat([torch.eye(d) / len(self.readout_layers)] * len(self.readout_layers), dim=1))
        assert self.readout_layers == tuple(sorted(set(self.readout_layers)))
        assert min(self.readout_layers) >= 1
        assert max(self.readout_layers) <= len(base.transformer.layers)

    @property
    def transformer(self):
        return self.base.transformer

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        base, tr = self.base, self.transformer
        x = base.patch_embed(pixels.float())
        x = torch.cat((x, base.cls_token.expand(x.shape[0], -1, -1)), dim=1)
        x = tr.input_proj(base.dropout(x))
        summaries = []
        for depth, block in enumerate(tr.layers, 1):
            x, _, _ = tr._run_block(block, x, None, None)
            if depth in self.readout_layers:
                # Same learned terminal RMSNorm for each selected depth;
                # No external controller normalization; fusion follows these states.
                summaries.append(tr.output_proj(tr.norm(x))[:, -1:])
        final = tr.output_proj(tr.norm(x))
        summary = self.fusion(torch.cat(summaries, dim=-1))
        return SimpleNamespace(last_hidden_state=torch.cat((summary, final[:, :-1]), dim=1))
