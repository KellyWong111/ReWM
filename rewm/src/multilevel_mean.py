"""Parameter-free depth mean on the whole-MoDA-disabled Full backbone."""
from types import SimpleNamespace
import torch
from torch import nn


class MultiLevelMeanEncoder(nn.Module):
    def __init__(self, base, readout_layers=(3, 6, 9, 12)):
        super().__init__()
        assert not base.transformer.use_moda
        assert all(not b.attn.qk_norm for b in base.transformer.layers)
        assert base.cls_token is not None
        self.base = base
        self.config = base.config
        self.readout_layers = tuple(readout_layers)
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
                # no new norm, projection, routing weights, or controller input.
                summaries.append(tr.output_proj(tr.norm(x))[:, -1:])
        final = tr.output_proj(tr.norm(x))
        summary = torch.stack(summaries, dim=0).mean(dim=0)
        return SimpleNamespace(last_hidden_state=torch.cat((summary, final[:, :-1]), dim=1))
