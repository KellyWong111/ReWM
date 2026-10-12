"""Full-family ablations for the ReWM visual encoder.
Each variant is built in place from an already constructed MoDAVisualEncoder.
No training is submitted by this module.
"""
import copy, math
import torch
from torch import nn
from types import SimpleNamespace

FULL_ARMS=(
    'full',
    'full_no_moda_qknorm',
    'full_abspos',
    'full_layernorm',
    'full_gelu_mlp',
)
ARMS = FULL_ARMS
MATRIX={
    'full': dict(
        family='full',
        role='main system',
        intervention='none',
        direct_training=True,
    ),
    'full_no_moda_qknorm': dict(
        family='full',
        role='core mechanism ablation',
        intervention='disable MoDA depth access AND QK normalization; retain all other Full components',
        direct_training=True,
    ),
    'full_abspos': dict(
        family='full',
        role='system component ablation',
        intervention='replace 1D RoPE with learned absolute positions; retain causal last summary readout',
        direct_training=True,
    ),
    'full_layernorm': dict(
        family='full',
        role='system component ablation',
        intervention='replace block/final RMSNorm with bias-free LayerNorm; retain QK norm',
        direct_training=True,
    ),
    'full_gelu_mlp': dict(
        family='full',
        role='system component ablation',
        intervention='replace SwiGLU MLP with bias-free GELU MLP at matched 4D width',
        direct_training=True,
    ),
}

class NoRotary(nn.Module):
    def forward(self,q,k,**kwargs):
        return q,k


class GELUMLP(nn.Module):
    def __init__(self,dim,depth,std):
        super().__init__()
        self.fc1=nn.Linear(dim,4*dim,bias=False)
        self.fc2=nn.Linear(4*dim,dim,bias=False)
        nn.init.normal_(self.fc1.weight,std=std)
        nn.init.normal_(self.fc2.weight,std=std/math.sqrt(2*depth))
    def forward(self,x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class AbsoluteFull(nn.Module):
    def __init__(self,base,std):
        super().__init__()
        self.base=base
        self.config=base.config
        self.position=nn.Parameter(torch.empty(1,base.patch_embed.num_patches+1,base.config.hidden_size))
        nn.init.trunc_normal_(self.position,std=std)
        for b in base.transformer.layers:
            b.attn.rotary=NoRotary()

    @property
    def transformer(self):
        return self.base.transformer

    def forward(self,pixels,interpolate_pos_encoding=True):
        x=self.base.patch_embed(pixels.float())
        x=torch.cat((x,self.base.cls_token.expand(x.shape[0],-1,-1)),dim=1)
        x=self.base.dropout(x+self.position.to(x.dtype))
        x=self.base.transformer(x,c=None)
        return SimpleNamespace(last_hidden_state=torch.cat((x[:,-1:],x[:,:-1]),dim=1))


def make_variant(anchor,arm,seed=3072,std=.02):
    assert arm in FULL_ARMS
    assert all(b.attn.qk_norm for b in anchor.transformer.layers)
    assert anchor.cls_token is not None
    base=copy.deepcopy(anchor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm=='full_layernorm':
            d=base.config.hidden_size
            base.transformer.norm=nn.LayerNorm(d,eps=1e-6,bias=False)
            for b in base.transformer.layers:
                b.attn_norm=nn.LayerNorm(d,eps=1e-6,bias=False)
                b.mlp_norm=nn.LayerNorm(d,eps=1e-6,bias=False)
        elif arm=='full_gelu_mlp':
            for b in base.transformer.layers:
                b.mlp=GELUMLP(base.config.hidden_size,len(base.transformer.layers),std)
        elif arm=='full_abspos':
            base=AbsoluteFull(base,std)
        elif arm=='full_no_moda_qknorm':
            base.transformer.use_moda=False
            for b in base.transformer.layers:
                b.attn.qk_norm=False
    return base
