"""Controlled LeWM reference and complete MoDA+RoPE insertion.

Only the two paper comparison arms are exposed by this release.
"""
import hashlib
import json
from types import SimpleNamespace
import torch
from torch import nn
from clean_vit_moda import CleanViTMoDAEncoder,_load_moda_kernel
from moda_module import RMSNorm,RotaryEmbedding
ARMS=['combo_base','combo_qrm']


class ComboEncoder(CleanViTMoDAEncoder):
    def __init__(self,arm,**kwargs):
        super().__init__(use_moda=False,**kwargs)
        self.arm=arm
        self.use_moda=arm=='combo_qrm'
        self.qk_rope=arm!='combo_base'
        self._moda_kernel=_load_moda_kernel()
        assert self._moda_kernel is not None
        self.q_norms=nn.ModuleList([RMSNorm(64,eps=1e-6) for _ in self._layers])
        self.k_norms=nn.ModuleList([RMSNorm(64,eps=1e-6) for _ in self._layers])
        self.rotary=RotaryEmbedding(dim=64,base=10000.)
        # Dormant parameters retained for paired initial hashes; never optimized.
        for m in [self.q_norms,self.k_norms]:
            m.requires_grad_(self.qk_rope)
        self.vit.embeddings.position_embeddings.requires_grad_(not self.qk_rope)
        self._last_k=None

    def _attention(self,q,k,v,cached_k,cached_v,scale):
        if self.qk_rope:
            q=self.q_norms[self._layer_index](q)
            k=self.k_norms[self._layer_index](k)
            # Physical CLS first; assign patch indices 0..N-1, summary index N.
            q=torch.cat([q[:,1:],q[:,:1]],1)
            k=torch.cat([k[:,1:],k[:,:1]],1)
            q,k=self.rotary(q,k,seqlen_offset=0,max_seqlen=q.shape[1])
            q=torch.cat([q[:,-1:],q[:,:-1]],1)
            k=torch.cat([k[:,-1:],k[:,:-1]],1)
        self._last_k=k
        return super()._attention(q,k,v,cached_k,cached_v,scale)

    def forward(self,pixel_values,bool_masked_pos=None,interpolate_pos_encoding=True,**kwargs):
        assert bool_masked_pos is None
        em=self.vit.embeddings
        if self.qk_rope:
            patches=em.patch_embeddings(pixel_values,interpolate_pos_encoding=interpolate_pos_encoding)
            hidden=em.dropout(torch.cat([em.cls_token.expand(patches.shape[0],-1,-1),patches],1))
        else:
            hidden=em(pixel_values,interpolate_pos_encoding=interpolate_pos_encoding)
        keys=[]
        values=[]
        for i,layer in enumerate(self._layers):
            self._layer_index=i
            ck=torch.stack(keys,dim=2).flatten(1,2) if keys else None
            cv=torch.stack(values,dim=2).flatten(1,2) if values else None
            fn=self._forward_old_layer if hasattr(layer.attention,'attention') else self._forward_new_layer
            hidden,_,v=fn(layer,hidden,ck,cv)
            if self.use_moda:
                keys.append(self._last_k)
                values.append(v)
            self._last_k=None
        hidden=self.vit.layernorm(hidden)
        return SimpleNamespace(last_hidden_state=hidden,pooled_state=hidden[:,0])


def make_encoder(arm,**kwargs):
    assert arm in ARMS
    with torch.random.fork_rng(devices=[]):
        return ComboEncoder(arm,**kwargs)


def assert_arm(model,arm):
    assert type(model) is ComboEncoder and model.arm==arm
    assert model.use_moda==(arm=='combo_qrm') and model.qk_rope==(arm!='combo_base')
    assert model.config.intermediate_size==768
    assert isinstance(model.vit.layernorm,nn.LayerNorm)
    assert model._moda_kernel is not None and model._last_k is None


def tensor_hashes(state):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            for k,v in state.items() if torch.is_tensor(v)}

def validate_training_pair(root,fit,cell,anchor):
    a=root/anchor['name']/anchor['arm']
    ref=json.loads((a/'initial_tensor_hashes.json').read_text())
    cur=json.loads((cell/'initial_tensor_hashes.json').read_text())
    assert ref==cur, 'Full model initial tensors must match between retrieval arms'
    downstream={k:v for k,v in ref.items() if not k.startswith('encoder.')}
    assert downstream and downstream=={k:v for k,v in cur.items() if not k.startswith('encoder.')}, 'Downstream initialization mismatch'
    # Identical first two augmented input/action batches on every DDP rank.
    for rank in range(8):
        assert (a/f'batch_hashes_rank{rank}.jsonl').read_bytes()==(cell/f'batch_hashes_rank{rank}.jsonl').read_bytes(), f'Input mismatch rank {rank}'
    (cell/'PAIR_VALIDATED.json').write_text(json.dumps(dict(anchor=anchor['name'],
        downstream_tensors=len(downstream),batch_ranks=8,first_batches=2,passed=True),indent=2))
