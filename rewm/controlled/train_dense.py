"""Run the controlled insertion using upstream training and budget hooks."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'official'))
import train as upstream
import lightning as pl
import torch
import hydra

def verify_source():
    manifest = json.loads((HERE / 'upstream_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((HERE/'official'/name).read_bytes()).hexdigest() == digest, name
    return manifest

class DenseCheckpoints(upstream.ModelObjectCallBack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = time.monotonic()
        self.io_seconds = 0.0
        self.budget = int(os.environ.get('TRAIN_BUDGET', '4096'))
        self.grid = set(json.loads(os.environ["SAVE_STEPS"]) + [0,self.budget])
        self.root = Path(os.environ['CELL_ROOT'])

    def save(self, trainer, module):
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        out = self.root/'checkpoints'/f'step_{step}_object.ckpt'
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            begin = time.monotonic()
            tmp = out.with_suffix('.tmp')
            torch.save(module.model, tmp)
            tmp.replace(out)
            self.io_seconds += time.monotonic() - begin

    def on_train_start(self, trainer, pl_module):
        # Share immutable full model initialization within each seed.
        anchor=os.environ.get('CANONICAL_INIT_CHECKPOINT')
        if anchor:
            assert hashlib.sha256(Path(anchor).read_bytes()).hexdigest()==os.environ['CANONICAL_INIT_SHA256']
            reference=torch.load(anchor,map_location='cpu',weights_only=False)
            pl_module.model.load_state_dict(reference.state_dict(),strict=True)
            del reference
        self.started = time.monotonic()
        epoch_target = int(os.environ.get('TRAIN_EPOCH_TARGET','0'))
        self.screen_grid = []
        if epoch_target:
            steps_per_epoch = int(trainer.num_training_batches)
            assert trainer.estimated_stepping_batches == steps_per_epoch * 100
            self.budget = steps_per_epoch * epoch_target
            self.grid = {0,2,4,8,16,32,64,128,*[max(1,round(steps_per_epoch*f)) for f in [0.25,0.5,0.75,1.0]],self.budget}
            if epoch_target==2:self.grid.add(round(1.5*steps_per_epoch))
            self.grid = {s for s in self.grid if s<=self.budget}
            self.screen_grid = sorted({round(steps_per_epoch*0.5),steps_per_epoch,self.budget})
        if trainer.is_global_zero:
            self.root.mkdir(parents=True, exist_ok=True)
            sched = pl_module.lr_schedulers()
            details = dict(upstream=verify_source(), world_size=trainer.world_size, per_gpu_batch=112, effective_batch_size=896,
                estimated_stepping_batches=trainer.estimated_stepping_batches,
                max_epochs=trainer.max_epochs, max_steps=trainer.max_steps,
                scheduler_class=type(sched).__name__, scheduler_state=sched.state_dict(),
                budget=self.budget, checkpoint_grid=sorted(self.grid), screen_grid=self.screen_grid, steps_per_epoch=int(trainer.num_training_batches),
                encoder_class=type(pl_module.model.encoder).__module__+'.'+type(pl_module.model.encoder).__name__,
                params=sum(p.numel() for p in pl_module.model.parameters()),
                optimizer_groups=[{k:g.get(k) for k in ['lr','initial_lr','weight_decay','betas']} for g in trainer.optimizers[0].param_groups])
            details['intervention'] = dict(warmup_steps=int(os.environ['WARMUP_STEPS']), lr=os.environ['TRAIN_LR'], wd=os.environ['TRAIN_WD'])
            assert sched.warmup_steps == int(os.environ['WARMUP_STEPS'])
            assert sched.max_steps == trainer.estimated_stepping_batches
            assert trainer.world_size == 8
            assert trainer.max_steps == -1, 'Short budgets must not compress scheduler'
            assert trainer.max_epochs == 100
            assert trainer.estimated_stepping_batches > self.budget
            (self.root/'training_manifest.json').write_text(json.dumps(details, indent=2, default=str)+'\n')
        if trainer.is_global_zero:
            from component_controls import tensor_hashes
            initial=tensor_hashes(pl_module.model.state_dict())
            (self.root/'initial_tensor_hashes.json').write_text(json.dumps(initial,sort_keys=True))
        self.save(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.lr_used = [group['lr'] for group in trainer.optimizers[0].param_groups]
        if trainer.global_step < 2:
            from component_controls import tensor_hashes
            row=dict(step=int(trainer.global_step),rank=trainer.global_rank,
                     hashes=tensor_hashes({k:v for k,v in batch.items() if k in ('pixels','action')}))
            with (self.root/f'batch_hashes_rank{trainer.global_rank}.jsonl').open('a') as f:
                f.write(json.dumps(row,sort_keys=True)+'\n')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if trainer.is_global_zero:
            losses = {k:float(v.detach()) for k,v in outputs.items()
                      if 'loss' in k and torch.is_tensor(v) and v.numel()==1}
            if not all(torch.isfinite(torch.tensor(v)) for v in losses.values()):
                raise RuntimeError(f'Nonfinite training loss at step {step}: {losses}')
            row = dict(step=step, lr_used=self.lr_used,
                lr_next=[g['lr'] for g in trainer.optimizers[0].param_groups],
                samples_seen=step*112*trainer.world_size,
                elapsed_seconds=time.monotonic()-self.started,
                checkpoint_io_seconds=self.io_seconds, **losses)
            with (self.root/'learning_curve.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
        if step in self.grid:
            self.save(trainer, pl_module)
        if step >= self.budget:
            # Stop observes an early prefix of the official run. Avoid the
            # extra full validation pass Lightning otherwise inserts on stop.
            # Official sanity validation and completed-epoch validation stay.
            if not trainer.is_last_batch:
                trainer.limit_val_batches = 0
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.root/'training_resume.ckpt')
        if trainer.is_global_zero:
            assert trainer.global_step >= self.budget
            (self.root/'TRAIN_COMPLETE').write_text(str(trainer.global_step)+'\n')

    def on_train_epoch_end(self, trainer, pl_module):
        # An early budget stop must not be mislabeled as a completed epoch.
        if not trainer.should_stop:
            super().on_train_epoch_end(trainer, pl_module)

@hydra.main(version_base=None, config_path=str(HERE/'official/config/train'), config_name='lewm')
def main(cfg):
    verify_source()
    from warmup_override import install
    install()
    from datetime import timedelta
    from lightning.pytorch.strategies import DDPStrategy, StrategyRegistry
    StrategyRegistry.register('ddp_clean_timeout', DDPStrategy, find_unused_parameters=True, timeout=timedelta(minutes=5))
    pl.seed_everything(int(os.environ['RUN_SEED']), workers=True)
    arm = os.environ['ARM']
    from component_controls import ARMS,make_encoder
    assert arm in ARMS
    def factory(size, **kwargs):
        assert size=='tiny' and kwargs['pretrained'] is False and kwargs['use_mask_token'] is False
        return make_encoder(arm,img_size=kwargs['image_size'],patch_size=kwargs['patch_size'])
    upstream.spt.backbone.utils.vit_hf=factory
    upstream.ModelObjectCallBack = DenseCheckpoints
    # training_manifest.json records per_gpu_batch=112 (effective 896 on 8 GPUs).
    assert int(cfg.loader.batch_size) == 112, cfg.loader.batch_size
    upstream.run.__wrapped__(cfg)

if __name__ == '__main__':
    main()
