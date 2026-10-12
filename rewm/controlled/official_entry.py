"""Official architecture/recipe with observation-only checkpoint hooks."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "official"))
import train as upstream
import hydra
import torch
import lightning as pl

class Checkpoints(upstream.ModelObjectCallBack):
    def save_step(self, trainer, module):
        if trainer.is_global_zero:
            dest = Path(os.environ["CELL_ROOT"]) / "checkpoints"
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"step_{trainer.global_step}_object.ckpt"
            torch.save(module.model, path)
    def on_train_start(self, trainer, module):
        self.save_step(trainer, module)
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        import json
        if int(trainer.global_step) in json.loads(os.environ["SAVE_STEPS"]):
            self.save_step(trainer, module)
        budget = int(os.environ.get("TRAIN_BUDGET", "0"))
        if budget and trainer.global_step >= budget:
            if not trainer.is_last_batch:
                trainer.limit_val_batches = 0
            trainer.should_stop = True
    def on_train_epoch_end(self, trainer, module):
        if not trainer.should_stop:
            super().on_train_epoch_end(trainer, module)
    def on_train_end(self, trainer, module):
        trainer.save_checkpoint(Path(os.environ["CELL_ROOT"]) / "training_resume.ckpt")

if __name__ == "__main__":
    pl.seed_everything(int(os.environ["RUN_SEED"]), workers=True)
    upstream.ModelObjectCallBack = Checkpoints
    # Resolve imported Hydra entrypoints against the shipped config directory.
    hydra.main(
        version_base=None,
        config_path=str(Path(__file__).resolve().parent / "official" / "config" / "train"),
        config_name="lewm",
    )(upstream.run.__wrapped__)()
