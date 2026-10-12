"""Dataset transforms and exact-step model-object checkpoint callbacks."""
import os
import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


def _get_save_steps():
    """Steps at which to save: an explicit list from SAVE_STEP_GRID, else an interval from SAVE_STEP_INTERVAL."""
    grid = os.environ.get("SAVE_STEP_GRID", "")
    if grid:
        return sorted(set(int(x) for x in grid.split(",") if x.strip()))
    interval = int(os.environ.get("SAVE_STEP_INTERVAL", "100"))
    return interval


class ModelObjectCallBack(Callback):
    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1, step_interval: int = 0, model_builder=None):
        super().__init__()
        self.dirpath = Path(dirpath) if not isinstance(dirpath, Path) else dirpath
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self.model_builder = model_builder
        self._saved_steps = set()
        self._save_steps = _get_save_steps()  # int (interval) or list (explicit grid)

        if isinstance(self._save_steps, list):
            self._save_set = set(self._save_steps)
        else:
            self._save_set = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        if self.step_interval > 0 and 0 not in self._saved_steps and trainer.is_global_zero:
            self._dump_model(pl_module.model, self.dirpath / f"{self.filename}_step_0_object.ckpt")
            self._saved_steps.add(0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.step_interval <= 0:
            return
        gstep = trainer.global_step
        should_save = False
        if self._save_set is not None:
            should_save = gstep in self._save_set
        else:
            interval = self._save_steps
            should_save = (interval > 0 and gstep % interval == 0)
        if should_save and gstep not in self._saved_steps and gstep > 0:
            if trainer.is_global_zero:
                self._dump_model(pl_module.model, self.dirpath / f"{self.filename}_step_{gstep}_object.ckpt")
            self._saved_steps.add(gstep)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        epoch_idx = trainer.current_epoch + 1
        if trainer.is_global_zero and epoch_idx % self.epoch_interval == 0:
            output_path = self.dirpath / f"{self.filename}_epoch_{epoch_idx}_object.ckpt"
            self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        clean_model = self.model_builder(model) if self.model_builder is not None else model
        torch.save(clean_model, path)
