"""Fixed-recipe checks, schedules, and initialization."""
import hashlib
import json
import math
import os
from pathlib import Path
import lightning as pl
import torch
from budget_schedule import BudgetCosine
from full_ablation import make_variant
from utils import ModelObjectCallBack


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def tensor_hashes(model):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()}


def install_schedule(protocol):
    import stable_pretraining.optim.lr_scheduler as sched
    if protocol["schedule"] == "delayed_cosine":
        sched.BudgetCosine = BudgetCosine
        sched.DEFAULT_SCHEDULER_FACTORIES["BudgetCosine"] = lambda module, optimizer: dict(
            warmup_steps=8, hold_until=128, decay_end=512, min_lr_ratio=.5)
        return "BudgetCosine", protocol["interval"]
    sched.DEFAULT_SCHEDULER_FACTORIES["LinearWarmupCosineAnnealingLR"] = lambda module, optimizer: dict(
        warmup_steps=protocol["warmup"], max_steps=module.trainer.estimated_stepping_batches,
        warmup_start_lr=0., eta_min=0.)
    return "LinearWarmupCosineAnnealingLR", protocol["interval"]


def build_arm(model, arm, seed, output, anchor=None):
    if anchor is not None:
        # Object checkpoints are pickle: only load checkpoints you trust.
        checkpoint = torch.load(anchor, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.state_dict(), strict=True)
    initial = tensor_hashes(model)
    multilevel = arm in ("serial_multilevel_mean", "serial_multilevel_concat")
    model.encoder = make_variant(model.encoder, "full_no_moda_qknorm" if multilevel else arm, seed=seed)
    if arm == "serial_multilevel_mean":
        from multilevel_mean import MultiLevelMeanEncoder
        model.encoder = MultiLevelMeanEncoder(model.encoder, (3, 6, 9, 12))
    elif arm == "serial_multilevel_concat":
        from multilevel_concat import MultiLevelConcatEncoder
        model.encoder = MultiLevelConcatEncoder(model.encoder, (3, 6, 9, 12))
    flags = arm not in ("full_no_moda_qknorm", "serial_multilevel_mean", "serial_multilevel_concat")
    assert model.encoder.transformer.use_moda == flags
    assert all(block.attn.qk_norm == flags for block in model.encoder.transformer.layers)
    actual = tensor_hashes(model)
    assert all(actual[k] == value for k, value in initial.items() if not k.startswith("encoder."))
    if arm == "full":
        assert actual == initial
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        write_json(output / "INITIALIZATION.json", dict(
            arm=arm, encoder_seed=seed,
            initialization="provided_step_0_anchor" if anchor else "seeded_from_scratch",
            anchor_sha256=sha256(anchor) if anchor else None,
            encoder_class=type(model.encoder).__module__ + "." + type(model.encoder).__qualname__,
            encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
            nonencoder_initialization_equal=True, moda_includes_qknorm=True, tensors=actual))
    return model


class ExactObjects(ModelObjectCallBack):
    def __init__(self, *args, steps_per_epoch, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_per_epoch = steps_per_epoch

    def on_train_epoch_end(self, trainer, module):
        # Early stopping can leave a partial epoch.
        if trainer.global_step % self.steps_per_epoch == 0:
            super().on_train_epoch_end(trainer, module)


class ProtocolAudit(pl.Callback):
    def __init__(self, protocol, plan, output):
        self.protocol, self.plan, self.output = protocol, plan, Path(output)

    def on_train_start(self, trainer, module):
        r = self.protocol
        assert trainer.world_size == r["devices"], "The released recipe uses eight GPUs."
        assert trainer.max_epochs == r["max_epochs"] and trainer.max_steps == -1
        assert trainer.estimated_stepping_batches == r["max_epochs"] * r["steps_per_epoch"], (
            "Dataset size/batching differs from the recorded recipe", trainer.estimated_stepping_batches)
        scheduler = module.lr_schedulers()
        assert scheduler.warmup_steps == r["warmup"]
        assert trainer.lr_scheduler_configs[0].interval == r["interval"]
        if r["schedule"] == "delayed_cosine":
            assert (scheduler.hold_until, scheduler.decay_end, scheduler.min_lr_ratio) == (128, 512, .5)
        else:
            assert scheduler.max_steps == r["max_epochs"] * r["steps_per_epoch"]
        if trainer.is_global_zero:
            write_json(self.output / "RUNTIME_PROTOCOL.json", dict(
                plan=self.plan, world_size=trainer.world_size, global_batch=r["global_batch"],
                scheduler_state=scheduler.state_dict(), interval=r["interval"],
                scheduler_advancement="each optimizer update (stable-pretraining manual optimization)"))

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        n, r = trainer.global_step, self.protocol
        scheduler = module.lr_schedulers()
        assert scheduler.last_epoch == n, "Scheduler must advance once per update."
        if r["schedule"] == "delayed_cosine":
            progress = min(1., max(0., (n - 128) / 384))
            ratio = n / 8 if n < 8 else .5 + .5 * (1 + math.cos(math.pi * progress)) / 2
        else:
            warm = r["warmup"]
            total = r["max_epochs"] * r["steps_per_epoch"]
            ratio = n / warm if n < warm else (1 + math.cos(math.pi * (n - warm) / (total - warm))) / 2
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        assert abs(lr - r["lr"] * ratio) < 1e-12, (n, lr, r["lr"] * ratio)
        if trainer.is_global_zero and (n in self.plan["save_steps"] or n % 100 == 0 or n <= 8):
            losses = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items() if "loss" in k}
            assert all(math.isfinite(v) for v in losses.values())
            with (self.output / "LR_TRACE.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=n, lr=lr, losses=losses)) + "\n")
        if n >= self.plan["stop_after"]:
            trainer.should_stop = True

    def on_train_end(self, trainer, module):
        assert trainer.global_step == self.plan["stop_after"]
        trainer.save_checkpoint(self.output / "training_resume.ckpt")
        if trainer.is_global_zero:
            records = []
            for step in self.plan["save_steps"]:
                path = self.output / f"lewm_full_ablation_step_{step}_object.ckpt"
                assert path.is_file(), path
                records.append(dict(step=step, checkpoint=path.name, sha256=sha256(path)))
            write_json(self.output / "TRAIN_COMPLETE.json", dict(actual_updates=trainer.global_step,
                       checkpoints=records, encoder_seed=self.plan["seed"], arm=self.plan["arm"]))
