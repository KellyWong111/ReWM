import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

from release_hooks import build_arm, install_schedule, ExactObjects, ProtocolAudit
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

def train(cfg, protocol, plan, output, anchor=None):
    """Original training computation with local input/output and protocol guards."""
    schedule_name, schedule_interval = install_schedule(protocol)
    pl.seed_everything(cfg.seed, workers=True)
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    # Preserve the HDF5 loader used by the experiments and its sequence schema.
    dataset_cfg.pop("name", None)
    cache_dir = os.environ.get("STABLEWM_HOME", None) or os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.HDF5Dataset(
        name=cfg.data.dataset.name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = build_arm(hydra.utils.instantiate(cfg.model), plan["arm"], int(cfg.seed), output, anchor)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": schedule_name},
            "interval": schedule_interval,
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(output)
    cfg.subdir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ExactObjects(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1, step_interval=1,
        steps_per_epoch=protocol["steps_per_epoch"],
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, ProtocolAudit(protocol, plan, output)],
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=False,
    )

    manager = spt.Manager(
        seed=int(cfg.seed),
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=None,
    )

    manager()
    return

