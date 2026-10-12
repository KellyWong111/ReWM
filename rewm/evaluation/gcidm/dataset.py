"""Embedding extraction and dataset for IDM training."""

from __future__ import annotations

import os
import sys
import time
import types
import json
from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401 - registers compressed HDF5 filters
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


class _PreProjectorNormalizer(nn.Module):
    """Frozen encoder-readout calibration inserted immediately before P(h)."""

    def __init__(self, projector: nn.Module, center, scale):
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float32).reshape(-1)
        scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if scale.numel() not in {1, center.numel()}:
            raise ValueError(
                f"interface scale has {scale.numel()} values for {center.numel()} dimensions"
            )
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("non-finite pre-projector interface statistics")
        if (scale <= 0).any():
            raise ValueError("pre-projector interface scale must be positive")
        self.projector = projector
        self.register_buffer("interface_center", center)
        self.register_buffer("interface_scale", scale)

    def forward(self, h):
        return self.projector((h - self.interface_center) / self.interface_scale)


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        if not part:
            continue
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _apply_interface_spec(model: nn.Module) -> nn.Module:
    spec_path = os.environ.get("GCIDM_INTERFACE_SPEC", "")
    if not spec_path:
        return model
    with open(spec_path) as handle:
        spec = json.load(handle)
    if spec.get("version") != 1 or spec.get("location") != "pre_projector":
        raise ValueError(f"unsupported GC-IDM interface spec: {spec_path}")
    jepa = model.model if hasattr(model, "model") else model
    projector = jepa.projector
    bn_state = spec.get("batchnorm_recalibration")
    if bn_state is not None:
        bn = _resolve_module(projector, bn_state["module_path"])
        if not isinstance(bn, nn.BatchNorm1d):
            raise TypeError(
                f"interface spec target {bn_state['module_path']} is {type(bn).__name__}, "
                "not BatchNorm1d"
            )
        mean = torch.as_tensor(bn_state["running_mean"], dtype=bn.running_mean.dtype)
        var = torch.as_tensor(bn_state["running_var"], dtype=bn.running_var.dtype)
        if mean.shape != bn.running_mean.shape or var.shape != bn.running_var.shape:
            raise ValueError("recalibrated BatchNorm statistics have the wrong shape")
        bn.running_mean.copy_(mean.to(bn.running_mean.device))
        bn.running_var.copy_(var.to(bn.running_var.device))
        bn.num_batches_tracked.fill_(int(bn_state.get("num_batches_tracked", 1)))
    jepa.projector = _PreProjectorNormalizer(
        projector, spec["center"], spec["scale"],
    ).to(next(jepa.parameters()).device)
    jepa.projector.eval()
    print(
        f"Applied frozen pre-projector interface: {spec_path} "
        f"mode={spec.get('mode')} bn_recalibrated={bn_state is not None}",
        flush=True,
    )
    return model


def select_encoder_readout(output):
    """Return the encoder global state according to GCIDM_READOUT.

    auto: use pooled_state when available, otherwise CLS.
    cls: force last_hidden_state[:, 0].
    pooled: require output.pooled_state.
    L{n}: read intermediate layer n (0-indexed) from hidden_states if available.
           Uses CLS token (index 0) of that layer's hidden state.
    """
    readout = os.environ.get("GCIDM_READOUT", "auto").lower()
    if readout == "cls":
        return output.last_hidden_state[:, 0]
    if readout == "pooled":
        if not hasattr(output, "pooled_state"):
            raise AttributeError("GCIDM_READOUT=pooled but encoder output has no pooled_state")
        return output.pooled_state
    if readout.startswith("l") and readout[1:].isdigit():
        layer_idx = int(readout[1:])
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise AttributeError(f"GCIDM_READOUT={readout} but encoder output has no hidden_states")
        if layer_idx >= len(hidden_states):
            raise IndexError(f"GCIDM_READOUT={readout} but only {len(hidden_states)} layers available")
        return hidden_states[layer_idx][:, 0]
    if readout != "auto":
        raise ValueError(f"Unsupported GCIDM_READOUT={readout!r}; expected auto, cls, pooled, or L{n}")
    return output.pooled_state if hasattr(output, "pooled_state") else output.last_hidden_state[:, 0]


class _CompatLeWM(nn.Module):
    """Compatibility wrapper for historical stable_worldmodel.wm.lewm.LeWM configs."""

    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()


def _install_lewm_compat() -> None:
    """Make downloaded LeWM configs importable in the current LeMo layout."""
    candidates = [
        os.environ.get("LEWM_CODE_DIR"),
        os.environ.get("LEWM_ROOT") and str(Path(os.environ["LEWM_ROOT"]) / "code"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        import stable_worldmodel.wm as wm
        import module as lemo_module
    except Exception:
        return
    lewm_mod = sys.modules.get("stable_worldmodel.wm.lewm")
    if lewm_mod is None:
        lewm_mod = types.ModuleType("stable_worldmodel.wm.lewm")
        sys.modules["stable_worldmodel.wm.lewm"] = lewm_mod
    lewm_mod.LeWM = _CompatLeWM
    setattr(wm, "lewm", lewm_mod)
    module_mod = sys.modules.get("stable_worldmodel.wm.lewm.module")
    if module_mod is None:
        module_mod = types.ModuleType("stable_worldmodel.wm.lewm.module")
        sys.modules["stable_worldmodel.wm.lewm.module"] = module_mod
    module_mod.Predictor = lemo_module.ARPredictor
    module_mod.ARPredictor = lemo_module.ARPredictor
    module_mod.Embedder = lemo_module.Embedder
    module_mod.MLP = lemo_module.MLP
    setattr(lewm_mod, "module", module_mod)


class EmbeddingTripleDataset(Dataset):
    """Dataset of (z_t, z_goal, steps_remaining, a_t) for GC-IDM training."""

    def __init__(
        self,
        embeddings_path: str,
        max_goal_horizon: int = 50,
        frameskip: int = 1,
        train_split: float = 1.0,
        split_seed: int = 42,
        split_partition: str = "train",
    ):
        """
        Args:
            embeddings_path: path to .npz file from extract_embeddings()
            max_goal_horizon: maximum horizon for goal sampling
            frameskip: number of raw steps between paired frames.
            train_split: fraction of episodes to use for training (1.0 = all).
                Set to 0.9 to hold out 10% of episodes.
            split_seed: RNG seed for episode-level split (default 42, matches paper).
            split_partition: "train" or "val" partition to use.
        """
        data = np.load(embeddings_path)
        self.embeddings = data["embeddings"]    # (N, D)
        self.actions = data["actions"]          # (N, act_dim)
        self.ep_ids = data["episode_ids"]       # (N,)
        self.max_goal_horizon = max_goal_horizon
        self.frameskip = frameskip

        # Episode-level split: filter frames to train/val episodes
        if train_split < 1.0:
            unique_eps = np.unique(self.ep_ids)
            n_holdout = max(1, round(len(unique_eps) * (1.0 - train_split)))
            rng = np.random.default_rng(split_seed)
            holdout_eps = set(rng.choice(unique_eps, size=n_holdout, replace=False).tolist())
            if split_partition == "train":
                keep_eps = set(unique_eps.tolist()) - holdout_eps
            else:
                keep_eps = holdout_eps
            ep_mask = np.isin(self.ep_ids, list(keep_eps))
            self.held_out_episodes = sorted(holdout_eps)
            print(f"  Episode split: {len(keep_eps)}/{len(unique_eps)} episodes "
                  f"({ep_mask.sum()}/{len(ep_mask)} frames, partition={split_partition})")
            print(f"  Held-out episode IDs ({len(self.held_out_episodes)}): {self.held_out_episodes}")
        else:
            ep_mask = np.ones(len(self.ep_ids), dtype=bool)
            self.held_out_episodes = []

        # Build valid pair indices
        n = len(self.ep_ids)
        same_ep = self.ep_ids[:-frameskip] == self.ep_ids[frameskip:]
        if frameskip == 1:
            no_nan = ~np.isnan(self.actions[:-frameskip]).any(axis=1)
        else:
            nan_mask = np.isnan(self.actions).any(axis=1)
            kernel = np.ones(frameskip, dtype=bool)
            has_nan = np.convolve(nan_mask, kernel, mode='valid')[:n - frameskip] > 0
            no_nan = ~has_nan
        valid = same_ep & no_nan & ep_mask[:-frameskip]
        self.valid_indices: list[int] = np.nonzero(valid)[0].tolist()

        self._build_episode_map()

    def _build_episode_map(self):
        """Map each episode ID to its frame range for goal sampling."""
        self.ep_ranges: dict[int, tuple[int, int]] = {}
        current_ep = self.ep_ids[0]
        start = 0
        for i in range(1, len(self.ep_ids)):
            if self.ep_ids[i] != current_ep:
                self.ep_ranges[current_ep] = (start, i)
                current_ep = self.ep_ids[i]
                start = i
        self.ep_ranges[current_ep] = (start, len(self.ep_ids))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        i = self.valid_indices[idx]
        k = self.frameskip
        z_t = torch.from_numpy(self.embeddings[i]).float()
        a_t = torch.from_numpy(self.actions[i : i + k].reshape(-1)).float()

        # Sample a future frame as goal
        ep = self.ep_ids[i]
        ep_start, ep_end = self.ep_ranges[ep]
        max_future = min(i + self.max_goal_horizon, ep_end - 1)

        if max_future <= i + 1:
            goal_idx = i + 1
        else:
            goal_idx = np.random.randint(i + 1, max_future + 1)

        z_goal = torch.from_numpy(self.embeddings[goal_idx]).float()
        steps_remaining = torch.tensor(goal_idx - i, dtype=torch.long)

        return {
            "z_t": z_t,
            "z_goal": z_goal,
            "steps_remaining": steps_remaining,
            "action": a_t,
        }


def load_lewm_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LeWM model from any supported format.

    Supports:
      1. *_object.ckpt — torch.save'd full object (standard le-wm output)
      2. Directory with config.json + weights.pt (HuggingFace download)
      3. HuggingFace repo ID like 'quentinll/lewm-tworooms' (auto-downloads)

    Returns the raw model (LeWM or spt.Module wrapper), moved to device.
    """
    from pathlib import Path

    path = Path(checkpoint_path)

    # Case 1: _object.ckpt file
    if path.is_file() and path.suffix == ".ckpt":
        print(f"Loading _object.ckpt: {path}")
        model = torch.load(str(path), map_location=device, weights_only=False)
        model.eval()
        return _apply_interface_spec(model)

    # Case 2: directory with config.json + weights.pt
    if path.is_dir():
        cfg_file = path / "config.json"
        wts_file = path / "weights.pt"
        if cfg_file.exists() and wts_file.exists():
            print(f"Loading from config.json + weights.pt in {path}")
            return _apply_interface_spec(
                _load_from_config_weights(str(cfg_file), str(wts_file), device)
            )

    # Case 3: HuggingFace repo ID (contains '/')
    if "/" in checkpoint_path and not path.exists():
        print(f"Downloading from HuggingFace: {checkpoint_path}")
        from huggingface_hub import hf_hub_download
        cfg_file = hf_hub_download(checkpoint_path, "config.json")
        wts_file = hf_hub_download(checkpoint_path, "weights.pt")
        return _apply_interface_spec(_load_from_config_weights(cfg_file, wts_file, device))

    # Case 4: maybe it's a glob pattern or directory with _object.ckpt inside
    if path.is_dir():
        ckpts = sorted(path.glob("*_object.ckpt"))
        if ckpts:
            print(f"Loading latest _object.ckpt from {path}: {ckpts[-1].name}")
            model = torch.load(str(ckpts[-1]), map_location=device, weights_only=False)
            model.eval()
            return _apply_interface_spec(model)

    raise FileNotFoundError(
        f"Cannot load checkpoint from '{checkpoint_path}'. "
        f"Expected: _object.ckpt file, dir with config.json+weights.pt, or HF repo ID."
    )


def _load_from_config_weights(
    cfg_path: str, wts_path: str, device: str
) -> torch.nn.Module:
    """Reconstruct a LeWM model from config.json + weights.pt (HuggingFace format)."""
    import json
    from hydra.utils import instantiate

    _install_lewm_compat()
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = instantiate(cfg)
    state_dict = torch.load(wts_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"  Model: {type(model).__name__}, params: {sum(p.numel() for p in model.parameters()):,}")
    return model


@torch.no_grad()
def extract_embeddings(
    checkpoint_path: str,
    h5_path: str,
    output_path: str,
    img_size: int = 224,
    batch_size: int = 2048,
    num_prefetch: int = 12,
    device: str = "cuda:0",
    frame_start: int = 0,
    frame_end: int | None = None,
) -> str:
    """Extract LeWM embeddings from all frames in an HDF5 dataset.

    Loads the trained LeWM model, encodes every frame through the frozen
    encoder + projector, and saves the result alongside actions and episode IDs.

    Args:
        checkpoint_path: path to checkpoint (_object.ckpt, dir, or HF repo ID)
        h5_path: path to the HDF5 dataset (e.g., pusht_expert_train.h5)
        output_path: where to save the .npz file
        img_size: image size for normalization
        batch_size: encoding batch size
        num_prefetch: number of batches to prefetch from HDF5 in background
        device: torch device

    Returns:
        output_path
    """
    from concurrent.futures import ThreadPoolExecutor

    model = load_lewm_model(checkpoint_path, device)

    # Access the encoder and projector from the JEPA model
    # The spt.Module wraps: model.model = JEPA(encoder, predictor, ...)
    jepa = model.model if hasattr(model, "model") else model
    encoder = jepa.encoder
    projector = jepa.projector
    extract_stage = os.environ.get("GCIDM_EXTRACT_STAGE", "projector")
    if extract_stage == "encoder":
        projector = nn.Identity()
    elif extract_stage != "projector":
        raise ValueError(
            f"unsupported GCIDM_EXTRACT_STAGE={extract_stage!r}; expected encoder|projector"
        )
    print(f"  Embedding extraction stage: {extract_stage}")

    # Wrap encoder+projector for multi-GPU if available
    class _EncoderProjector(torch.nn.Module):
        def __init__(self, enc, proj):
            super().__init__()
            self.enc = enc
            self.proj = proj
        def forward(self, x):
            out = self.enc(x, interpolate_pos_encoding=True)
            return self.proj(select_encoder_readout(out))

    enc_proj = _EncoderProjector(encoder, projector).to(device)
    enc_proj.eval()

    # The launcher pins one GPU via CUDA_VISIBLE_DEVICES; DataParallel engages only when more are visible.
    n_gpus = torch.cuda.device_count()
    use_data_parallel = n_gpus > 1 and device != "cpu"
    data_parallel_ready = False

    print(f"Loading HDF5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        h5_keys = list(f.keys())
        print(f"  HDF5 keys: {h5_keys}")

        pix_dset = f["pixels"]
        total_frames = int(pix_dset.shape[0])
        frame_start = int(frame_start)
        frame_end = total_frames if frame_end is None else int(frame_end)
        if not 0 <= frame_start < frame_end <= total_frames:
            raise ValueError(
                f"invalid frame range [{frame_start}, {frame_end}) for {total_frames} frames"
            )

        # Slice frame-aligned arrays so isolated single-GPU processes can
        # extract disjoint shards without duplicating output rows.
        action_key = "actions" if "actions" in f else "action"
        actions = f[action_key][frame_start:frame_end]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

        if "ep_idx" in f:
            episode_ids = f["ep_idx"][frame_start:frame_end].astype(np.int64)
        elif "episode_idx" in f:
            episode_ids = f["episode_idx"][frame_start:frame_end].astype(np.int64)
        else:
            full_episode_ids = np.zeros(total_frames, dtype=np.int64)
            for ei in range(len(ep_len)):
                start = ep_offset[ei]
                end = start + ep_len[ei]
                full_episode_ids[start:end] = ei
            episode_ids = full_episode_ids[frame_start:frame_end]

        state = f["state"][frame_start:frame_end] if "state" in f else None
        proprio = f["proprio"][frame_start:frame_end] if "proprio" in f else None

        # Pixel metadata — don't load pixels yet
        n_frames = frame_end - frame_start
        pix_shape = pix_dset.shape
        is_hwc = (pix_dset.ndim == 4 and pix_dset.shape[-1] == 3)
        is_chw = (pix_dset.ndim == 4 and pix_dset.shape[1] == 3)
        print(f"  Pixel shape: {pix_shape} ({'HWC' if is_hwc else 'CHW' if is_chw else '???'})")

        if not (is_hwc or is_chw):
            raise ValueError(f"Unexpected pixel shape: {pix_shape}")

        # Normalize pixels: uint8 → float32, apply ImageNet-style normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        all_embeddings = []
        n_batches = (n_frames + batch_size - 1) // batch_size

        # Prefetch: read HDF5 batches in background threads while GPU encodes
        def _read_batch(batch_idx):
            s = frame_start + batch_idx * batch_size
            e = min(s + batch_size, frame_end)
            pix = pix_dset[s:e]
            if is_hwc:
                pix = np.transpose(pix, (0, 3, 1, 2))
            return pix

        print(f"Encoding frame range [{frame_start}, {frame_end}) "
              f"({n_frames} frames) in {n_batches} batches "
              f"(batch_size={batch_size}, prefetch={num_prefetch})...")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=num_prefetch) as pool:
            # Submit first N batches
            futures = {}
            for i in range(min(num_prefetch, n_batches)):
                futures[i] = pool.submit(_read_batch, i)

            for b in range(n_batches):
                if (b + 1) % 20 == 0 or b == n_batches - 1:
                    elapsed = time.time() - t0
                    fps = (b + 1) * batch_size / max(elapsed, 1e-6)
                    eta = (n_batches - b - 1) * elapsed / max(b + 1, 1)
                    print(f"  batch {b + 1}/{n_batches}  ({fps:.0f} frames/s, ETA {eta:.0f}s)")

                # Get prefetched pixels
                batch_pix = futures[b].result()

                # Submit next batch for prefetch
                next_b = b + num_prefetch
                if next_b < n_batches:
                    futures[next_b] = pool.submit(_read_batch, next_b)

                # Clean up used future
                del futures[b]

                batch_pix = torch.from_numpy(batch_pix).float().to(device) / 255.0
                batch_pix = (batch_pix - mean) / std

                # Some custom Triton kernels (notably the MoDA RMSNorm) keep
                # mutable autotuner state.  Let one replica compile/tune every
                # kernel before DataParallel invokes replicas concurrently.
                # The warm-up uses the exact per-replica shape of the real
                # DataParallel call, so the cached configuration is reusable.
                if use_data_parallel and not data_parallel_ready:
                    per_replica = max(1, (len(batch_pix) + n_gpus - 1) // n_gpus)
                    print(
                        f"  Warming up custom kernels on one GPU "
                        f"(per-replica batch={per_replica})...",
                        flush=True,
                    )
                    enc_proj(batch_pix[:per_replica])
                    torch.cuda.synchronize(device)
                    enc_proj = torch.nn.DataParallel(enc_proj)
                    data_parallel_ready = True
                    print(f"  Using DataParallel on {n_gpus} GPUs", flush=True)

                emb = enc_proj(batch_pix)
                all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_dict = dict(
        embeddings=embeddings,
        actions=actions.astype(np.float32),
        episode_ids=episode_ids,
        embed_dim=embeddings.shape[1],
        frame_start=np.array(frame_start, dtype=np.int64),
        frame_end=np.array(frame_end, dtype=np.int64),
        total_frames=np.array(total_frames, dtype=np.int64),
    )
    if state is not None:
        save_dict["state"] = state.astype(np.float32)
    if proprio is not None:
        save_dict["proprio"] = proprio.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved: {output_path}")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  actions:    {actions.shape}")
    if state is not None:
        print(f"  state:      {state.shape}")
    if proprio is not None:
        print(f"  proprio:    {proprio.shape}")
    print(f"  episodes:   {len(ep_len)}")
    return output_path
