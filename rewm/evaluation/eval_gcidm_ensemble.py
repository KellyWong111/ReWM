#!/usr/bin/env python3
"""Evaluate a frozen LeWM encoder with an audited ensemble of GC-IDMs.

This evaluator intentionally contains no CEM path.  It preserves the canonical
PushT sampler and success predicate from eval_gcidm_planner.py while adding two
controller-side operations: model aggregation and use of overlapping actions
already predicted by a multi-action GC-IDM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import stable_worldmodel as swm

from eval_gcidm_planner import (
    DATASET_CONFIGS,
    GoalConditionedPolicy,
    build_normalizers,
    get_img_transform,
)
from gcidm.dataset import load_lewm_model, select_encoder_readout
from train_gcidm_planner import load_idm


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def sample_eval_episodes_vectorized(
    dataset,
    num_eval: int,
    goal_offset: int,
    seed: int = 42,
    train_split: float = 1.0,
    split_seed: int = 42,
):
    """Bit-identical, vectorized form of the canonical GC-IDM sampler.

    The canonical implementation scans all dataset rows once per episode to
    obtain episode lengths.  PushT has 18,685 episodes and 2.34M rows, so that
    quadratic-style Python loop dominates repeated validation runs.  This
    version preserves the original valid-index ordering and, intentionally,
    its ``rng.choice(len(valid_indices) - 1, ...)`` sampling convention.
    """
    col_name = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    ep_indices, inverse = np.unique(episode_idx, return_inverse=True)

    max_step = np.full(len(ep_indices), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(max_step, inverse, np.asarray(step_idx, dtype=np.int64))
    max_start_per_row = max_step[inverse] - int(goal_offset)
    valid_mask = step_idx <= max_start_per_row

    if train_split < 1.0:
        n_holdout = max(1, round(len(ep_indices) * (1.0 - train_split)))
        rng_split = np.random.default_rng(split_seed)
        holdout_eps = rng_split.choice(ep_indices, size=n_holdout, replace=False)
        valid_mask = valid_mask & np.isin(episode_idx, holdout_eps)
        print(
            f"  Eval restricted to {len(holdout_eps)} held-out episodes "
            f"(split_seed={split_seed}, train_split={train_split})"
        )

    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No valid eval starts (goal_offset={goal_offset}, "
            f"train_split={train_split})"
        )

    rng = np.random.default_rng(seed)
    n = min(num_eval, len(valid_indices) - 1)
    if n < num_eval:
        print(
            f"  WARNING: reduced num_eval {num_eval} -> {n} "
            f"(only {len(valid_indices)} valid starts)"
        )
    sampled = rng.choice(len(valid_indices) - 1, size=n, replace=False)
    sampled = np.sort(valid_indices[sampled])
    rows = dataset.get_row_data(sampled)
    return rows[col_name], rows["step_idx"]


class AggregatedIDM(torch.nn.Module):
    """Aggregate identically shaped action blocks from frozen GC-IDMs."""

    def __init__(self, models: list[torch.nn.Module], method: str):
        super().__init__()
        if not models:
            raise ValueError("at least one GC-IDM is required")
        self.models = torch.nn.ModuleList(models)
        self.method = method
        horizons = {int(model.max_horizon) for model in models}
        if len(horizons) != 1:
            raise ValueError(f"GC-IDM max_horizon mismatch: {horizons}")
        self.max_horizon = horizons.pop()

    def forward(self, z_t, z_goal, steps_remaining):
        predictions = torch.stack(
            [model(z_t, z_goal, steps_remaining) for model in self.models], dim=0
        )
        if self.method == "mean":
            return predictions.mean(dim=0)
        if self.method == "median":
            ordered = predictions.sort(dim=0).values
            count = ordered.shape[0]
            if count % 2:
                return ordered[count // 2]
            return 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])
        if self.method == "trimmed":
            if predictions.shape[0] < 3:
                return predictions.mean(dim=0)
            ordered = predictions.sort(dim=0).values
            return ordered[1:-1].mean(dim=0)
        raise ValueError(f"unknown aggregation method: {self.method}")


class OverlapAwarePolicy(GoalConditionedPolicy):
    """Replan every step while optionally averaging overlapping action blocks."""

    def __init__(
        self,
        *args,
        temporal: str,
        temporal_decay: float,
        horizon_mode: str,
        goal_horizon: int,
        action_clamp: str,
        record_trace: bool = False,
        recovery_idm: torch.nn.Module | None = None,
        stall_window: int | None = None,
        stall_epsilon: float | None = None,
        recovery_deadline: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temporal = temporal
        self.temporal_decay = float(temporal_decay)
        self.horizon_mode = horizon_mode
        self.goal_horizon = int(goal_horizon)
        self.action_clamp = action_clamp
        self.record_trace = bool(record_trace)
        self.recovery_idm = recovery_idm
        self.stall_window = stall_window
        self.stall_epsilon = stall_epsilon
        self.recovery_deadline = recovery_deadline
        self._block_history: list[torch.Tensor] = []
        self.action_values = 0
        self.action_values_outside_bounds = 0
        self.max_abs_action = 0.0
        self.trace_latent_distance: list[list[float]] = []
        self.trace_horizon: list[int] = []
        self.trace_action_norm: list[list[float]] = []
        self._initial_distance: torch.Tensor | None = None
        self._distance_history: list[torch.Tensor] = []
        self._recovery_mask: torch.Tensor | None = None
        self.recovery_trigger_step: list[int] | None = None

    def _remaining(self) -> int:
        if self.horizon_mode == "budget":
            value = self.eval_budget - self._step_count
        elif self.horizon_mode == "goal":
            value = self.goal_horizon - self._step_count
        elif self.horizon_mode == "fixed25":
            value = self.goal_horizon
        else:
            raise ValueError(self.horizon_mode)
        return max(1, min(int(value), int(self.idm.max_horizon)))

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        del kwargs
        t0 = time.perf_counter()
        info_dict = self._prepare_info(info_dict)
        pixels = info_dict["pixels"][:, -1].to(self.device)
        goal = info_dict["goal"][:, -1].to(self.device)

        enc_out = self.jepa.encoder(pixels, interpolate_pos_encoding=True)
        z_current = self.jepa.projector(select_encoder_readout(enc_out))

        if (
            self.cache_goal_encoding
            and self._cached_goal_input is not None
            and self._cached_goal_input.shape == goal.shape
            and torch.equal(self._cached_goal_input, goal)
        ):
            z_goal = self._cached_z_goal
        else:
            enc_out_g = self.jepa.encoder(goal, interpolate_pos_encoding=True)
            z_goal = self.jepa.projector(select_encoder_readout(enc_out_g))
            if self.cache_goal_encoding:
                self._cached_goal_input = goal
                self._cached_z_goal = z_goal

        remaining = self._remaining()
        steps_remaining = torch.full(
            (z_current.shape[0],), remaining, device=self.device, dtype=torch.long
        )
        flat = self.idm(z_current, z_goal, steps_remaining)
        block = flat.reshape(z_current.shape[0], -1, self.raw_action_dim)

        latent_distance = torch.linalg.vector_norm(z_current - z_goal, dim=-1)
        if self.recovery_idm is not None:
            if self._initial_distance is None:
                self._initial_distance = latent_distance.detach().clamp_min(1e-12)
                self._recovery_mask = torch.zeros_like(latent_distance, dtype=torch.bool)
                self.recovery_trigger_step = [-1] * int(latent_distance.shape[0])
            normalized_distance = latent_distance / self._initial_distance
            self._distance_history.append(normalized_distance.detach())
            assert self.stall_window is not None
            assert self.stall_epsilon is not None
            assert self.recovery_deadline is not None
            stall = torch.zeros_like(latent_distance, dtype=torch.bool)
            if len(self._distance_history) > self.stall_window:
                improvement = (
                    self._distance_history[-self.stall_window - 1]
                    - self._distance_history[-1]
                )
                stall = improvement <= self.stall_epsilon
            deadline = torch.full_like(
                stall, self._step_count >= self.recovery_deadline
            )
            trigger = stall | deadline
            assert self._recovery_mask is not None
            newly_triggered = trigger & ~self._recovery_mask
            if bool(newly_triggered.any()):
                assert self.recovery_trigger_step is not None
                for index in newly_triggered.nonzero(as_tuple=False).flatten().tolist():
                    self.recovery_trigger_step[int(index)] = int(self._step_count)
            self._recovery_mask = self._recovery_mask | trigger
            recovery_remaining = max(
                1,
                min(
                    int(self.eval_budget - self._step_count),
                    int(self.recovery_idm.max_horizon),
                ),
            )
            recovery_steps = torch.full(
                (z_current.shape[0],), recovery_remaining,
                device=self.device, dtype=torch.long,
            )
            recovery_flat = self.recovery_idm(
                z_current, z_goal, recovery_steps
            )
            recovery_block = recovery_flat.reshape(
                z_current.shape[0], -1, self.raw_action_dim
            )
            block = torch.where(
                self._recovery_mask.view(-1, 1, 1), recovery_block, block
            )

        candidates = [block[:, 0]]
        weights = [1.0]
        if self.temporal != "none":
            for lag, old_block in enumerate(reversed(self._block_history), start=1):
                if lag >= old_block.shape[1]:
                    break
                candidates.append(old_block[:, lag])
                if self.temporal == "uniform":
                    weights.append(1.0)
                elif self.temporal == "exp":
                    weights.append(self.temporal_decay**lag)
                else:
                    raise ValueError(self.temporal)
        action_stack = torch.stack(candidates, dim=0)
        weight_tensor = torch.as_tensor(
            weights, dtype=action_stack.dtype, device=action_stack.device
        ).view(-1, 1, 1)
        action = (action_stack * weight_tensor).sum(dim=0) / weight_tensor.sum()

        if self.record_trace:
            action_norm = torch.linalg.vector_norm(action, dim=-1)
            self.trace_latent_distance.append(
                latent_distance.detach().float().cpu().tolist()
            )
            self.trace_horizon.append(int(remaining))
            self.trace_action_norm.append(action_norm.detach().float().cpu().tolist())

        max_history = max(0, block.shape[1] - 1)
        self._block_history.append(block.detach())
        if max_history == 0:
            self._block_history = []
        elif len(self._block_history) > max_history:
            self._block_history = self._block_history[-max_history:]

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._plan_times.append(time.perf_counter() - t0)
        self._step_count += 1
        self.action_values += int(action.numel())
        self.action_values_outside_bounds += int((action.abs() > 1.0).sum().item())
        self.max_abs_action = max(self.max_abs_action, float(action.abs().max().item()))
        if self.action_clamp == "unit":
            action = action.clamp(-1.0, 1.0)
        elif self.action_clamp != "none":
            raise ValueError(self.action_clamp)
        return action.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--idms", nargs="+", required=True)
    parser.add_argument("--aggregate", choices=("mean", "median", "trimmed"), default="mean")
    parser.add_argument("--temporal", choices=("none", "uniform", "exp"), default="none")
    parser.add_argument("--temporal-decay", type=float, default=0.5)
    parser.add_argument("--horizon-mode", choices=("budget", "goal", "fixed25"), default="budget")
    parser.add_argument("--action-clamp", choices=("none", "unit"), default="unit")
    parser.add_argument("--dataset", choices=("pusht",), default="pusht")
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--recovery-idm")
    parser.add_argument("--stall-window", type=int)
    parser.add_argument("--stall-epsilon", type=float)
    parser.add_argument("--recovery-deadline", type=int)
    args = parser.parse_args()

    if args.goal_offset != 25 or args.eval_budget != 50 or args.num_eval != 100:
        raise ValueError("locked protocol requires goal_offset=25, eval_budget=50, num_eval=100")
    if not (0.0 < args.temporal_decay <= 1.0):
        raise ValueError("temporal decay must be in (0, 1]")
    hybrid_values = (
        args.recovery_idm, args.stall_window,
        args.stall_epsilon, args.recovery_deadline,
    )
    hybrid_enabled = args.recovery_idm is not None
    if hybrid_enabled != all(value is not None for value in hybrid_values):
        raise ValueError("hybrid recovery arguments must be provided together")
    if hybrid_enabled:
        if args.horizon_mode != "goal" or args.temporal != "none":
            raise ValueError("hybrid requires primary goal horizon and no temporal overlap")
        if args.stall_window < 1 or args.stall_epsilon < 0.0:
            raise ValueError("invalid stall rule")
        if not 1 <= args.recovery_deadline < args.eval_budget:
            raise ValueError("recovery deadline must be inside the evaluation budget")

    dcfg = DATASET_CONFIGS[args.dataset]
    device = torch.device(args.device)
    data_root = os.environ["STABLEWM_HOME"]
    dataset = swm.data.HDF5Dataset(dcfg.dataset_name, keys_to_cache=dcfg.cache_keys)
    img_tf = get_img_transform(dcfg.img_size)
    transform = {"pixels": img_tf, "goal": img_tf}
    process = build_normalizers(dataset, dcfg.cache_keys)
    eval_episodes, eval_start_idx = sample_eval_episodes_vectorized(
        dataset, args.num_eval, args.goal_offset, args.seed,
        train_split=1.0, split_seed=42,
    )

    model = load_lewm_model(args.checkpoint, str(device))
    jepa = model.model if hasattr(model, "model") else model
    jepa.eval().requires_grad_(False)
    idm_models = [load_idm(path, str(device)) for path in args.idms]
    output_dims = {
        int(model.cfg.action_dim * model.cfg.frameskip) for model in idm_models
    }
    if len(output_dims) != 1:
        raise ValueError(f"action-block shape mismatch across ensemble: {output_dims}")
    aggregate = AggregatedIDM(idm_models, args.aggregate).to(device).eval()
    recovery_idm = None
    if hybrid_enabled:
        recovery_idm = load_idm(args.recovery_idm, str(device)).to(device).eval()
        if int(aggregate.max_horizon) != 25 or int(recovery_idm.max_horizon) != 50:
            raise ValueError("hybrid requires H25 primary and H50 recovery readers")

    policy = OverlapAwarePolicy(
        jepa_model=jepa,
        idm=aggregate,
        eval_budget=args.eval_budget,
        process=process,
        transform=transform,
        device=device,
        cache_goal_encoding=True,
        raw_action_dim=dcfg.raw_action_dim,
        temporal=args.temporal,
        temporal_decay=args.temporal_decay,
        horizon_mode=args.horizon_mode,
        goal_horizon=args.goal_offset,
        action_clamp=args.action_clamp,
        record_trace=args.record_trace,
        recovery_idm=recovery_idm,
        stall_window=args.stall_window,
        stall_epsilon=args.stall_epsilon,
        recovery_deadline=args.recovery_deadline,
    )
    world = swm.World(
        env_name=dcfg.env,
        num_envs=args.num_eval,
        image_shape=(dcfg.img_size, dcfg.img_size),
        max_episode_steps=2 * max(args.eval_budget, args.goal_offset) + 5,
        **(dcfg.world_kwargs or {}),
    )
    world.set_policy(policy)

    dummy = torch.randn(min(args.num_eval, 8), 3, dcfg.img_size, dcfg.img_size, device=device)
    with torch.no_grad():
        for _ in range(3):
            jepa.encoder(dummy, interpolate_pos_encoding=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del dummy

    ep_arr = np.asarray(eval_episodes.tolist())
    start_arr = np.asarray(eval_start_idx.tolist())
    _ = dataset.load_chunk(ep_arr, start_arr, start_arr + args.goal_offset + 1)
    del _

    started = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=args.goal_offset,
        eval_budget=args.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=dcfg.callables,
        save_video=False,
    )
    elapsed = time.time() - started
    result = {
        "label": args.label,
        "status": "complete",
        "protocol": {
            "dataset": args.dataset,
            "num_eval": args.num_eval,
            "goal_offset": args.goal_offset,
            "eval_budget": args.eval_budget,
            "rollout_seed": args.seed,
        },
        "controller": {
            "idms": [str(Path(path).resolve()) for path in args.idms],
            "idm_sha256": [sha256(path) for path in args.idms],
            "aggregate": args.aggregate,
            "temporal": args.temporal,
            "temporal_decay": args.temporal_decay,
            "horizon_mode": args.horizon_mode,
            "action_clamp": args.action_clamp,
        },
        "encoder": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "sha256": sha256(args.checkpoint),
        },
        "metrics": jsonable(metrics),
        "elapsed_seconds": elapsed,
        "avg_plan_time_ms": policy.avg_plan_time_ms,
        "action_diagnostics": {
            "values": policy.action_values,
            "outside_unit_bounds": policy.action_values_outside_bounds,
            "outside_unit_fraction": (
                policy.action_values_outside_bounds / max(policy.action_values, 1)
            ),
            "max_abs_before_clamp": policy.max_abs_action,
        },
        "eval_episodes": jsonable(eval_episodes),
        "eval_start_idx": jsonable(eval_start_idx),
    }
    if args.record_trace:
        result["trace"] = {
            "latent_distance": policy.trace_latent_distance,
            "horizon": policy.trace_horizon,
            "action_norm": policy.trace_action_norm,
        }
    if hybrid_enabled:
        result["controller"]["hybrid_recovery"] = {
            "recovery_idm": str(Path(args.recovery_idm).resolve()),
            "recovery_idm_sha256": sha256(args.recovery_idm),
            "stall_window": args.stall_window,
            "stall_epsilon": args.stall_epsilon,
            "recovery_deadline": args.recovery_deadline,
            "primary_horizon": "goal",
            "recovery_horizon": "budget",
            "latched": True,
            "episode_specific_branching": False,
        }
        result["recovery_trigger_step"] = policy.recovery_trigger_step
    output = Path(args.result_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    success = float(result["metrics"]["success_rate"])
    print(f"GC-IDM-ENSEMBLE {success:.1f}%")
    print(f"RESULT_JSON={output}")


if __name__ == "__main__":
    main()
