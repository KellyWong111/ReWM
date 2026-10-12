#!/usr/bin/env python3
"""Auditable frozen-GC-IDM Puzzle rollout with optional action ensembling.

The task, start/goal construction, action horizon, physical success predicate,
and image transform are the historical Puzzle-3x3 protocol.  Multiple IDMs
only change how their predicted action blocks are aggregated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gcidm.dataset import select_encoder_readout
from train_gcidm_planner import load_idm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--idm", action="append", required=True)
    parser.add_argument("--aggregation", choices=("mean", "median"), default="mean")
    parser.add_argument(
        "--npz",
        required=True,
    )
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--frameskip", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import ogbench
    if args.num_eval not in (1,50) or args.goal_offset != 25 or args.eval_budget != 50:
        raise ValueError("Puzzle protocol is locked to n=50, offset=25, budget=50")
    if args.frameskip != 3 or args.action_dim != 5:
        raise ValueError("Puzzle controller interface is locked to frameskip=3/action_dim=5")
    device = torch.device(args.device)

    jepa = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(device).eval()
    idms = [load_idm(path, device=device) for path in args.idm]
    means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def expand_image(image: np.ndarray) -> np.ndarray:
        upsampled = np.repeat(np.repeat(image, 3, axis=0), 3, axis=1)
        pad = 224 - 192
        low = pad // 2
        high = pad - low
        return np.pad(upsampled, ((low, high), (low, high), (0, 0)), mode="edge")

    @torch.no_grad()
    def encode(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(expand_image(image)).permute(2, 0, 1).float() / 255.0
        tensor = ((tensor - means) / stds).unsqueeze(0).to(device)
        output = jepa.encoder(tensor, interpolate_pos_encoding=True)
        return jepa.projector(select_encoder_readout(output))

    data = np.load(args.npz, allow_pickle=False)
    observations = data["observations"]
    qpos = data["qpos"]
    qvel = data["qvel"]
    terminals = data["terminals"]
    buttons = data["button_states"]
    valid = []
    for index in range(0, len(observations) - args.goal_offset - 1):
        if terminals[index]:
            continue
        if not terminals[index + 1 : index + args.goal_offset + 1].any():
            valid.append(index)
    valid = np.asarray(valid, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    starts = valid[rng.integers(len(valid), size=args.num_eval)]

    env = ogbench.make_env_and_datasets("visual-puzzle-3x3-play-v0", env_only=True)
    records = []
    successes = 0
    for episode_index, start_value in enumerate(starts.tolist()):
        start = int(start_value)
        goal = start + args.goal_offset
        goal_z = encode(observations[goal])
        env.reset(seed=args.seed + episode_index)
        unwrapped = env.unwrapped
        unwrapped.set_state(qpos[start].copy(), qvel[start].copy(), buttons[start].copy())
        unwrapped._target_button_states = buttons[goal].copy()
        trivial = bool(np.array_equal(buttons[start], buttons[goal]))
        current_z = encode(observations[start])
        success = False
        first_success_step = None
        for step in range(args.eval_budget):
            horizon = max(1, min(args.eval_budget - step, idms[0].max_horizon))
            steps_remaining = torch.full((1,), horizon, device=device, dtype=torch.long)
            with torch.no_grad():
                outputs = [model(current_z, goal_z, steps_remaining) for model in idms]
                stack = torch.stack(outputs, dim=0)
                if args.aggregation == "mean":
                    action_block = stack.mean(dim=0)
                else:
                    action_block = stack.median(dim=0).values
                action = action_block.reshape(1, args.frameskip, args.action_dim)[0, 0]
            observation, _, terminated, truncated, _ = env.step(
                np.clip(action.cpu().numpy(), -1.0, 1.0)
            )
            if all(unwrapped._compute_successes()):
                success = True
                first_success_step = step + 1
                break
            current_z = encode(observation)
            if terminated or truncated:
                break
        successes += int(success)
        record = {
            "episode_index": episode_index,
            "start_index": start,
            "goal_index": goal,
            "trivial": trivial,
            "success": success,
            "first_success_step": first_success_step,
        }
        records.append(record)
        print(
            f"episode={episode_index:02d} start={start} goal={goal} "
            f"trivial={int(trivial)} success={int(success)} step={first_success_step}",
            flush=True,
        )

    env.close()
    result = {
        "tag": args.tag,
        "checkpoint": args.ckpt,
        "idms": args.idm,
        "aggregation": args.aggregation,
        "rollout_seed": args.seed,
        "num_eval": args.num_eval,
        "successes": successes,
        "success_rate": successes / args.num_eval,
        "failed_episode_indices": [r["episode_index"] for r in records if not r["success"]],
        "failed_start_indices": [r["start_index"] for r in records if not r["success"]],
        "trivial_count": sum(int(r["trivial"]) for r in records),
        "episodes": records,
        "protocol": {
            "environment": "visual-puzzle-3x3-play-v0",
            "goal_offset": args.goal_offset,
            "eval_budget": args.eval_budget,
            "frameskip": args.frameskip,
            "action_dim": args.action_dim,
            "success": "all(env.unwrapped._compute_successes())",
            "controller": "frozen encoder plus GC-IDM; first action of predicted block",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"RESULT {args.tag} {successes}/{args.num_eval}", flush=True)
    print(f"GC-IDM {100*successes/args.num_eval:.1f}%", flush=True)


if __name__ == "__main__":
    main()
