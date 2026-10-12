#!/usr/bin/env python3
"""Train one GC-IDM while persisting every optimization choice in its artifact."""

from __future__ import annotations

import argparse
import os

import torch

from train_gcidm_planner import train_idm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-goal-horizon", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--embedding-normalization", default="none", choices=["none","rms","zscore"])
    args = parser.parse_args()

    locked = {
        "embed_dim": 192,
        "action_dim": 2,
        "frameskip": 3,
        "hidden_dim": 1024,
        "n_layers": 4,
        "noise_sigma": 0.0,
        "noise_schedule": "fixed",
        "max_goal_horizon": args.max_goal_horizon,
        "train_split": 1.0,
        "split_seed": 42,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": 8192,
        "epochs": 300,
        "val_split": 0.1,
        "seed": args.seed,
        "feature_mode": "concat",
        "embedding_normalization": args.embedding_normalization,
    }
    train_idm(
        embeddings_path=args.embeddings,
        output_path=args.output,
        embed_dim=locked["embed_dim"],
        action_dim=locked["action_dim"],
        frameskip=locked["frameskip"],
        hidden_dim=locked["hidden_dim"],
        n_layers=locked["n_layers"],
        noise_sigma=locked["noise_sigma"],
        noise_schedule=locked["noise_schedule"],
        max_goal_horizon=locked["max_goal_horizon"],
        train_split=locked["train_split"],
        split_seed=locked["split_seed"],
        lr=locked["lr"],
        weight_decay=locked["weight_decay"],
        batch_size=locked["batch_size"],
        epochs=locked["epochs"],
        val_split=locked["val_split"],
        seed=locked["seed"],
        device=args.device,
        feature_mode=locked["feature_mode"],
        save_every=0,
        embedding_normalization=locked["embedding_normalization"],
    )
    payload = torch.load(args.output, map_location="cpu", weights_only=False)
    payload["training_protocol"] = locked
    temporary = args.output + ".audit_tmp"
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(f"[LOCKED_GCIDM_SAVED] {args.output} protocol={locked}")


if __name__ == "__main__":
    main()
