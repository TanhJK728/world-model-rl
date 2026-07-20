#!/usr/bin/env python3
"""Evaluate trained PPO checkpoints and a random-policy baseline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ppo_core import (
    build_agent_from_checkpoint,
    evaluate_agent,
    evaluate_random_policy,
    select_device,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/ppo_summary/evaluation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    env_id = "Pendulum-v1"
    ppo_seed_means: list[float] = []

    for checkpoint_path in args.checkpoints:
        agent, checkpoint, _ = build_agent_from_checkpoint(checkpoint_path, device)
        config = checkpoint.get("config", {})
        env_id = config.get("env_id", env_id)
        seed = int(config.get("seed", -1))
        values = evaluate_agent(
            agent,
            env_id=env_id,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            device=device,
        )
        stats = summarize(values)
        ppo_seed_means.append(float(stats["mean"]))
        rows.append(
            {
                "label": f"ppo_seed{seed}",
                "checkpoint": str(checkpoint_path),
                "policy_type": "deterministic_actor_mean",
                "episodes": args.episodes,
                **stats,
            }
        )
        print(
            f"{checkpoint_path}: mean={stats['mean']:.2f} "
            f"std={stats['std']:.2f} over {args.episodes} episodes"
        )


    if len(ppo_seed_means) > 1:
        import numpy as np

        seed_array = np.asarray(ppo_seed_means, dtype=np.float64)
        rows.append(
            {
                "label": "ppo_across_seeds",
                "checkpoint": "",
                "policy_type": "aggregate_seed_means",
                "episodes": args.episodes,
                "mean": float(seed_array.mean()),
                "std": float(seed_array.std(ddof=1)),
                "min": float(seed_array.min()),
                "max": float(seed_array.max()),
            }
        )

    random_values = evaluate_random_policy(
        env_id=env_id,
        episodes=args.episodes,
        seed=args.seed,
    )
    random_stats = summarize(random_values)
    rows.append(
        {
            "label": "random_policy",
            "checkpoint": "",
            "policy_type": "uniform_random",
            "episodes": args.episodes,
            **random_stats,
        }
    )
    print(
        f"random_policy: mean={random_stats['mean']:.2f} "
        f"std={random_stats['std']:.2f} over {args.episodes} episodes"
    )

    fieldnames = [
        "label",
        "checkpoint",
        "policy_type",
        "episodes",
        "mean",
        "std",
        "min",
        "max",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
