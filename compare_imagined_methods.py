#!/usr/bin/env python3
"""Compare model return, real return, exploitation gap, and training horizon."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from imagined_env import load_state_pool
from ppo_core import (
    build_agent_from_checkpoint,
    evaluate_agent,
    pendulum_reward_torch,
    select_device,
)
from world_model import WorldModelEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--state-dataset", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--evaluation-horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/comparison")
    return parser.parse_args()


def checkpoint_label(checkpoint: dict, path: str) -> str:
    imagined = checkpoint.get("imagined_config")
    if not imagined:
        seed = checkpoint.get("config", {}).get("seed", "x")
        return f"Real PPO seed {seed}"
    method = imagined["method"]
    horizon = imagined["horizon"]
    if method == "fixed":
        return f"Fixed H={horizon}"
    if method == "uncertainty":
        return "Uncertainty termination"
    if method == "weighted":
        return f"Weighted advantage H={horizon}"
    return Path(path).parent.name


def training_horizon_from_summary(checkpoint_path: str, checkpoint: dict) -> float:
    summary_path = Path(checkpoint_path).parent / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "final_mean_segment_horizon_20" in summary:
                return float(summary["final_mean_segment_horizon_20"])
        except (json.JSONDecodeError, ValueError):
            pass
    imagined = checkpoint.get("imagined_config")
    if imagined:
        return float(imagined.get("horizon", float("nan")))
    return float("nan")


@torch.no_grad()
def evaluate_in_model(
    agent,
    world_model: WorldModelEnsemble,
    initial_obs: np.ndarray,
    horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    states = torch.as_tensor(initial_obs, dtype=torch.float32, device=device)
    returns = torch.zeros(len(states), dtype=torch.float32, device=device)
    uncertainties: list[torch.Tensor] = []
    for _ in range(horizon):
        actions = agent.deterministic_action(states)
        returns += pendulum_reward_torch(states, actions)
        states, uncertainty, _ = world_model.predict(states, actions)
        uncertainties.append(uncertainty)
    mean_uncertainty = torch.stack(uncertainties, dim=1).mean(dim=1)
    return returns.cpu().numpy(), mean_uncertainty.cpu().numpy()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    rng = np.random.default_rng(args.seed)

    world_model, _ = WorldModelEnsemble.load_checkpoint(args.world_model, device)
    initial_pool = load_state_pool(args.state_dataset, initial_only=True)
    if len(initial_pool) == 0:
        initial_pool = load_state_pool(args.state_dataset, initial_only=False)
    indices = rng.choice(
        len(initial_pool), size=args.episodes, replace=len(initial_pool) < args.episodes
    )
    initial_obs = initial_pool[indices]

    rows: list[dict[str, object]] = []
    for checkpoint_path in args.checkpoints:
        agent, checkpoint, _ = build_agent_from_checkpoint(checkpoint_path, device)
        label = checkpoint_label(checkpoint, checkpoint_path)
        env_id = checkpoint.get("config", {}).get("env_id", "Pendulum-v1")
        model_returns, model_uncertainty = evaluate_in_model(
            agent,
            world_model,
            initial_obs,
            args.evaluation_horizon,
            device,
        )
        real_returns = evaluate_agent(
            agent,
            env_id=env_id,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            initial_observations=initial_obs,
            device=device,
            max_steps=args.evaluation_horizon,
        )
        gap_values = model_returns - real_returns
        row = {
            "method": label,
            "checkpoint": checkpoint_path,
            "episodes": args.episodes,
            "evaluation_horizon": args.evaluation_horizon,
            "model_return_mean": float(model_returns.mean()),
            "model_return_std": float(model_returns.std(ddof=1)),
            "real_return_mean": float(real_returns.mean()),
            "real_return_std": float(real_returns.std(ddof=1)),
            "exploitation_gap_mean": float(gap_values.mean()),
            "exploitation_gap_std": float(gap_values.std(ddof=1)),
            "mean_model_uncertainty": float(model_uncertainty.mean()),
            "mean_training_horizon": training_horizon_from_summary(
                checkpoint_path, checkpoint
            ),
        }
        rows.append(row)
        print(
            f"{label}: model={row['model_return_mean']:.2f}, "
            f"real={row['real_return_mean']:.2f}, "
            f"gap={row['exploitation_gap_mean']:.2f}"
        )

    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown_lines = [
        "| Method | Model return | Real return | Exploitation gap | Mean training horizon |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row['method']} | {row['model_return_mean']:.2f} ± {row['model_return_std']:.2f} "
            f"| {row['real_return_mean']:.2f} ± {row['real_return_std']:.2f} "
            f"| {row['exploitation_gap_mean']:.2f} ± {row['exploitation_gap_std']:.2f} "
            f"| {row['mean_training_horizon']:.2f} |"
        )
    (output_dir / "comparison.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )

    labels = [str(row["method"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2,
        [float(row["model_return_mean"]) for row in rows],
        width,
        label="model return",
    )
    plt.bar(
        x + width / 2,
        [float(row["real_return_mean"]) for row in rows],
        width,
        label="real return",
    )
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("200-step return")
    plt.title("Learned-model return versus real-environment return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_vs_real_return.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(x, [float(row["exploitation_gap_mean"]) for row in rows])
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Model return − real return")
    plt.title("Model exploitation gap")
    plt.tight_layout()
    plt.savefig(output_dir / "exploitation_gap.png", dpi=180)
    plt.close()

    print(f"Saved comparison to {output_dir}")


if __name__ == "__main__":
    main()
