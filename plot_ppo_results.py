#!/usr/bin/env python3
"""Plot multi-seed PPO learning curves and evaluation results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--evaluation", default="runs/ppo_summary/evaluation.csv")
    parser.add_argument("--output-dir", default="runs/ppo_summary")
    return parser.parse_args()


def read_metric(path: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[int] = []
    returns: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row["mean_return_20"]
            if value.lower() != "nan":
                steps.append(int(row["global_step"]))
                returns.append(float(value))
    if not steps:
        raise RuntimeError(f"No finite returns in {path}")
    return np.asarray(steps), np.asarray(returns)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = [read_metric(path) for path in args.metrics]
    max_common_step = min(int(steps[-1]) for steps, _ in curves)
    min_common_step = max(int(steps[0]) for steps, _ in curves)
    grid = np.linspace(min_common_step, max_common_step, 200)
    interpolated = np.stack(
        [np.interp(grid, steps, returns) for steps, returns in curves], axis=0
    )
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)

    plt.figure(figsize=(8, 5))
    for steps, returns in curves:
        plt.plot(steps, returns, alpha=0.35)
    plt.plot(grid, mean, linewidth=2.0, label="seed mean")
    if len(curves) > 1:
        plt.fill_between(grid, mean - std, mean + std, alpha=0.2, label="±1 std")
    plt.xlabel("Environment steps")
    plt.ylabel("Mean episodic return (last 20)")
    plt.title("PPO on Pendulum-v1: multi-seed learning curve")
    plt.legend()
    plt.tight_layout()
    learning_path = output_dir / "learning_curve_mean_std.png"
    plt.savefig(learning_path, dpi=180)
    plt.close()

    evaluation_path = Path(args.evaluation)
    if evaluation_path.exists():
        labels: list[str] = []
        means: list[float] = []
        stds: list[float] = []
        with evaluation_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels.append(row["label"])
                means.append(float(row["mean"]))
                stds.append(float(row["std"]))
        x = np.arange(len(labels))
        plt.figure(figsize=(8, 5))
        plt.bar(x, means, yerr=stds, capsize=4)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Deterministic evaluation return")
        plt.title("PPO checkpoints vs random policy")
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_bar.png", dpi=180)
        plt.close()

    print(f"Saved {learning_path}")


if __name__ == "__main__":
    main()
