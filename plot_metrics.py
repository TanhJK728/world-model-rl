#!/usr/bin/env python3
"""Compatibility plot for a single PPO run."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="runs/ppo_seed42/metrics.csv")
    parser.add_argument("--out", default="runs/ppo_seed42/learning_curve.png")
    args = parser.parse_args()

    steps, returns = [], []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row["mean_return_20"]
            if value.lower() != "nan":
                steps.append(int(row["global_step"]))
                returns.append(float(value))
    if not steps:
        raise RuntimeError("No completed episodes found in metrics.csv")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, returns)
    plt.xlabel("Environment steps")
    plt.ylabel("Mean episodic return (last 20)")
    plt.title("PPO on Pendulum-v1")
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
