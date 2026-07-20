#!/usr/bin/env python3
"""Evaluate one-step and multi-step dynamics errors on policy, random, and OOD data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo_core import select_device
from world_model import WorldModelEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--horizons", default="1,5,10,20")
    parser.add_argument("--max-segments", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/world_model_eval")
    return parser.parse_args()


def dataset_label(path: str) -> str:
    metadata_path = Path(path).with_suffix(".json")
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "mode", Path(path).stem
            )
        except json.JSONDecodeError:
            pass
    return Path(path).stem


def build_segments(
    data: np.lib.npyio.NpzFile,
    max_horizon: int,
    max_segments: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(data["obs"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    next_obs = np.asarray(data["next_obs"], dtype=np.float32)
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    timestep = np.asarray(data["timestep"], dtype=np.int64)

    candidates: list[tuple[int, np.ndarray]] = []
    for episode in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == episode)
        indices = indices[np.argsort(timestep[indices])]
        if len(indices) < max_horizon:
            continue
        for start in range(0, len(indices) - max_horizon + 1):
            window = indices[start : start + max_horizon]
            expected = np.arange(timestep[window[0]], timestep[window[0]] + max_horizon)
            if np.array_equal(timestep[window], expected):
                candidates.append((window[0], window))

    if not candidates:
        raise RuntimeError("No contiguous rollout segments found.")
    if len(candidates) > max_segments:
        chosen = rng.choice(len(candidates), size=max_segments, replace=False)
        candidates = [candidates[int(i)] for i in chosen]

    initial_obs = np.stack([obs[first] for first, _ in candidates], axis=0)
    action_sequences = np.stack([actions[window] for _, window in candidates], axis=0)
    true_sequences = np.stack([next_obs[window] for _, window in candidates], axis=0)
    return initial_obs, action_sequences, true_sequences


def main() -> None:
    args = parse_args()
    horizons = sorted({int(value) for value in args.horizons.split(",")})
    max_horizon = max(horizons)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    model, checkpoint = WorldModelEnsemble.load_checkpoint(args.checkpoint, device)

    one_step_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []

    for dataset_path in args.datasets:
        label = dataset_label(dataset_path)
        data = np.load(dataset_path)
        obs_np = np.asarray(data["obs"], dtype=np.float32)
        actions_np = np.asarray(data["actions"], dtype=np.float32)
        next_obs_np = np.asarray(data["next_obs"], dtype=np.float32)

        sample_errors: list[torch.Tensor] = []
        sample_uncertainties: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(obs_np), args.batch_size):
                end = start + args.batch_size
                obs = torch.as_tensor(obs_np[start:end], dtype=torch.float32, device=device)
                actions = torch.as_tensor(
                    actions_np[start:end], dtype=torch.float32, device=device
                )
                targets = torch.as_tensor(
                    next_obs_np[start:end], dtype=torch.float32, device=device
                )
                predictions, uncertainty, _ = model.predict(obs, actions)
                sample_errors.append(
                    (predictions - targets).square().mean(dim=-1).cpu()
                )
                sample_uncertainties.append(uncertainty.cpu())
        one_step_error = torch.cat(sample_errors).numpy()
        one_step_uncertainty = torch.cat(sample_uncertainties).numpy()
        one_step_rows.append(
            {
                "source": label,
                "transitions": len(obs_np),
                "one_step_mse": float(one_step_error.mean()),
                "one_step_mse_std": float(one_step_error.std(ddof=1)),
                "mean_uncertainty": float(one_step_uncertainty.mean()),
            }
        )

        initial_obs_np, action_seq_np, true_seq_np = build_segments(
            data, max_horizon, args.max_segments, rng
        )
        state = torch.as_tensor(initial_obs_np, dtype=torch.float32, device=device)
        action_sequences = torch.as_tensor(action_seq_np, dtype=torch.float32, device=device)
        true_sequences = torch.as_tensor(true_seq_np, dtype=torch.float32, device=device)
        errors_by_step: list[torch.Tensor] = []
        uncertainty_by_step: list[torch.Tensor] = []

        with torch.no_grad():
            for h in range(max_horizon):
                state, uncertainty, _ = model.predict(state, action_sequences[:, h])
                error = (state - true_sequences[:, h]).square().sum(dim=-1)
                errors_by_step.append(error.cpu())
                uncertainty_by_step.append(uncertainty.cpu())

        errors = torch.stack(errors_by_step, dim=1).numpy()
        uncertainties = torch.stack(uncertainty_by_step, dim=1).numpy()
        for horizon in horizons:
            rollout_rows.append(
                {
                    "source": label,
                    "horizon": horizon,
                    "segments": len(initial_obs_np),
                    "cumulative_squared_error": float(errors[:, :horizon].mean()),
                    "endpoint_squared_error": float(errors[:, horizon - 1].mean()),
                    "mean_uncertainty": float(uncertainties[:, :horizon].mean()),
                }
            )
        print(
            f"{label}: one-step MSE={one_step_error.mean():.8f}, "
            f"H={max_horizon} cumulative squared error={errors.mean():.8f}"
        )

    one_step_path = output_dir / "one_step_results.csv"
    with one_step_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(one_step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(one_step_rows)

    rollout_path = output_dir / "rollout_results.csv"
    with rollout_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rollout_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rollout_rows)

    plt.figure(figsize=(8, 5))
    for label in sorted({str(row["source"]) for row in rollout_rows}):
        selected = [row for row in rollout_rows if row["source"] == label]
        selected.sort(key=lambda row: int(row["horizon"]))
        plt.plot(
            [int(row["horizon"]) for row in selected],
            [float(row["cumulative_squared_error"]) for row in selected],
            marker="o",
            label=label,
        )
    plt.xlabel("Rollout horizon H")
    plt.ylabel("Cumulative squared state error")
    plt.title("World-model error accumulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rollout_error.png", dpi=180)
    plt.close()

    summary = {
        "checkpoint": args.checkpoint,
        "uncertainty_quantiles": checkpoint.get("uncertainty_quantiles", {}),
        "one_step_results": one_step_rows,
        "rollout_results": rollout_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
