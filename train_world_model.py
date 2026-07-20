#!/usr/bin/env python3
"""Train a five-member bootstrapped dynamics ensemble on contiguous transition data."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ppo_core import select_device
from world_model import WorldModelEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", default="runs/world_model")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_datasets(paths: list[str]) -> dict[str, np.ndarray]:
    obs_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    next_obs_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    episode_offset = 0

    for source_index, path in enumerate(paths):
        data = np.load(path)
        obs = np.asarray(data["obs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        next_obs = np.asarray(data["next_obs"], dtype=np.float32)
        episode = np.asarray(data["episode_id"], dtype=np.int64)
        unique_episode, inverse = np.unique(episode, return_inverse=True)
        remapped_episode = inverse.astype(np.int64) + episode_offset
        episode_offset += len(unique_episode)

        obs_parts.append(obs)
        action_parts.append(actions)
        next_obs_parts.append(next_obs)
        episode_parts.append(remapped_episode)
        source_parts.append(np.full(len(obs), source_index, dtype=np.int64))

    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "actions": np.concatenate(action_parts, axis=0),
        "next_obs": np.concatenate(next_obs_parts, axis=0),
        "episode_id": np.concatenate(episode_parts, axis=0),
        "source": np.concatenate(source_parts, axis=0),
    }


def batched_member_mse(
    model: WorldModelEnsemble,
    member: int,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    batch_size: int,
) -> float:
    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            end = start + batch_size
            prediction = model.predict_member(member, obs[start:end], actions[start:end])
            errors.append((prediction - next_obs[start:end]).square().mean(dim=-1).cpu())
    return float(torch.cat(errors).mean().item())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    arrays = load_datasets(args.datasets)
    obs_np = arrays["obs"]
    actions_np = arrays["actions"]
    next_obs_np = arrays["next_obs"]
    episode_np = arrays["episode_id"]
    unique_episodes = np.unique(episode_np)
    rng.shuffle(unique_episodes)
    num_val_episodes = max(1, int(round(len(unique_episodes) * args.val_fraction)))
    val_episodes = set(unique_episodes[:num_val_episodes].tolist())
    val_mask = np.asarray([episode in val_episodes for episode in episode_np], dtype=bool)
    train_mask = ~val_mask
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)

    x_train = np.concatenate([obs_np[train_indices], actions_np[train_indices]], axis=-1)
    y_train = next_obs_np[train_indices] - obs_np[train_indices]
    input_mean = torch.as_tensor(x_train.mean(axis=0), dtype=torch.float32)
    input_std = torch.as_tensor(x_train.std(axis=0) + 1e-6, dtype=torch.float32)
    target_mean = torch.as_tensor(y_train.mean(axis=0), dtype=torch.float32)
    target_std = torch.as_tensor(y_train.std(axis=0) + 1e-6, dtype=torch.float32)

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    deltas = next_obs - obs

    model = WorldModelEnsemble(
        obs_dim=obs.shape[-1],
        act_dim=actions.shape[-1],
        ensemble_size=args.ensemble_size,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    ).to(device)

    metrics_path = output_dir / "training_metrics.csv"
    rows: list[dict[str, float | int]] = []
    train_index_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    val_index_tensor = torch.as_tensor(val_indices, dtype=torch.long, device=device)

    for member_index, member in enumerate(model.models):
        optimizer = torch.optim.AdamW(
            member.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        bootstrap_np = rng.choice(train_indices, size=len(train_indices), replace=True)
        bootstrap = torch.as_tensor(bootstrap_np, dtype=torch.long, device=device)
        best_val = float("inf")
        best_state = copy.deepcopy(member.state_dict())
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            member.train()
            permutation = bootstrap[torch.randperm(len(bootstrap), device=device)]
            train_loss_sum = 0.0
            train_count = 0

            for start in range(0, len(permutation), args.batch_size):
                idx = permutation[start : start + args.batch_size]
                x_norm = model.normalized_inputs(obs[idx], actions[idx])
                target_norm = model.normalize_targets(deltas[idx])
                prediction_norm = model.forward_normalized_member(member_index, x_norm)
                loss = F.mse_loss(prediction_norm, target_norm)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(member.parameters(), 10.0)
                optimizer.step()

                train_loss_sum += float(loss.item()) * len(idx)
                train_count += len(idx)

            member.eval()
            val_mse = batched_member_mse(
                model,
                member_index,
                obs[val_index_tensor],
                actions[val_index_tensor],
                next_obs[val_index_tensor],
                args.batch_size,
            )
            train_loss = train_loss_sum / max(train_count, 1)
            rows.append(
                {
                    "member": member_index,
                    "epoch": epoch,
                    "train_normalized_mse": train_loss,
                    "validation_one_step_mse": val_mse,
                }
            )

            if val_mse < best_val - 1e-8:
                best_val = val_mse
                best_state = copy.deepcopy(member.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"member={member_index} epoch={epoch:03d} "
                    f"train_norm_mse={train_loss:.6f} val_mse={val_mse:.8f}"
                )
            if epochs_without_improvement >= args.patience:
                print(f"member={member_index} early stop at epoch {epoch}")
                break

        member.load_state_dict(best_state)
        member.eval()
        print(f"member={member_index} best validation MSE={best_val:.8f}")

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "member",
                "epoch",
                "train_normalized_mse",
                "validation_one_step_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    uncertainty_values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(val_indices), args.batch_size):
            idx = val_index_tensor[start : start + args.batch_size]
            _, uncertainty, _ = model.predict(obs[idx], actions[idx])
            uncertainty_values.append(uncertainty.cpu())
    uncertainty_np = torch.cat(uncertainty_values).numpy()
    quantiles = {
        "q50": float(np.quantile(uncertainty_np, 0.50)),
        "q90": float(np.quantile(uncertainty_np, 0.90)),
        "q95": float(np.quantile(uncertainty_np, 0.95)),
        "q99": float(np.quantile(uncertainty_np, 0.99)),
    }

    checkpoint_path = output_dir / "world_model.pt"
    torch.save(
        model.checkpoint_dict(
            uncertainty_quantiles=quantiles,
            metadata={
                "datasets": args.datasets,
                "train_transitions": int(len(train_indices)),
                "validation_transitions": int(len(val_indices)),
                "seed": args.seed,
            },
        ),
        checkpoint_path,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "uncertainty_quantiles": quantiles,
                "train_transitions": int(len(train_indices)),
                "validation_transitions": int(len(val_indices)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 5))
    for member_index in range(args.ensemble_size):
        member_rows = [row for row in rows if row["member"] == member_index]
        plt.plot(
            [int(row["epoch"]) for row in member_rows],
            [float(row["validation_one_step_mse"]) for row in member_rows],
            label=f"member {member_index}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation one-step MSE")
    plt.title("Dynamics ensemble validation error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "world_model_training.png", dpi=180)
    plt.close()

    print(f"Saved world model to {checkpoint_path}")
    print(f"Uncertainty quantiles: {quantiles}")


if __name__ == "__main__":
    main()
