#!/usr/bin/env python3
"""Refine a real PPO policy inside a learned world model."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from imagined_env import ImaginedVectorEnv, load_state_pool
from ppo_core import build_agent_from_checkpoint, select_device
from world_model import WorldModelEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--state-dataset", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--method", choices=["fixed", "uncertainty", "weighted"], required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--uncertainty-threshold", type=float, default=-1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initial-states-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    world_model, world_checkpoint = WorldModelEnsemble.load_checkpoint(
        args.world_model, device
    )
    world_model.eval()
    agent, init_checkpoint, action_space = build_agent_from_checkpoint(
        args.init_checkpoint, device
    )
    agent.train()
    init_config = init_checkpoint.get("config", {})
    env_id = init_config.get("env_id", "Pendulum-v1")
    hidden_dim = int(init_config.get("hidden_dim", 128))

    state_pool = load_state_pool(args.state_dataset, initial_only=args.initial_states_only)
    quantiles = world_checkpoint.get("uncertainty_quantiles", {})
    calibrated_q95 = float(quantiles.get("q95", 1e-4))
    threshold = (
        args.uncertainty_threshold
        if args.uncertainty_threshold > 0
        else calibrated_q95
    )
    termination_threshold = threshold if args.method == "uncertainty" else None

    imagined_env = ImaginedVectorEnv(
        world_model=world_model,
        state_pool=state_pool,
        num_envs=args.num_envs,
        horizon=args.horizon,
        device=device,
        seed=args.seed,
        uncertainty_threshold=termination_threshold,
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_dim = world_model.obs_dim
    act_dim = world_model.act_dim
    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    if num_updates < 1:
        raise ValueError("total_timesteps is smaller than one rollout batch")

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    uncertainty_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs = imagined_env.states.clone()
    next_done = torch.zeros(args.num_envs, device=device)
    global_step = 0
    start_time = time.time()
    uncertainty_terminations = 0

    metrics_path = output_dir / "metrics.csv"
    fields = [
        "update",
        "global_step",
        "segments_completed",
        "model_segment_return_20",
        "mean_segment_horizon_20",
        "mean_uncertainty",
        "uncertainty_terminations",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "clip_fraction",
        "mean_advantage_weight",
        "sps",
    ]

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for update in range(1, num_updates + 1):
            fraction_remaining = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = fraction_remaining * args.learning_rate

            for step in range(args.num_steps):
                global_step += args.num_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value

                next_obs, rewards, next_done, uncertainty, info = imagined_env.step(action)
                rewards_buf[step] = rewards
                uncertainty_buf[step] = uncertainty
                uncertainty_terminations += int(info["uncertainty_done"].sum().item())

            with torch.no_grad():
                next_value = agent.get_value(next_obs)
                advantages = torch.zeros_like(rewards_buf)
                last_gae = torch.zeros(args.num_envs, device=device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones_buf[t + 1]
                        next_values = values_buf[t + 1]
                    delta = (
                        rewards_buf[t]
                        + args.gamma * next_values * next_nonterminal
                        - values_buf[t]
                    )
                    last_gae = (
                        delta
                        + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                    )
                    advantages[t] = last_gae
                returns = advantages + values_buf

            b_obs = obs_buf.reshape(-1, obs_dim)
            b_actions = actions_buf.reshape(-1, act_dim)
            b_old_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_old_values = values_buf.reshape(-1)
            b_uncertainty = uncertainty_buf.reshape(-1)
            if args.method == "weighted":
                b_weights = torch.exp(-args.beta * b_uncertainty / max(threshold, 1e-12))
                b_weights = b_weights.clamp(min=0.02, max=1.0)
            else:
                b_weights = torch.ones_like(b_uncertainty)

            indices = np.arange(batch_size)
            clip_fractions: list[float] = []
            policy_loss_v = value_loss_v = approx_kl_v = 0.0

            for _epoch in range(args.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, minibatch_size):
                    mb_idx = indices[start : start + minibatch_size]
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_idx], b_actions[mb_idx]
                    )
                    log_ratio = new_logprob - b_old_logprobs[mb_idx]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fractions.append(
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        )

                    mb_adv = b_advantages[mb_idx]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    mb_adv = mb_adv * b_weights[mb_idx]
                    pg_loss_1 = -mb_adv * ratio
                    pg_loss_2 = -mb_adv * torch.clamp(
                        ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                    )
                    policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                    value_unclipped = (new_value - b_returns[mb_idx]).square()
                    value_clipped_prediction = b_old_values[mb_idx] + torch.clamp(
                        new_value - b_old_values[mb_idx],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    value_clipped = (
                        value_clipped_prediction - b_returns[mb_idx]
                    ).square()
                    value_loss = 0.5 * torch.max(value_unclipped, value_clipped).mean()
                    loss = (
                        policy_loss
                        + args.vf_coef * value_loss
                        - args.ent_coef * entropy.mean()
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    policy_loss_v = float(policy_loss.item())
                    value_loss_v = float(value_loss.item())
                    approx_kl_v = float(approx_kl.item())

                if approx_kl_v > args.target_kl:
                    break

            segment_return_20 = (
                float(np.mean(imagined_env.completed_returns[-20:]))
                if imagined_env.completed_returns
                else float("nan")
            )
            segment_horizon_20 = (
                float(np.mean(imagined_env.completed_lengths[-20:]))
                if imagined_env.completed_lengths
                else float("nan")
            )
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            writer.writerow(
                {
                    "update": update,
                    "global_step": global_step,
                    "segments_completed": len(imagined_env.completed_returns),
                    "model_segment_return_20": segment_return_20,
                    "mean_segment_horizon_20": segment_horizon_20,
                    "mean_uncertainty": float(uncertainty_buf.mean().item()),
                    "uncertainty_terminations": uncertainty_terminations,
                    "policy_loss": policy_loss_v,
                    "value_loss": value_loss_v,
                    "approx_kl": approx_kl_v,
                    "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                    "mean_advantage_weight": float(b_weights.mean().item()),
                    "sps": sps,
                }
            )
            f.flush()

            if update == 1 or update % 10 == 0 or update == num_updates:
                print(
                    f"update={update:04d}/{num_updates} step={global_step:8d} "
                    f"segment_return20={segment_return_20:8.2f} "
                    f"mean_horizon20={segment_horizon_20:5.2f} "
                    f"uncertainty={uncertainty_buf.mean().item():.6g}"
                )

    checkpoint_config = dict(init_config)
    checkpoint_config.update(
        {
            "env_id": env_id,
            "hidden_dim": hidden_dim,
            "seed": args.seed,
        }
    )
    imagined_config = {
        "method": args.method,
        "horizon": args.horizon,
        "uncertainty_threshold": threshold,
        "beta": args.beta,
        "world_model": args.world_model,
        "state_dataset": args.state_dataset,
        "init_checkpoint": args.init_checkpoint,
        "total_timesteps": global_step,
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "config": checkpoint_config,
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "imagined_config": imagined_config,
        },
        checkpoint_path,
    )
    summary = {
        **imagined_config,
        "checkpoint": str(checkpoint_path),
        "segments_completed": len(imagined_env.completed_returns),
        "final_model_segment_return_20": segment_return_20,
        "final_mean_segment_horizon_20": segment_horizon_20,
        "uncertainty_terminations": uncertainty_terminations,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved imagined PPO checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
