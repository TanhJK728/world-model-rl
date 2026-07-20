#!/usr/bin/env python3
"""Collect contiguous Pendulum transition datasets from policy, random, or OOD actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ppo_core import build_agent_from_checkpoint, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["policy", "random", "ood"], required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "policy" and not args.checkpoint:
        raise ValueError("--checkpoint is required for --mode policy")

    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    agent = None
    if args.mode == "policy":
        agent, checkpoint, _ = build_agent_from_checkpoint(args.checkpoint, device)
        args.env_id = checkpoint.get("config", {}).get("env_id", args.env_id)

    envs = [gym.make(args.env_id) for _ in range(args.num_envs)]
    observations: list[np.ndarray] = []
    episode_ids = np.arange(args.num_envs, dtype=np.int64)
    next_episode_id = args.num_envs
    timesteps = np.zeros(args.num_envs, dtype=np.int64)

    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=args.seed + i)
        env.action_space.seed(args.seed + i)
        observations.append(np.asarray(obs, dtype=np.float32))

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    next_obs_rows: list[np.ndarray] = []
    done_rows: list[bool] = []
    terminated_rows: list[bool] = []
    truncated_rows: list[bool] = []
    episode_rows: list[int] = []
    timestep_rows: list[int] = []

    collected = 0
    try:
        while collected < args.steps:
            obs_batch = np.stack(observations, axis=0)
            if args.mode == "policy":
                obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
                with torch.no_grad():
                    if args.deterministic_policy:
                        actions_t = agent.deterministic_action(obs_t)
                    else:
                        actions_t, _, _, _ = agent.get_action_and_value(obs_t)
                action_batch = actions_t.cpu().numpy()
            elif args.mode == "random":
                action_batch = np.stack([env.action_space.sample() for env in envs], axis=0)
            else:
                # Legal but deliberately extreme torques: distribution shift relative to a trained policy.
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=(args.num_envs, 1))
                magnitudes = rng.uniform(0.85, 1.0, size=(args.num_envs, 1))
                highs = np.stack([env.action_space.high for env in envs], axis=0)
                action_batch = signs * magnitudes * highs

            for i, env in enumerate(envs):
                if collected >= args.steps:
                    break
                current_obs = observations[i]
                action = np.asarray(action_batch[i], dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                done = bool(terminated or truncated)

                obs_rows.append(current_obs.copy())
                action_rows.append(action.copy())
                reward_rows.append(float(reward))
                next_obs_rows.append(next_obs.copy())
                done_rows.append(done)
                terminated_rows.append(bool(terminated))
                truncated_rows.append(bool(truncated))
                episode_rows.append(int(episode_ids[i]))
                timestep_rows.append(int(timesteps[i]))
                collected += 1

                if done:
                    reset_obs, _ = env.reset()
                    observations[i] = np.asarray(reset_obs, dtype=np.float32)
                    episode_ids[i] = next_episode_id
                    next_episode_id += 1
                    timesteps[i] = 0
                else:
                    observations[i] = next_obs
                    timesteps[i] += 1

            if collected % 10_000 < args.num_envs:
                print(f"Collected {collected}/{args.steps} transitions")
    finally:
        for env in envs:
            env.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        obs=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        next_obs=np.asarray(next_obs_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.bool_),
        terminated=np.asarray(terminated_rows, dtype=np.bool_),
        truncated=np.asarray(truncated_rows, dtype=np.bool_),
        episode_id=np.asarray(episode_rows, dtype=np.int64),
        timestep=np.asarray(timestep_rows, dtype=np.int64),
    )
    metadata = {
        "mode": args.mode,
        "env_id": args.env_id,
        "steps": collected,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "deterministic_policy": args.deterministic_policy,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
