#!/usr/bin/env python3
"""From-scratch continuous-control PPO for Pendulum-v1."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ppo_core import Agent, make_env, select_device


@dataclass
class Config:
    env_id: str = "Pendulum-v1"
    seed: int = 42
    total_timesteps: int = 200_000
    num_envs: int = 8
    num_steps: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden_dim: int = 128
    output_dir: str = "runs/ppo_seed42"
    device: str = "auto"
    save_every: int = 25
    anneal_lr: int = 1  # 1: linear decay to 0 over training; 0: constant learning_rate
    normalize_reward: int = 0  # 1: NormalizeReward on the training signal (obs untouched)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_def in Config.__dataclass_fields__.items():
        default = field_def.default
        flag = "--" + field_name.replace("_", "-")
        parser.add_argument(flag, type=type(default), default=default)
    return Config(**vars(parser.parse_args()))


def save_checkpoint(
    path: Path,
    cfg: Config,
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    episodes_completed: int,
    best_mean_return: float,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "episodes_completed": episodes_completed,
            "best_mean_return": best_mean_return,
        },
        path,
    )


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = select_device(cfg.device)
    print(f"Using device: {device}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed, i) for i in range(cfg.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("Continuous Box action space required.")

    # RecordEpisodeStatistics (inner) observes raw rewards, so logged episode returns
    # stay on the true environment scale even when NormalizeReward rescales the reward
    # fed to GAE/value learning. NormalizeReward only touches the training signal; the
    # observation space (and therefore the saved policy interface) is unchanged, so the
    # downstream world-model pipeline is unaffected.
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    if cfg.normalize_reward:
        envs = gym.wrappers.vector.NormalizeReward(envs, gamma=cfg.gamma)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(obs_dim, envs.single_action_space, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    batch_size = cfg.num_envs * cfg.num_steps
    if batch_size % cfg.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches.")
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_timesteps // batch_size
    if num_updates < 1:
        raise ValueError("total_timesteps must be at least num_envs * num_steps.")

    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((cfg.num_steps, cfg.num_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    global_step = 0
    start_time = time.time()
    best_mean_return = -float("inf")

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "update",
        "global_step",
        "episodes_completed",
        "mean_return_20",
        "mean_length_20",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "learning_rate",
        "sps",
    ]

    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                fraction_remaining = 1.0 - (update - 1.0) / num_updates
                current_lr = fraction_remaining * cfg.learning_rate
            else:
                current_lr = cfg.learning_rate
            optimizer.param_groups[0]["lr"] = current_lr

            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value

                obs_np, reward_np, terminated_np, truncated_np, info = envs.step(
                    action.cpu().numpy()
                )
                done_np = np.logical_or(terminated_np, truncated_np)
                rewards_buf[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)

                # RecordEpisodeStatistics reports true (raw) episode returns on the step
                # an episode ends; "_episode" masks which envs finished this step.
                if "episode" in info:
                    finished = info["_episode"]
                    episode_returns.extend(
                        float(x) for x in np.asarray(info["episode"]["r"])[finished]
                    )
                    episode_lengths.extend(
                        int(x) for x in np.asarray(info["episode"]["l"])[finished]
                    )

                next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            # GAE-Lambda backward recursion.
            with torch.no_grad():
                next_value = agent.get_value(next_obs)
                advantages = torch.zeros_like(rewards_buf)
                last_gae = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones_buf[t + 1]
                        next_values = values_buf[t + 1]
                    delta = (
                        rewards_buf[t]
                        + cfg.gamma * next_values * next_nonterminal
                        - values_buf[t]
                    )
                    last_gae = (
                        delta
                        + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                    )
                    advantages[t] = last_gae
                returns = advantages + values_buf

            b_obs = obs_buf.reshape(-1, obs_dim)
            b_actions = actions_buf.reshape(-1, act_dim)
            b_old_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_old_values = values_buf.reshape(-1)

            indices = np.arange(batch_size)
            clip_fractions: list[float] = []
            policy_loss_v = value_loss_v = entropy_v = approx_kl_v = 0.0
            target_kl_stop = False

            for _epoch in range(cfg.update_epochs):
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
                        clip_fraction = (
                            ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                        )
                        clip_fractions.append(clip_fraction)

                    mb_adv = b_advantages[mb_idx]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss_1 = -mb_adv * ratio
                    pg_loss_2 = -mb_adv * torch.clamp(
                        ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                    )
                    policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                    value_loss_unclipped = (new_value - b_returns[mb_idx]).square()
                    value_prediction_clipped = b_old_values[mb_idx] + torch.clamp(
                        new_value - b_old_values[mb_idx],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    value_loss_clipped = (
                        value_prediction_clipped - b_returns[mb_idx]
                    ).square()
                    value_loss = 0.5 * torch.max(
                        value_loss_unclipped, value_loss_clipped
                    ).mean()
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss
                        + cfg.vf_coef * value_loss
                        - cfg.ent_coef * entropy_mean
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    policy_loss_v = float(policy_loss.item())
                    value_loss_v = float(value_loss.item())
                    entropy_v = float(entropy_mean.item())
                    approx_kl_v = float(approx_kl.item())

                if approx_kl_v > cfg.target_kl:
                    target_kl_stop = True
                    break

            mean_return_20 = (
                float(np.mean(episode_returns[-20:])) if episode_returns else float("nan")
            )
            mean_length_20 = (
                float(np.mean(episode_lengths[-20:])) if episode_lengths else float("nan")
            )
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            writer.writerow(
                {
                    "update": update,
                    "global_step": global_step,
                    "episodes_completed": len(episode_returns),
                    "mean_return_20": mean_return_20,
                    "mean_length_20": mean_length_20,
                    "policy_loss": policy_loss_v,
                    "value_loss": value_loss_v,
                    "entropy": entropy_v,
                    "approx_kl": approx_kl_v,
                    "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                    "learning_rate": current_lr,
                    "sps": sps,
                }
            )
            csv_file.flush()

            if mean_return_20 > best_mean_return:
                best_mean_return = mean_return_20
                save_checkpoint(
                    output_dir / "best_checkpoint.pt",
                    cfg,
                    agent,
                    optimizer,
                    global_step,
                    len(episode_returns),
                    best_mean_return,
                )

            if update == 1 or update % 10 == 0 or update == num_updates:
                stop_text = " target-kl-stop" if target_kl_stop else ""
                print(
                    f"update={update:04d}/{num_updates} "
                    f"step={global_step:8d} episodes={len(episode_returns):4d} "
                    f"return20={mean_return_20:9.2f} kl={approx_kl_v:.5f} "
                    f"sps={sps}{stop_text}"
                )

            if update % cfg.save_every == 0 or update == num_updates:
                save_checkpoint(
                    output_dir / "checkpoint.pt",
                    cfg,
                    agent,
                    optimizer,
                    global_step,
                    len(episode_returns),
                    best_mean_return,
                )

    envs.close()
    summary = {
        "seed": cfg.seed,
        "requested_timesteps": cfg.total_timesteps,
        "actual_timesteps": global_step,
        "episodes_completed": len(episode_returns),
        "final_mean_return_20": mean_return_20,
        "best_mean_return_20": best_mean_return,
        "device": str(device),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved checkpoint to {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
