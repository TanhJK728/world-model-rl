#!/usr/bin/env python3
"""Shared PPO components used by training, evaluation, and imagined RL."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


def safe_torch_load(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a trusted local checkpoint onto a chosen device."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def layer_init(
    layer: nn.Linear,
    std: float = math.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-critic with a tanh-squashed Gaussian continuous policy."""

    def __init__(self, obs_dim: int, action_space: gym.spaces.Box, hidden_dim: int = 128):
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def policy_parameters(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.policy_parameters(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.policy_parameters(obs)
        dist = Normal(mean, std)

        if action is None:
            raw_action = dist.rsample()
            squashed = torch.tanh(raw_action)
            env_action = squashed * self.action_scale + self.action_bias
        else:
            squashed = (action - self.action_bias) / self.action_scale
            squashed = squashed.clamp(-0.999999, 0.999999)
            raw_action = torch.atanh(squashed)
            env_action = action

        log_prob = dist.log_prob(raw_action)
        correction = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6)
        log_prob = (log_prob - correction).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)  # Pre-squash diagnostic proxy.
        value = self.get_value(obs)
        return env_action, log_prob, entropy, value


def make_env(env_id: str, seed: int, index: int):
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed + index)
        env.observation_space.seed(seed + index)
        return env

    return thunk


def build_agent_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Agent, dict[str, Any], gym.spaces.Box]:
    checkpoint = safe_torch_load(checkpoint_path, device)
    config = checkpoint.get("config", {})
    env_id = config.get("env_id", "Pendulum-v1")
    hidden_dim = int(config.get("hidden_dim", 128))

    env = gym.make(env_id)
    try:
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("Continuous Box action space required.")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_space = env.action_space
        agent = Agent(obs_dim, action_space, hidden_dim).to(device)
    finally:
        env.close()

    model_state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if model_state is None:
        raise KeyError(f"No model state found in checkpoint: {checkpoint_path}")
    agent.load_state_dict(model_state)
    agent.eval()
    return agent, checkpoint, action_space


def pendulum_reward_torch(obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Exact Pendulum-v1 reward, evaluated from current observation and torque."""
    theta = torch.atan2(obs[..., 1], obs[..., 0])
    theta_dot = obs[..., 2]
    torque = action[..., 0].clamp(-2.0, 2.0)
    return -(theta.square() + 0.1 * theta_dot.square() + 0.001 * torque.square())


def pendulum_reward_numpy(obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    theta = np.arctan2(obs[..., 1], obs[..., 0])
    theta_dot = obs[..., 2]
    torque = np.clip(action[..., 0], -2.0, 2.0)
    return -(theta**2 + 0.1 * theta_dot**2 + 0.001 * torque**2)


def observation_to_pendulum_state(obs: np.ndarray) -> np.ndarray:
    """Convert [cos(theta), sin(theta), theta_dot] to the internal [theta, theta_dot]."""
    obs = np.asarray(obs, dtype=np.float64)
    return np.asarray([np.arctan2(obs[1], obs[0]), obs[2]], dtype=np.float64)


def evaluate_agent(
    agent: Agent,
    env_id: str = "Pendulum-v1",
    episodes: int = 20,
    seed: int = 10_000,
    deterministic: bool = True,
    initial_observations: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
) -> np.ndarray:
    """Evaluate a policy in the real Gymnasium environment."""
    device = torch.device(device)
    returns: list[float] = []
    env = gym.make(env_id)
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            if initial_observations is not None:
                initial_obs = np.asarray(initial_observations[episode], dtype=np.float64)
                env.unwrapped.state = observation_to_pendulum_state(initial_obs)
                obs = initial_obs.astype(np.float32)

            total_return = 0.0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and (max_steps is None or steps < max_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    if deterministic:
                        action_t = agent.deterministic_action(obs_t)
                    else:
                        action_t, _, _, _ = agent.get_action_and_value(obs_t)
                action = action_t.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                steps += 1
            returns.append(total_return)
    finally:
        env.close()
    return np.asarray(returns, dtype=np.float64)


def evaluate_random_policy(
    env_id: str = "Pendulum-v1",
    episodes: int = 20,
    seed: int = 20_000,
) -> np.ndarray:
    returns: list[float] = []
    env = gym.make(env_id)
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            env.action_space.seed(seed + episode)
            total_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
            returns.append(total_return)
    finally:
        env.close()
    return np.asarray(returns, dtype=np.float64)


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
