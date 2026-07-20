#!/usr/bin/env python3
"""Torch-native vectorized imagined environment backed by a dynamics ensemble."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ppo_core import pendulum_reward_torch
from world_model import WorldModelEnsemble


def load_state_pool(dataset_path: str | Path, initial_only: bool = False) -> np.ndarray:
    data = np.load(dataset_path)
    obs = np.asarray(data["obs"], dtype=np.float32)
    if initial_only and "timestep" in data:
        timestep = np.asarray(data["timestep"], dtype=np.int64)
        initial = obs[timestep == 0]
        if len(initial) > 0:
            return initial
    return obs


class ImaginedVectorEnv:
    """Runs ensemble-mean rollouts and exposes ensemble disagreement."""

    def __init__(
        self,
        world_model: WorldModelEnsemble,
        state_pool: np.ndarray,
        num_envs: int,
        horizon: int,
        device: torch.device,
        seed: int = 0,
        uncertainty_threshold: float | None = None,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.world_model = world_model
        self.state_pool = torch.as_tensor(state_pool, dtype=torch.float32, device=device)
        self.num_envs = num_envs
        self.horizon = horizon
        self.device = device
        self.uncertainty_threshold = uncertainty_threshold
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.states = torch.empty((num_envs, self.state_pool.shape[-1]), device=device)
        self.elapsed = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.running_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []
        self.reset()

    def _sample_states(self, count: int) -> torch.Tensor:
        indices = torch.randint(
            low=0,
            high=len(self.state_pool),
            size=(count,),
            generator=self.generator,
            device="cpu",
        ).to(self.device)
        return self.state_pool[indices]

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            self.states = self._sample_states(self.num_envs)
            self.elapsed.zero_()
            self.running_returns.zero_()
        else:
            mask = mask.bool()
            count = int(mask.sum().item())
            if count:
                self.states[mask] = self._sample_states(count)
                self.elapsed[mask] = 0
                self.running_returns[mask] = 0.0
        return self.states.clone()

    @torch.no_grad()
    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        current_states = self.states
        rewards = pendulum_reward_torch(current_states, actions)
        predicted_next, uncertainty, _ = self.world_model.predict(current_states, actions)

        self.elapsed += 1
        self.running_returns += rewards
        horizon_done = self.elapsed >= self.horizon
        if self.uncertainty_threshold is None:
            uncertainty_done = torch.zeros_like(horizon_done)
        else:
            uncertainty_done = uncertainty > self.uncertainty_threshold
        done = torch.logical_or(horizon_done, uncertainty_done)

        finished_returns = self.running_returns[done].detach().cpu()
        finished_lengths = self.elapsed[done].detach().cpu()
        self.completed_returns.extend(float(x) for x in finished_returns)
        self.completed_lengths.extend(int(x) for x in finished_lengths)

        self.states = predicted_next
        final_predicted_states = predicted_next.clone()
        self.reset(mask=done)

        info = {
            "horizon_done": horizon_done,
            "uncertainty_done": uncertainty_done,
            "final_predicted_states": final_predicted_states,
        }
        return self.states.clone(), rewards, done.float(), uncertainty, info
