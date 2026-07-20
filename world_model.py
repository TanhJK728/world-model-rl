#!/usr/bin/env python3
"""Normalized MLP dynamics ensemble for Pendulum observations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ppo_core import safe_torch_load


class DynamicsMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, layers: int = 3):
        super().__init__()
        modules: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(layers):
            modules.extend([nn.Linear(last_dim, hidden_dim), nn.SiLU()])
            last_dim = hidden_dim
        modules.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def project_pendulum_observation(obs: torch.Tensor) -> torch.Tensor:
    """Project predicted [cos(theta), sin(theta), theta_dot] onto valid geometry."""
    if obs.shape[-1] != 3:
        return obs
    xy = obs[..., :2]
    norm = torch.linalg.vector_norm(xy, dim=-1, keepdim=True).clamp_min(1e-6)
    xy = xy / norm
    angular_velocity = obs[..., 2:3].clamp(-8.0, 8.0)
    return torch.cat([xy, angular_velocity], dim=-1)


class WorldModelEnsemble(nn.Module):
    """Bootstrap ensemble predicting state deltas."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 256,
        layers: int = 3,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ensemble_size = ensemble_size
        self.hidden_dim = hidden_dim
        self.layers = layers
        input_dim = obs_dim + act_dim
        self.models = nn.ModuleList(
            [DynamicsMLP(input_dim, obs_dim, hidden_dim, layers) for _ in range(ensemble_size)]
        )

        self.register_buffer(
            "input_mean",
            torch.zeros(input_dim) if input_mean is None else input_mean.float().clone(),
        )
        self.register_buffer(
            "input_std",
            torch.ones(input_dim) if input_std is None else input_std.float().clone(),
        )
        self.register_buffer(
            "target_mean",
            torch.zeros(obs_dim) if target_mean is None else target_mean.float().clone(),
        )
        self.register_buffer(
            "target_std",
            torch.ones(obs_dim) if target_std is None else target_std.float().clone(),
        )

    def normalized_inputs(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return (x - self.input_mean) / self.input_std

    def normalize_targets(self, deltas: torch.Tensor) -> torch.Tensor:
        return (deltas - self.target_mean) / self.target_std

    def denormalize_targets(self, normalized_deltas: torch.Tensor) -> torch.Tensor:
        return normalized_deltas * self.target_std + self.target_mean

    def forward_normalized_member(
        self, member: int, normalized_inputs: torch.Tensor
    ) -> torch.Tensor:
        return self.models[member](normalized_inputs)

    def predict_member(
        self,
        member: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        project: bool = True,
    ) -> torch.Tensor:
        x_norm = self.normalized_inputs(obs, actions)
        delta_norm = self.models[member](x_norm)
        delta = self.denormalize_targets(delta_norm)
        next_obs = obs + delta
        return project_pendulum_observation(next_obs) if project else next_obs

    def predict_all(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        project: bool = True,
    ) -> torch.Tensor:
        predictions = [
            self.predict_member(i, obs, actions, project=project)
            for i in range(self.ensemble_size)
        ]
        return torch.stack(predictions, dim=0)  # [ensemble, batch, obs_dim]

    def predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        project: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_predictions = self.predict_all(obs, actions, project=project)
        mean_prediction = all_predictions.mean(dim=0)
        variance_per_dimension = all_predictions.var(dim=0, unbiased=False)
        scalar_uncertainty = variance_per_dimension.mean(dim=-1)
        return mean_prediction, scalar_uncertainty, all_predictions

    def checkpoint_dict(
        self,
        uncertainty_quantiles: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "ensemble_size": self.ensemble_size,
                "hidden_dim": self.hidden_dim,
                "layers": self.layers,
            },
            "model_states": [model.state_dict() for model in self.models],
            "input_mean": self.input_mean.detach().cpu(),
            "input_std": self.input_std.detach().cpu(),
            "target_mean": self.target_mean.detach().cpu(),
            "target_std": self.target_std.detach().cpu(),
            "uncertainty_quantiles": uncertainty_quantiles or {},
            "metadata": metadata or {},
        }

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> tuple["WorldModelEnsemble", dict[str, Any]]:
        checkpoint = safe_torch_load(path, device)
        config = checkpoint["config"]
        model = cls(
            obs_dim=int(config["obs_dim"]),
            act_dim=int(config["act_dim"]),
            ensemble_size=int(config["ensemble_size"]),
            hidden_dim=int(config["hidden_dim"]),
            layers=int(config["layers"]),
            input_mean=checkpoint["input_mean"],
            input_std=checkpoint["input_std"],
            target_mean=checkpoint["target_mean"],
            target_std=checkpoint["target_std"],
        ).to(device)
        for member, state in zip(model.models, checkpoint["model_states"], strict=True):
            member.load_state_dict(state)
        model.eval()
        return model, checkpoint
