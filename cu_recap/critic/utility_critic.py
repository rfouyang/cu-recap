from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UtilityCriticConfig:
    feature_dim: int = 1024
    state_dim: int = 32
    hidden_dim: int = 384
    history_len: int = 4
    num_layers: int = 2
    num_heads: int = 6


class UtilityCritic(nn.Module):
    """
    Fast pre-query critic.

    Inputs are ONLY available at query time:
      visual/policy feature history: [B, L, feature_dim]
      robot state history:           [B, L, state_dim]

    Outputs:
      utility_mean: expected downstream learning gain if corrected now
      utility_logvar: aleatoric utility uncertainty
      failure_logit: execution failure risk (separate from learning utility)
      correction_cost: predicted human-control seconds (positive scalar)

    Epistemic uncertainty should be obtained with an ensemble of these models.
    """

    def __init__(self, cfg: UtilityCriticConfig):
        super().__init__()
        self.cfg = cfg

        self.feature_proj = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.state_proj = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.history_len, cfg.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

        self.utility_head = nn.Linear(cfg.hidden_dim, 2)  # mean, log-variance
        self.failure_head = nn.Linear(cfg.hidden_dim, 1)
        self.cost_head = nn.Linear(cfg.hidden_dim, 1)

    def encode(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or state.ndim != 3:
            raise ValueError("features/state must be [B, L, D]")
        if features.shape[1] > self.cfg.history_len:
            features = features[:, -self.cfg.history_len:]
            state = state[:, -self.cfg.history_len:]

        x = self.feature_proj(features) + self.state_proj(state)
        x = x + self.pos[:, : x.shape[1]]
        x = self.temporal(x)
        return self.norm(x[:, -1])

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(features, state)
        uv = self.utility_head(z)
        return {
            "utility_mean": uv[:, 0],
            "utility_logvar": uv[:, 1].clamp(-8.0, 5.0),
            "failure_logit": self.failure_head(z).squeeze(-1),
            "correction_cost": F.softplus(self.cost_head(z).squeeze(-1)) + 1e-3,
        }
