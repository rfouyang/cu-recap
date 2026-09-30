from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class CriticLossConfig:
    propensity_clip_min: float = 0.05
    max_ipw: float = 10.0
    utility_weight: float = 1.0
    failure_weight: float = 0.25
    cost_weight: float = 0.10


def gaussian_nll(mean, logvar, target):
    return 0.5 * (logvar + (target - mean).pow(2) * torch.exp(-logvar))


def selective_feedback_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: CriticLossConfig = CriticLossConfig(),
) -> dict[str, torch.Tensor]:
    """
    Utility labels are observed only for queried states.

    q_i ~ Bernoulli(p_i)
    utility loss uses clipped inverse-propensity weighting:

        sum q_i / p_i * l_i / sum q_i / p_i

    Required batch fields:
      queried:          [B] 0/1
      query_propensity: [B] probability used at collection time
      utility_target:   [B] valid when queried=1
      failure_label:    [B] 0/1
      correction_cost:  [B] valid when queried=1
    """
    q = batch["queried"].float()
    p = batch["query_propensity"].float().clamp_min(cfg.propensity_clip_min)

    w = (q / p).clamp_max(cfg.max_ipw)
    w_norm = w.sum().clamp_min(1.0)

    util_per_item = gaussian_nll(
        pred["utility_mean"],
        pred["utility_logvar"],
        batch["utility_target"].float(),
    )
    utility_loss = (w * util_per_item).sum() / w_norm

    failure_loss = F.binary_cross_entropy_with_logits(
        pred["failure_logit"],
        batch["failure_label"].float(),
    )

    cost_per_item = F.smooth_l1_loss(
        pred["correction_cost"],
        batch["correction_cost"].float(),
        reduction="none",
    )
    cost_loss = (w * cost_per_item).sum() / w_norm

    total = (
        cfg.utility_weight * utility_loss
        + cfg.failure_weight * failure_loss
        + cfg.cost_weight * cost_loss
    )
    return {
        "total": total,
        "utility": utility_loss,
        "failure": failure_loss,
        "cost": cost_loss,
        "effective_queried_weight": w_norm.detach(),
    }
