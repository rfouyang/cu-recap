from __future__ import annotations

import torch


@torch.no_grad()
def ensemble_statistics(predictions: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Combine M independently trained utility critics.

    total utility uncertainty =
      epistemic variance across means
      + mean aleatoric variance.
    """
    means = torch.stack([p["utility_mean"] for p in predictions], dim=0)
    aleatoric = torch.stack([p["utility_logvar"].exp() for p in predictions], dim=0)

    mean = means.mean(dim=0)
    epistemic_var = means.var(dim=0, unbiased=False)
    aleatoric_var = aleatoric.mean(dim=0)
    total_std = (epistemic_var + aleatoric_var).clamp_min(1e-12).sqrt()

    failure_prob = torch.stack(
        [torch.sigmoid(p["failure_logit"]) for p in predictions], dim=0
    ).mean(dim=0)

    correction_cost = torch.stack(
        [p["correction_cost"] for p in predictions], dim=0
    ).mean(dim=0)

    return {
        "utility_mean": mean,
        "utility_std": total_std,
        "failure_prob": failure_prob,
        "correction_cost": correction_cost,
        "epistemic_var": epistemic_var,
        "aleatoric_var": aleatoric_var,
    }
