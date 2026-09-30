from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class AcquisitionConfig:
    beta_ucb: float = 1.0
    temperature: float = 0.25
    min_propensity: float = 0.05
    max_propensity: float = 0.95
    safety_risk_threshold: float = 0.95

    dual_init: float = 0.10
    dual_lr: float = 0.01
    target_cost_per_decision: float = 0.05


class BudgetedQueryPolicy:
    """
    Stochastic utility-per-cost acquisition with a dual budget controller.

    Learning query:
        score = utility_UCB - lambda * predicted_cost
        p(query) = sigmoid(score / temperature)

    Safety query:
        deterministic override based on failure risk.
        Safety is intentionally separated from learning utility.

    The returned propensity MUST be logged for IPW training.
    """

    def __init__(self, cfg: AcquisitionConfig = AcquisitionConfig(), seed: int = 0):
        self.cfg = cfg
        self.dual = cfg.dual_init
        self.rng = random.Random(seed)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def decide(
        self,
        utility_mean: float,
        utility_std: float,
        predicted_cost: float,
        failure_risk: float,
    ) -> dict:
        if failure_risk >= self.cfg.safety_risk_threshold:
            return {
                "query": True,
                "propensity": 1.0,
                "reason": "safety",
                "acquisition_score": float("inf"),
            }

        ucb = utility_mean + self.cfg.beta_ucb * utility_std
        score = ucb - self.dual * max(predicted_cost, 1e-3)
        p = self._sigmoid(score / max(self.cfg.temperature, 1e-6))
        p = min(self.cfg.max_propensity, max(self.cfg.min_propensity, p))

        query = self.rng.random() < p
        return {
            "query": query,
            "propensity": p,
            "reason": "utility" if query else "none",
            "acquisition_score": score,
            "utility_ucb": ucb,
        }

    def update_dual(self, observed_cost: float) -> None:
        violation = observed_cost - self.cfg.target_cost_per_decision
        self.dual = max(0.0, self.dual + self.cfg.dual_lr * violation)
