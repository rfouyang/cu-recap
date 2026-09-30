from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


Tensor = torch.Tensor


@dataclass
class InfluenceResult:
    correction_loss: float
    probe_loss_before: float
    grad_dot: float
    grad_cosine: float
    first_order_gain: float


def _grads(loss: Tensor, params: Sequence[torch.nn.Parameter]) -> list[Tensor]:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    out = []
    for p, g in zip(params, grads):
        out.append(torch.zeros_like(p) if g is None else g.detach())
    return out


def _dot(gs1: Sequence[Tensor], gs2: Sequence[Tensor]) -> Tensor:
    return sum((a * b).sum() for a, b in zip(gs1, gs2))


def _norm(gs: Sequence[Tensor]) -> Tensor:
    return torch.sqrt(sum((g * g).sum() for g in gs).clamp_min(1e-20))


def first_order_correction_utility(
    correction_loss_fn: Callable[[], Tensor],
    probe_loss_fn: Callable[[], Tensor],
    params: Sequence[torch.nn.Parameter],
    step_size: float,
) -> InfluenceResult:
    """
    First-order approximation of one correction update's improvement on a
    held-out hard-state probe buffer.

        theta' = theta - eta * g_c

        L_H(theta') ~= L_H(theta) - eta * g_H^T g_c

    Therefore the predicted *loss reduction* is

        U_FO = eta * g_H^T g_c

    Positive gradient alignment means the correction is expected to improve
    the probe objective after one SGD/LoRA step.
    """
    corr_loss = correction_loss_fn()
    g_c = _grads(corr_loss, params)

    probe_loss = probe_loss_fn()
    g_h = _grads(probe_loss, params)

    dot = _dot(g_h, g_c)
    cosine = dot / (_norm(g_h) * _norm(g_c)).clamp_min(1e-20)
    gain = float(step_size * dot.item())

    return InfluenceResult(
        correction_loss=float(corr_loss.detach().item()),
        probe_loss_before=float(probe_loss.detach().item()),
        grad_dot=float(dot.item()),
        grad_cosine=float(cosine.item()),
        first_order_gain=gain,
    )


@torch.no_grad()
def _snapshot(params: Sequence[torch.nn.Parameter]) -> list[Tensor]:
    return [p.detach().clone() for p in params]


@torch.no_grad()
def _restore(params: Sequence[torch.nn.Parameter], snapshot: Sequence[Tensor]) -> None:
    for p, saved in zip(params, snapshot):
        p.copy_(saved)


def one_step_probe_gain(
    correction_loss_fn: Callable[[], Tensor],
    probe_loss_fn: Callable[[], Tensor],
    params: Sequence[torch.nn.Parameter],
    step_size: float,
) -> float:
    """
    More expensive post-hoc utility target.

    Applies exactly one temporary SGD step on the selected parameter subset
    (typically LoRA adapters), measures held-out probe loss reduction, then
    restores the model.

        U_1step(c) = L_H(theta) - L_H(theta - eta * grad L_c)

    This is used as the high-quality supervision target for queried states.
    It does NOT mutate the model after returning.
    """
    params = list(params)
    snap = _snapshot(params)

    try:
        with torch.no_grad():
            before = float(probe_loss_fn().item())

        corr_loss = correction_loss_fn()
        grads = torch.autograd.grad(
            corr_loss,
            params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )

        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None:
                    p.add_(g, alpha=-step_size)

            after = float(probe_loss_fn().item())

        return before - after
    finally:
        _restore(params, snap)
