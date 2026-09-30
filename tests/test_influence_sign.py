import torch

from cu_recap.utility.grad_influence import (
    first_order_correction_utility,
    one_step_probe_gain,
)


def test_aligned_gradient_predicts_positive_probe_gain():
    w = torch.nn.Parameter(torch.tensor([0.0]))

    def corr_loss():
        return ((w - 1.0) ** 2).mean()

    def probe_loss():
        return ((w - 1.0) ** 2).mean()

    fo = first_order_correction_utility(
        corr_loss, probe_loss, [w], step_size=0.1
    )
    exact = one_step_probe_gain(
        corr_loss, probe_loss, [w], step_size=0.1
    )

    assert fo.first_order_gain > 0
    assert exact > 0


def test_opposed_gradient_predicts_negative_probe_gain():
    w = torch.nn.Parameter(torch.tensor([0.0]))

    def corr_loss():
        return ((w - 1.0) ** 2).mean()

    def probe_loss():
        return ((w + 1.0) ** 2).mean()

    fo = first_order_correction_utility(
        corr_loss, probe_loss, [w], step_size=0.1
    )
    exact = one_step_probe_gain(
        corr_loss, probe_loss, [w], step_size=0.1
    )

    assert fo.first_order_gain < 0
    assert exact < 0
