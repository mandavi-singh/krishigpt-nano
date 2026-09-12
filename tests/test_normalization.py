"""Tests for LayerNorm (from scratch)."""
import torch
import torch.nn as nn

from model.normalization import LayerNorm

D = 16


def make(dim: int = D, eps: float = 1e-5) -> LayerNorm:
    torch.manual_seed(0)
    return LayerNorm(dim, eps=eps)


def test_hand_example_4_2() -> None:
    """Lesson check: mu=3, var=1 -> [4,2] maps to [1,-1]."""
    ln = LayerNorm(2, eps=0.0)
    out = ln(torch.tensor([4.0, 2.0]))
    assert torch.allclose(out, torch.tensor([1.0, -1.0]), atol=1e-6)


def test_irrigation_example_output_stats() -> None:
    ln = LayerNorm(4, eps=0.0)
    out = ln(torch.tensor([3.0, 7.0, 5.0, 1.0]))
    # closed form from the lesson: [-0.447, 1.342, 0.447, -1.342]
    expected = torch.tensor([-0.4472136, 1.3416408, 0.4472136, -1.3416408])
    assert torch.allclose(out, expected, atol=1e-6)
    assert abs(out.mean().item()) < 1e-6
    assert abs(out.var(correction=0).item() - 1.0) < 1e-6


def test_output_zero_mean_unit_variance() -> None:
    ln = make(eps=0.0)
    x = torch.randn(3, 7, D) * 10 + 3
    out = ln(x)
    means = out.mean(dim=-1)
    vars_ = out.var(dim=-1, correction=0)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-5)


def test_matches_nn_layernorm_with_shared_weights() -> None:
    ln = make()
    ref = nn.LayerNorm(D)
    ref.weight.data.copy_(ln.gamma.data)
    ref.bias.data.copy_(ln.beta.data)
    x = torch.randn(2, 5, D)
    assert torch.allclose(ln(x), ref(x), atol=1e-6)


def test_per_token_independence() -> None:
    """Normalizing one token must not depend on any other token/batch element."""
    ln = make()
    x = torch.randn(2, 6, D)
    y = ln(x)
    for b in range(2):
        for t in range(6):
            assert torch.allclose(y[b, t], ln(x[b, t]), atol=1e-6)
    # and editing batch element (1,0,...) changes only its own output
    x2 = x.clone()
    x2[1, 0] = torch.randn(D) * 100
    y2 = ln(x2)
    assert torch.allclose(y[0], y2[0], atol=1e-6)
    assert torch.allclose(y[1, 1:], y2[1, 1:], atol=1e-6)
    assert not torch.allclose(y[1, 0], y2[1, 0])


def test_zero_variance_token_is_safe() -> None:
    ln = make(4)
    out = ln(torch.tensor([[2.0, 2.0, 2.0, 2.0]]))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros(1, 4), atol=1e-5)  # all deviations 0


def test_biased_variance_convention() -> None:
    """LayerNorm divides var by D (correction=0), not D-1."""
    ln = LayerNorm(3, eps=0.0)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = ln(x)
    # biased var = 2/3; std = sqrt(2/3). (x-mu)/std: (-1, 0, +1)/sqrt(2/3)
    expected = torch.tensor([[-1.2247449, 0.0, 1.2247449]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_gradient_flows_through_gamma_beta_and_input() -> None:
    ln = make()
    x = torch.randn(2, 4, D, requires_grad=True)
    ln(x).sum().backward()
    assert ln.gamma.grad is not None and torch.any(ln.gamma.grad != 0)
    assert ln.beta.grad is not None and torch.any(ln.beta.grad != 0)
    assert x.grad is not None and torch.any(x.grad != 0)


def test_train_eval_identical() -> None:
    ln = make()
    x = torch.randn(2, 5, D)
    ln.train()
    a = ln(x).detach()
    ln.eval()
    b = ln(x).detach()
    assert torch.equal(a, b)


def test_gamma_beta_can_learn_affine() -> None:
    """With learned gamma/beta the output need not be mean-0/var-1."""
    ln = LayerNorm(4, eps=0.0)
    with torch.no_grad():
        ln.gamma.copy_(torch.tensor([2.0, 0.5, 1.0, 3.0]))
        ln.beta.copy_(torch.tensor([1.0, -1.0, 0.0, 0.5]))
    out = ln(torch.tensor([[3.0, 7.0, 5.0, 1.0]]))
    # normalized = [-0.447, 1.342, 0.447, -1.342]; then gamma*x + beta
    expected = torch.tensor([[2 * -0.4472 + 1.0, 0.5 * 1.3416 - 1.0,
                              1.0 * 0.4472, 3.0 * -1.3416 + 0.5]])
    assert torch.allclose(out, expected, atol=1e-4)
