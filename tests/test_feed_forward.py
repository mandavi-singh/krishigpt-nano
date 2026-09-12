"""Tests for FeedForward and GELU (exact + GPT-2 approximation)."""
import math

import pytest
import torch
import torch.nn.functional as F

from model.feed_forward import FeedForward, gelu_approx, gelu_exact

D = 32


def test_gelu_exact_reference_values() -> None:
    # Phi(1) = 0.84134..., Phi(-1) = 0.15865...
    assert gelu_exact(torch.tensor(1.0)).item() == pytest.approx(0.84134, abs=1e-5)
    assert gelu_exact(torch.tensor(-1.0)).item() == pytest.approx(-0.15865, abs=1e-5)
    assert gelu_exact(torch.tensor(0.0)).item() == 0.0
    # odd symmetry: gelu_exact(-x) != -gelu_exact(x), but check identity x*Phi(x)
    x = torch.tensor(2.0)
    phi = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    assert torch.allclose(gelu_exact(x), x * phi)


def test_gelu_exact_matches_torch_reference() -> None:
    """Independent reference: nn.functional.gelu (exact mode)."""
    xs = torch.linspace(-6, 6, 1001)
    assert torch.allclose(gelu_exact(xs), F.gelu(xs, approximate="none"), atol=1e-6)


def test_gelu_approx_matches_torch_reference() -> None:
    xs = torch.linspace(-6, 6, 1001)
    assert torch.allclose(gelu_approx(xs), F.gelu(xs, approximate="tanh"), atol=1e-6)


def test_approx_close_to_exact_but_not_equal() -> None:
    xs = torch.linspace(-10, 10, 20001)
    err = (gelu_exact(xs) - gelu_approx(xs)).abs()
    assert err.max() < 1e-3                      # tight enough for training
    assert err.max() > 1e-5                      # but a genuine approximation


def test_soft_gate_negative_survives() -> None:
    assert gelu_exact(torch.tensor(-1.0)).item() < 0       # negative output
    assert gelu_exact(torch.tensor(-1.0)).item() > -0.2
    # nonzero gradient in the negative region (no dead units)
    x = torch.tensor(-2.0, requires_grad=True)
    gelu_exact(x).backward()
    assert x.grad.item() != 0.0
    # contrast: ReLU grad at -2 is exactly 0
    y = torch.tensor(-2.0, requires_grad=True)
    F.relu(y).backward()
    assert y.grad.item() == 0.0


def test_ff_shape() -> None:
    ff = FeedForward(D)
    out = ff(torch.randn(3, 7, D))
    assert out.shape == (3, 7, D)                  # D in, D out per position


def test_ff_parameter_count() -> None:
    nob = FeedForward(D, hidden_mult=4, bias=False)
    assert sum(p.numel() for p in nob.parameters()) == 8 * D * D
    withb = FeedForward(D, hidden_mult=4, bias=True)
    assert sum(p.numel() for p in withb.parameters()) == 8 * D * D + D + 4 * D


def test_position_independence() -> None:
    torch.manual_seed(0)
    ff = FeedForward(D)
    x = torch.randn(2, 5, D)
    y = ff(x)
    for b in range(2):
        for t in range(5):
            assert torch.allclose(y[b, t], ff(x[b, t]), atol=1e-6)


def test_per_token_mixing_only() -> None:
    """Editing one position's input must not change any other position's output."""
    ff = FeedForward(D)
    x = torch.randn(1, 6, D)
    y = ff(x)
    x2 = x.clone()
    x2[0, 3] = torch.randn(D) * 100
    y2 = ff(x2)
    mask = torch.ones(6, dtype=torch.bool); mask[3] = False
    assert torch.allclose(y[0, mask], y2[0, mask], atol=1e-6)
    assert not torch.allclose(y[0, 3], y2[0, 3])


def test_gradient_flows() -> None:
    ff = FeedForward(D)
    x = torch.randn(2, 4, D, requires_grad=True)
    ff(x).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)
    for n, p in ff.named_parameters():
        assert p.grad is not None and torch.any(p.grad != 0), n
