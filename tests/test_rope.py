"""Tests for RotaryEmbedding / apply_rotary."""
import pytest
import torch

from model.rope import RotaryEmbedding, apply_rotary

DIM = 16


def test_output_shape_and_position_zero_identity() -> None:
    rot = RotaryEmbedding(DIM, max_len=64)
    x = torch.randn(3, DIM)
    assert rot.rotate(x, torch.zeros(3, dtype=torch.long)).allclose(x)
    assert rot.rotate(x, torch.tensor([5, 6, 7])).shape == x.shape


def test_matches_complex_multiplication() -> None:
    rot = RotaryEmbedding(DIM, max_len=32)
    x = torch.randn(5, DIM)
    pos = torch.tensor([0, 1, 7, 13, 31])
    got = rot.rotate(x, pos)
    z = torch.view_as_complex(x.reshape(5, DIM // 2, 2))
    theta = rot.inv_freq.to(torch.float64)
    expected = torch.view_as_real(
        z.to(torch.complex128) * torch.exp(1j * pos.to(torch.float64)[:, None] * theta[None, :])
    ).reshape(5, DIM).to(x.dtype)
    assert torch.allclose(got, expected, atol=1e-5)


def test_scores_depend_on_offset_only() -> None:
    q, k = torch.randn(1, DIM), torch.randn(1, DIM)
    for d in (1, 4, 9):
        scores = []
        for start in (0, 3, 17, 55, 101):
            qi = apply_rotary(q, torch.tensor([start]), DIM)
            kj = apply_rotary(k, torch.tensor([start + d]), DIM)
            scores.append((qi[0] @ kj[0]).item())
        assert max(scores) - min(scores) < 1e-5, d


def test_norm_preserved() -> None:
    x = torch.randn(7, DIM)
    xr = apply_rotary(x, torch.arange(7), DIM)
    assert torch.allclose(x.norm(dim=1), xr.norm(dim=1), atol=1e-5)


def test_frequency_ladder() -> None:
    rot = RotaryEmbedding(DIM, max_len=8)
    expected = 1.0 / 10000.0 ** (torch.arange(0, DIM, 2) / DIM)
    assert torch.allclose(rot.inv_freq, expected, atol=1e-7)
    # first pair fastest, last pair slowest
    assert rot.inv_freq[0] == 1.0
    assert rot.inv_freq[-1] < rot.inv_freq[0]


def test_additivity_over_composed_offsets() -> None:
    """R(a) @ R(b) == R(a+b): rotating twice composes the angles."""
    rot = RotaryEmbedding(DIM, max_len=64)
    x = torch.randn(1, DIM)
    a, b = 7, 11
    once = rot.rotate(rot.rotate(x, torch.tensor([b])), torch.tensor([a]))
    direct = rot.rotate(x, torch.tensor([a + b]))
    assert torch.allclose(once, direct, atol=1e-5)


def test_batch_broadcast() -> None:
    rot = RotaryEmbedding(DIM, max_len=32)
    x = torch.randn(4, 6, DIM)                       # (B, T, D)
    pos = torch.arange(6)
    out = rot.rotate(x, pos)
    assert out.shape == x.shape
    # equals applying each batch row with the same positions
    for b in range(4):
        assert torch.allclose(out[b], rot.rotate(x[b], pos), atol=1e-6)


def test_rejects_odd_dim() -> None:
    with pytest.raises(ValueError):
        RotaryEmbedding(DIM + 1)
