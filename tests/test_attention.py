"""Tests for scaled_dot_product_attention (with optional causal mask)."""
import math

import pytest
import torch

from model.attention import causal_mask, scaled_dot_product_attention


def test_shapes_2d() -> None:
    T, dk, dv = 7, 6, 4
    out, w = scaled_dot_product_attention(
        torch.randn(T, dk), torch.randn(T, dk), torch.randn(T, dv)
    )
    assert out.shape == (T, dv)
    assert w.shape == (T, T)


def test_shapes_batched() -> None:
    out, w = scaled_dot_product_attention(
        torch.randn(3, 5, 8), torch.randn(3, 5, 8), torch.randn(3, 5, 4)
    )
    assert out.shape == (3, 5, 4)
    assert w.shape == (3, 5, 5)


def test_weights_are_distributions_rowwise() -> None:
    _, w = scaled_dot_product_attention(
        torch.randn(2, 6, 8), torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    )
    assert torch.allclose(w.sum(-1), torch.ones(2, 6), atol=1e-5)
    assert (w >= 0).all()


def test_matches_manual_rowwise_loop() -> None:
    torch.manual_seed(0)
    q, k, v = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 3)
    out, _ = scaled_dot_product_attention(q, k, v)
    for i in range(5):
        scores = q[i] @ k.transpose(0, 1) / math.sqrt(4)   # (5,)
        manual_w = scores.exp() / scores.exp().sum()
        manual_out = manual_w @ v
        assert torch.allclose(out[i], manual_out, atol=1e-6)


def test_scaling_reduces_saturation() -> None:
    torch.manual_seed(1)
    q, k = torch.randn(1, 1, 128), torch.randn(1, 64, 128)
    v = torch.randn(1, 64, 8)
    _, w_off = scaled_dot_product_attention(q, k, v, scale=False)
    _, w_on = scaled_dot_product_attention(q, k, v, scale=True)
    assert w_off.max() >= w_on.max()                       # scaled distribution softer
    ent_off = -(w_off * w_off.log()).sum().item()
    ent_on = -(w_on * w_on.log()).sum().item()
    assert ent_on > ent_off                                # higher entropy after scaling


def test_uniform_keys_give_mean_of_values() -> None:
    """If all keys are equal, q matches them equally -> output = mean(v)."""
    q = torch.randn(2, 3, 8)
    k0 = torch.randn(1, 1, 8)
    k = k0.expand(2, 5, 8)
    v = torch.randn(2, 5, 4)
    out, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w, torch.full((2, 3, 5), 1.0 / 5), atol=1e-6)
    assert torch.allclose(out, v.mean(dim=1, keepdim=True).expand(2, 3, 4), atol=1e-6)


def test_single_key_passes_value_through() -> None:
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 1, 8)
    v = torch.randn(2, 1, 4)
    out, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w, torch.ones(2, 3, 1), atol=1e-6)
    assert torch.allclose(out, v.expand(2, 3, 4), atol=1e-6)


def test_mismatched_dk_raises() -> None:
    with pytest.raises(AssertionError):
        scaled_dot_product_attention(torch.randn(2, 4), torch.randn(2, 5), torch.randn(2, 3))


def test_gradient_flows_to_all_inputs() -> None:
    q = torch.randn(4, 4, requires_grad=True)
    k = torch.randn(4, 4, requires_grad=True)
    v = torch.randn(4, 4, requires_grad=True)
    out, _ = scaled_dot_product_attention(q, k, v)
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.any(t.grad != 0)


# ---------------- causal masking ----------------

def test_causal_mask_structure() -> None:
    m = causal_mask(4)
    assert m.shape == (4, 4)
    idx = torch.triu_indices(4, 4, offset=1)           # strictly upper cells
    assert torch.all(m[idx[0], idx[1]].isneginf())     # future banned = -inf
    assert torch.tril(m).eq(0).all()                   # on/under diagonal = 0
    assert m[0, 0] == 0 and m[3, 3] == 0               # diagonal allowed


def test_masked_weights_zero_above_diagonal() -> None:
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 6, 8), torch.randn(2, 6, 8), torch.randn(2, 6, 4)
    _, w = scaled_dot_product_attention(q, k, v, mask=causal_mask(6))
    for i in range(6):
        assert torch.all(w[:, i, i + 1:] == 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 6), atol=1e-5)  # renormalized


def test_row0_attends_only_to_itself() -> None:
    q, k, v = torch.randn(1, 5, 8), torch.randn(1, 5, 8), torch.randn(1, 5, 4)
    out, w = scaled_dot_product_attention(q, k, v, mask=causal_mask(5))
    assert torch.allclose(w[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(out[0, 0], v[0, 0], atol=1e-6)   # pass-through


def test_causal_blocks_future_information_flow() -> None:
    """Corrupting tokens 3.. must never change outputs 0..2."""
    torch.manual_seed(1)
    T = 6
    q, k, v = torch.randn(1, T, 8), torch.randn(1, T, 8), torch.randn(1, T, 4)
    out, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask(T))
    k2 = k.clone(); v2 = v.clone()
    k2[0, 3:] = torch.randn(3, 8) * 1000
    v2[0, 3:] = torch.randn(3, 4) * 1000
    out2, _ = scaled_dot_product_attention(q, k2, v2, mask=causal_mask(T))
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-5)
    assert not torch.allclose(out[:, 3:], out2[:, 3:], atol=1e-3)


def test_unmasked_leaks_future() -> None:
    torch.manual_seed(2)
    T = 5
    q, k, v = torch.randn(1, T, 8), torch.randn(1, T, 8), torch.randn(1, T, 4)
    k2 = k.clone(); v2 = v.clone()
    k2[0, 2:] = torch.randn(3, 8) * 1000
    out_a, _ = scaled_dot_product_attention(q, k, v)       # no mask
    out_b, _ = scaled_dot_product_attention(q, k2, v2)
    assert not torch.allclose(out_a[:, :2], out_b[:, :2], atol=1e-3)  # leak!


def test_last_row_equals_unmasked_row() -> None:
    """Row T-1 sees everything, so causal output == unmasked output there."""
    torch.manual_seed(3)
    T = 4
    q, k, v = torch.randn(2, T, 8), torch.randn(2, T, 8), torch.randn(2, T, 4)
    out_c, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask(T))
    out_u, _ = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out_c[:, -1], out_u[:, -1], atol=1e-6)


def test_mask_broadcasts_over_batch_and_heads() -> None:
    q = torch.randn(3, 2, 5, 8)              # (B, H, T, d)
    k = torch.randn(3, 2, 5, 8)
    v = torch.randn(3, 2, 5, 4)
    out, w = scaled_dot_product_attention(q, k, v, mask=causal_mask(5))  # (T, T) broadcasts
    assert out.shape == (3, 2, 5, 4) and w.shape == (3, 2, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(3, 2, 5), atol=1e-5)
