"""Tests for MultiHeadAttention (from scratch, no nn.MultiheadAttention)."""
import pytest
import torch

from model.attention import causal_mask, scaled_dot_product_attention
from model.multi_head import MultiHeadAttention

D, H, T, B = 32, 4, 12, 2


def make() -> MultiHeadAttention:
    torch.manual_seed(0)
    return MultiHeadAttention(D, H)


def test_shapes() -> None:
    m = make()
    out, w = m(torch.randn(B, T, D))
    assert out.shape == (B, T, D)
    assert w.shape == (B, H, T, T)


def test_parameter_count_is_four_full_matrices() -> None:
    m = make()
    assert sum(p.numel() for p in m.parameters()) == 4 * D * D


def test_weights_are_causal_distributions() -> None:
    m = make()
    _, w = m(torch.randn(B, T, D))
    for i in range(T - 1):
        assert torch.all(w[:, :, i, i + 1:] == 0)
    assert torch.allclose(w.sum(-1), torch.ones(B, H, T), atol=1e-5)


def test_single_head_equals_base_attention() -> None:
    torch.manual_seed(1)
    m = MultiHeadAttention(D, n_heads=1, use_rope=False)
    x = torch.randn(B, T, D)
    out, w = m(x)
    q, k, v = m.wq(x), m.wk(x), m.wv(x)
    out_ref, w_ref = scaled_dot_product_attention(q, k, v, mask=causal_mask(T))
    assert torch.allclose(w[:, 0], w_ref, atol=1e-6)
    assert torch.allclose(out, m.wo(out_ref), atol=1e-5)


def test_causality_blocks_future() -> None:
    m = make()
    x = torch.randn(1, 6, D)
    out, _ = m(x)
    x2 = x.clone()
    x2[0, 3:] = torch.randn(3, D) * 100
    out2, _ = m(x2)
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-5)
    assert not torch.allclose(out[:, 3:], out2[:, 3:], atol=1e-3)


def test_rope_translation_invariance() -> None:
    """Same content at shifted positions -> identical attention weights."""
    m = MultiHeadAttention(D, H)
    content = torch.randn(1, 4, D)
    _, wa = m(content, positions=torch.tensor([0, 1, 2, 3]))
    _, wb = m(content, positions=torch.tensor([10, 11, 12, 13]))
    assert torch.allclose(wa, wb, atol=1e-5)


def test_rope_breaks_pure_content_invariance() -> None:
    """With RoPE, changing ONLY positions (same x) changes weights."""
    m = MultiHeadAttention(D, H)
    content = torch.randn(1, 4, D)
    _, wa = m(content, positions=torch.tensor([0, 1, 2, 3]))
    _, wb = m(content, positions=torch.tensor([0, 2, 4, 6]))   # gaps changed
    assert not torch.allclose(wa, wb, atol=1e-4)


def test_v_not_rotated() -> None:
    """Rotating positions changes WHERE heads attend, but if all heads attended
    uniformly (equal keys), outputs would be the unrotated mean of V. Instead
    we check directly: rope is applied only to q/k in code (structural test) --
    here we verify changing positions leaves V-path content intact by comparing
    outputs under a uniform-key trick at H=1 without RoPE vs with RoPE at gap 0.
    Simpler invariant: with positions all zero, RoPE is the identity, so the
    module must equal its use_rope=False twin given shared weights."""
    torch.manual_seed(2)
    m_rope = MultiHeadAttention(D, H, use_rope=True)
    m_plain = MultiHeadAttention(D, H, use_rope=False)
    # strict=False: the rope twin has extra fixed buffers (rope.cos/sin);
    # the four weight matrices are what we are comparing.
    m_plain.load_state_dict(m_rope.state_dict(), strict=False)
    for name, param in m_plain.named_parameters():
        assert torch.equal(param, dict(m_rope.named_parameters())[name])
    x = torch.randn(1, 4, D)
    out_r, w_r = m_rope(x, positions=torch.zeros(4, dtype=torch.long))  # identity rotation
    out_p, w_p = m_plain(x)
    assert torch.allclose(w_r, w_p, atol=1e-5)
    assert torch.allclose(out_r, out_p, atol=1e-5)


def test_rejects_bad_head_split() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(30, 4)


def test_gradient_flows_to_all_four_matrices() -> None:
    m = make()
    x = torch.randn(B, T, D, requires_grad=True)
    out, _ = m(x)
    out.sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.any(p.grad != 0), name
    assert x.grad is not None
