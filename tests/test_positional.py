"""Tests for sinusoidal PositionalEncoding."""
import math

import pytest
import torch

from model.positional import PositionalEncoding


def make_pe(d: int = 16, max_len: int = 128) -> PositionalEncoding:
    return PositionalEncoding(d, max_len)


def test_output_shape_and_addition() -> None:
    pe = make_pe()
    x = torch.randn(3, 10, 16)
    out = pe(x)
    assert out.shape == x.shape
    assert torch.allclose(out, x + pe.pe[:10])        # exactly additive


def test_formula_matches_closed_form() -> None:
    d, L = 8, 20
    pe = make_pe(d, L)
    for pos in range(L):
        for k in range(d // 2):
            div = 10000.0 ** (2 * k / d)
            assert pe.pe[pos, 2 * k].item() == pytest.approx(math.sin(pos / div), abs=1e-6)
            assert pe.pe[pos, 2 * k + 1].item() == pytest.approx(math.cos(pos / div), abs=1e-6)


def test_bounded_and_unique() -> None:
    pe = make_pe(16, 512)
    assert pe.pe.abs().max() <= 1.0
    # every position gets its own vector
    assert torch.unique(pe.pe, dim=0).shape[0] == 512


def test_relative_offset_is_rotation() -> None:
    """PE(pos+k) equals a FIXED rotation of PE(pos) per frequency pair.

    For one frequency w, the pair (sin, cos) at pos+k equals R(k) @ pair(pos),
    R(k) = [[cos kw, sin kw], [-sin kw, cos kw]], independent of pos.
    Verify numerically for several (pos, pos') at the same k.
    """
    d = 8
    pe = make_pe(d, 64)
    k = 5
    for pair in range(d // 2):
        w = 1.0 / 10000.0 ** (2 * pair / d)
        R = torch.tensor([[math.cos(k * w), math.sin(k * w)],
                          [-math.sin(k * w), math.cos(k * w)]])
        for pos in range(40):
            got = pe.pe[pos + k, 2 * pair : 2 * pair + 2]
            expected = R @ pe.pe[pos, 2 * pair : 2 * pair + 2]
            assert torch.allclose(got, expected, atol=1e-5)


def test_order_visible_in_embedding_stream() -> None:
    from model.embeddings import TokenEmbedding
    torch.manual_seed(0)
    emb = TokenEmbedding(10, 8)
    pe = make_pe(8, 32)
    a = torch.tensor([[4, 7]])
    b = torch.tensor([[7, 4]])
    ha, hb = pe(emb(a)), pe(emb(b))
    # multisets of rows must differ once PE is added
    sa = {tuple(v.tolist()) for v in ha[0]}
    sb = {tuple(v.tolist()) for v in hb[0]}
    assert sa != sb
    # but without PE the multisets coincide
    assert {tuple(v.tolist()) for v in emb(a)[0]} == {tuple(v.tolist()) for v in emb(b)[0]}


def test_no_learned_parameters() -> None:
    pe = make_pe()
    assert list(pe.parameters()) == []          # buffer only, nothing to learn
    assert pe.pe.grad is None                   # buffers receive no gradient
    # With no grad-requiring input, the output is outside any graph and
    # backward() raises (autograd rule from Lab 03: needs a tracked graph).
    out = pe(torch.randn(2, 5, 16))
    assert out.requires_grad is False
    with pytest.raises(RuntimeError):
        out.sum().backward()
    # But gradients DO flow through PE untouched when the input is trainable
    # (real usage: x is the embedding output, which requires grad).
    x = torch.randn(2, 5, 16, requires_grad=True)
    pe(x).sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x))
    assert pe.pe.grad is None                      # the buffer still gets no grad
