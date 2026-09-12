"""Tests for TransformerBlock (pre-norm decoder block)."""
import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock

D, H, T, B = 32, 4, 10, 2


def make(dropout: float = 0.0, bias: bool = True) -> TransformerBlock:
    torch.manual_seed(0)
    return TransformerBlock(D, H, dropout=dropout, bias=bias)


def test_shapes() -> None:
    blk = make()
    out, w = blk(torch.randn(B, T, D))
    assert out.shape == (B, T, D)
    assert w.shape == (B, H, T, T)


def test_parameter_count_bias_true() -> None:
    blk = make(bias=True)
    expected = 4 * D * D + 4 * D + 8 * D * D + 5 * D + 4 * D
    assert sum(p.numel() for p in blk.parameters()) == expected


def test_parameter_count_bias_false_matches_lesson() -> None:
    blk = make(bias=False)
    # the lesson's 197,120 formula at D=128: 4D^2 + 8D^2 + 4D
    assert sum(p.numel() for p in blk.parameters()) == 4 * D * D + 8 * D * D + 4 * D
    blk128 = TransformerBlock(128, 4, bias=False)
    assert sum(p.numel() for p in blk128.parameters()) == 197_120


def test_zero_init_sublayers_give_identity() -> None:
    """block(x) == x when both sublayers output zero: the stream passes through
    the residuals untouched, independent of what LN/LN2 compute."""
    blk = make()
    with torch.no_grad():
        for name, p in blk.named_parameters():
            if any(s in name for s in ("wq", "wk", "wv", "wo", "fc1", "fc2")):
                p.zero_()
    x = torch.randn(1, 6, D)
    out, _ = blk(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_causality_through_block() -> None:
    blk = make()
    x = torch.randn(1, 6, D)
    out, _ = blk(x)
    x2 = x.clone()
    x2[0, 3:] = torch.randn(3, D) * 100
    out2, _ = blk(x2)
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-5)
    assert not torch.allclose(out[:, 3:], out2[:, 3:], atol=1e-3)


def test_prenorm_sublayer_input_is_normalized() -> None:
    """Hook LN1: whatever x's scale, the attention sublayer receives unit-scale."""
    blk = make()
    received = {}
    handle = blk.attn.register_forward_pre_hook(
        lambda mod, inp: received.setdefault("x", inp[0].detach().clone())
    )
    x = torch.randn(1, 4, D) * 37 + 5             # deliberately messy scale
    blk(x)
    handle.remove()
    normed = received["x"]
    means = normed.mean(dim=-1)
    vars_ = normed.var(dim=-1, correction=0)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)
    assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-4)


def test_output_is_raw_stream_plus_correction() -> None:
    """out - x equals the two (dropout-free) sublayer corrections exactly."""
    blk = make(dropout=0.0)
    x = torch.randn(1, 4, D)
    out, _ = blk(x)
    # recompute corrections manually following the documented order
    h, _ = blk.attn(blk.ln1(x))
    x_mid = x + h
    f = blk.ffn(blk.ln2(x_mid))
    assert torch.allclose(out, x_mid + f, atol=1e-6)


def test_dropout_only_active_in_train() -> None:
    x = torch.randn(1, 4, D)
    blk = make(dropout=0.5)
    blk.eval()
    a, _ = blk(x)
    b, _ = blk(x)
    assert torch.equal(a, b)                       # deterministic in eval
    blk.train()
    runs = [blk(x)[0] for _ in range(3)]
    assert not (torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2]))


def test_gradient_flows_to_all_parameters() -> None:
    blk = make()
    x = torch.randn(1, 4, D, requires_grad=True)
    out, _ = blk(x)
    out.sum().backward()
    dead = [n for n, p in blk.named_parameters()
            if p.grad is None or not torch.any(p.grad != 0)]
    assert dead == [], f"no gradient for: {dead}"
    assert x.grad is not None and torch.any(x.grad != 0)


def test_positions_argument_supported() -> None:
    blk = make()
    content = torch.randn(1, 4, D)
    _, wa = blk(content, positions=torch.tensor([0, 1, 2, 3]))
    _, wb = blk(content, positions=torch.tensor([7, 8, 9, 10]))
    assert torch.allclose(wa, wb, atol=1e-5)       # RoPE translation invariance
