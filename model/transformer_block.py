"""Transformer decoder block (pre-norm), from scratch — the KrishiGPT unit cell.

    x ──┬──────────────────────┐        x ──┬────────────────┐
        v                      │            v                │
      LN1 -> MultiHeadAttn -- (+)         LN2 -> FFN ------ (+)

    forward(x):                                   # (B, T, D)
        h   = LN1(x)
        a   = dropout(MHA(h))                     # RoPE on Q/K, causal inside
        x   = x + a                               # RESIDUAL: raw stream untouched
        h   = LN2(x)
        f   = dropout(FFN(h))                     # D -> 4D -> D with GELU
        return x + f, attn_weights

WHY RESIDUALS: grad(x + F(x)) = I + J_F. Stacked N blocks give 2^N gradient
paths, one of which is the pure identity (grad 1) -> deep training is possible.
WHY PRE-NORM: the identity path must carry x UNnormalized; LN sits on the
sublayer INPUT so the sublayer is well-scaled while the stream stays pristine.
The slow norm growth of the stream is cleaned up by the FINAL LayerNorm that
lives after the last block (inside the GPT, next component).

Parameter count per block (D=d_model):
    no bias : 4D^2 (MHA) + 8D^2 (FFN) + 4D (two LN) = 197,120 at D=128
    with bias (GPT-2 style, our default): + 4D (MHA) + 5D (FFN) = 198,272

Experiment:  python model/transformer_block.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.feed_forward import FeedForward  # noqa: E402
from model.multi_head import MultiHeadAttention  # noqa: E402
from model.normalization import LayerNorm  # noqa: E402


class TransformerBlock(nn.Module):
    """One GPT-style pre-norm decoder block (attention sublayer + FFN sublayer)."""

    def __init__(self, d_model: int, n_heads: int, max_len: int = 512,
                 dropout: float = 0.0, ffn_mult: int = 4,
                 bias: bool = True) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_len=max_len, bias=bias)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_mult=ffn_mult, bias=bias)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                positions: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D) -> (out (B, T, D), attention weights (B, H, T, T))."""
        h, weights = self.attn(self.ln1(x), positions)
        x = x + self.drop1(h)                      # residual 1
        f = self.ffn(self.ln2(x))
        x = x + self.drop2(f)                      # residual 2
        return x, weights


def experiment() -> None:
    from tokenizer import BPETokenizer

    torch.manual_seed(0)

    # -- 1. nano-config shapes ---------------------------------------------------
    B, T, D, H = 2, 256, 128, 4
    blk = TransformerBlock(D, H)
    x = torch.randn(B, T, D)
    out, w = blk(x)
    n_params = sum(p.numel() for p in blk.parameters())
    print(f"block @ B={B} T={T} D={D} H={H}:")
    print(f"  in {tuple(x.shape)} -> out {tuple(out.shape)}, weights {tuple(w.shape)}")
    print(f"  params (bias=True) = {n_params:,}  (bias=False would be 197,120)")
    assert out.shape == x.shape and w.shape == (B, H, T, T)
    assert n_params == 4 * D * D + 4 * D + 8 * D * D + 5 * D + 4 * D

    # -- 2. zero-init sublayers -> block is EXACTLY the identity ------------------
    # Proof of the residual concept: the stream passes through untouched when
    # both sublayers output zero, whatever LN does to them.
    blk0 = TransformerBlock(16, 4)
    with torch.no_grad():
        for name, p in blk0.named_parameters():
            if any(s in name for s in ("wq", "wk", "wv", "wo", "fc1", "fc2")):
                p.zero_()
    x0 = torch.randn(1, 6, 16)
    out0, _ = blk0(x0)
    identity = torch.allclose(out0, x0, atol=1e-6)
    print(f"\nzero-initialized sublayers -> block(x) == x exactly: {identity}")
    assert identity

    # -- 3. causality survives the full block ---------------------------------------
    blk3 = TransformerBlock(16, 4)
    x3 = torch.randn(1, 6, 16)
    o3, _ = blk3(x3)
    x3b = x3.clone()
    x3b[0, 3:] = torch.randn(3, 16) * 50
    o3b, _ = blk3(x3b)
    past = torch.allclose(o3[:, :3], o3b[:, :3], atol=1e-5)
    print(f"edit tokens 3..5 -> outputs 0..2 unchanged: {past}")
    assert past

    # -- 4. stream norm drifts slowly across stacked blocks --------------------------
    blk4 = TransformerBlock(32, 4)
    xs = torch.randn(1, 16, 32)
    norms = [xs.norm(dim=-1).mean().item()]
    h = xs
    for _ in range(4):
        h, _ = blk4(h)
        norms.append(h.norm(dim=-1).mean().item())
    print("\nresidual-stream mean norm after 0..4 blocks:",
          [f"{v:.2f}" for v in norms], " <- slow drift, final LN cleans it up")

    # -- 5. agriculture attention through the block -----------------------------------
    corpus = " ".join([
        "apply fertilizer before irrigation"] * 2 + [
        "wheat rust is a disease", "soil needs nitrogen"])
    tok = BPETokenizer.train(corpus, num_merges=30)
    ids = torch.tensor([tok.encode("soil needs nitrogen")])
    pieces = [tok.id2token[i] for i in ids[0].tolist()]
    emb = torch.randn(tok.vocab_size, 32) * 0.1
    blk5 = TransformerBlock(32, 4)
    _, w5 = blk5(emb[ids])
    print(f"\ntokens: {pieces}")
    q_pos = len(pieces) - 1
    print(f"last-token attention rows for '{pieces[q_pos]}' heads 0..3:")
    for hh in range(4):
        print("  head", hh, ":", ["%.3f" % v for v in w5[0, hh, q_pos].tolist()])

    print("\nEXPERIMENT COMPLETE: transformer block verified.")


if __name__ == "__main__":
    experiment()
