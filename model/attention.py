"""Scaled dot-product attention, from scratch (optionally causal).

    scores  = Q K^T / sqrt(d_k)          (... , T, T)
    scores += mask                          causal mask, -inf in future cells
    weights = softmax(scores, row-wise)  each row sums to 1
    out     = weights @ V                (..., T, d_k)

Q (queries) = "what am I looking for?", K (keys) = "what do I advertise?",
V (values) = "what content I hand over". Attention is a soft lookup: every
position's output becomes a blend of ALL values, weighted by learned match
scores. This is where information starts moving between tokens.

WHY /sqrt(d_k): a dot product of d_k independent unit-variance components has
variance ~ d_k. Large scores saturate softmax (one-hot, near-zero gradients).
Scaling restores unit variance so softmax stays in its sensitive region.

WHY mask BEFORE softmax: exp(-inf) = 0 removes the entry from the numerator AND
the denominator, so allowed weights renormalize to sum to 1. Zeroing weights
after softmax would leave them un-normalized and already contaminated by the
future scores in the denominator. (fp16 kernels use a large finite value like
-1e9 because -inf can produce NaNs.)

WHY CAUSAL MATTERS: a GPT predicting token i must not read tokens > i.
Training with the mask yields ALL T prefix predictions in one parallel pass,
and the trained behavior matches autoregressive generation by construction.

Experiment:  python model/attention.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MINUS_INF = float("-inf")


def causal_mask(seq_len: int) -> torch.Tensor:
    """Upper-triangular mask (T, T): 0 on/under the diagonal, -inf above.

    Diagonal allowed: a token may attend to itself. Strict future (j > i) masked.
    """
    return torch.triu(torch.full((seq_len, seq_len), MINUS_INF), diagonal=1)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: bool = True,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core attention math with explicit Q, K, V.

    Args:
        q: (..., T_q, d_k)  queries
        k: (..., T_k, d_k)  keys
        v: (..., T_k, d_v)  values
        mask: optional broadcastable (T_q, T_k) additive mask (-inf for banned cells)
    Returns:
        out:     (..., T_q, d_v)  blended values
        weights: (..., T_q, T_k)  row-wise softmax attention probabilities
    """
    assert q.shape[-1] == k.shape[-1], "q and k must share d_k"
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1)            # (..., T_q, T_k)
    if scale:
        scores = scores / math.sqrt(d_k)        # variance control
    if mask is not None:
        scores = scores + mask                  # -inf kills future cells
    weights = torch.softmax(scores, dim=-1)     # rows sum to 1
    out = weights @ v                           # (..., T_q, d_v)
    return out, weights


def experiment() -> None:
    """Replicate the lesson's hand examples and prove causality numerically."""
    # -- 1. the hand-worked 'wheat rust spreads' example -----------------------
    T = 3
    scores = torch.tensor([[1.0, 2.0, 3.0],
                           [0.5, 1.5, 2.5],
                           [0.1, 0.9, 1.9]])               # pre-scaled scores
    w = torch.softmax(scores + causal_mask(T), dim=-1)
    print("masked weights for 'wheat rust spreads':")
    print("  row 0 (wheat):  ", [round(x, 3) for x in w[0].tolist()], "(self only)")
    print("  row 1 (rust):   ", [round(x, 3) for x in w[1].tolist()])
    print("  row 2 (spreads):", [round(x, 3) for x in w[2].tolist()])
    assert w[0, 1].item() == 0.0 and w[0, 2].item() == 0.0
    assert w[1, 2].item() == 0.0
    assert torch.allclose(w.sum(-1), torch.ones(3), atol=1e-6)

    # -- 2. PROOF of causality: editing the future cannot change the past ------
    torch.manual_seed(0)
    T, dk, dv = 6, 4, 4
    q = torch.randn(1, T, dk)
    k = torch.randn(1, T, dk)
    v = torch.randn(1, T, dv)
    mask = causal_mask(T)
    out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    # corrupt every future token (positions 3..5) however we like
    k2 = k.clone(); v2 = v.clone()
    k2[0, 3:] = torch.randn(3, dk) * 100
    v2[0, 3:] = torch.randn(3, dv) * 100
    out2, _ = scaled_dot_product_attention(q, k2, v2, mask=mask)
    past_same = torch.allclose(out[:, :3], out2[:, :3], atol=1e-6)
    future_diff = not torch.allclose(out[:, 3:], out2[:, 3:], atol=1e-6)
    print("\nedit future tokens 3..5 -> outputs 0..2 unchanged:", past_same)
    print("                       -> outputs 3..5 did change:", future_diff)
    assert past_same and future_diff

    # -- 3. row i depends ONLY on values j <= i (weight structure check) -------
    _, w2 = scaled_dot_product_attention(q, k, v, mask=mask)
    ok = all(torch.all(w2[0, i, i + 1:] == 0) for i in range(T - 1))
    print("weights above diagonal are exactly zero:", ok)
    assert ok

    # -- 4. without the mask the future leaks (control) --------------------------
    out_nomask, _ = scaled_dot_product_attention(q, k2, v2)
    leak = not torch.allclose(out_nomask[:, :3], out2[:, :3], atol=1e-3)
    print("control WITHOUT mask: future edit leaks into past rows:", leak)
    assert leak

    # -- 5. saturation + variance checks carried over from the core lesson ------
    d_k = 128
    q5 = torch.randn(1, 1, d_k)
    k5 = torch.randn(1, 50, d_k)
    _, w_unscaled = scaled_dot_product_attention(q5, k5, torch.randn(1, 50, d_k), scale=False)
    _, w_scaled = scaled_dot_product_attention(q5, k5, torch.randn(1, 50, d_k), scale=True)
    ent = lambda p: -(p * p.clamp_min(1e-12).log()).sum().item()
    print("\nsaturation demo (d_k=128, 50 keys):")
    print(f"  unscaled max weight {w_unscaled.max():.4f} entropy {ent(w_unscaled):.4f}"
          f" | scaled max {w_scaled.max():.4f} entropy {ent(w_scaled):.4f}")
    assert w_unscaled.max() > w_scaled.max()

    print("\nEXPERIMENT COMPLETE: causal masking verified.")


if __name__ == "__main__":
    experiment()

