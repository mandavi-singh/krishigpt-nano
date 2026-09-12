"""LayerNorm, from scratch.

    LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta      per token, over D

WHY: stacked blocks drift in scale; normalizing each token's feature vector to
mean 0 / variance 1 before every sublayer stabilizes deep training. gamma
(scale) and beta (shift) are learned per feature, so the layer can undo the
restriction wherever useful (init gamma=1, beta=0 => pure normalization).

STATS LIVE IN THE TENSOR: mean/var are computed from the token itself, not
from running averages -- so train() and eval() behave identically (contrast
with BatchNorm). eps (inside the sqrt) guards zero-variance vectors.

Note: normalization uses the BIASED variance (divide by D, PyTorch convention).

Experiment:  python model/normalization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LayerNorm(nn.Module):
    """Per-token (feature-axis) normalization with learnable affine.

    Input/shape: (..., D) -- normalizes over the last dim, independently for
    every (D,) slice. For our layouts this is each token of (B, T, D).
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))   # scale, init 1 (identity-ish)
        self.beta = nn.Parameter(torch.zeros(dim))   # shift, init 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # statistics over the feature axis ONLY -- never across batch or time
        mean = x.mean(dim=-1, keepdim=True)          # (..., 1)
        var = x.var(dim=-1, keepdim=True, correction=0)  # biased variance (÷D)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # eps guards var -> 0
        return self.gamma * x_norm + self.beta


def experiment() -> None:
    torch.manual_seed(0)

    # -- 1. the hand example: x = [4, 2] -> [1, -1] ----------------------------
    ln = LayerNorm(2, eps=0.0)                       # eps=0 for exact arithmetic
    out = ln(torch.tensor([4.0, 2.0]))
    print("hand example [4, 2] ->", [round(v, 4) for v in out.tolist()], "(expect [1, -1])")
    assert torch.allclose(out, torch.tensor([1.0, -1.0]), atol=1e-6)

    # -- 2. zero-variance token: eps saves us ------------------------------------
    ln2 = LayerNorm(4)
    const = torch.tensor([2.0, 2.0, 2.0, 2.0])
    out2 = ln2(const)
    print("constant token [2,2,2,2] ->", out2.tolist(), "(finite, all zeros)",
          "eps =", ln2.eps)
    assert torch.isfinite(out2).all() and torch.allclose(out2, torch.zeros(4))

    # -- 3. matches torch.nn.LayerNorm with the same weights ---------------------
    D = 16
    mine = LayerNorm(D)
    ref = nn.LayerNorm(D)
    ref.weight.data.copy_(mine.gamma.data)
    ref.bias.data.copy_(mine.beta.data)
    x = torch.randn(3, 7, D) * 5 + 2                 # messy scale
    assert torch.allclose(mine(x), ref(x), atol=1e-6)
    print("\nidentical to nn.LayerNorm on (3,7,16) random messy input [ok]")

    # -- 4. irrigation hand example from the lesson --------------------------------
    x_irr = torch.tensor([3.0, 7.0, 5.0, 1.0])
    ln4 = LayerNorm(4, eps=0.0)
    out4 = ln4(x_irr)
    print("'irrigation' [3,7,5,1] ->", ["%+.3f" % v for v in out4.tolist()],
          "(mean %.1e, var %.4f)" % (out4.mean().item(), out4.var(correction=0).item()))
    assert abs(out4.mean().item()) < 1e-6
    assert abs(out4.var(correction=0).item() - 1.0) < 1e-6

    # -- 5. gamma/beta learnability demo ---------------------------------------------
    ln5 = LayerNorm(3)
    x5 = torch.randn(4, 3)
    y = ln5(x5)
    assert y.requires_grad
    y.sum().backward()
    print("gamma.grad != 0:", bool(torch.any(ln5.gamma.grad != 0)),
          "| beta.grad != 0:", bool(torch.any(ln5.beta.grad != 0)))

    # -- 6. train() == eval() ---------------------------------------------------------
    ln6 = LayerNorm(D)
    x6 = torch.randn(2, 5, D)
    ln6.train(); a = ln6(x6).detach()
    ln6.eval(); b = ln6(x6).detach()
    assert torch.equal(a, b)
    print("train() output == eval() output: True  (no running stats)")

    print("\nEXPERIMENT COMPLETE: LayerNorm verified.")


if __name__ == "__main__":
    experiment()
