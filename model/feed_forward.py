"""Feed-forward network (the "think" layer), from scratch.

    FFN(x) = W2 · gelu(W1 x)          per position, position-independent

Attention mixes BETWEEN tokens (linear re-weighting); the FFN transforms each
token's gathered context INDEPENDENTLY with a learned non-linearity. A block
is "read (attention) then think (FFN)". It also holds 2*D*4D = 8D^2 params
per block -- 2x attention's 4D^2 -- and is where fact-like associations
(e.g. 'urea' <-> '46%% nitrogen') are believed to live.

GELU (exact):      x * Phi(x) = 0.5 x (1 + erf(x/sqrt(2)))
GELU (GPT-2 approx): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

GELU is a SOFT self-gate: output = input scaled by confidence about its own
size. Unlike ReLU, it keeps small nonzero gradients for negative inputs, so
units never die permanently.

Experiment:  python model/feed_forward.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SQRT_2 = math.sqrt(2.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """GELU(x) = x * Phi(x), via the erf form of the normal CDF."""
    return 0.5 * x * (1.0 + torch.erf(x / SQRT_2))


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """The tanh approximation used by GPT-2 (max error ~3e-4 vs exact)."""
    return 0.5 * x * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)))


class FeedForward(nn.Module):
    """D -> d_hidden -> D MLP with GELU, applied independently per position.

    In/bias choices: GPT-2 uses biases in attention+FFN linear layers; LLaMA
    drops them. For the nano model we keep the GPT-2 convention (bias=True)
    so parameter counts match the classic reference when we compare later.
    """

    def __init__(self, d_model: int, hidden_mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_mult * d_model, bias=bias)
        self.fc2 = nn.Linear(hidden_mult * d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, 4D) -> (B, T, D); no position mixing happens here
        return self.fc2(gelu_exact(self.fc1(x)))


def experiment() -> None:
    torch.manual_seed(0)

    # -- 1. exact vs approximate GELU -------------------------------------------
    xs = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    print("x      exact     approx    diff")
    for v in xs.tolist():
        t = torch.tensor(v)
        e, a = gelu_exact(t), gelu_approx(t)
        print(f"{v:+.1f}  {e:+.4f}  {a:+.4f}  {(e - a).abs():.2e}")
    grid = torch.linspace(-10, 10, 20001)
    err = (gelu_exact(grid) - gelu_approx(grid)).abs().max()
    print(f"max |exact-approx| on [-10,10]: {err:.2e}  (approx is fine for training)")
    assert err < 1e-3
    # known reference values
    assert abs(gelu_exact(torch.tensor(1.0)).item() - 0.8413) < 1e-4
    assert abs(gelu_exact(torch.tensor(-1.0)).item() + 0.1587) < 1e-4

    # -- 2. soft gating in action ----------------------------------------------
    neg = gelu_exact(torch.tensor(-1.0))
    print(f"\nGELU(-1) = {neg:+.4f}  <- negative input survives softly")
    print(f"ReLU(-1) = 0 exactly   <- hard gate kills it (and its gradient)")

    # -- 3. hand-worked agriculture example (D=2, hidden=4) ----------------------
    torch.manual_seed(1)
    ff = FeedForward(2, hidden_mult=2, bias=False)
    with torch.no_grad():
        ff.fc1.weight.copy_(torch.tensor([[1.5, 0.0],
                                          [-1.0, 0.0],
                                          [2.0, 0.0],
                                          [-3.0, 0.0]]))
        ff.fc2.weight.copy_(torch.tensor([[0.4, 0.3, 0.5, 0.2],
                                          [0.1, 0.1, 0.1, 0.1]]))
    soil_vec = torch.tensor([[1.0, 0.0]])           # 'soil' after attention
    hidden = ff.fc1(soil_vec)
    gated = gelu_exact(hidden)
    out = ff(soil_vec)
    print("\n'soil' vector [1, 0] through FFN:")
    print("  expanded :", ["%+.3f" % v for v in hidden[0].tolist()], "(4 detectors)")
    print("  after GELU:", ["%+.3f" % v for v in gated[0].tolist()], "(negatives soft-gated)")
    print("  projected :", ["%+.3f" % v for v in out[0].tolist()], "(back to D=2)")
    assert hidden[0, 2] > hidden[0, 0] > 0 and hidden[0, 3] < 0

    # -- 4. position independence -------------------------------------------------
    ff4 = FeedForward(16)
    x = torch.randn(2, 7, 16)
    y = ff4(x)
    same = all(torch.allclose(y[b, t], ff4(x[b, t]), atol=1e-6)
               for b in range(2) for t in range(7))
    print(f"\nposition independence: y[b,t] == FFN(x[b,t]) for all (b,t): {same}")
    assert same

    # -- 5. parameter accounting: FFN = 8D^2 (no bias) --------------------------------
    D = 128
    nob = FeedForward(D, bias=False)
    assert sum(p.numel() for p in nob.parameters()) == 8 * D * D
    print(f"\nD={D}: FFN params = {sum(p.numel() for p in nob.parameters()):,} = 8*D^2 "
          f"(attention has 4*D^2 = {4 * D * D:,})")

    print("\nEXPERIMENT COMPLETE: FFN + GELU verified.")


if __name__ == "__main__":
    experiment()
