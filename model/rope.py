"""RoPE — Rotary Position Embedding, from scratch (Su et al. 2021).

IDEA: do not add position to embeddings. Rotate Q and K by their positions,
so the attention score depends ONLY on the distance between tokens:

    <R(i)q, R(j)k> = q^T R(i)^T R(j) k = q^T R(j-i) k        [offset-only]

because 2D rotations satisfy R(a)^T R(b) = R(b-a).

Per dimension pair (2k, 2k+1) read as a complex number z = x_even + i*x_odd:

    z' = z * e^(i * m * theta_k),     theta_k = 1 / 10000^(2k/D)

In real coordinates (what we implement):

    x'_even = x_even*cos(m*theta_k) - x_odd*sin(m*theta_k)
    x'_odd  = x_even*sin(m*theta_k) + x_odd*cos(m*theta_k)

V is deliberately NOT rotated: position belongs in the scores (which tokens
attend), not in the values (what content is aggregated).

Experiment:  python model/rope.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for rotary rotations, no learned params."""

    def __init__(self, dim: int, max_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even (rotations act on pairs)")
        # frequency ladder: theta_k for pair k = 0..dim/2-1 (same ladder as sinusoidal PE)
        self.inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)  # (D/2,)
        pos = torch.arange(max_len).float()                                    # (max_len,)
        angles = pos.unsqueeze(1) * self.inv_freq.unsqueeze(0)                 # (max_len, D/2)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate even/odd pairs of x by the angle of their position.

        Named `rotate`, NOT `apply`: `nn.Module.apply(fn)` is a framework
        method (recursively applies a function to submodules) and shadowing
        it breaks model.apply()-based weight init — a real bug we hit.

        Args:
            x: (..., dim) content vectors (queries or keys).
            positions: (T,) long positions matching the sequence axis before ...
                       (we keep it simple here: works for (T, dim) inputs).
        Returns:
            x rotated: same shape as x.
        """
        cos = self.cos[positions]          # (T, D/2)
        sin = self.sin[positions]          # (T, D/2)
        while cos.dim() < x.dim():         # allow broadcasting on batch axes
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        xe, xo = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = xe * cos - xo * sin     # rotation of each pair
        out[..., 1::2] = xe * sin + xo * cos
        return out


def apply_rotary(x: torch.Tensor, positions: torch.Tensor, dim: int,
                 base: float = 10000.0) -> torch.Tensor:
    """Stateless helper used by the experiments (builds tables on the fly)."""
    if positions.max() >= 512:
        raise ValueError("extend max_len for this demo")
    rot = RotaryEmbedding(dim, max_len=max(512, int(positions.max()) + 1), base=base)
    return rot.rotate(x, positions)


def experiment() -> None:
    """Verify the lesson's math on agriculture content vectors."""
    torch.manual_seed(0)

    # -- 1. clock-hand check (D=2, theta=1): score depends on offset only ----
    dim, base = 2, 10000.0
    # single pair only -> theta_0 = 10000^0 = 1 rad/step (use base 1 for the toy)
    q = torch.tensor([[1.0, 0.0]])          # 'wheat' content
    k = torch.tensor([[1.0, 0.0]])          # 'fertilizer' content
    def score(i: int, j: int) -> float:
        return apply_rotary(q, torch.tensor([i]), dim, base=1.0)[0] \
             @ apply_rotary(k, torch.tensor([j]), dim, base=1.0)[0]
    print("D=2 toy (theta=1 rad/step):")
    print(f"  offset 3 at (2,5):   {score(2, 5).item():+.4f}   (expect cos 3 = {-0.9900:.4f})")
    print(f"  offset 3 at (20,23): {score(20, 23).item():+.4f}   (expect cos 3 = {-0.9900:.4f})")
    print(f"  offset 5 at (2,7):   {score(2, 7).item():+.4f}   (expect cos 5 = {0.2837:.4f})")
    assert abs(score(2, 5) - torch.cos(torch.tensor(3.0))) < 1e-6
    assert abs(score(20, 23) - score(2, 5)) < 1e-6
    assert abs(score(2, 7) - torch.cos(torch.tensor(5.0))) < 1e-6

    # -- 2. distance-only identity at full dimension --------------------------
    dim = 8
    q = torch.randn(1, dim)                  # content vector of 'wheat'
    k = torch.randn(1, dim)                  # content vector of 'fertilizer'
    print("\nidentity <R(i)q, R(j)k> == <R(k)q, R(l)k> whenever j-i == l-k:")
    for d in (1, 3, 7):
        s: list[float] = []
        for start in (0, 5, 33, 100, 250):
            qi = apply_rotary(q, torch.tensor([start]), dim)
            kj = apply_rotary(k, torch.tensor([start + d]), dim)
            s.append((qi[0] @ kj[0]).item())
        spread = max(s) - min(s)
        print(f"  offset {d}: scores across 5 positions -> mean {sum(s)/len(s):+.4f}, "
              f"spread {spread:.2e}")
        assert spread < 1e-5

    # -- 3. rotations preserve norms (orthogonality) --------------------------
    x = torch.randn(4, dim)
    xr = apply_rotary(x, torch.tensor([10, 20, 30, 40]), dim)
    assert torch.allclose(x.norm(dim=1), xr.norm(dim=1), atol=1e-6)
    print("\nnorm preservation: max relative error",
          f"{((x.norm(dim=1) - xr.norm(dim=1)).abs() / x.norm(dim=1)).max():.2e}")

    # -- 4. position 0 is the identity -----------------------------------------
    assert torch.allclose(apply_rotary(x, torch.zeros(4, dtype=torch.long), dim), x)
    print("position 0 rotation = identity [ok]")

    # -- 5. matches complex multiplication (closed form) ------------------------
    z = torch.view_as_complex(x.reshape(4, dim // 2, 2))
    for t in range(4):
        theta = 1.0 / base ** (torch.arange(0, dim, 2) / dim)
        phase = torch.exp(1j * t * theta).to(z.dtype)
        z_expected = z[t] * phase
        x_expected = torch.view_as_real(z_expected).reshape(dim)
        got = apply_rotary(x[t : t + 1], torch.tensor([t]), dim)[0]
        assert torch.allclose(got, x_expected, atol=1e-6)
    print("complex-form equivalence across all pairs/positions [ok]")

    print("\nEXPERIMENT COMPLETE: rotary math verified.")


if __name__ == "__main__":
    experiment()
