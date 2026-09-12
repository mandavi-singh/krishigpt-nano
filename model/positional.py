"""Sinusoidal positional encoding, from scratch.

    h_pos = embedding(token) + PE(pos),   PE(pos, 2k)   = sin(pos / 10000^(2k/D))
                                          PE(pos, 2k+1) = cos(pos / 10000^(2k/D))

WHY: self-attention is permutation equivariant -- it only knows WHAT tokens are
present, not WHERE. "wheat crop" and "crop wheat" would attend identically
without an injected position signal.

WHY THIS FORM: angle-sum identities make PE(pos+k) a fixed linear transform
(rotations per frequency pair) of PE(pos), so relative order is learnable;
bounded in [-1, 1]; geometric frequency range gives each position a unique
multi-scale "odometer" fingerprint.

NOTE: fixed here (not learned); RoPE (next lesson) supersedes this idea by
rotating Q/K instead of adding to embeddings.

Experiment:  python model/positional.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position vectors, precomputed up to `max_len`.

    forward(x): x of shape (B, T, D) -> x + PE[0:T].
    PE is a registered buffer: saved with checkpoints, moved with .to(device),
    but NEVER trained (no gradient).
    """

    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = self._build_table(embed_dim, max_len)          # (max_len, D)
        self.register_buffer("pe", pe)

    @staticmethod
    def _build_table(embed_dim: int, max_len: int) -> torch.Tensor:
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for sin/cos pairs")
        pos = torch.arange(max_len, dtype=torch.float32)    # (max_len,)
        # frequency divisors: 10000^(2k/D), k = 0..D/2-1
        div = 10000.0 ** (torch.arange(0, embed_dim, 2) / embed_dim)  # (D/2,)
        angles = pos.unsqueeze(1) / div.unsqueeze(0)        # (max_len, D/2)
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(angles)                     # even cols
        pe[:, 1::2] = torch.cos(angles)                     # odd cols
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position vectors: (B, T, D) + (T, D) -> (B, T, D)."""
        return x + self.pe[0 : x.size(1)]


def experiment() -> None:
    """Show that PE lets the model's input distinguish token ORDER."""
    from model.embeddings import TokenEmbedding
    from tokenizer import BPETokenizer

    torch.manual_seed(0)
    corpus = "wheat wheat crop fertilizer irrigation"
    tok = BPETokenizer.train(corpus, num_merges=25)
    D = 16
    emb = TokenEmbedding(tok.vocab_size, D)
    pe = PositionalEncoding(D, max_len=64)

    ids_a = torch.tensor([tok.encode("wheat crop")])
    ids_b = torch.tensor([tok.encode("crop wheat")])

    h_a, h_b = emb(ids_a), emb(ids_b)                       # content only
    h_a_pe, h_b_pe = pe(h_a), pe(h_b)                       # content + position

    print("tokens:  'wheat crop'  vs  'crop wheat'")
    print("same token set? ", sorted(ids_a.tolist()[0]) == sorted(ids_b.tolist()[0]))

    # without PE: same vectors, different order => sets are identical
    same_vectors = {tuple(v.tolist()) for v in h_a[0]} == {tuple(v.tolist()) for v in h_b[0]}
    print("without PE: same multisets of vectors ->", same_vectors, "(order invisible)")

    # with PE: the multisets differ -> order is now visible
    same_vectors_pe = {tuple(v.tolist()) for v in h_a_pe[0]} == {tuple(v.tolist()) for v in h_b_pe[0]}
    print("with PE   : same multisets of vectors ->", same_vectors_pe, "(order visible)")
    assert not same_vectors_pe and same_vectors

    # position fingerprint: PE vectors unique up to max_len and bounded
    assert torch.unique(pe.pe, dim=0).shape[0] == 64
    assert pe.pe.abs().max() <= 1.0

    # relative-offset property: PE(pos+k) is a FIXED rotation of PE(pos) per
    # frequency pair (not a translation), so difference vectors depend on pos.
    d1 = pe.pe[1 + 4] - pe.pe[1]
    d2 = pe.pe[2 + 4] - pe.pe[2]
    print("PE(pos+4)-PE(pos) depends on pos (rotation, not translation):",
          not torch.allclose(d1, d2))
    # numeric rotation check for frequency pair 0, offset k=4:
    k = 4
    w = 1.0 / 10000.0 ** (0 / D)                # angular frequency of pair 0
    R = torch.tensor([[math.cos(k * w), math.sin(k * w)],
                      [-math.sin(k * w), math.cos(k * w)]])
    got = pe.pe[1 + k, 0:2]
    expected = R @ pe.pe[1, 0:2]
    print("rotation R(k)@PE(pos) == PE(pos+k) (pair 0):", torch.allclose(got, expected, atol=1e-5))
    print("\nPE(0) :", [round(v, 3) for v in pe.pe[0].tolist()])
    print("PE(1) :", [round(v, 3) for v in pe.pe[1].tolist()])
    print("PE(63):", [round(v, 3) for v in pe.pe[63].tolist()])
    print("\nEXPERIMENT COMPLETE: position now visible in the input.")


if __name__ == "__main__":
    experiment()
