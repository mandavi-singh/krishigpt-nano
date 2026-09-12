"""Token embedding layer, from scratch.

    ids (B, T)  ->  lookup in a learned table W of shape (V, D)  ->  h (B, T, D)

The ids carry no meaning; each row of W is a D-dim point that starts random and
is moved into a meaningful geometry ONLY by training (next-token prediction
loss + backprop + AdamW). This module is the first model component of System A.

Experiment:  python model/embeddings.py   (no new concepts, just this layer)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TokenEmbedding(nn.Module):
    """Educational embedding table: W of shape (vocab_size, embed_dim).

    Implementations:
        forward():        W[ids]                     (advanced indexing)
        forward_manual(): one-hot @ W view           (same result, explicit math)
    Both are numerically identical to nn.Embedding with the same weights;
    the tests prove this.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # One learned vector per token id. Init scale ~1/sqrt(D) keeps the
        # activations' magnitude O(1) relative to what LayerNorm expects later.
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim) / math.sqrt(embed_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) long -> (B, T, D) float. A pure row-gather."""
        return self.weight[ids]

    def forward_manual(self, ids: torch.Tensor) -> torch.Tensor:
        """The same lookup expressed as one-hot @ W (the explicit math view)."""
        # (B, T) -> (B, T, V) one-hots; only one 1.0 per row.
        onehot = torch.nn.functional.one_hot(ids, self.vocab_size).to(self.weight.dtype)
        # (B, T, V) @ (V, D) -> (B, T, D): a weighted sum where the weight is 1.
        return onehot @ self.weight


def experiment() -> None:
    """Run the agriculture embedding experiment described in the mentor notes."""
    from tokenizer import BPETokenizer

    torch.manual_seed(0)

    # -- 1. build our agriculture vocab -------------------------------------
    corpus = " ".join([
        "wheat", "wheat crop", "wheat fertilizer", "fertilizer", "irrigation",
    ])
    tok = BPETokenizer.train(corpus, num_merges=25)
    V, D = tok.vocab_size, 8          # D=8 keeps the table printable
    print(f"vocab_size={V}  embed_dim={D}")

    emb = TokenEmbedding(V, D)
    print("\nW (random init) - first 8 rows (id: token -> vector):")
    for i in list(tok.id2token)[:12]:
        row = emb.weight[i]
        print(f"  id {i:>2} {tok.id2token[i]:>15} ->",
              " ".join(f"{v:+.3f}" for v in row.tolist()))

    # -- 2. lookup shapes ----------------------------------------------------
    text = "wheat wheat crop"
    ids = torch.tensor([tok.encode(text)])              # (1, T)
    h = emb(ids)
    print(f"\n'{text}' -> ids {ids.tolist()[0]}")
    print(f"lookup shape: (1, {ids.shape[1]}, {D}) == {tuple(h.shape)}")
    assert h.shape == (1, ids.shape[1], D)

    # batch view (B=2): same sentence twice, proves (B, T, D) layout
    h2 = emb(torch.stack([ids[0], ids[0]]))
    assert h2.shape == (2, ids.shape[1], D)
    print(f"batch shape : {tuple(h2.shape)}")

    # -- 3. manual one-hot view == indexing == nn.Embedding -------------------
    assert torch.allclose(emb.forward_manual(ids), h)
    ref = nn.Embedding(V, D)
    ref.weight.data.copy_(emb.weight.data)
    assert torch.allclose(ref(ids), h)
    print("one-hot@W, W[ids], and nn.Embedding all EXACTLY equal -> the math view holds")

    # -- 4. random rows are ~orthogonal: no meaning yet -----------------------
    wheat = emb.weight[tok.token2id.get("wheat</w>", 4)]
    crop = emb.weight[tok.token2id.get("c", 4) if "c" in tok.token2id else 5]
    cos = torch.nn.functional.cosine_similarity(wheat.unsqueeze(0), crop.unsqueeze(0))
    print(f"\ncosine(wheat, crop) at random init = {cos.item():+.3f}  (near 0: unrelated)")

    # -- 5. gradient sparsity: only used rows are updated ----------------------
    emb_fresh = TokenEmbedding(V, D)
    loss = emb_fresh(ids).sum()     # toy scalar loss touching ids only
    loss.backward()
    used = set(ids.flatten().tolist())
    touched = [i for i in range(V) if torch.any(emb_fresh.weight.grad[i] != 0)]
    print(f"\ntokens in batch : {sorted(used)}")
    print(f"rows with grad  : {touched}  <- exactly the used rows (one-hot view proved)")
    assert set(touched) == used

    print("\nEXPERIMENT COMPLETE: embedding layer works, math verified.")


if __name__ == "__main__":
    experiment()
