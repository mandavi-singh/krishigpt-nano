"""Multi-head attention, from scratch (no nn.MultiheadAttention).

One head = one attention distribution = one "theory" of what to look at.
H heads run H independent Q/K/V projection circuits in parallel, then their
outputs are concatenated and mixed by one output projection:

    Q,K,V = X Wq, X Wk, X Wv                 (B, T, d_model) each
    split : (B, T, H, d_h) -> transpose -> (B, H, T, d_h)      [zero-cost view]
    RoPE  : rotate Q and K by positions (V is NEVER rotated)
    attn  : softmax(Q K^T / sqrt(d_h) + causal_mask) @ V  per (batch, head)
    merge : transpose back, reshape (B, T, H*d_h) = (B, T, d_model)
    out   : merged @ Wo

WHY d_model IS SPLIT: same FLOPs as one full-width head, but H independent
similarity functions can attend to different tokens simultaneously; Wo then
selects/combines head findings. Scaling uses sqrt(d_h) because each dot
product lives in the head's d_h-dimensional space.

WHY .contiguous()/reshape after transpose: transpose only changes strides
(logical view); the merge reshape materializes head concatenation in memory.

Experiment:  python model/multi_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.attention import causal_mask, scaled_dot_product_attention  # noqa: E402
from model.rope import RotaryEmbedding  # noqa: E402


class MultiHeadAttention(nn.Module):
    """GPT-style causal multi-head self-attention with RoPE on Q/K.

    Shapes (B=batch, T=seq, D=d_model, H=heads, dh=D//H):
        in  x:               (B, T, D)
        Q, K, V after proj:  (B, T, D)
        after split:         (B, H, T, dh)
        scores/weights:      (B, H, T, T)
        head outputs:        (B, H, T, dh)
        after merge + Wo:    (B, T, D)
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int = 512,
                 use_rope: bool = True, bias: bool = False) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Four full (D, D) matrices total -- NOT one set per head. The split
        # into heads is a reshape; specialization comes from training.
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        # RoPE frequency ladder is sized by the HEAD dimension: rotations act
        # on pairs inside each head's dh-dim slice.
        self.rope = RotaryEmbedding(self.head_dim, max_len) if use_rope else None
        self.register_buffer("mask", causal_mask(max_len))

    # ------------------------------------------------------------- reshaping
    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, dh): view + transpose, no math."""
        b, t_len, _ = t.shape
        t = t.view(b, t_len, self.n_heads, self.head_dim)
        return t.transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, H, T, dh) -> (B, T, D): heads concatenated along feature axis."""
        b, _, t_len, _ = t.shape
        t = t.transpose(1, 2)                      # (B, T, H, dh)
        return t.reshape(b, t_len, self.d_model)   # copies into row-major order

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor,
                positions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (out (B,T,D), weights (B,H,T,T))."""
        b, t_len, _ = x.shape
        if positions is None:
            positions = torch.arange(t_len, device=x.device)
        q = self._split_heads(self.wq(x))          # (B, H, T, dh)
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))
        if self.rope is not None:
            q = self.rope.rotate(q, positions)      # rotate queries
            k = self.rope.rotate(k, positions)      # rotate keys; V untouched
        out, weights = scaled_dot_product_attention(
            q, k, v, mask=self.mask[:t_len, :t_len]
        )
        return self.wo(self._merge_heads(out)), weights


def experiment() -> None:
    """Shapes, equivalence to single-head math, causality, head diversity."""
    from tokenizer import BPETokenizer

    torch.manual_seed(0)

    # -- 1. shapes at KrishiGPT-nano config ------------------------------------
    B, T, D, H = 2, 256, 128, 4
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    out, w = mha(x)
    n_params = sum(p.numel() for p in mha.parameters())
    print(f"config B={B} T={T} D={D} H={H} dh={D // H}")
    print(f"  out {tuple(out.shape)}  weights {tuple(w.shape)}  params {n_params} (= 4*D^2)")
    assert out.shape == (B, T, D) and w.shape == (B, H, T, T)
    assert n_params == 4 * D * D

    # -- 2. with H=1 this equals plain causal attention exactly -----------------
    D2, T2 = 12, 8
    mha1 = MultiHeadAttention(D2, 1, use_rope=False)
    x2 = torch.randn(2, T2, D2)
    out1, w1 = mha1(x2)
    q, k, v = mha1.wq(x2), mha1.wk(x2), mha1.wv(x2)
    out_ref, w_ref = scaled_dot_product_attention(q, k, v, mask=causal_mask(T2))
    out_ref = mha1.wo(out_ref)
    print("\nH=1 equivalence with base attention:")
    print("  weights equal:", torch.allclose(w1[:, 0], w_ref, atol=1e-6),
          "| outputs equal:", torch.allclose(out1, out_ref, atol=1e-5))
    assert torch.allclose(w1[:, 0], w_ref, atol=1e-6)
    assert torch.allclose(out1, out_ref, atol=1e-5)

    # -- 3. causality survives the head split -------------------------------------
    T3 = 6
    mha3 = MultiHeadAttention(16, 4)
    x3 = torch.randn(1, T3, 16)
    out3, _ = mha3(x3)
    x3b = x3.clone()
    x3b[0, 3:] = torch.randn(3, 16) * 50
    out3b, _ = mha3(x3b)
    past_same = torch.allclose(out3[:, :3], out3b[:, :3], atol=1e-5)
    print(f"\nedit tokens 3..5 -> outputs 0..2 unchanged: {past_same}")
    assert past_same

    # -- 4. RoPE translation invariance inside full MHA ----------------------------
    # Same content at positions [0,1,2] vs [5,6,7] must give identical weights.
    mha4 = MultiHeadAttention(16, 4)
    content = torch.randn(1, 3, 16)
    _, w_a = mha4(content, positions=torch.tensor([0, 1, 2]))
    _, w_b = mha4(content, positions=torch.tensor([5, 6, 7]))
    inv = torch.allclose(w_a, w_b, atol=1e-6)
    print(f"same content shifted by 5 positions -> identical weights (all heads): {inv}")
    assert inv

    # -- 5. agriculture head-diversity demo ----------------------------------------
    corpus = " ".join([
        "apply fertilizer before irrigation",
        "apply fertilizer before irrigation",
        "wheat rust is a disease",
        "soil needs nitrogen",
    ])
    tok = BPETokenizer.train(corpus, num_merges=30)
    ids = torch.tensor([tok.encode("apply fertilizer before irrigation")])
    pieces = [tok.id2token[i] for i in ids[0].tolist()]
    torch.manual_seed(7)
    mha5 = MultiHeadAttention(32, 4)
    emb = torch.randn(tok.vocab_size, 32) * 0.1
    _, w5 = mha5(emb[ids])
    # BPE reality check: 'apply' split into ['a', 'pply</w>'] and 'before'
    # into ['b', 'efore</w>'] — rare words stay fragmented. 'fertilizer' (2x in
    # corpus) survived as ONE token, so we anchor the diversity demo on it.
    q_pos = pieces.index("fertilizer</w>")
    print(f"\ntokens: {pieces}")
    print("(note: 'apply' and 'before' were split into pieces by BPE; frequent")
    print(" 'fertilizer' stayed whole - subword granularity in action)")
    print(f"attention rows for query '{pieces[q_pos]}' (heads 0..3):")
    for h in range(4):
        row = w5[0, h, q_pos].tolist()
        print("  head", h, ":", ["%.3f" % r for r in row])
    rows = w5[0, :, q_pos]
    distinct = not any(torch.allclose(rows[i], rows[j], atol=1e-5)
                       for i in range(4) for j in range(i + 1, 4))
    print("all four heads produce DIFFERENT distributions:", distinct)
    print("(at init this is random; training is what makes heads specialize)")

    print("\nEXPERIMENT COMPLETE: multi-head attention verified.")


if __name__ == "__main__":
    experiment()
