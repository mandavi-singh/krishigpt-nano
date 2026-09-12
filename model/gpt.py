"""KrishiGPT-nano: the full GPT, assembled from our from-scratch components.

    token ids (B, T)
      -> TokenEmbedding               (B, T, D)   W_te: (V, D)
      -> TransformerBlock x N         (B, T, D)   pre-norm, RoPE, causal
      -> final LayerNorm              (B, T, D)   cleans residual-stream drift
      -> lm_head                      (B, T, V)   one logit per vocab token

DESIGN DECISIONS (user-confirmed):
  - N=6 blocks, D=128, H=4 (head_dim=32), RoPE, pre-norm, biases on
  - WEIGHT TYING: lm_head.weight IS the embedding matrix W_te (GPT-2 style),
    saving V*D params and making input/output vocab geometries identical.
  - GPT-2 init: Normal(0, 0.02) everywhere, biases 0; sublayer OUTPUT
    projections (wo, fc2) additionally scaled by 1/sqrt(2N) so the residual
    stream starts near identity.
  - No learned positional table: RoPE inside attention is the only position
    signal; therefore the model is exactly translation-equivariant in positions.

Param budget at V=500, D=128, N=6 (bias=True block = 198,272, verified):
    blocks 6*198,272 = 1,189,632 ; embed 64,000 ; final LN 256 ; head 0 (tied)
    TOTAL = 1,253,888  (~1.25M, CPU-trainable nano scale)

Experiment:  python model/gpt.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.embeddings import TokenEmbedding  # noqa: E402
from model.normalization import LayerNorm  # noqa: E402
from model.transformer_block import TransformerBlock  # noqa: E402


@dataclass
class GPTConfig:
    """All dials of the model in one place (configurable per project rule)."""
    vocab_size: int
    max_len: int = 512          # context window; mask + RoPE tables sized by it
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.0        # 0.0 for first nano run (user-confirmed)
    tie_weights: bool = True    # share embedding matrix with lm_head


class GPT(nn.Module):
    """Decoder-only GPT: embedding -> N pre-norm blocks -> final LN -> head."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.drop_emb = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model, cfg.n_heads, max_len=cfg.max_len,
                             dropout=cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        self.ln_final = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # Same Parameter object at both ends: input embedding and output
            # projection share one vocabulary geometry by construction.
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ----------------------------------------------------------------- init
    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 convention: Normal(0, 0.02), biases zero, LN gamma=1/beta=0."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, TokenEmbedding):
            module.weight.data.normal_(0.0, 0.02)   # overrides lab-era 1/sqrt(D)
        elif isinstance(module, LayerNorm):
            module.gamma.data.fill_(1.0)
            module.beta.data.zero_()

    def _scale_residual_projections(self) -> None:
        """wo and fc2 get an extra 1/sqrt(2N) so the stream starts near
        identity: each block's correction is initially small (ResNet-style)."""
        scale = 1.0 / (2 * self.cfg.n_layers) ** 0.5
        for blk in self.blocks:
            blk.attn.wo.weight.data.mul_(scale)
            blk.ffn.fc2.weight.data.mul_(scale)

    # -------------------------------------------------------------- forward
    def forward(self, ids: torch.Tensor,
                positions: torch.Tensor | None = None) -> torch.Tensor:
        """ids (B, T) -> logits (B, T, V). No PE table: RoPE lives in blocks."""
        x = self.drop_emb(self.tok_emb(ids))
        for blk in self.blocks:
            x, _ = blk(x, positions)
        return self.lm_head(self.ln_final(x))


def experiment() -> None:
    from tokenizer import BPETokenizer

    torch.manual_seed(0)

    # -- 1. build on the real agriculture BPE vocab ------------------------------
    corpus = " ".join([
        "wheat wheat wheat", "wheat crop", "wheat crop disease",
        "apply fertilizer before irrigation", "soil needs nitrogen",
    ] * 3)
    tok = BPETokenizer.train(corpus, num_merges=50)
    V = tok.vocab_size
    cfg = GPTConfig(vocab_size=V)
    model = GPT(cfg)
    print(f"agri vocab V={V}, config {cfg}")

    # -- 2. parameter accounting ---------------------------------------------------
    parts = {
        "blocks": sum(p.numel() for b in model.blocks for p in b.parameters()),
        "embedding": model.tok_emb.weight.numel(),
        "final_ln": 2 * cfg.d_model,
        "lm_head": 0 if cfg.tie_weights else V * cfg.d_model,
    }
    total = sum(p.numel() for p in model.parameters())
    print("\nparam budget:")
    for k, v in parts.items():
        print(f"  {k:>10}: {v:,}")
    print(f"  {'TOTAL':>10}: {total:,}")
    assert parts["blocks"] == 6 * 198_272
    assert parts["embedding"] == V * cfg.d_model
    assert total == parts["blocks"] + parts["embedding"] + parts["final_ln"]

    # -- 3. weight tying is structural ----------------------------------------------
    tied = model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    print(f"\nlm_head.weight is tok_emb.weight (same storage): {tied}")
    assert tied
    del parts["lm_head"]

    # -- 4. forward pass shapes --------------------------------------------------------
    ids = torch.tensor([tok.encode("soil needs nitrogen")])
    logits = model(ids)
    print(f"'soil needs nitrogen' -> ids {ids.tolist()[0]}")
    print(f"logits shape: {tuple(logits.shape)}  (one row per position, {V} cols)")
    assert logits.shape == (1, ids.shape[1], V)

    # -- 5. loss is computable on shifted targets ----------------------------------------
    ids2 = torch.tensor([tok.encode("apply fertilizer before irrigation")])
    logits2 = model(ids2)
    inputs = logits2[:, :-1].reshape(-1, V)              # positions 0..T-2
    targets = ids2[:, 1:].reshape(-1)                    # successors 1..T-1
    loss = nn.functional.cross_entropy(inputs, targets)
    ppl = torch.exp(loss).item()
    print(f"\ninit loss = {loss.item():.4f}, init perplexity = {ppl:.2f}")
    print(f"(untrained model ~= uniform over {V} tokens; ln({V}) = "
          f"{torch.log(torch.tensor(float(V))).item():.4f})")
    assert abs(loss.item() - torch.log(torch.tensor(float(V))).item()) < 0.15

    # -- 6. gradients reach everything ------------------------------------------------------
    loss.backward()
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or not torch.any(p.grad != 0)]
    print("parameters with zero gradient:", dead if dead else "none")
    assert dead == []

    # -- 7. translation invariance (RoPE only -> positions shift = identity) ------------------
    a = model(ids, positions=torch.arange(ids.shape[1]))
    b = model(ids, positions=torch.arange(ids.shape[1]) + 100)
    inv = torch.allclose(a, b, atol=1e-5)
    print(f"content shifted +100 positions -> identical logits (eval mode): {inv}")

    # -- 8. CPU budget preview: tokens/sec for a training-sized step -----------------------------
    B, T = 8, 128
    xb = torch.randint(0, V, (B, T))
    tb = torch.randint(0, V, (B, T))
    model.eval()
    with torch.no_grad():
        model(xb)                                   # warmup
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.time()
    steps = 3
    for _ in range(steps):
        opt.zero_grad()
        lg = model(xb)
        l = nn.functional.cross_entropy(lg[:, :-1].reshape(-1, V),
                                        tb[:, 1:].reshape(-1))
        l.backward()
        opt.step()
    dt = time.time() - t0
    tok_per_s = steps * B * T / dt
    print(f"\nCPU preview: {steps} train steps of (B={B}, T={T}) in {dt:.1f}s "
          f"-> ~{tok_per_s:,.0f} tokens/sec (forward+backward+AdamW)")

    print("\nEXPERIMENT COMPLETE: KrishiGPT-nano assembled and verified.")


if __name__ == "__main__":
    experiment()
