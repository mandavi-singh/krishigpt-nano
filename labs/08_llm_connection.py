"""Lab 08 — LLM connection: every lab concept, placed inside a GPT.

This lab does NOT implement the transformer. It shows, with real tensor shapes
and real autograd, where each Phase-1 primitive will live in the model we build
next. Every block below prints a shape; together they read like the blueprint
of the whole project.

    tokens ──► embedding lookup ──► Q/K/V projections (matmul) ──► attention
      ──► transformer blocks ──► LM head (matmul) ──► logits ──► cross entropy
      ──► backward() ──► AdamW ──► trained GPT

Run:  python labs/08_llm_connection.py
"""
import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(0)
    V, C, D, H, B, T = 100, 64, 32, 2, 4, 8      # vocab, ctx, embed, heads, batch, seq

    # 1. TOKENS ARE TENSORS (Lab 1) ---------------------------------------
    # The char tokenizer (Phase 2) turns text into ids: shape (batch, seq)
    tokens = torch.randint(0, V, (B, T))
    print(f"1  token ids            {tuple(tokens.shape)}  int64")

    # 2. EMBEDDING = lookup table (Lab 1: indexing into a 2D tensor)
    # Phase 3: nn.Embedding(V, D), one learned vector per token id.
    emb = nn.Embedding(V, D)
    x = emb(tokens)                              # (B, T) -> (B, T, D)
    assert x.shape == (B, T, D)
    manual = emb.weight[tokens]                  # embedding lookup IS indexing
    assert torch.equal(x, manual)
    print(f"2  token embeddings     {tuple(x.shape)}  (lookup/indexing)")

    # 3. Q/K/V PROJECTIONS = matmul (Lab 2)
    # Phase 4-5: three linear layers, one per role. (B,T,D) @ (D,Dh)^T
    wq = nn.Linear(D, D, bias=False)             # "x Wq^T"
    q = wq(x)                                    # (B, T, D)
    assert q.shape == (B, T, D)
    print(f"3  Q projection         {tuple(x.shape)} -> {tuple(q.shape)}  (matmul)")

    # 4. ATTENTION SCORES = batch matmul (Lab 2)
    # Phase 4: scores = Q K^T / sqrt(dk), shape (B, heads, T, T)
    q4 = q.view(B, T, H, D // H).transpose(1, 2)          # (B, H, T, Dh)
    k4 = q4                                              # k=v=q stand-in (Phase 4-5 code)
    scores = q4 @ k4.transpose(-2, -1) / (D // H) ** 0.5
    assert scores.shape == (B, H, T, T)
    print(f"4  attention scores     {tuple(scores.shape)}  (batch matmul)")

    # 5. ATTENTION OUTPUT + RESIDUAL = softmax + matmul + element-wise add
    # (Labs 1, 2; Phase 4-6). attn = softmax(scores) @ V, then heads rejoin.
    attn = torch.softmax(scores, dim=-1) @ k4            # (B, H, T, Dh)
    attn = attn.transpose(1, 2).reshape(B, T, D)         # merge heads -> (B, T, D)
    h = x + attn                                         # residual connection
    assert h.shape == x.shape
    print(f"5  attention + residual {tuple(h.shape)}  (softmax, matmul, add)")

    # 6. LM HEAD = matmul to vocabulary (Lab 2)
    lm_head = nn.Linear(D, V, bias=False)        # Phase 7
    logits = lm_head(h)                          # (B, T, V): score per vocab token
    assert logits.shape == (B, T, V)
    print(f"6  logits               {tuple(logits.shape)}  (matmul to vocab)")

    # 7. LOSS = cross entropy over next tokens (Lab 5)
    targets = torch.randint(0, V, (B, T))        # Phase 8: shifted-by-one targets
    loss = nn.CrossEntropyLoss()(logits.view(-1, V), targets.view(-1))
    print(f"7  cross entropy        {loss.item():.4f}  (ppl = {torch.exp(loss).item():.2f})")

    # 8. BACKPROP = autograd (Lab 3)
    loss.backward()                              # chain rule through everything above
    assert emb.weight.grad is not None
    assert lm_head.weight.grad is not None
    print(f"8  backward()           grads on embedding {tuple(emb.weight.grad.shape)} + head {tuple(lm_head.weight.grad.shape)}")

    # 9. UPDATE = AdamW (Lab 6)
    opt = torch.optim.AdamW(list(emb.parameters()) + list(lm_head.parameters()), lr=1e-3)
    before = emb.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, emb.weight.detach())
    print("9  AdamW step           embeddings changed [ok]")

    print("\nEvery primitive from Labs 1-7 just ran in GPT order.")
    print("Phases 2-9 replace the stand-ins with the real implementations.")


def verify() -> None:
    """Re-run main() silently; raises if any assertion inside fails."""
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        main()


if __name__ == "__main__":
    main()
