# Phase 1 — PyTorch Basics Labs

Small, independent, runnable lessons covering exactly the PyTorch we need for
the from-scratch LLM. Each lab has a `verify()` with numeric assertions and a
`main()` demo, and can be run on its own.

| Lab | Topic | Key takeaways |
|-----|-------|---------------|
| 01 | Tensor fundamentals | creation, shape, dtype, indexing, view/reshape, transpose, broadcasting |
| 02 | Matrix operations | element-wise ops, dot product, matmul, batch matmul, einsum |
| 03 | Autograd | requires_grad, computational graph, backward(), grad accumulation, no_grad |
| 04 | nn.Module | parameters(), forward(), train()/eval(), dropout behaviour |
| 05 | Loss functions | MSE vs cross entropy, why CE = next-token loss, perplexity, CE gradient |
| 06 | Optimizers | manual GD == SGD, momentum, Adam moments, AdamW weight decay |
| 07 | Complete training loop | Dataset, DataLoader, forward/loss/backward/step/zero_grad, validation |
| 08 | LLM connection | every lab primitive placed in GPT-pipeline order |

Run one lab:

```powershell
.\.venv\Scripts\Activate.ps1
python labs/01_tensors.py
```

Run all labs + tests:

```powershell
pytest tests/test_labs.py -q
```

## Concept → GPT map

```
Tensor                       -> everything: token ids, weights, activations
   ↓
Matrix multiplication        -> embeddings are lookups; Wq/Wk/Wv/Wo and FFN and
                                the LM head are all matmuls: (B,T,D) @ (D,N)
   ↓
Embeddings                   -> nn.Embedding: id -> learned vector  (Phase 3)
   ↓
Q/K/V projections            -> x Wq, x Wk, x Wv                      (Phase 4)
   ↓
Attention                    -> softmax(QKᵀ/√dk)V, batch matmul        (Phase 4-5)
   ↓
Transformer                  -> pre-norm block: attn + FFN + residuals (Phase 6)
   ↓
Logits                       -> LM head: (B,T,D) -> (B,T,V)           (Phase 7)
   ↓
Cross Entropy                -> loss = -log p(next token); ppl = e^CE (Phase 9)
   ↓
Backpropagation              -> backward() fills .grad for every parameter
   ↓
AdamW                        -> adaptive optimizer + weight decay      (Phase 9)
   ↓
Training                     -> zero_grad → forward → loss → backward → step (Phase 9)
```

## Mental model to keep

- Shapes rule the project: `(batch, seq, dim)` is the universal layout;
  attention adds a head dim: `(batch, heads, seq, seq)`.
- `view/transpose` moves data logically at zero cost; matmul needs the layout
  it expects — hence the constant reshaping around multi-head attention.
- Training is one 5-line loop (Lab 7); everything from a 212-param MLP to our
  2M-param GPT executes those same 5 lines.
