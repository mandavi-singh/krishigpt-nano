# TODO

## DIRECTION (updated 2026-08-16): KrishiGPT
Final goal: domain-specific Agriculture LLM / AI Assistant (System A from-scratch
GPT for learning + System B KrishiGPT application: KB + RAG + tools + API + UI).
Tiny Shakespeare remains educational-only. Currently in teaching mode:
no new components until each concept is explained and confirmed.

## Awaiting user confirmation
- Understanding of 7 direction questions (what we build, Shakespeare role,
  A/B/C/D distinction, final architecture, 2M model limits, data flow,
  why not train a final assistant from scratch on CPU)
- Next teaching step: manual BPE on agriculture mini-corpus (wheat/fertilizer/
  irrigation morphology) before any new code

## Phase 0 — Environment (DONE)
- [x] Inspect hardware (CPU-only: i5-1335U, 16 GB RAM, ~55 GB disk)
- [x] Decide Python version (3.12, not 3.14)
- [x] Create project skeleton
- [x] README.md, TODO.md, CHANGELOG.md
- [x] User confirmed plan (nano ~2M on Tiny Shakespeare, CPU-only)
- [x] Created Python 3.12.5 venv at myllm/.venv
- [x] requirements.txt: torch 2.9.1 CPU, numpy, pytest (minimal for Phase 1)
- [x] Verified: python/torch versions, CUDA=False, torch=10 CPU threads
- [x] Sanity tests passed: matmul shapes, autograd, one gradient step

## Phase 1 — PyTorch basics (DONE)
- [x] labs/01_tensors.py — creation, shape, dtype, indexing, view/reshape,
      transpose/contiguous, broadcasting
- [x] labs/02_matrix_ops.py — element-wise, dot, matmul, batch matmul (attn
      shape), bmm, einsum
- [x] labs/03_autograd.py — manual vs autograd derivatives, grad accumulation,
      retain_grad, no_grad/detach
- [x] labs/04_nnmodule.py — TinyMLP, parameter discovery, train()/eval(),
      dropout semantics
- [x] labs/05_losses.py — MSE, cross entropy derived by hand, CE vs confidence,
      CE gradient = softmax-onehot, perplexity, ignore_index
- [x] labs/06_optimizers.py — manual GD == SGD, momentum, Adam moments,
      AdamW weight decay (+ lesson: default wd=0.01 if omitted)
- [x] labs/07_training_loop.py — Dataset, DataLoader, full train/val loop
- [x] labs/08_llm_connection.py — all primitives replayed in GPT order
- [x] labs/README.md (concept -> GPT map)
- [x] tests/test_labs.py — 8/8 labs verified (assertions + standalone demos)
- [ ] User confirms -> start Phase 2 (Tokenization)

## Phase 2 — Tokenization (DONE)
- [x] tokenizer/special.py — <PAD>/<UNK>/<BOS>/<EOS> at fixed ids 0..3
- [x] tokenizer/base.py — Tokenizer ABC (vocab, encode/decode, BOS/EOS wrap)
- [x] tokenizer/char_tokenizer.py — exact round-trip
- [x] tokenizer/word_tokenizer.py — regex split + min_freq vocab cutoff
- [x] tokenizer/bpe_tokenizer.py — BPE from scratch: pre-tokenize, pair
      counting, frequent-pair selection, merging, learned-rule encode,
      </w>-aware decode, JSON save/load; step-by-step verbose traces
- [x] data/preprocessing.py — Tiny Shakespeare downloader
- [x] tests: 32 new tokenizer tests (round-trips, special/unknown/empty/
      repeated/punct/determinism, merge-order semantics, save/load)
- [x] tokenizer/demo.py — classic "low lower lowest" trace + measured
      Tiny Shakespeare comparison (char 69 vocab / word 13,335 / bpe-300 368)
- [ ] User confirms -> start Phase 3 (Embeddings + positional + RoPE)

## Phase 3 — Embeddings & positions
- [ ] model/embeddings.py (token embeddings)
- [ ] Sinusoidal absolute positional encoding
- [ ] model/rope.py (rotary embeddings)
- [ ] Tests + shape walkthroughs

## Phase 4 — Self-attention
- [ ] model/attention.py: scaled_dot_product_attention (naive, explicit QKV)
- [ ] Causal mask, tests, masking visualization

## Phase 5 — Multi-head attention
- [ ] MultiHeadAttention from scratch (no nn.MultiheadAttention)
- [ ] Shape assertions + tests

## Phase 6 — Transformer block
- [ ] model/normalization.py (LayerNorm, RMSNorm)
- [ ] model/feed_forward.py (MLP)
- [ ] model/transformer_block.py (pre-norm + residual)
- [ ] Tests

## Phase 7 — Mini-GPT
- [ ] model/gpt.py (configurable GPT)
- [ ] configs/nano_gpt.json (~2M params)
- [ ] Parameter count + forward-shape tests

## Phase 8 — Dataset
- [ ] data/preprocessing.py (download Tiny Shakespeare, split)
- [ ] data/dataset.py (fixed-length sequences, DataLoader)

## Phase 9 — Training
- [ ] training/loss.py, training/checkpoint.py, training/train.py
- [ ] CPU-friendly config, perplexity + tokens/sec logging
- [ ] Validate: loss ↓, samples coherent-ish

## Phase 10 — Generation
- [ ] inference/sampling.py (greedy, temperature, top-k, top-p)
- [ ] inference/generate.py
- [ ] Compare sampling strategies

## Phase 11 — Prefill & decode
- [ ] Explicit prefill() / decode_step() functions
- [ ] Measure TTFT, TPOT, latency, throughput

## Phase 12 — KV cache
- [ ] inference/kv_cache.py from scratch
- [ ] Benchmarks: cached vs uncached (TTFT/TPOT/tokens-per-sec/memory)

## Phase 13 — Batching
- [ ] optimization/batching.py (static + dynamic)
- [ ] Throughput/latency tables vs batch size

## Phase 14 — Caching types
- [ ] Experiments: KV cache vs prompt cache vs prefix cache

## Phase 15 — Quantization
- [ ] FP32/FP16/BF16/INT8/INT4 explainer + educational INT8 experiment
- [ ] memory/speed/quality comparison

## Phase 16 — RAG
- [ ] rag/: chunking, embeddings, vector store, retrieval, generation

## Phase 17 — Fine-tuning (pretrained model, first production-lib use)
- [ ] SFT + LoRA + QLoRA on small HF model

## Phase 18 — Tool calling
- [ ] agents/tools.py: calculator, python eval (sandboxed), doc search

## Phase 19 — Agent
- [ ] Simple plan/act/observe loop

## Phase 20 — API
- [ ] api/server.py: /generate, /chat, /health (FastAPI)

## Phase 21 — Production comparison
- [ ] Compare vs HF Transformers / vLLM (CPU caveats for vLLM)

## Phase 22 — Final integration + docs

## Open questions for user
- (resolved) Model tier: nano (~2M) on Tiny Shakespeare — confirmed.
- vLLM: kept on roadmap for later GPU/cloud environment (Phase 21), not dropped.
