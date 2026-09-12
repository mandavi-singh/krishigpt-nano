# CHANGELOG

## 2026-08-22 — Project freeze (submission-ready; no further training)
- FINAL checkpoint frozen: krishigpt_v3/best.pt (step 3800, val 4.0804 / ppl
  59.17, 1,860,224 params) — SHA-256 manifest final/FREEZE_MANIFEST.json,
  file set read-only. Clean-environment verifier (foreign cwd, hash check,
  strict load, end-to-end greedy + reproducible seeded sampling) PASSED.
- Final evaluation (final/final_eval.py): val 4.0804/59.17; params 1,860,224;
  top-p0.9 rep 0.190 / d1 0.812 / d2 0.994 / fab 4.5%; topk40 rep 0.308 / fab
  4.0%; throughput ~21-36 tok/s (CPU); artifacts final/final_eval.json.
- Showcase: 15 agriculture prompts x {topp0.9, topk40}, fixed seed 0, up to 96
  tokens, every sample labeled MODEL OUTPUT (final/showcase_samples.md/.json).
  Fabrication metric attested vs the model's own cumulative training texts;
  prior-row reference numbers unchanged.
- Deliverables: final/MODEL_CARD.md (positioning, architecture, tokenizer,
  corpus/provenance, v1->v2->v3 lineage, results, limitations, licensing),
  final/DIAGRAMS.md (ASCII + Mermaid: architecture, training pipeline),
  final/FINAL_REPORT.md (submission report incl. v1/v2/v2-ext/v3 comparison
  table), final/verify_final_checkpoint.py.
- Integrity: all 274 v1/v2 corpus/checkpoint/colab baselines re-hashed
  BYTE-IDENTICAL; full suite 177 passed. Training, corpus, tokenizer, and all
  checkpoints frozen; no further experiments will be started.

## 2026-08-22 — Corpus-v3 warm-start run (local, bounded; user-confirmed)
- New run `checkpoints/krishigpt_v3/`: warm-start from
  krishigpt_nano_v2_extended/best.pt (step 4600, 9,420,800 tokens, val 4.3761),
  4 epochs / 3,840 steps on corpus v3 (1,966,105 train tokens / 960 steps per
  epoch); best.pt step 3800: val 4.0804 / ppl 59.2 (final 4.0857 @3840);
  baseline reproduced pre-run; no overfitting (val still improving at exit).
- Wiring add (v2 pattern): train.py + run_eval.py gained a "v3" corpus branch
  (same paths/cache-dir convention; no algorithm/hyperparameter change).
  update_warm_start/extend_schedule cannot wrap a new corpus (horizon must
  exceed donor step), so counters restart with epochs=4 as the invariant -
  documented in the config notes.
- Pre-flight: precheck_v3.py (strict load, 99 AdamW states, vocab 5237, arch,
  v3 train token count == report, v3 valid ids byte-identical to the v2 cache,
  baseline 4.3761 reproduced); 30-step stats-only dry run verified
  loading/counters/loss/ckpt/token accounting (61,440 = 30x16x128). Full suite:
  177 passed.
- Eval: run_eval.py --corpus v3 (same prompts/seeds) + fabrication_eval.py
  (attestation pool now corpus-cumulative per checkpoint; prior rows reproduce
  exactly). Fabrication: greedy 0.0%, topk40 4.0% (~=), topp0.9 15.3->4.5%
  (largest improvement yet), temp0.9 7.6%. Repetition/diversity ~= v2-ext for
  sampling variants; greedy degraded (wiki section-token loops '= ='/'See
  also'). Coherence: first factoid-adjacent phrases ('Rice is grown in East
  Asia', 'clay horizon', 'sowing') appear; novel fragments persist under
  sampling.
- Verdict: larger gap-targeted corpus helped substantially on val loss/ppl and
  reduced top-p fabrication, but did NOT solve fabrication (capacity +
  data-scale floor). Results + recommendation:
  evaluation/corpus_v3/EXPERIMENT_v3_warmstart.md.
- Integrity: SHA-256 snapshot of all v1/v2 corpus files, tokenizer, all prior
  checkpoints and the Colab notebook re-verified zero-diff after the run.
  Total lineage token exposure: 17,285,128. No further training started.

## 2026-08-22 — Corpus v3 gap-fill expansion built (data step, NO training)
- New corpus version `data/corpus_v3/` = v2 train (83 docs, byte-reused) + 101
  gap-fill docs (97 Wikipedia CC BY-SA 4.0 + 4 public-domain Gutenberg books):
  crop diseases/pests, fertilizer chemistry & nutrient deficiencies, soil-science
  depth, irrigation methods, farming practices, state-level Indian agriculture,
  20 crop-specific articles, agriculture terminology, clean modern English prose.
- Provenance per v2 rules: per-doc meta + sources_v3.json (license/URL/access);
  17 fetch skips recorded; 2 fetch-time redirect dupes removed; 1 redirect
  collision with v2 train skipped as corpus duplicate; 1 Gutenberg book
  (gutenberg:57457, 1523 Husbandry) dropped on measured archaic/short-line
  quality markers but kept with provenance.
- Split/leak-safety: TRAIN 184 docs / 7,015,724 chars / 1,163,400 words /
  1,966,105 BPE tokens (+64.3% vs v2 train); VALID = exact byte copy of the
  6-doc v2 validation set (22,026 ids, never trained in any run) so val loss
  stays directly comparable with 4.3711. Hard asserts: new ids disjoint from
  all history, 844 cross-corpus dup lines removed, 0 line-overlap with v2 valid,
  verify_v3_compat.py: fast-encoded valid id stream == v2 cached tensor exactly.
- Tokenizer decision: REUSE the 5,237-vocab BPE (per instruction, no retrain);
  measured UNK 0.052% on the new part, 0.100% combined, compression 280.2
  tok/kchar vs v2's 276.4 — coverage well inside acceptable. Compatibility gate
  passed. Scripts: evaluation/corpus_v3/{fetch_sources_v3.py, build_corpus_v3.py,
  verify_v3_compat.py}; report: evaluation/corpus_v3/REPORT.md.
- Perf note: naive BPE encode is O(5000 merges) per word — measured over 7 MB
  it hangs for hours (two prior build runs hit 40-min timeouts before the fix).
  The build uses a rank-based fast merge loop with a per-word cache, strict-
  equality-verified against tok.encode on 1,508 sampled word types before use;
  tokenizer module NOT modified.
- Integrity: SHA-256 snapshots of every v1/v2 artifact (274 files: corpora,
  tokenizer, checkpoints, colab/) taken before and after: zero differences;
  krishigpt_nano_v2_extended/best.pt untouched. Full suite: 177 passed.
- STOPPED per user instruction: no v3 training until explicit confirmation.

## 2026-08-21 — KrishiGPT-nano v2 CONTINUATION run (local, bounded; no Colab)
- Executes the written recommendation of `evaluation/corpus_v2/EXPERIMENT_v2_warmstart.md`:
  continue warm-start training on corpus v2 from `krishigpt_nano_v2/best.pt`
  (step 2300, tokens 4,710,400, val 4.5068). Architecture/tokenizer/corpus/optimizer unchanged.
- New: `training/precheck_v2_extended.py`, `configs/nano_agri_v2_extended.json`,
  `configs/nano_agri_v2_extended_dry.json`, `evaluation/fabrication_eval.py`
  (local, deterministic word-level fabricated-word eval; v1/ext + v2 reference rows
  reproduce exactly), `evaluation/corpus_v2/EXPERIMENT_v2_extended.md` (design + results).
- Training code: two bounded features, each test-pinned (suite now 177 passed):
  1. `train_stats_only`: dry-run mode — final.pt is metadata-only, no weights
     persisted; load_checkpoint refuses stats-only payloads.
  2. `update_warm_start` + `extend_schedule`: warm-start load that KEEPS donor
     step/tokens/best-val and re-normalizes the cosine schedule to the extended
     horizon (true continuation; the donor's bookkeeping is not discarded).
- Dry run (disposable `checkpoints/v2ext_dry`, stats-only, 30 steps): verified
  resume seam, in-range loss, correct counters — before expensive training.
- Full run: `checkpoints/krishigpt_nano_v2_extended/` — start step 2300 (4,710,400
  tokens), +2,336 steps → **step 4636, 9,494,528 tokens**; best.pt at step 4600:
  **val 4.3761 / ppl 79.5** (was 4.5068 / 90.6), final.pt 4.3711 at 4636. Val loss
  improved at 14/23 evaluations, earliest 300 steps warm-up perturbation only,
  no overfitting (never exceeded baseline after step ~2700 and still decreasing at
  exit). Train ppl ~45-55 vs val 79.5: small corpus memorization below threshold.
- Generation (same 5 prompts × greedy/temp0.9/topk40/topp0.9, same seeds):
  topk40 rep-rate 0.333→0.302, topp0.9 0.178→0.165; greedy still degenerate
  (rep 0.829). Fabrication (word-level, corpus-v1+v2 attested): greedy 0.0%
  (unchanged), topk40 3.7→3.6% (flat), topp0.9 15.5→15.3% (flat), temp0.9
  8.1→13.5% (small-sample noise; same-seed ref rows reproduce exactly). The
  capacity/data-scale floor on fabricated spelling from the v2 report stands.
- Next step not yet chosen (awaiting confirmation): one more ~1-2 epoch min-LR
  extension vs moving to a bigger corpus/vocab. Extended design+results in
  `evaluation/corpus_v2/EXPERIMENT_v2_extended.md`; artifacts in
  `evaluation/results/krishigpt_nano_v2_extended_best/`. All v1/v2 artifacts
  and the Colab notebook untouched.

## 2026-08-17 — Agriculture pretraining corpus built (data step, not training)
- New: data/build_agriculture_corpus.py (download -> metadata -> clean ->
  dedupe -> doc-level split -> BPE trained on TRAIN only), tests/test_corpus.py
  (10 offline tests for clean/dedupe/banner-strip/split).
- Sources recorded (37 docs, per-doc JSON in data/sources/agri_sources.json):
  32 Wikipedia agri articles (CC BY-SA 4.0, attribution recorded) + 5 Project
  Gutenberg public-domain agriculture books. 1 wiki article (Drought) skipped
  after retries. ICAR/FAO excluded: reuse terms not machine-verifiable.
- Cleaning/dedup/corpus stats (verified on disk):
  - chars after clean: 3,804,835 -> after dedupe: 3,799,633 (-5,202)
  - TRAIN: 33 docs, 3,162,028 chars, 535,320 words
  - VALID: 4 docs, 637,605 chars, 111,365 words
  - split: document-level, deterministic seed 0, no leakage (tested)
- BPE (from scratch) trained on train split only: 5,000 merges -> vocab 5,237;
  train encodes to 897,225 tokens (283.7/kchar). Saved to
  data/processed/agri_bpe_tokenizer.json. (5,000 merges ~= 25+ min on this CPU
  with the deliberately naive pairwise BPE; 2,000 merges took 759 s.)
- With V=5,237 the full KrishiGPT-nano = 1,860,224 params (~1.86M, within the
  planned ~2M budget): blocks 1,189,632 + tied embedding/head 670,336 + LN 256.
- End-to-end smoke test PASSED: saved tokenizer loads, encodes "Wheat is a crop
  that needs fertilizer and irrigation." and round-trips; real train excerpt
  runs through GPT -> logits (1,125,5237), init CE 8.576 ~= ln(5237) = 8.564.
- Bugs fixed during the build (rule 18):
  1. BPETokenizer.save/load used Windows cp1252 default encoding -> Unicode
     errors on real corpus characters; both now explicit UTF-8.
  2. gutendex format keys carry charset suffixes ("text/plain; charset=...")
     -> exact-key lookup never matched; now prefix-matched. Corpus went from
     0 to 5 public-domain books (~2.8M chars).
  3. Wikipedia API rate-limited (429) 23/33 articles at 0.2 s spacing ->
     retry-with-backoff fetcher (exp backoff, 4 attempts) + 1 s politeness +
     incremental cache (re-downloads skip already-saved docs). All re-fetched.
  4. split_documents: char-fill policy could produce an empty train side with
     tiny corpora -> replaced with document-count split clamped to [1, n-1];
     deterministic via sorted-by-id pre-shuffle.
  5. 5,000-merge BPE run exceeded the shell timeout AFTER saving the tokenizer;
     stats JSON reconciled with the on-disk tokenizer (measured values only).
- Full suite: 137 passed.

## 2026-08-16 — System A component 10/11: KrishiGPT-nano assembled
- User decisions: N=6 blocks, weight tying ON, GPT-2 init 0.02 (embedding
  init switched from lab-era 1/sqrt(D) to 0.02 — flagged per rule 5).
- New: `model/gpt.py` (GPTConfig dataclass; GPT = TokenEmbedding -> dropout ->
  6x TransformerBlock(RoPE, causal, pre-norm) -> final LayerNorm -> tied
  lm_head; residual projections scaled 1/sqrt(2N)), `tests/test_gpt.py` (10 tests).
- Verified:
  - param budget: blocks 1,189,632 + embed + final LN; tied head costs 0.
    Demo vocab V=74 -> TOTAL 1,199,360 (~1.2M); at real V~500 total ~1.25M
  - weight tying structural: lm_head.weight and tok_emb.weight share storage
    (same data_ptr); gradient through head updates embedding
  - logits (1, T, V); init CE loss 4.3396 ~= ln(74) = 4.3041 (untrained ~ uniform)
  - causality through full model; RoPE translation invariance (in-range shift);
    gradients reach every parameter
  - CPU preview: (B=8, T=128) train step incl. AdamW ~= 6,882 tokens/sec
- REAL BUG found & fixed (rule 18): RotaryEmbedding defined `apply(x, pos)`,
  which SHADOWED nn.Module.apply(fn); GPT's model.apply(init) crashed with
  TypeError across all 10 tests. Renamed to `rotate()` in rope.py,
  multi_head.py, tests/test_rope.py; docstring records the lesson.
- Test bug fixed: translation-invariance test shifted positions by +100 with
  max_len=64 — out-of-bounds is CORRECT behavior (RoPE table = context window
  boundary); test now shifts +20 and documents the boundary.
- Full suite: 127 passed.

## 2026-08-16 — System A component 9: transformer decoder block (pre-norm)
- Teaching cycle completed: residual math -> pre-norm placement -> full block
  walkthrough with shape audit -> comprehension check -> implement.
- New: `model/transformer_block.py` (TransformerBlock: LN1 -> MHA(RoPE, causal)
  -> dropout -> residual; LN2 -> FFN(GELU) -> dropout -> residual; exposes
  attention weights), `tests/test_transformer_block.py` (10 tests).
- Verified:
  - shapes at nano config: (2,256,128) in/out, weights (2,4,256,256)
  - params = 198,272 bias=True (lesson's 197,120 + biases); bias=False test
    pins 197,120 exactly at D=128
  - zero-initialized sublayers -> block(x) == x EXACTLY (residual identity proof)
  - causality survives the block; LN1 output is unit-scale (forward-pre-hook);
    out == x + attn_corr + ffn_corr recomputed independently
  - dropout deterministic in eval / stochastic in train; positions kwarg gives
    RoPE translation invariance through the block
  - stream norm drift measured: 5.66 -> 8.52 over 4 blocks (slow drift; final
    LayerNorm inside GPT cleans it up before lm-head)
- Full suite: 117 passed.
- Design decisions confirmed by user: dropout 0.0 for first nano run,
  biases on (GPT-2 convention), attention weights exposed for visualization.

## 2026-08-16 — System A component 8: LayerNorm
- Teaching cycle completed: formula -> gamma/beta/eps -> comprehension check -> implement.
- New: `model/normalization.py` (LayerNorm: biased variance correction=0,
  learnable gamma/beta, eps inside sqrt), `tests/test_normalization.py`
  (10 tests, incl. hand-computed cases).
- Verified:
  - hand example [4,2] -> [1,-1] exactly; irrigation example
    [3,7,5,1] -> [-0.447, 1.342, 0.447, -1.342] (mean 0, var 1)
  - biased-variance convention pinned numerically ([1,2,3] -> +-1.2247)
  - identical to nn.LayerNorm with shared weights; per-token independence
  - zero-variance token -> finite all-zeros output (eps guard)
  - gradients flow through gamma, beta, and the input
  - train() output == eval() output (no running statistics)
- Full suite: 107 passed.

## 2026-08-16 — System A component 7: feed-forward network + GELU
- Teaching cycle completed: concept -> math -> comprehension check -> implement.
- New: `model/feed_forward.py` (gelu_exact via erf form, GPT-2-style
  gelu_approx, FeedForward = Linear -> GELU -> Linear), `tests/test_feed_forward.py`
  (10 tests, incl. independent references via F.gelu exact/tanh modes).
- Verified:
  - gelu_exact reference values: GELU(1) = 0.84134, GELU(-1) = -0.15865
  - exact matches F.gelu(approximate='none'); approx matches
    F.gelu(approximate='tanh'); max |exact-approx| = 4.73e-04 on [-10,10]
  - soft gate: negative inputs produce small negative outputs with NONZERO
    gradient; ReLU contrast confirmed (grad at -2 is exactly 0)
  - hand example reproduced: expand [1.5,-1.0,2.0,-3.0] -> GELU
    [1.400,-0.159,1.954,-0.004] -> project [1.489, 0.319]
  - strict position independence (per-token computation); editing one
    position's input changes only that position's output
  - parameter accounting: FFN = 8D^2 (131,072 at D=128) vs attention 4D^2
- Design decision: GPT-2 conventions for the nano model (biases on, GELU-exact
  default; LLaMA-style bias-free/SwiGLU noted as later ablation).

## 2026-08-16 — System A component 6: multi-head attention
- Teaching cycle completed: concept -> math -> comprehension check -> implement.
- New: `model/multi_head.py` (MultiHeadAttention: Wq/Wk/Wv/Wo, head split via
  view+transpose, RoPE on Q/K only, causal mask, scaled_dot_product_attention,
  merge + output projection), `tests/test_multi_head.py` (10 tests).
- Verified at nano config (B=2, T=256, D=128, H=4):
  - shapes out (2,256,128), weights (2,4,256,256); params = 4*D^2 = 65,536
  - H=1 case EXACTLY equals base causal attention (weights + outputs)
  - causality survives head split (future edits cannot change past outputs)
  - RoPE translation invariance inside full MHA: same content shifted by 5
    positions gives identical weights; changing gaps changes weights
  - position-0 rotation == identity equals the use_rope=False twin (strict=False
    load_state_dict needed: rope buffers exist only in one module — fixed)
  - gradients reach all four matrices
- Agriculture demo surfaced real BPE behavior: 'apply' -> ['a','pply</w>'],
  'before' -> ['b','efore</w>'] (rare words fragmented), 'fertilizer' -> whole
  token (freq 2). Demo rewritten to anchor on 'fertilizer</w>'; at init all
  heads are near-uniform and only slightly different — specialization is a
  TRAINING outcome, not an init property (documented in demo output).
- Full suite: 87 passed.

## 2026-08-16 — System A component 5: causal masking
- Teaching cycle completed: concept -> math -> comprehension check -> implement.
- Extended `model/attention.py`: `causal_mask(T)` (upper triangle -inf,
  diagonal allowed), optional `mask` argument in scaled_dot_product_attention;
  added masked hand example ('wheat rust spreads'). tests/test_attention.py
  grew 9 -> 16 tests.
- Verified:
  - masked hand-example weights: row0 [1,0,0], row1 [0.269,0.731,0],
    row2 [0.108,0.240,0.652] — rows renormalize to 1
  - PROOF of causality: corrupting tokens 3..5 leaves outputs 0..2 exactly
    unchanged while outputs 3..5 change; without the mask future edits leak
    into past rows (control)
  - weights above diagonal are exactly 0; last row equals unmasked output;
    mask broadcasts over (B, H, T, T)
- Two test bugs fixed while verifying (rule 18): `tril()` comparison wrongly
  tested zeroed cells against -inf; `triu_indices` needs offset=1 to exclude
  the diagonal. Mask structure test now uses strict-upper indices + isneginf().
- Note: user re-pasted the full KrishiGPT brief mid-phase; direction unchanged
  and already recorded — treating as confirmation of the active plan.

## 2026-08-16 — System A component 4: scaled dot-product attention
- Teaching cycle completed: concept -> math -> comprehension check -> implement.
- New: `model/attention.py` (scaled_dot_product_attention(q, k, v) returning
  (out, weights); no mask, no heads yet), `tests/test_attention.py` (9 tests),
  model/__init__ export updated.
- Verified against the hand-worked 'soil needs nitrogen' example:
  - raw scores [0.5, 0.3, 0.9] -> scaled -> weights [0.313, 0.272, 0.415];
    'soil' attends most to 'nitrogen'
  - /sqrt(d_k) effect: d_k=128, 50 keys -> unscaled max weight 0.991
    (entropy 0.05, near one-hot), scaled max 0.173 (entropy 3.23)
  - measured Var(q.k): std tracks sqrt(d_k) (3.94/8.12/15.85 vs 4/8/16)
  - uniform keys => output = exact mean of values; single key => pass-through
  - row-wise softmax sums to 1; gradients flow to q, k, v
- Uniform-key / single-key tests pin the "soft lookup" semantics exactly.

## 2026-08-16 — System A component 3: RoPE (rotary embeddings)
- Teaching cycle completed: concept -> math -> comprehension check -> implement.
- New: `model/rope.py` (RotaryEmbedding: cos/sin buffer tables, no learned
  params; `apply(x, positions)` rotates interleaved even/odd pairs),
  `tests/test_rope.py` (8 tests), model/__init__ exports updated.
- Verified numerically:
  - toy D=2: offset-only scores; offset 3 at (2,5) and (20,23) both = cos 3;
    offset 5 = cos 5
  - offset-only identity holds across 5 absolute positions: spread <= 1e-6
  - rotation preserves norms (orthogonal); position 0 is the identity
  - real-array rotation == complex multiply for all pairs/positions
  - R(a)R(b) == R(a+b) composition; (B, T, D) broadcast works
  - frequency ladder invariant: theta_k = 1/10000^(2k/D)
- Design decision: interleaved pair convention matching the complex-form math;
  V is not rotated (position belongs in scores, not values).

## 2026-08-16 — System A component 2: sinusoidal positional encoding
- Teaching cycle completed (concept -> math -> comprehension check -> implement).
- New: `model/positional.py` (PositionalEncoding: fixed table, registered
  buffer, no learned params), `tests/test_positional.py` (6 tests).
- Verified:
  - closed-form formula matches sin/cos table to 1e-6
  - bounded |PE| <= 1, all 512 position vectors unique
  - rotation property: PE(pos+k) == R(k) @ PE(pos) per frequency pair, R fixed
  - "wheat crop" vs "crop wheat": identical vector multisets WITHOUT PE,
    distinct WITH PE (order now visible to the model)
- One test bug fixed: backward() on a graph with no grad-requiring tensor
  raises RuntimeError (autograd rule from Lab 03); test now asserts that and
  also proves gradients flow through PE untouched when input requires grad.

## 2026-08-16 — System A component 1: token embeddings
- Teaching cycle completed: concept -> manual math -> comprehension check ->
  smallest implementation -> experiment.
- New: `model/__init__.py`, `model/embeddings.py` (TokenEmbedding: W (V,D),
  init N(0, 1/sqrt(D)) via manual scale; forward = W[ids]; forward_manual =
  one-hot @ W), `tests/test_embeddings.py` (7 tests).
- Verified on the agriculture BPE vocab (V=45):
  - shapes (1,T,D) and (B,T,D) correct
  - indexing == one-hot@W == nn.Embedding, exactly
  - gradient sparsity: only used rows receive gradient; row used 3x gets 3x grad
  - random-init rows ~orthogonal (cos ~0.13): no meaning before training
- One test bug found & fixed: `make_ids()` consumed RNG state mid-test,
  so the reference and ours compared different id batches. Fixed by drawing
  ids once and reusing.
- Next: positional encoding (not started).

## 2026-08-16 — Direction change: KrishiGPT (Agriculture LLM)
- Final goal changed from generic mini-LLM to KrishiGPT: agriculture-domain
  assistant = from-scratch GPT (System A) + KB/RAG/tools/API/UI (System B).
- Tiny Shakespeare retained ONLY as the Phase 1-2 educational corpus; clearly
  marked as non-domain data in data/README.md and README.md.
- Created data/{raw,processed,sources} and ui/ directories; no data code added.
- Teaching mode engaged: explain -> example -> comprehension check ->
  implement -> experiment. No Phase 3+ code until confirmed.
- Documented the four mechanisms that will be kept separate:
  A) pretraining from scratch, B) instruction fine-tuning, C) RAG, D) tools.
- Data governance rule added: record source/URL/license/title/collection date
  in data/sources/ for every external document before use; no fabricated facts.

## 2026-08-16 — Phase 2 complete: tokenization from scratch
- Built `tokenizer/` package (no HF/tiktoken/sentencepiece):
  - special.py: <PAD>/<UNK>/<BOS>/<EOS> at fixed ids 0..3.
  - char_tokenizer.py: exact round-trip, unseen chars -> <UNK>.
  - word_tokenizer.py: regex word/punct split, min_freq vocab cutoff.
  - bpe_tokenizer.py: full educational BPE (pre-tokenize, pair counting,
    merging, learned-rule encode, decode, JSON save/load, verbose trace).
  - demo.py: classic example trace + Tiny Shakespeare comparison.
- Tests: 32 tokenizer tests; full suite 40 passed.
- Measured Tiny Shakespeare: char vocab=69, word vocab=13,335, bpe-300 vocab=368.
- Failures found & fixed (rule 18): punctuation-lossiness behavior documented,
  merge-trace assertions pinned to measured values, console encoding issues fixed.

## 2026-08-16 — Phase 1 complete: PyTorch basics labs
- 8 runnable labs with numeric assertions; all pass. See labs/README.md.

## 2026-08-16 — Phase 0 complete: venv + verification
- Python 3.12.5 venv, torch 2.9.1 CPU, numpy, pytest; sanity checks passed.
