# KrishiGPT-nano — Model Card

**Final frozen checkpoint:** `checkpoints/krishigpt_v3/best.pt` (SHA-256 in `final/FREEZE_MANIFEST.json`, file set read-only) · training step 3800 · validation loss **4.0804**, perplexity **59.2**, on the byte-fixed 172-window validation split · **1,860,224 parameters** · evaluated 2026-08-22.

---

## 1. What KrishiGPT-nano is

KrishiGPT-nano is a **1.86M-parameter decoder-only transformer language model**, built **entirely from scratch** (embeddings, sinusoidal constants, RoPE, causal multi-head attention, LayerNorm, FFN, GPT block, BPE tokenizer, dataset pipeline, AdamW training loop, sampling, checkpointing — no nn.MultiheadAttention, no HuggingFace, no tiktoken). It is pretrained on a curated agriculture corpus and generates agriculture-themed continuations from short prompts.

Answer disposition: it is a **next-token prediction model, not a question-answering system**: no retrieval, no fact checking, no tool use. Instruction following is minimal: a post-freeze QA fine-tune lineage (v4 → v4_qa → v4_qa2) moved constraint-format compliance from 0% to 20% on the project's 10-test instruction bench (both list-of-three tests pass; the model starts answers in the right shape but cannot reliably stop), with domain MCQ unchanged at chance. See `evaluation/corpus_v4/EXPERIMENT_v4_qa.md`.

## 2. SLM vs LLM positioning

Small Language Model (SLM), deliberately:

- **Parameter scale:** 1.86M vs GPT-2-small's 124M (~67× smaller) and GPT-3's 175B (~94,000× smaller).
- **Context window:** 128 tokens.
- **Vocabulary:** 5,237 BPE tokens, domain-specialized, general-English coverage.
- **Compute:** CPU-only (Intel Core i5-1335U, ~800–2,100 tokens/sec). Trains in minutes to a couple hours per run — if it can't run on a laptop, it wasn't nano.
- **Lineage goal:** reproducibility and learning-first: the whole run lineage is 5 checkpoints, all on one laptop, with every number below reproducible from artifacts on disk.

An SLM with this much CPU-quality training over a curated domain corpus matches that spirit: domain-flavored calibrated language modeling, not instruction or factuality answering.

## 3. Architecture

6-block transformer decoder, GPT-2-style pre-norm, ~1.86M params.

| Component | Detail |
|---|---|
| Blocks | 6 |
| d_model | 128 |
| Attention heads | 4 (d_k = 32) |
| Position encoding | RoPE (rotary, cos/sin tables, applied to Q and K only) |
| Normalization | LayerNorm, pre-norm, biased variance, before attention and before FFN |
| Feed-forward | Linear → GELU (exact) → Linear, hidden 256 |
| Vocabulary / embedding | 5,237 BPE tokens, token embedding tied with LM head (zero-cost tied head) |
| Dropout | 0.0 |
| Causality | strict triangular mask |
| Max context | 128 |

Parameter breakdown: 6 blocks × 198,272 = 1,189,632 + tied embedding/head 670,336 +
final LN 256 = **1,860,224** (the tying is structural — `lm_head.weight` shares storage with `tok_emb.weight`, confirmed by `data_ptr()` equality).

Architecture diagram: see `final/DIAGRAMS.md` (ASCII + Mermaid).

## 4. Tokenizer

From-scratch **BPE**, one tokenizer for the whole run lineage:

- **Shape:** 4 special tokens (PAD/UNK/BOS/EOS at ids 0–3) + 159 base characters + 5,000 learned merges = **5,237 vocab**.
- Trained on the v1 train split only; saved once; never modified since. Directly compatible with every checkpoint.
- No HF/tiktoken/sentencepiece; `</w>`-marked end-of-word; unknown characters → whole-word `<UNK>`.
- Measured quality on final corpus text: 280.2 tokens/1000 chars, word-level UNK 0.100%, single-token rate 76.58%, avg 1.398 tokens/word — well inside the established reuse threshold, so no retraining across v1→v2→v3, keeping the embedding table learned through the full run lineage.

## 5. Training corpus and sources

Three versioned corpora under `data/`, each with full provenance (`*.meta.json` per document, `sources*.json` with provider/URL/license/access date). Sources picked so reuse rights were machine-verifiable; ICAR/FAO/USDA/government extension released on rights policy.

| corpus | documents | chars | BPE train tokens | added focus |
|---|---|---|---|---|
| v1 | 37 (32 Wikipedia + 5 Gutenberg PD) | 3,799,633 | 897,225 | first corpus; cleaned+dedup'd; audited |
| v2 | 89 (82 wiki + 7 Gutenberg PD) | 4,413,829 | 1,150,725 | gap-fills machines/plant/India/pests; leak-safe split |
| v3 | 184 (97 new wiki + 4 new Gutenberg PD) | 7,015,724 | 1,966,105 | diseases/pests/nutrient chemistry/soil science depth/state-level Indian agriculture/farming practices/crop-specific pages/terminology/clean modern prose |

Corpus hygiene: Wikipedia sections citation (See also/References/…) cut; Gutenberg license banners removed; long repeated cross-line boilerplates removed; document-level split, seed 0.

**Validation set** (byte-fixed since v2): `wiki:Combine harvester`, `wiki:No-till farming`, `wiki:Loam`, `wiki:Phosphorus cycle`, `wiki:Rainwater harvesting`, `wiki:Rabi crop` — 6 documents, 22,026 BPE tokens, never included in training in any run, byte-identical across v2/v2-ext/v3 so every validation loss since v2 is directly comparable.

## 6. Corpus/model progression v1 → v2 → v3

| run | warm-start from | corpus | epochs/ steps | val loss | val ppl | best checkpoint |
|---|---|---|---|---|---|---|
| v1 full scratch | — | v1 | — / 800 | 5.7076 | 301.1 | krishigpt_nano_full |
| v1 extended | v1 full | v1 | +4 / 2,552 total | 5.2720 | 194.8 | krishigpt_nano_extended |
| v2 warm-start | v1 extended | v2 | 4 / 2,336 | 4.5068 | 90.63 | krishigpt_nano_v2 |
| v2 extended | v2 | v2 | +4 / 4,636 total | 4.3761 | 79.52 | krishigpt_nano_v2_extended |
| **v3 warm-start (FINAL)** | v2 extended | v3 | 4 / 3,840 | **4.0804** | **59.17** | **krishigpt_v3/best.pt** |
| *v4 (post-freeze, not submitted)* | *v3 (frozen)* | *v4* | *4 / 4,440 (GPU)* | *3.9752* | *53.3* | *krishigpt_v4/best.pt* |
| *v4_qa2 (post-freeze, instruction-tune)* | *v4_qa* | *qa_v2* | *3 / 63 (GPU)* | — | — | *krishigpt_v4_qa2/best.pt* |

(val losses from v2 onward on the same byte-fixed validation set; v1 losses on the different v1 validation documents and not comparable). Cumulative gradient exposure: ~17.3M tokens (through v3). The v4 row is post-freeze research: it improves val loss but with visibly diminishing returns. The v4_qa2 row is the instruction-tune endpoint: 63 steps on 801 constraint-format QA pairs (yes/no, one-word, list-of-three + definitional) — instruction bench compliance 0% → 20%, MCQ unchanged at chance; QA-valid loss 1.74 (its own valid set, not comparable to corpus rows). The submitted checkpoint stays v3. Full record: `evaluation/corpus_v4/EXPERIMENT_v4_qa.md`.

Full comparison table (val, generation, fabrication) → `final/FINAL_REPORT.md`, §5.

## 7. Final results (frozen checkpoint)

Measured with the standard evaluation (`final/final_eval.py`):

| metric | value |
|---|---|
| parameters | 1,860,224 |
| val loss / ppl (byte-fixed, 172 windows) | **4.0804 / 59.17** |
| greedy rep-rate / distinct-1 | 0.879 / 0.134 (degenerate — not recommended) |
| top-p 0.9 rep-rate / distinct-1 / distinct-2 | **0.190 / 0.812 / 0.994** |
| top-k 40 rep-rate / distinct-1 / distinct-2 | 0.308 / 0.697 / 0.943 |
| fabricated words, top-p 0.9 (attested against own training text) | **4.5%** |
| fabricated words, top-k 40 | 4.0% |
| end-to-end decode throughput | ~21–36 tokens/sec (CPU, varies with load) |
| recommended decoding | **top-p 0.9 (default temperature, fixed seed)** for reports; top-k 40 as secondary |

Samples: `final/showcase_samples.md` (15 prompts × 2 decoders, every sample labeled **MODEL OUTPUT**, not human-authored, not from the corpus, not verified agricultural advice).

## 8. Limitations

1. **No factuality.** 1.86M params can store distribution-shapes but cannot support fact claims; generated sentences can be fluent, grammatical, and wrong in the same breath. Samples must never be read as agricultural guidance.
2. **Hallucinated tokens remain present under sampling.** ~4–4.5% of sampled tokens are novel wordforms assembled from frequent BPE pieces, a capacity + data-scale phenomenon (documented in `evaluation/corpus_v3/EXPERIMENT_v3_warmstart.md`). Failure rate tracks corpus size — larger helps, data-scale floor never fully resolves.
3. **Greedy decoding is degenerate.** Repetitive loops near all from trial one, and after the heavy Wikipedia v3 exposure the greedy-style wiki section token loops (`= = =`, "See also") become prominent. Use sampling.
4. **128-token window.** Samples truncate mid-sentence at 64–96 tokens; the model cannot maintain a long-range narrative and there is no KV-cache-in-a-portable-format, no quantization, no batching; CPU throughput sufficient for classroom demo but not for service.
5. **Validation set is small** (6 documents, 22,026 tokens) — perplexity is stable but the eval set is narrow.
6. **Domain vocabulary only.** Outside the agriculture/text distribution in training, outputs degrade quickly (careful not to overstate domain breadth).
7. **Determinism limited to seeded sampling.** un-seeded sampling varies run-to-run.

## 9. Ethical and licensing considerations

- **Composition:** model weights and artifact code are authored for this project. Wikipedia text was used under **CC BY-SA 4.0** — attribution per article is recorded; corpus lists and per-document metadata (`sources_v2.json`, `sources_v3.json`, `*.meta.json`) must **remain with the corpus** if redistributed. Gutenberg books are public domain (US) per provider metadata, recorded similarly.
- **No ICAR/FAO/USDA/government-extension content is included**: reuse rights not machine-verifiable; excluded deliberately and documented. This decision limits corpus size for machine-readability honesty.
- **Fabricated tokens:** this model produces fluently fake words (imHeralia, paramarchorus). Whoever reuses it should frame all output as illustrative, retain labels, not present fabricated samples as fact.
- **Not for real-world advice.** This is an educational artifact; deploying it for agricultural advice for farmers would be irresponsible. Farming decisions affect livelihoods and the model has zero grounding.
- **Reproducibility:** corpus artifacts, training configs/logs, checkpoint manifests on disk; everything here was reproducible offline from local files.

## 10. Artifacts

| what | where |
|---|---|
| final weights (frozen) | `checkpoints/krishigpt_v3/best.pt` + `final/FREEZE_MANIFEST.json` |
| tokenizer | `data/processed/agri_bpe_tokenizer.json` (vocab 5,237) |
| corpus | `data/corpus_v3/agri_{train,valid}_v3.txt` (train/cache) with provenance in `data/corpus_v3/` |
| run config | `configs/nano_agri_v3.json` |
| run log / checkpoints | `checkpoints/krishigpt_v3/{train_log.jsonl, step_*.pt, best.pt, final.pt}` |
| evaluation harness | `evaluation/run_eval.py`, `evaluation/fabrication_eval.py`, `final/final_eval.py` |
| final eval artifacts | `final/final_eval.json`, `final/showcase_samples.md/.json`, `final/FINAL_REPORT.md`, `final/DIAGRAMS.md` |
| experiment notes | `evaluation/corpus_v2/EXPERIMENT_v2_warmstart.md`, `evaluation/corpus_v2/EXPERIMENT_v2_extended.md`, `evaluation/corpus_v3/EXPERIMENT_v3_warmstart.md`, `evaluation/corpus_v4/EXPERIMENT_v4_qa.md` |
