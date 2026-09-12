# KrishiGPT-nano — Final Experiment Report

*Building a 1.86M-parameter agriculture language model from scratch: corpus, training, warm-starting, and honest limitation analysis.*

Date: 2026-08-22 · Machine: CPU-only (i5-1335U, 16 GB) · Framework: PyTorch 2.9.1 CPU, Python 3.12 · All components implemented from scratch (no HF Transformers, no tiktoken); 177 automated tests.

---

## 1. Objective

Build **KrishiGPT-nano**: a small language model (SLM) for the agriculture domain, entirely from scratch — tokenizer, transformer architecture, training loop, sampling, evaluation — trained on license-clean, provenance-recorded agriculture text. Study how much domain-focused data (corpus v1 → v2 → v3), warm-starting, and bounded compute can improve validation loss and generation quality, and measure honestly where the approach stops helping (fabrication, grammar, greedy decoding).

## 2. System

- **Model:** 6-block decoder-only transformer, d_model 128, 4 heads (d_k 32), RoPE positions, pre-norm LayerNorm, GELU FFN (128→256→128), causal mask, token embedding tied with the LM head. **1,860,224 parameters** verified structurally (tied head asserts shared storage).
- **Tokenizer:** from-scratch BPE, 5,237 vocab (4 specials + 159 base chars + 5,000 merges), trained once on v1 train, reused for v2/v3 (measured UNK on final text 0.10% — reuse justified; retraining would have broken checkpoint compatibility).
- **Data:** Wikipedia (CC BY-SA 4.0) + Project Gutenberg (US public domain). ICAR/FAO/USDA material excluded: reuse terms not machine-verifiable. Per-document provenance JSON (provider, URL, license, access date) recorded before any use. Cleaning, exact-line dedupe, deterministic document-level splits (seed 0), no train/valid leakage at any stage.
- **Training:** AdamW, betas (0.9, 0.95), weight decay 0.1 on 2D params only, grad clip 1.0, cosine LR 6e-4 → 6e-5 with 200-step warmup, seq_len 128, batch 16, CPU ~400–3,100 tok/s. Warm starts carry weights + optimizer states.
- **Evaluation (identical protocol at every checkpoint):** 5 fixed prompts × {greedy, temp 0.9, top-k 40, top-p 0.9}, fixed seeds; validation loss/perplexity on a **byte-fixed validation set** (6 documents, 22,026 tokens, never trained in any run, identical across v2/v2-ext/v3 → all val losses directly comparable); repetition rate, distinct-1/-2, max run; **word-level fabrication metric**: each generated word checked by word-boundary regex against the corpus the model actually trained on.

## 3. Corpus & training lineage

| run | start | corpus | steps (new) | tokens (at best) | best ckpt | val loss | val ppl |
|---|---|---|---|---|---|---|---|
| v1 full | scratch | v1 (37 docs, 897k tok) | 800 | 1,638,400 | nano_full | 5.7076* | 301.1* |
| v1 extended | v1 full | v1 | +1,752 (best at 2,400 total) | 4,915,200 | nano_extended | 5.2720* | 194.8* |
| v2 warm-start | v1 ext | **v2 (89 docs, 1.20M tok)** | 2,336 | 4,710,400 | nano_v2 | 4.5068 | 90.63 |
| v2 extended | v2 | v2 | +2,336 | 9,420,800 | nano_v2_extended | 4.3761 | 79.52 |
| **v3 (FINAL)** | v2 ext | **v3 (184 docs, 1.97M tok)** | 3,840 | 7,782,400 | **krishigpt_v3** | **4.0804** | **59.17** |

\* v1 runs used a different (v1-era) validation document set — not comparable with the byte-fixed v2/v3 numbers; v2/v2-ext/v3 are comparable with each other. Cumulative gradient exposure through the lineage ≈ 17.3M tokens. Corpus sizes: v1→v2 +28% tokens; v2→v3 +64% tokens (gap-targeted: crop diseases, pests, nutrient chemistry, soil depth, irrigation methods, farming practices, Indian state agriculture, 20 crop-specific pages, terminology, modern prose).

## 4. Key results (final frozen checkpoint)

**Validation:** 4.0804 loss / 59.17 ppl (+6.8% improvement over v2-ext on the identical 172-window set). Validation improved at 18 of the 22 post-warmup evaluations and was still decreasing at exit — **no overfitting**; the first ~200 steps ran warm while the LR re-warmed from 6e-5 through the cosine schedule, recovered by step ~700.

**Generation (5 prompts, fixed seeds, best.pt):**

| decoding | rep-rate | distinct-1 | distinct-2 | fabricated words |
|---|---|---|---|---|
| greedy | 0.879 | 0.134 | 0.181 | 0.0% (but degenerate loops) |
| top-k 40 | 0.308 | 0.697 | 0.943 | 4.0% |
| **top-p 0.9** | **0.190** | **0.812** | **0.994** | **4.5%** |

Fabrication is measured against the model's own cumulative training text (v1+v2+v3), same definition as all previous checkpoints (prior rows reproduce exactly under the same seeds). Inference throughput: ~21–36 tok/s end-to-end on CPU, varying with machine load.

**Qualitative:** top-k/top-p samples reached factoid-adjacent constructions for the first time at v3 ("Rice is grown in East Asia", "clay horizon", "Compost improves the soil by increasing soil productivity"), with frequent grammatical glue; but novel fragments persist ("paramarchorus", "imHeralia", "mandelic") and 96-token windows truncate mid-thought. Greedy collapses into loops, now dominated by Wikipedia section-token artifacts (`= = =`, "See also") — a phase-transition cost of heavy wiki exposure.

## 5. What worked and what did not

**Worked:**
1. Larger, gap-targeted corpus: each corpus version moved val loss substantially (−0.43 with v2 despite new docs; −0.30 with v3 on the identical eval set); larger corpus ≈ lower fabrication floor.
2. Warm-starting from prior best checkpoints: batch-scale cost, never destabilized training beyond the warm LR transient; weights + AdamW state transferred cleanly across corpus switches.
3. Fixed byte-identical validation: honest cross-run comparison; no eval-set gaming possible, leakage structurally guarded (v1-valid reserved from v1; v2-valid never trained).
4. From-scratch stack with 177 tests: several real bugs found and fixed via measurement (RoPE `apply` shadowing `nn.Module.apply`; Windows cp1252 tokenizer saves; gutendex charset-suffixed keys; equivalence-proof verification of the rank-based fast BPE decode; naive BPE encode measured hours-slow on 7 MB and replaced by an equality-checked fast path).

**Did not work / honest limits:**
1. **Fabrication was reduced, never solved.** Best-case floor ≈ 4% of sampled words are novel forms; greedy ≈ 0% but unusable. The failure is capacity + data-scale: a 1.86M-parameter model with a 5,237-vocab cannot guarantee spelling of words rarely co-occurring.
2. **Greedy decoding stayed degenerate through all 5 runs** — a decoding limitation, not fixable by more compute; top-p 0.9 is the recommended setting.
3. Grammar improved but remains broken in ~half of sampled sentences; no further epoch-extensions are expected to be high-yield (late-run val deltas shrank to ~−0.002–0.006).

## 6. Validation-methodology notes (what made the numbers trustworthy)

- Every pre-training claim was pre-flighted: strict checkpoint load, architecture/vocab asserts, baseline val-loss reproduction to 1e-5 relative error, 30-step dry runs with stats-only checkpoints before spending compute.
- Every post-run artifact (logs, per-step checkpoints, eval summaries, samples, hash manifests) is on disk; corpus texts unchanged since build; all v1/v2 corpora+checkpoints SHAs verified unchanged after final run.
- Sample sizes are small (5 prompts × 64 tokens); small metric differences between decoding runs at the same checkpoint are noise; the stable claims are the order-of-magnitude ones (ppl, fabrication floor class, greedy-vs-sampling behavior).

## 7. Reproducibility

Everything runs locally from this repository: `requirements.txt` (torch CPU, numpy, pytest), `python -m pytest tests` (177 tests), `final/verify_final_checkpoint.py` (clean-cwd load + generation check), `final/final_eval.py`, `final/showcase_samples.py`. Final checkpoint frozen with SHA-256 manifest and read-only attribute (`final/FREEZE_MANIFEST.json`).

## 8. Limitations & ethics (summary)

- 128-token context, CPU-only, no KV-cache/quantization/batching in portable form; no instruction tuning; domain-specialized vocabulary coverage only.
- **No factual reliability: samples are labeled MODEL OUTPUT and must not be read as agricultural advice.**
- Licensing: CC BY-SA 4.0 attribution + public-domain provenance recorded per document and kept with the corpus; ICAR/FAO/USDA-government sources deliberately excluded.

## 9. Conclusion & next steps (not executed)

KrishiGPT-nano demonstrates a complete, tested, reproducible from-scratch SLM pipeline: corpus governance → tokenization → architecture → training → warm-start lineage → honest evaluation. Final val ppl 59.2 with fabrication ~4% under top-p.

Recommended (deferred, requires new decisions): (1) a decoding-only study at the frozen checkpoint to quantify the decoding share of residual fabrication; (2) clean human-authored "how-to" agriculture text as the highest-leverage remaining corpus lever; (3) instruction tuning (Phase 17 on the roadmap) if the goal shifts from pretraining to assistant.

## 10. Post-freeze addendum — corpus v4 + QA fine-tune (2026-09-07, Colab Tesla T4)

Executed after the v3 freeze; the submitted artifact was never touched (v3 `best.pt` SHA-256 re-verified byte-identical before and after the entire session). Full record: `evaluation/corpus_v4/EXPERIMENT_v4_qa.md`.

- **Corpus v4 pretraining** (33 new how-to/practice docs, 2.27M tokens, warm start from frozen v3, 4,440 steps on GPU): byte-fixed val loss **4.0804 → 3.9752** (ppl 59.2 → 53.3) — third consecutive corpus-expansion gain, now with visibly diminishing returns (−0.43 → −0.30 → −0.105).
- **QA fine-tune, round 1** (616 auto-generated definitional pairs, corrected 63-step/3-epoch schedule after a first 240-step attempt overfit and was archived): learned the Q/A surface **format** (Q:/A: turn-taking in samples) but moved nothing else — instruction compliance 0% → 0%, domain MCQ chance → chance (24.8% → 20.8%, paired questions, ~1 SE).
- **QA fine-tune, round 2** (`krishigpt_v4_qa2`, 63 steps on 801 *constraint-format* pairs — yes/no balanced 251/118, one-word, list-of-three + cleaned definitional; bench phrasing deliberately held out): **instruction compliance 0% → 20%** — both list-of-three tests pass, the project's first instruction-following movement. Passes are surface-format learning (the model shapes answer starts correctly, e.g. "Does rice need water? → yes", but cannot reliably stop); MCQ unchanged at chance (21.8%). Round 1's negative result is thereby diagnosed as a data-format failure, fixed once.
- **Verdict:** the 3-level funnel caught the round-1 failure exactly as designed (only L1 improved), and the round-2 fix moved L2 at zero knowledge cost — a complete diagnose → fix → re-measure loop at 1.86M parameters. v3 remains the frozen submission; v4/v4_qa/v4_qa2 are documented research artifacts (`evaluation/corpus_v4/EXPERIMENT_v4_qa.md`).

---

### Companion artifacts

| deliverable | file |
|---|---|
| model card | `final/MODEL_CARD.md` |
| diagrams (ASCII + Mermaid) | `final/DIAGRAMS.md` |
| final numbers (JSON) | `final/final_eval.json` |
| 15-prompt labeled samples | `final/showcase_samples.md` / `.json` |
| freeze manifest | `final/FREEZE_MANIFEST.json` |
| per-experiment records | `evaluation/corpus_v2/EXPERIMENT_v2_warmstart.md`, `.../EXPERIMENT_v2_extended.md`, `evaluation/corpus_v3/EXPERIMENT_v3_warmstart.md`, `evaluation/corpus_v4/EXPERIMENT_v4_qa.md` |
| corpus build reports | `evaluation/corpus_v2/REPORT.md`, `evaluation/corpus_v3/REPORT.md` |
