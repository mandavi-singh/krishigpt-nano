# Corpus-v2 Warm-Start Experiment — Results & Verdict

Run: `checkpoints/krishigpt_nano_v2/` · warm-start from `krishigpt_nano_extended/best.pt` (SHA-verified copy as `warm_start.pt`) · 4 epochs / 2,336 steps on corpus v2 · same 6x128x4 RoPE tied model (1,860,224 params), same 5,237-vocab BPE, same AdamW/optimizer & training code. No architecture/tokenizer/training-algorithm changes.

## Pre-flight verification (all passed)
- v2 token IDs built with the EXISTING tokenizer and match the build report exactly: train 1,196,617, valid 22,026 ids (cached under `data/corpus_v2/`; v1 caches untouched).
- Warm-start checkpoint strict-loads (model + 2-group AdamW state, 99 optimizer states); vocab 5237 == tokenizer vocab; architecture 6/128/4/tied; forward sanity OK.
- Leak-safe v2 split: valid drawn ONLY from documents the v1-trained model never saw (60-doc pool; 29 v1-seen docs kept in train). This fixed an initial leakage where 5 v1-trained articles had landed in v2 validation.

## Validation (v2 split, 172 windows) — apples-to-apples on the same eval set
| Point | val loss | val perplexity |
|---|---|---|
| warm-start baseline (before any v2 step) | **5.1916** | 179.8 |
| best.pt (step 2300) | **4.5068** | **90.6** |
| final (step 2336) | 4.5040 | ~90.3 |

Improvement over baseline: **-0.6848 loss (-13.2%)**, perplexity 179.8 → 90.6. Validation improved at every one of the 23 in-run evaluations and never increased → **no overfitting**. (Note: v1 vs v2 val losses use different eval doc sets and are not directly comparable; the 4-epoch delta above is on one fixed set, so it is the meaningful number.)

Throughput: training ~3,700–4,700 tok/s (CPU), generation eval 41.4 tok/s.

## Generation — same 5 prompts × {greedy, temp0.9, topk40, topp0.9}, same seeds
Repetition / distinctness (mean over 5 prompts):

| Variant | metric | v1 ext best | v2 best | Δ |
|---|---|---|---|---|
| greedy | rep-rate | 0.876 | 0.822 | ↓ slightly better, still degenerate |
| greedy | distinct-1 | 0.14 | 0.19 | ↑ but still loops |
| temp0.9 | rep-rate | 0.219 | 0.184 | ↓ better |
| temp0.9 | distinct-2 | 0.96 | 0.97 | ≈ |
| topk40 | rep-rate | 0.340 | 0.333 | ≈ |
| topp0.9 | rep-rate | 0.159 | 0.178 | ≈ |

Greedy remains a degenerate loop in both checkpoints (v1: "is a plant that is a plant…"; v2: "to be used to be used…", "= = =" runs). Sampling variants stay diverse.

## Fabricated-word diagnostic (word-level, vs full v1+v2 corpus; deterministic regen)
Words absent from ALL training text:
| Variant | v1 ext best | v2 best | Δ |
|---|---|---|---|
| greedy | 0.0% | 0.0% | — |
| temp0.9 | **14.8%** | **8.1%** | ↓ ~half |
| topk40 | 5.8% | **3.7%** | ↓ |
| topp0.9 | 14.6% | **15.5%** | ≈ (did not improve) |

## Grammar / coherence (honest read of the actual samples)
Improved but not fixed. v2 samples now form more plausible agriculture phrasing ("…soil supplies on a natural transport of carbon…", "…production of soybeans…"), but still contain broken constructions ("does not picking", "are kept the year", "the same farmer is not the best") and fabricated spelling fragments persist under sampling (imtype, Bosdian, missiforage, Stamther, Finently).

## Verdict
**The larger/better corpus helped — substantially on loss and partly on generation — but it does NOT fully fix the fabricated-word or grammar problem.**

- Validation loss improved 13.2% (ppl 179.8 → 90.6) with no overfitting: the model is better calibrated on held-out agriculture text.
- Token-level fabrication dropped from ~0 in v1-ext to ~0 in v2 greedy and was roughly halved for temp0.9 (14.8→8.1%) and topk40 (5.8→3.7%), but top-p stayed flat (14.6→15.5%) — i.e. the problem is attenuated, not eliminated.
- Repetition/diversity on sampling is comparable-or-better; greedy is still unusable (decoding limitation, unaffected by corpus).

**Root cause assessment:** the fabricated words are built from common BPE sub-word pieces, so they are a **capacity + data-scale** artifact, not a decoding bug. More/larger corpus helps because it exposes more real word forms, but 4.4M chars is still small for a 5,237-vocab model to lock down spelling. Real reduction of fabrication likely needs: larger corpus AND/or more epochs at min-LR, and possibly a larger vocab (finer merges reduce novel sub-word joins). Decoding (top-k/top-p/temperature) controls diversity but not factual spelling.

**Recommendation for the next experiment (highest leverage):** another bounded warm-start run over corpus v2 (the last ~6 validation deltas were -0.01 to -0.03, still improving → more steps are justified and cheap, ~20 min per 4 epochs). If loss still improves, keep extending; only then move to a bigger corpus or a vocab retrain. Greedy decoding should be retired in favor of top-p/top-k sampling for reporting.

Artifacts: `checkpoints/krishigpt_nano_v2/{step_*.pt, best.pt, final.pt, warm_start.pt, train_log.jsonl}`; `evaluation/results/krishigpt_nano_v2_best/{summary.json, samples.json, report.md}`; v2 corpus in `data/corpus_v2/` (v1 corpus, tokenizer, and all earlier checkpoints untouched).
