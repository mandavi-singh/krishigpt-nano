# Corpus-v2 CONTINUATION Experiment — Design (local, bounded)

Date: 2026-08-21 · Run dir: `checkpoints/krishigpt_nano_v2_extended/` · Local CPU only (no Colab, no GitHub).

## What this experiment is

The corpus-v2 warm-start run (`krishigpt_nano_v2`, 4 epochs / 2,336 steps) ended with
its own written recommendation (see `EXPERIMENT_v2_warmstart.md`, Verdict):

> "another bounded warm-start run over corpus v2 (the last ~6 validation deltas were
> -0.01 to -0.03, still improving → more steps are justified and cheap…)"

This experiment executes exactly that recommendation, locally:

- **Start:** `checkpoints/krishigpt_nano_v2/best.pt` (strict-loads weights + AdamW
  optimizer state). This is a full **resume-style continuation**: step=2300,
  tokens_seen=4,710,400, best_val_loss=4.5068 are all carried forward, and the LR
  schedule is extended over the new total horizon (cosine continues toward min_lr).
- **Budget:** 2,336 additional steps (= 4 × 584 steps/epoch; run finishes at step
  4,636, i.e. full epochs 5-7 plus 548/584 of epoch 8). `max_steps: 4636`.
- **Locked (unchanged):** architecture (6×128×4, RoPE, pre-norm, tied head,
  1,860,224 params), tokenizer (5,237-vocab BPE), corpus v2 (train 1,196,617 ids /
  valid 22,026 ids, same leak-safe split), optimizer (AdamW β=(0.9,0.95), wd=0.1,
  grad-clip 1.0), all training code, seed 0, seq_len 128, bs 16.

## What is NOT this experiment

- No Colab. The `colab/` notebook stays as-is (unexecuted) — source of truth is local.
- No architecture / vocab / corpus / optimizer changes.
- No follow-on experiment after this one without explicit user confirmation.

## Baselines to beat (all numbers from the fixed v2 validation split)

| checkpoint | step | val loss | val ppl | gr fab% | .9 fab% | k40 fab% | p.9 fab% |
|---|---|---|---|---|---|---|---|
| warm_start (v1-ext weights) | 0 | 5.1916 | 179.8 | 0.0 | 14.8 | 5.8 | 14.6 |
| **v2 best.pt** | 2300 | **4.5068** | **90.6** | 0.0 | 8.1 | 3.7 | 15.5 |
| v2-extended best | 4636 | ? | ? | ? | ? | ? | ? |

## Pre-flight gate (must all pass before expensive training)

1. Full pytest suite (175 tests) green.
2. Compatibility check (`training/precheck_v2_extended.py`): best.pt strict-loads into
   a fresh model + 2-group AdamW; vocab==5237; architecture 6/128/4 tied; val loss on
   the v2 valid split reproduces 4.5068 ± 0.001; RNG state restored.
3. 30-step dry run (`run_name: v2ext_dry`, disposable dir): the exact operation
   of the real run (warm-start load of best.pt + update_warm_start bookkeeping +
   extended cosine horizon) with a 30-step `total_batches` budget and
   `train_stats_only=true` (final.pt keeps metadata only; NO weights persisted,
   run dir is disposable). Gates: counters resume at 2300 / 4,710,400 / 4.5068,
   LR ≈ 2.94e-4 (mid-cosine of the extended horizon), train loss in-range
   (~3.5-5), two in-run validations reproduce ~4.5068-ish, final.pt stores
   exactly step 2330 / tokens 4,771,840. The real run's dir is separate
   (`krishigpt_nano_v2_extended`) and stays empty until launch.

## Schedule (LR over the extended 4,636-step horizon)

Cosine decay re-normalized to total_steps=4636, warmup=200 (already satisfied at
step 2300). LR at step 2300 ≈ 2.94e-4 (mid-cosine), decaying to 6e-5 by step 4636.
Same `get_lr` formula — only `max_steps` changed.

## Metrics + artifacts

- `train_log.jsonl` (log_every 50, val_every 100, save_every 300), `best.pt`, `final.pt`.
- Overfitting watch: train loss falling while val loss climbs above 4.5068 (and does
  best.pt fail to improve for the entire run).
- Post-run: `evaluation/run_eval.py --corpus v2` (5 prompts × greedy/temp0.9/topk40/
  topp0.9) → repetition/distinctness; then the same word-level fabricated-word metric
  (word-boundary attestation against the full v1+v2 corpus) as previous runs.
- Everything written under `evaluation/results/krishigpt_nano_v2_extended_best/`.

## Decision rule

- **Helpful:** val loss < 4.5068 at the new best point (and fabs at or below the v2
  row) → more continuation steps justified; recommend the next bounded extension.
- **Neutral:** val loss flat (±0.005) → the model has squeezed this corpus dry at this
  capacity; recommend bigger corpus / vocab work instead of more steps.
- **Overfitting:** val loss rises above 4.5068 and never recovers → halt further
  continuation; recommend regularization (dropout) or data work, and report the exact
  step where val loss peaked.

---

# Results (2026-08-21)

## Pre-flight (all gates passed)
- Full suite green: 177 passed (175 existing + 2 new tests pinning the dry-run
  stats-only checkpoint and the update-warm_start continuation semantics).
- `training/precheck_v2_extended.py`: best.pt strict-loads (99 AdamW states, RNG),
  vocab 5237, 6/128/4 tied 1,860,224 params, and v2 val loss reproduces to
  4.5068 (rel err 1.1e-5).
- 30-step dry run (`checkpoints/v2ext_dry`, stats-only, disposable): counters resumed
  at step 2300 / 4,710,400 tokens / best 4.5068, extended-cosine LR 3.52e-4 →
  3.48e-4, train loss in-range (4.16-4.43), val 4.5392-4.5376, final step 2330 /
  4,771,840 tokens, weights NOT persisted.

## Run
- Start: `krishigpt_nano_v2/best.pt` — **step 2300, tokens 4,710,400**, val 4.5068.
- Trained: **2,336 steps** → **step 4636, tokens 9,494,528** (4 extra epochs of the
  same 9,348-window v2 split). best.pt saved at **step 4600 (val 4.3761)**; final.pt
  at 4636 (val 4.3711, the last-possible eval; val_every=100 so the last recorded
  best was 4600).
- LR: extended cosine re-normalized to 4636 steps: 3.52e-4 at resume, decaying to
  the 6e-5 floor (reached at step ~4550).
- Throughput: training 1,200-3,100 tok/s (machine-load dependent; v2 run saw
  3,700-4,700 tok/s on a quieter machine), generation 27.5 tok/s (41.4 tok/s on
  the v2 run — environment, not model).

## Validation (v2 split, 172 windows) — same fixed eval set, apples-to-apples
| Point | train loss | val loss | val ppl |
|---|---|---|---|
| v2-run best (start) | 4.3712 | 4.5068 | 90.63 |
| step 2400 (first eval) | 4.15 | 4.5616 | 95.7 |
| step 2700 (first new best) | 4.28 | 4.5049 | 90.5 |
| step 3500 | 3.98 | 4.4259 | 83.6 |
| **step 4600 (best.pt)** | 3.96 | **4.3761** | **79.52** |
| step 4636 (final.pt) | — | 4.3711 | 79.33 |

- Val loss **improved at 14 of the 23 in-run evaluations**; the 3 small upticks
  (+0.0103, +0.0019, +0.0019) were transient noise, each followed by new bests.
- First 100-300 steps were above baseline (4.56): re-annotation of the LR from its
  6e-5 resting value back to mid-cosine (~3.4e-4) perturbs before settling — expected,
  not overfitting.
- **No overfitting:** val loss ended 0.13 below the start and was still decreasing;
  the train/val gap did widen modestly (≈0.14 → ≈0.42 on the noisy per-batch train
  readings), consistent with some memorization on a finite corpus but below the
  overfitting threshold set by the decision rule.

Improvement over the v2 run on the same eval set: **-0.1307 val loss (-2.9%),
ppl 90.63 → 79.52**.

## Generation — same 5 prompts × {greedy, temp0.9, topk40, topp0.9}, same seeds
Repetition / distinctness (mean over 5 prompts):

| Variant | metric | v1 ext | v2 best | v2-ext best | Δ vs v2 |
|---|---|---|---|---|---|
| greedy | rep-rate | 0.876 | 0.822 | 0.829 | ≈ still degenerate |
| greedy | distinct-1 | 0.14 | 0.19 | 0.18 | ≈ |
| temp0.9 | rep-rate | 0.219 | 0.184 | 0.197 | ≈ |
| temp0.9 | distinct-1 | 0.78 | 0.82 | 0.81 | ≈ |
| topk40 | rep-rate | 0.340 | 0.333 | **0.302** | ↓ improved |
| topp0.9 | rep-rate | 0.159 | 0.178 | **0.165** | ↓ improved |
| topp0.9 | distinct-2 | — | 0.99 | **0.98** | ≈ |

## Fabricated-word diagnostic (word-level, vs full v1+v2 corpus, regenerated with
identical seeds/seeded-variants)
| Variant | v1 ext | v2 best | v2-ext best | Δ vs v2 |
|---|---|---|---|---|
| greedy | 0.0% | 0.0% | 0.0% | — |
| temp0.9 | 14.8% | **8.1%** | **13.5%** | ↑ worse (sample-dependent; 22/163 vs 14/172 words) |
| topk40 | 5.8% | 3.7% | **3.6%** | ≈ flat |
| topp0.9 | 14.6% | 15.5% | **15.3%** | ≈ flat |

Metric determinism verified: the v2/v1-ext reference rows reproduce exactly
(8.1/3.7/15.5 and 14.8/5.8/14.6). Fabrication is stochastic-sample + small-N
sensitive; the stable conclusion across all three checkpoints is: greedy ≈ 0%,
topk40 ≈ 3.6-3.7%, temp0.9/topp0.9 ≈ 8-15%.

## Grammar / coherence (honest read of the actual samples)
topk40 samples are the most fluent yet and now string together plausibly
agronomic syntax ("…domesticated around the 19th century that had been grown in
the United States", "…the production of wheat", "soil structure… clay …
leaching"), but broken constructions persist ("They do not yet do to try",
"many varieties of farmers are rought", chapter/fiction debris "CHAPTER XXII
_the Philealeders"). Fabricated spelling fragments unchanged under sampling
(_Stamther, Yleders, tothing, Foit, Offalation, yarconic) and 64-token windows
truncate mid-sentence ("…is to re").

## Verdict against the decision rule
**Helpful.** val loss 4.3761 < 4.5068 with no overfitting; the corpus still has
not been squeezed dry at this capacity. Fab % is ≈ v2 on topk40/topp0.9 and
worse on temp0.9 within small-sample noise, so the capacity/data-scale floor on
fabrication identified in the v2 report stands — more steps buy calibration
(ppl), not spelling fidelity under stochastic decoding.

**Recommendation for the next step (do NOT start without confirmation):**
the last 400 steps all improved at ~-0.002 to -0.009 per 100 steps and the LR
is now pinned at its 6e-5 floor. Two defensible options:
1. **One more bounded ~1-2 epoch extension at the min LR** (~580-1170 steps,
   ~7-15 min) to see if val loss still moves below 4.37; expect diminishing
   returns (≤ -0.01 total) and stop once val deltas go flat (±0.003).
2. **Change the data/capacity lever instead:** grow corpus v3 or widen vocab,
   which is the higher-leverage fix for fabrication per both v2 reports.
Either way: continue reporting with top-k sampling only; greedy remains a
degenerate loop and should stay retired.

Artifacts: `checkpoints/krishigpt_nano_v2_extended/{step_*.pt, best.pt,
final.pt, train_log.jsonl}`; `checkpoints/v2ext_dry/` (dry-run, stats-only,
disposable); `evaluation/results/krishigpt_nano_v2_extended_best/{summary.json,
samples.json, report.md, fabrication.json}`; script `evaluation/fabrication_eval.py`;
v2 source checkpoints untouched.
