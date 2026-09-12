# Corpus-v4 Pretraining + QA Fine-Tune Experiment — Results & Verdict

Date: 2026-09-07 · Machine: Google Colab, Tesla T4 GPU (first non-CPU run in the lineage; same code, `device: cuda` in config). Runs: `checkpoints/krishigpt_v4` (corpus-v4 pretraining, warm start from frozen `krishigpt_v3/best.pt`) and `checkpoints/krishigpt_v4_qa` (QA fine-tune, warm start from `krishigpt_v4/best.pt`). Same 6x128x4 RoPE tied model (1,860,224 params), same 5,237-vocab BPE (UNK word rate on the new text measured at 0.089% — no retraining justified), same AdamW settings and training code as all prior runs.

## Pre-flight (all gates passed)
- Frozen v3 `best.pt` SHA-256 verified **before any v4 step**: `137cc83da53dd873cbab22c82e7344c7c84882d79b37017a37c765ff38e14536` (matches FREEZE_MANIFEST), and re-verified after the entire session — the submitted artifact was never touched.
- Corpus v4: v3 train byte-reused + 33 new how-to/practice docs (4 Gutenberg public domain, 29 Wikipedia CC BY-SA 4.0); 1,191 duplicate lines removed by exact-line dedupe; train = 217 docs / 2,269,941 ids / 17,733 windows; **valid byte-identical to the v3 valid set** (22,026 ids) so val losses remain directly comparable across v2 / v2-ext / v3 / v4.
- Schedule: 4 epochs x 1,110 steps = 4,440 steps, cosine 6e-4 -> 6e-5, 200-step warmup. Phase 1 completed in the first Colab session; the verified second session resumed and exited in 16 s (restart-safe step checkpoints + `find_latest` worked exactly as designed).
- QA data: 87,684 sentences scanned from corpus-v4 train -> 666 pairs (616 train / 50 valid, seed 0), 3 deterministic formats per fact ("What is X?", "Define X.", "Explain what X is."), corpus-valid lines excluded.

## Phase 1 — corpus-v4 pretraining (funnel L1)

| checkpoint | corpus val loss (byte-fixed set) | val ppl |
|---|---|---|
| krishigpt_v3 best (start) | 4.0804 | 59.2 |
| **krishigpt_v4 best** | **3.9752** | **53.3** |

-0.1052 loss (-2.6%), ppl -10% on the identical eval set — the third consecutive corpus-expansion gain, with clearly diminishing returns (v2: -0.43; v3: -0.30; v4: -0.105). ~9.1M new tokens this run (cumulative lineage exposure ~26.4M). Step-by-step log: `checkpoints/krishigpt_v4/train_log.jsonl`.

## Phase 2 — QA fine-tuning (two attempts)

- **Attempt 1 (240 steps, first Colab session): OVERFIT.** 240 steps = ~11 epochs on 21 steps/epoch (bs 8); train loss collapsed 3.84 -> 1.31. Old best archived as `best_overfit_v1.pt` for the record.
- **Attempt 2 (corrected schedule, verified session): 63 steps = 3 real epochs**, val/save every 21 steps, LR 1e-4 -> 1e-5, warm start from the v4 pretraining best. `best.pt` selected on held-out QA-valid loss, with the byte-fixed corpus val loss logged as a regression guard (vs 3.9752). Per-epoch QA-val / corpus-val numbers: `checkpoints/krishigpt_v4_qa/train_log.jsonl`.

## Funnel evaluation (v3 frozen baseline vs v4_qa)

| metric | krishigpt_v3 | krishigpt_v4_qa | delta |
|---|---|---|---|
| L2 instruction compliance (10 tests) | 0% | 0% | none |
| L3 MCQ overall accuracy (101 paired Qs, chance 25%) | 24.8% | 20.8% | -4.0pp |
| L3 MCQ weighted domain score | 39.4% | 36.5% | -2.9pp |

Both MCQ evaluations use the **same deterministic 101 questions** (seed 0, built from v1+v2+v3 train), so the comparison is paired. v3 itself sat at chance (24.8% vs 25%, SE ~4.3pp); the -4pp move is ~1 SE — statistically indistinguishable from chance, though its direction is consistent with QA-tuning shifting option likelihoods away from corpus prose.

**Smoke generation (top-p 0.9, seed 0) — the decisive observation:**
- **Format learned:** the model now emits Q:/A: turn-taking (opens a fresh "Q : ..." after an answer) — precisely the "format learning, not knowledge injection" goal stated in the config.
- **Content unchanged:** answers remain word-salad ("Treatise is soaked by compost , ambi , soybean tools ..."). No factual capability was gained — expected at 1.86M params / ppl ~53.

L1 health reports were also generated for both v4 runs (`evaluation/results/krishigpt_v4{,_qa}/health_report.json`, Drive-synced).

## Honest read: why L2 stayed at 0% (three causes, in order of importance)

1. **Format coverage.** All 616 pairs are three templates of ONE format family — free-form definitional QA with sentence-length answers. The benchmark's constraint formats (one word, yes/no, JSON, exactly-three, <=10 words, stop word) are *never demonstrated* in the fine-tune data, so this run could not teach them. A base LM mimics demonstrated surface structure; it does not induce constraint semantics from a single unbounded-answer style.
2. **Pair quality.** The heuristic subject extractor accepts clause-internal "X is" patterns. Actual pairs in the 50-pair valid set include "What is whatever soil?", "Explain what as each cow is.", "Explain what here the soil is." (the NON_DEFINITIONAL blocklist misses "here"/"whatever"/"as each ..."). A fraction of the 616 training pairs carries the same noise.
3. **Capacity.** At 1.86M params / 5,237 vocab / 128-token context, a 63-step format tune can shift surface structure, not semantics — the capacity limit documented in FINAL_REPORT section 5 was never in scope for this phase to fix.

## Verdict

- **L1 improved a third time:** corpus v4 pretraining lowered the byte-fixed val loss 4.0804 -> 3.9752 (ppl 59.2 -> 53.3) with no regression; diminishing returns are now visible.
- **L2/L3 unmoved:** QA fine-tuning at this scale moved neither instruction compliance (0% -> 0%) nor domain MCQ (chance -> chance). Its only demonstrated gain is Q/A surface format. This is a **measured negative result on instruction tuning at 1.86M parameters** — exactly the failure the 3-level funnel was built to catch before it could be mistaken for progress.
- **Submission unchanged:** krishigpt_v3 remains the frozen submitted artifact; its SHA-256 was verified byte-identical before and after the entire v4 effort. v4 / v4_qa are documented research artifacts, not the model-card checkpoint.

**Recommendation if instruction-following is retried:**
1. Train on *constraint-format* pairs, not more definitional pairs: yes/no, one-word, list-of-exactly-three generated from the same 87k-sentence pool (a few thousand pairs, same corrected 3-epoch / 63-step schedule, low LR).
2. Fix pair quality first: require sentence-initial subjects, extend the NON_DEFINITIONAL blocklist, keep the domain-lexicon head gate.
3. Guard against benchmark overfitting: hold out at least one constraint format family from training; keep the 10-test bench frozen as-is.
4. Calibrate expectations: partial compliance is the realistic ceiling; capacity + data-scale remain the binding constraints.

**Commitment honored:** no further training started after this experiment; the v3 checkpoint SHA re-verified post-session (verify cell re-runnable anytime).

---

# Round 2 — constraint-format QA fine-tune (`krishigpt_v4_qa2`, executed 2026-09-11, Colab Tesla T4)

Prepared in direct response to the verdict above; execution requires the Colab GPU (the v4/v4_qa checkpoints live on Drive). Everything below is built, verified locally, and staged in `v4qa2_upload/` (bundle README has upload instructions).

**Data — `data/qa_v2/` (801 train / 50 valid pairs, generated by `evaluation/corpus_v4/make_qa_pairs_v2.py`, deterministic seed 0):**
- Definitional: 399 — kept from round 1 but with the fixed extractor (sentence-initial subject required, extended blocklist, head must itself be a domain term; 0 bad subjects vs round-1's "whatever soil"/"as each cow" class).
- Yes/no: 369 — **balanced 251 yes / 118 no** via cross-category negatives ("Is a tractor a crop? → no"); round 1 was 100%-yes, which would have taught yes-bias.
- One-word: 18 — "What is loam? Reply with a single word. → soil" (term-category).
- List-of-three: 15 — "Name three farm practices. → grafting, weeding, rotation".

**Benchmark-overfit guard:** training phrasing deliberately differs from the frozen bench's exact surfaces ("Answer yes or no." vs bench "Answer with only yes or no."; "Reply with a single word." vs "Answer in exactly one word."; list families/phrasing disjoint from the bench's "grain crops"/"soil types"); the ≤10-words, JSON, and stop-word formats are never trained.

**Schedule (`configs/nano_agri_v4_qa2.json`):** 801 train pairs → trainer-measured 174 seq-128 windows → **21 steps/epoch** at bs 8; 3 real epochs = **63 steps**; LR 1e-4 → 1e-5; val/save every 21; warm start from round-1 `krishigpt_v4_qa/best.pt`. `training/train_qa.py` gained a `qa_dir` config key (backward-compatible; absent → `data/qa`, round-1 behavior).

**Delivery — zero-upload self-provisioning notebook (`colab/KrishiGPT_v4_qa2_gpu.ipynb`, builder `scripts/build_v4qa2_notebook.py`):** nothing new needs to be uploaded to Drive. The notebook embeds byte-identical copies of both generators and the patched trainer (base64), generates `data/qa_v2` in Colab from the corpus-v4 text already on Drive (seed 0, deterministic), **hard-asserts the pair counts against the verified local run (851/801/50)** before training, writes the config with the runtime device, trains, runs the frozen funnel bench, post-verifies the run config inside `best.pt` (qa_dir / 63 steps / donor), and syncs checkpoints + results + the generated data back to Drive. Frozen v3 SHA-verified before and after.

**Verified locally before staging (full end-to-end dry run, CPU):** scripts decode + write byte-identical (UTF-8-safe on Windows and Colab); pair generation reproduces 851/801/50 exactly; a full 63-step training run completed (train qa_loss 5.02 → 2.28, held-out qa_val 5.19 → 2.27 ppl 9.7, corpus_val 4.16 → 4.23 — the expected mild corpus regression from QA-format specialization); post-train checkpoint config verification passed; all four funnel scripts ran to completion against the dry checkpoint (health gate, instruction bench, MCQ bench, funnel report). Dry-run bench on the *fake* donor (v3 weights standing in for round-1) already showed 10% compliance (1/10) vs 0% for v3 — evidence that the constraint-format data moves the metric, though the real donor's number is the one that counts.

**Expected outcome (honest):** movement on the yes/no and one-word test families is the realistic ceiling; ≤10-words/JSON likely stay at 0% (never trained, and capacity remains the binding constraint). Any compliance > 0% would be the project's first instruction-following signal. If it stays 0%, the funnel verdict is confirmed and v3 remains the frozen submission — either way the round closes the question the funnel opened.

## Round 2 results (executed)

**Run integrity:** self-provisioning notebook verified everything inline — v3 SHA untouched (pre + post), pair counts reproduced exactly on Colab (851/801/50 from the Drive corpus), run config inside `best.pt` confirmed (qa_dir=qa_v2, 63 steps, round-1 donor), donor SHA `9390ed74…bcd20` recorded. Training: 63 steps / 3.00 epochs on T4 in ~2-3 min, best QA val loss **1.7380** (round 1: ~2.27 territory on its data; not directly comparable — different valid sets).

**Funnel (frozen 10-test bench, seeded):**

| metric | krishigpt_v3 | krishigpt_v4_qa (R1) | **krishigpt_v4_qa2 (R2)** |
|---|---|---|---|
| instruction compliance | 0% | 0% | **20%** |
| MCQ accuracy (101 paired Qs) | 24.8% | 20.8% | 21.8% (weighted 31.9%) |

Per-test: `list_three` and `list_three2` PASS (both list-of-exactly-three tests — the format with the most shaped training data: "X, Y, Z" three-comma answers). `yes_no`/`one_word` still FAIL despite 369/18 training pairs — outputs show the model emits "yes"/"no" tokens readily (smoke test: "Does rice need water?" → "yes") but then continues instead of stopping, so the post-processed answer is not exactly {yes,no} or one word. ≤10-words/JSON/stop-word: FAIL as predicted (never trained).

MCQ 21.8% ≈ round-1's 20.8%, both within ~1 SE of chance — unchanged conclusion: no domain-knowledge movement, QA-tuning keeps likelihoods slightly off corpus prose.

**Interpretation (honest):**
- The **first genuine instruction-following movement in the project**: 0% → 20%, gained by training on constraint-shaped answers (format coverage was indeed the binding cause, not capacity alone).
- The two passing tests are exactly the format family with the strongest shape signal (short, rigid, 3-item comma lists) — consistent with surface-format learning, not semantic instruction understanding.
- The pass mechanism is visible in samples: the model now *starts* answers in the right shape ("yes", "Trees are…") but cannot *stop* — a 128-token window model with no stop/EOS discipline. This is the next concrete bottleneck if anyone continues (train stop behavior / answer-length control, not more formats).
- Capacity limit stands: one-word answers are often nonsense ("Trees are locally generally nutritious…"), grammar remains broken — FINAL_REPORT §5 limits unchanged.

**Round-2 verdict:** constraint-format data moved Level 2 from 0% to 20% at zero knowledge cost; the negative result of round 1 is now understood as a *data-format* failure, fixable, and fixed once. v3 remains the frozen submitted artifact (its SHA verified untouched post-run); v4_qa2 is the project's best instruction-following checkpoint and the documented endpoint of the v4 lineage.

Artifacts: bundle `v4qa2_upload/` (upload contents into the Drive myllm folder, same subfolder structure); notebook `colab/KrishiGPT_v4_qa2_gpu.ipynb` (builder `scripts/build_v4qa2_notebook.py`); generator `evaluation/corpus_v4/make_qa_pairs_v2.py`; config `configs/nano_agri_v4_qa2.json`; data `data/qa_v2/`. Post-execution artifacts (Drive-synced by the notebook): `checkpoints/krishigpt_v4_qa2/{best.pt, final.pt, step_*.pt, train_log.jsonl}`, `evaluation/results/krishigpt_v4_qa2/{health_report.json, instruction_bench.json, mcq_bench.json, funnel_report.json}`, `data/qa_v2/` (generated in Colab, synced for the record).

Artifacts (all Drive-synced): `checkpoints/krishigpt_v4/{best.pt, final.pt, step_*.pt, train_log.jsonl}`; `checkpoints/krishigpt_v4_qa/{best.pt, final.pt, best_overfit_v1.pt, train_log.jsonl}`; `evaluation/results/krishigpt_v4{,_qa}/{health_report.json, instruction_bench.json, mcq_bench.json, funnel_report.json}`; build scripts `evaluation/corpus_v4/{fetch_sources_v4.py, build_corpus_v4.py, make_qa_pairs.py}`; QA data + stats `data/qa/{qa_pairs_train.jsonl, qa_pairs_valid.jsonl, qa_stats.json}`; corpus stats `data/corpus_v4/corpus_v4_stats.json`.
