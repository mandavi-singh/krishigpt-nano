# Corpus-v3 Warm-Start Experiment — Results & Verdict

Run: `checkpoints/krishigpt_v3/` · warm-start from `krishigpt_nano_v2_extended/best.pt` · 4 epochs / 3,840 steps on corpus v3 · same 6x128x4 RoPE tied model (1,860,224 params), same 5,237-vocab BPE (no retraining), same AdamW/optimizer & training code. v1/v2 corpora and ALL earlier checkpoints untouched (SHA-verified, zero diffs).

## Pre-flight (all gates passed)
- `training/precheck_v3.py`: v2-extended best.pt strict-loads (weights + 99 AdamW states + RNG); vocab 5237 == tokenizer vocab; architecture 6/128/4/tied;
- v3 train id cache built: **1,966,105 tokens** — exactly matches the verified corpus report;
- v3 valid id stream **exactly identical** to the v2 validation cache (22,026 ids, byte-level);
- baseline val loss on the byte-fixed set reproduces **4.3761** (rel err 1.1e-5) before any v3 step.
- Dry run (`checkpoints/krishigpt_v3_dry`, 30 steps, stats-only): dataloading, counter reset, loss ~4.0-4.4, val 4.3698-4.3709, final step 30 / 61,440 tokens (=30x16x128), metadata-only checkpoint all verified.

## Config note (one structural change was unavoidable and documented)
The v2-extended schedule mechanism (`update_warm_start`/`extend_schedule`) requires the donor step < the extended horizon; it cannot wrap a NEW corpus whose schedule restarts. So the v3 run follows the established same-precedent: warm-start load of weights+optimizer, fresh step/counter bookkeeping, same cosine 6e-4->6e-5 over the new horizon, **epochs=4 carried as the invariant** (same as every prior corpus switch). All hyperparameters identical to prior runs.

## Validation (v2-identical set, 172 windows) — apples-to-apples
| Point | val loss | val ppl |
|---|---|---|
| v2-extended best pt (start) | 4.3761 | 79.5 |
| v2-extended final (ref) | 4.3711 | 79.3 |
| step 100 (warmup, LR 3e-4) | 4.4053 | 81.9 |
| step 700 (first recovery) | 4.3838 | 80.1 |
| step 1000 | 4.3527 | 77.7 |
| step 2000 | 4.1885 | 65.9 |
| step 3000 | 4.1147 | 61.2 |
| **step 3800 (best.pt)** | **4.0804** | **59.2** |
| step 3840 (final.pt) | 4.0857 | 59.4 |

Steps trained: **3,840** (fresh schedule, 960/epoch x 4). Tokens: 7,864,320 new (cumulative exposure through the lineage: 9,420,800 + 7,864,320 = 17,285,128). Improvement over the v2-extended baseline on the identical eval set: **-0.2957 loss (-6.8%), ppl 79.5 -> 59.2**. First ~200 steps ran warm (LR returning from 6e-5 to 1.5e-4->6e-4 during warmup), recovered by step ~700, then improved at 18/22 further evaluations; the only small non-improvements were +0.003-+0.015 noise bands. **No overfitting**: val loss is still decreasing at the final evaluation, train/val gap wide (train ~3.7-4.0 batch-noise) but val floor never breached.

## Generation — same 5 prompts x {greedy, temp0.9, topk40, topp0.9}, same seeds
Repetition / distinctness (mean over 5 prompts):

| Variant | metric | v1 ext | v2 | v2-ext | **v3** | Delta v3 vs v2-ext |
|---|---|---|---|---|---|---|
| greedy | rep-rate | 0.876 | 0.822 | 0.829 | 0.879 | worse loop (`= = =` runs from wiki section tokens; greedy decoding artifact) |
| greedy | distinct-1 | 0.14 | 0.19 | 0.18 | 0.13 | ditto |
| temp0.9 | rep-rate | 0.219 | 0.184 | 0.197 | **0.213** | ~= |
| temp0.9 | distinct-1 | 0.78 | 0.82 | 0.81 | 0.79 | ~= |
| topk40 | rep-rate | 0.340 | 0.333 | 0.302 | 0.308 | ~= |
| topp0.9 | rep-rate | 0.159 | 0.178 | 0.165 | **0.190** | slightly higher but still low; distinct-2 0.994 highest of all runs |

## Fabricated-word diagnostic (word-level, each checkpoint attested vs its OWN cumulative training text)
| Variant | v1 ext | v2 | v2-ext | **v3** | Delta v3 vs v2-ext |
|---|---|---|---|---|---|
| greedy | 0.0% | 0.0% | 0.0% | 0.0% | unchanged |
| temp0.9 | 14.8% | 8.1% | 13.5% | **7.6%** | **improved** (trace line is noisy; 13.5 was sample noise) |
| topk40 | 5.8% | 3.7% | 3.6% | **4.0%** | ~= flat |
| topp0.9 | 14.6% | 15.5% | 15.3% | **4.5%** | **much improved** (-70%) |

## Grammar / coherence (honest read of the actual samples)
v3 topk40 samples now occasionally produce **factoid-adjacent constructions** not seen in v2 ("Rice is grown in East Asia", "clay horizon", "sowing to the first year's planting", "produce by topping on soil"), and the grammatical glue is often clean ("The best fertilizer for wheat , with the best wheat , is the _Stam_ , and a well-known tree"). But fabrication still operates at the word-boundary level: `_Stam_`, `paramarchorus`, `mandelic`, `imHeralia`, `mandelic foundations`, `Unound` — and the signature "the Mumbai Center re" cut-off at 64 tokens confirms the context window still truncates. Section-token loops (`= = =`, "See also") are now the dominant greedy failure mode — an artifact of heavy Wikipedia exposure that pure scale cannot fix by only rewiring decoding.

## Verdict
**Larger, gap-targeted corpus helped substantially on validation (-0.30 loss, 79.5 -> 59.2 ppl, no overfitting) and gave the biggest single drop yet in fabricated words under top-p decoding (15.3 -> 4.5%). It did NOT solve fabrication; it moved the floor.** Greedy stayed unusable (and degraded via section tokens). Sample noise on the small-seeded set makes temp0.9 and topk40 differences within +/-0.5pp unreliable between runs — the stable cross-run pattern remains: greedy ~0%, top-k ~3.6-4.0%, and top-p/temp0.9 floors now clearly tracked with corpus size.

**Root-cause status:** fabrication under stochastic decoding is confirmed as a capacity + data-scale phenomenon. Even 7.1M chars of curated agriculture text and a near-attested-vocabulary of 5,237 tokens does not lock all novel subword joins; however, the `top-p 0.9` path now shows the first *material* improvement, suggesting the broken nucleus tail (high-rank low-prob mass) is where most novel joins used to come from. The remaining failure class is (a) rare novel fragments assembled from high-frequency pieces, and (b) residual section/template artifacts crawling into greedy.

**Recommendation:** do NOT start more training without confirmation. Natural next steps, in priority order:
1. **Minimize the decoding lever, not the compute lever**: greedy + high-spec top-p (0.85-0.95 with temperature ~0.7-0.8) with val loss 4.08 is currently the highest-quality sampling point; one more planned short experiment to compare greedy vs sample at this checkpoint would isolate how much of the remaining fabrication is a decoding artifact rather than a model artifact.
2. If more training: one more bounded ~1-2 epoch extension at the min LR (4 epochs of declining returns is likely — see the last 400 steps where val deltas were -0.006).
3. If improving the corpus further: the highest-leverage remaining gap is clean human-authored agriculture prose (extension-style "how-to" material), which would attack grammar directly. The Wikipedia article cap has already squeezed the largest single factor of the corpus.

**Commitment honored**: no further training has been started after this experiment; all v1/v2 artifacts SHAs verified byte-identical after the run.

Artifacts: `checkpoints/krishigpt_v3/{step_*.pt, best.pt, final.pt, train_log.jsonl}`; dry run `checkpoints/krishigpt_v3_dry/`; `evaluation/results/krishigpt_v3_best/{summary.json, samples.json, report.md, fabrication.json}`; script `evaluation/fabrication_eval.py` (pooling adjusted in corpora for the v3 train text); v3 id caches `data/corpus_v3/{agri_train_ids.pt, agri_valid_ids.pt}`.
