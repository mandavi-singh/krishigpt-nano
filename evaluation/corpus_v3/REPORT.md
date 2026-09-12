# KrishiGPT-nano Corpus v3 — Gap-Fill Expansion Study

Date: 2026-08-22 · Built in `data/corpus_v3/` · v1 corpus, v2 corpus, tokenizer, and ALL checkpoints untouched (SHA-verified, see §7). NO training started — awaiting confirmation. NO tokenizer retraining (per instruction).

## 1. Sources and provenance (requirement 1)

Fetchers: same verified v2 pipeline — Wikipedia MediaWiki extracts (CC BY-SA 4.0, attribution per article) + Project Gutenberg gutendex shelf with `copyright=false` enforced; per-doc `*.meta.json` + `sources_v3.json` record provider/URL/license/license_url/access date. No licensing invented — every doc's license string comes from its provider.

| Outcome | count | notes |
|---|---|---|
| Wikipedia fetched & kept | **97** | gap lists: v2's 7 v2-skip retries, crop diseases (late blight, rust, mildews, rice blast, ergot, smut, Fusarium wilt), pests (aphid, whitefly, thrips, mealybug, fall armyworm, cutworm, wireworm, desert locust), fertilizers/nutrients (urea, potash, ammonium nitrate/sulfate, superphosphate, bone meal, vermicompost, biofertilizer, micronutrient, zinc deficiency), soil depth (structure, compaction, organic matter, microbiology, horizon, CEC, humus, clay minerals), irrigation methods (center pivot, furrow, water table, water conservation, watershed mgmt), farming practices (intercropping, companion planting, terracing, precision/hydroponics/vertical, greenhouse, regenerative, stubble burning, sowing), Indian agriculture (Maharashtra, Karnataka, Rajasthan, Odisha, White Revolution, coffee), crops (jute, tea, millet/pearl millet, peanut, lentil, pea, common bean, sunflower, rapeseed, oat, rye, cassava, sweet potato, coconut, banana, mango, onion, garlic), terminology (agronomy, horticulture, cultivar, seed bank, germplasm, green manure, monoculture, crop yield, agribusiness, machinery, food security), climate/post-harvest (El Niño, food storage, cold chain) |
| Gutenberg fetched & kept | **4** | Prairie Farmer Vol.56 ×2 (clean, early-modern journalism), The Stewardship of the Soil (address), Adobe Days (clean, 13% short lines) |
| Quality drop (kept as file, excluded from corpus) | 1 | `gutenberg:57457` "The Book of Husbandry" (1523): measured 12.7 archaic markers/kW + 65% short lines — worse density than v1's flagged worst book |
| Corpus-duplicate skip | 1 | `wiki:Surface irrigation` redirected to an article already present in v2 train — existing text stays canonical |
| Total new docs in v3 | **101** | |
| Skipped (missing/short at fetch) | 17 | incl. both `Agriculture in Punjab/West Bengal`, `Groundnut/Peanut`, `Wheat rust/Urea fertilizer` (redirected siblings already fetched), recorded in `sources_v3.json` |
| Redirect dupes removed at fetch | 2 | `Phytophthora infestans`→Late blight, `Soil microorganisms`→Soil microbiology |

Provenance: `data/corpus_v3/sources_v3.json` (46.8 KB) + 102 per-doc meta files + raw texts in `data/corpus_v3/raw/`.

## 2. Train/validation document counts (requirement 2)

| Split | docs | policy |
|---|---|---|
| **train** | **184** | = 83 v2 train docs (byte-reused, unchanged) + 101 new |
| **valid** | **6** | = the **exact v2 validation set**, byte-copied: `wiki:Combine harvester`, `wiki:No-till farming`, `wiki:Loam`, `wiki:Phosphorus cycle`, `wiki:Rainwater harvesting`, `wiki:Rabi crop` — docs never seen by any trained model in any run |

## 3. Chars / words / tokens (requirement 3)

| Measure | v2 train | **v3 train** | Δ | v3 valid |
|---|---|---|---|---|
| docs | 83 | 184 | **+101** | 6 |
| chars | 4,329,381 | **7,015,724** | **+62.0%** | 84,458 |
| words | 726,740 | **1,163,400** | +60.1% | 13,054 |
| BPE tokens (existing vocab) | 1,196,617 | **1,966,105** | **+64.3%** | 22,026 (identical) |

Train: 15,360 seq-128 windows → **960 steps/epoch** @ bs=16 (v2 had 584).

## 4. Leakage audit (requirement 4) — all passed as hard asserts

- new doc ids disjoint from ALL history (v1 pool ∪ v2 train ∪ v2 valid ∪ v1-validation reserved set);
- redirect collisions skipped (1 case, above);
- exact-line dedupe (lines ≥25 chars) of new docs against all previously-trained text: **844 duplicate lines removed** (mostly cross-Wikipedia boilerplate);
- `agri_valid_v3.txt` byte-identical to `agri_valid_v2.txt`;
- remaining new-doc lines cross-checked against v2 valid: **0 overlaps**;
- `verify_v3_compat.py`: fast-encoded v3 valid stream == cached v2 valid tensor **exactly** (22,026 ids).

## 5. Existing 5,237-vocab BPE quality (requirement 5) — no retraining

Measured with a rank-based fast encoder, **strictly equality-verified against `tok.encode` on 1,508 sampled word types** before use (naive encode is O(5000 merges) per word — hours over 7 MB — the timeout reason; tokenizer module unchanged).

| Text | tokens | tok/kchar | UNK words | single-token words | avg tok/word |
|---|---|---|---|---|---|
| v3 new train part | 769,488 | 286.5 | **0.052%** | 74.09% | 1.456 |
| v3 combined train | 1,966,105 | 280.2 | **0.100%** | 76.58% | 1.398 |
| v3 valid | 22,026 | 260.8 | 0.013% | 75.53% | 1.394 |
| v2 train (ref) | 1,196,617 | 276.4 | 0.129% | 78.08% | 1.362 |

**Coverage is acceptable** (UNK 0.05–0.13% of words ≪ the 1–2% retrain threshold established in §5 of `evaluation/corpus_v2/REPORT.md`; compression equal-or-better on new text: 286.5 vs 276.4 tok/kchar). **Recommendation: REUSE the 5,237-vocab tokenizer** → `krishigpt_nano_v2_extended/best.pt` stays directly compatible.

## 6. v3 vs v2 comparison (requirement 6)

Size: train tokens 1,196,617 → **1,966,105 (+64.3%)**; validation is the same fixed 6-doc/22,026-token set, so any v3 val loss is directly comparable with 4.3711/4.3761.

Topic hits per 10k words (new-part | combined) vs v2's known weak areas:
- pests/diseases: **63.6** | 49.5  (v2 combined was the weakest modern topic)
- fertilizers/nutrients: **47.9** | 43.0
- plant physiology: **56.3** | 54.3
- irrigation: **104.3** | 99.2
- crops/cultivation: **153.1** | 134.5 (incl. 20 new crop-specific articles)
- indian agriculture: **32.8** | 28.2 (state-level articles added)
- agronomy/practices: **58.3** | 56.9 (precision/vertical/hydro/regenerative/terracing/intercropping)
- livestock: 22.6 | 26.3; machinery: 21.3 | 33.4 (least changed by design — not the gap)

Clean modern English: gap articles were chosen specifically as modern explanatory prose (disease pages, nutrient chemistry, farming practices); the one archaic book was measured and dropped.

## 7. v2 artifacts and checkpoints untouched (requirement 7) — SHA-verified

`baseline_hashes.txt` (274 files across `data/corpus_v2`, `data/processed`, `data/raw`, `data/sources`, `checkpoints/`, `colab/`) was snapshotted before any fetch/build work and re-hashed after: **zero differences**. `checkpoints/krishigpt_nano_v2_extended/best.pt` (step 4600, val 4.3761) is byte-identical.

## 8. Compatibility gate (training-ready check, not training)

`verify_v3_compat.py` PASSED:
- existing BPE encodes v3 valid to *exactly* the v2-cached 22,026-id stream → every checkpoint's current val loss stays valid as the v3 comparison baseline;
- v3 train encodes to exactly 1,966,105 ids → 15,360 windows, 960 steps/epoch at seq_len 128 / bs 16.

## 9. What happens next (NOT started)

Awaiting explicit confirmation. The natural design, for decision only: warm-start from `krishigpt_nano_v2_extended/best.pt` on corpus v3 with a fresh LR schedule (same architecture/tokenizer/optimizer/code), bounded budget, same eval + fabrication harness. No file in v1/v2 was altered; the Colab notebook remains unexecuted.

Artifacts: `data/corpus_v3/{agri_train_v3.txt, agri_valid_v3.txt, corpus_v3_stats.json, sources_v3.json, raw/}` · scripts `evaluation/corpus_v3/{fetch_sources_v3.py, build_corpus_v3.py, verify_v3_compat.py}` · hash snapshots `evaluation/corpus_v3/{baseline_hashes.txt, after_hashes.txt}`.
