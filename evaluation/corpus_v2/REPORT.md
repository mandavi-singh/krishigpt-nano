# KrishiGPT-nano Corpus v2 — Audit, Build, and Tokenizer Decision

Date: 2026-08-17 · Built in `data/corpus_v2/` · v1 corpus and all checkpoints untouched.

## 1. Audit of the existing corpus (v1)

| Weakness | Evidence |
|---|---|
| Tiny document pool | 33 train docs; 4 valid docs |
| Skewed toward old books | Top-3 docs = 52% of corpus chars |
| Archaic language in training data | gutenberg:51764 (Tusser 1580, 18.6%) has 12 archaic markers/1k words; gutenberg:22973 (Markham 1613) 3/kW |
| OCR/scan noise | 3 Gutenberg books are 22–80% short lines; gutenberg:16594 has 43 reference-noise markers |
| Topic gaps (per 10k words) | plant physiology 3.3, agronomy 7.5, pests/diseases 7.8, Indian agriculture 8.7, machinery 12.9 |
| Validation set imbalance | v1 valid contains one huge archaic book (Roman Farm Management, 13.7% of corpus) |
| Duplication | Low: only 11 cross-doc duplicated long lines — sources are clean, the problem is scale + composition |

## 2. What v2 adds (licensing verified per document)

| Source | License | Docs | Notes |
|---|---|---|---|
| Wikipedia (new + re-used) | CC BY-SA 4.0 | 82 | machinery (Plough, Tractor, Combine harvester, Seed drill, Tillage, No-till), physiology (Plant physiology, Germination, Pollination, Photosynthesis), soil (Soil science, Soil pH, Loam, Topsoil, Soil erosion), pests (IPM, Locust, Fungicide, Herbicide), climate (Climate-smart agriculture, Rainfed agriculture, Drought mitigation, Water scarcity, Rainwater harvesting), Indian agriculture (Green Revolution in India, Kharif/Rabi crop, Agriculture in Tamil Nadu, ICAR), livestock (Dairy/Poultry farming, Fodder, Pasture, Animal husbandry, Silvopasture) |
| Project Gutenberg (public domain, `copyright=false` enforced via gutendex) | Public domain in the USA | 7 | screened for fiction, OCR noise, scan junk; per-bucket cap 2 |

Provenance for every document: `data/corpus_v2/sources_v2.json` + per-doc `*.meta.json` (id, provider, authors, url, license, license_url, accessed date).
**Excluded on rights:** ICAR/USDA extension PDFs, Indian government portals — reuse rights not machine-verifiable, so not scraped (same policy as v1). 7 Wikipedia titles missing/short were skipped (recorded in sources_v2.json).

## 3. Split and leakage guarantees

- v1 validation documents (`gutenberg:12140`, `wiki:Harvest`, `wiki:Monsoon`, `wiki:Organic farming`) are **reserved out of v2 entirely** — none can appear in either v2 split.
- v2 split is document-level (90/10, seed 0, deterministic): no document or duplicated line in both splits; exact-line dedupe applied cross-doc (lines ≥25 chars).
- Note: v2 validation docs differ from v1's, so validation losses are **not directly comparable** across corpus versions.

## 4. Corpus statistics

| Metric | v1 | v2 |
|---|---|---|
| Total docs | 37 | **89** |
| Chars after clean | 3,804,835 | 4,446,373 |
| Chars after exact-line dedupe | 3,799,633 (removed 5,202 / 0.14%) | **4,413,829** (removed 32,544 / **0.73%**) |
| Words | 646,685 | 739,794 |
| Train docs / chars / words | 33 / 3,162,028 / 535,320 | **80 / 4,150,989 / 699,762** |
| Valid docs / chars / words | 4 / 637,605 / 111,365 | **9 / 262,840 / 40,032** |
| Train tokens (existing 5,237 vocab) | 897,225 | **1,150,725 (+28.3%)** |
| Valid tokens | 178,018 | 67,918 |
| Tokens per 1,000 chars (train) | 283.7 | **277.2** |

Valid docs: `wiki:Agriculture`, `wiki:Agriculture in India`, `wiki:Green Revolution`, `wiki:Micro-irrigation`, `wiki:Crop`, `wiki:Barley`, `wiki:Fodder`, `wiki:Sugarcane`, `wiki:Locust` — modern, clean, topic-diverse (replaces the archaic-book-dominated v1 validation set).

Topic coverage (keyword hits / 10k train words, v1 → v2): plant physiology **3.3 → 31.8**, machinery 12.9 → 40.9, soil science 34.1 → 59.6, irrigation 13.7 → 55.9, climate 12.5 → 21.6, livestock 22.2 → 26.9, Indian agriculture 7.9 → 12.1, pests 8.9 → 35.7.

## 5. Tokenizer decision: REUSE the 5,237-token BPE (do not retrain)

Evidence — the existing tokenizer measured on v2 train (no retraining performed):

| Metric | v1 train (ref) | v2 train |
|---|---|---|
| Tokens / 1,000 chars | 283.7 | 277.2 (compression slightly *better*) |
| UNK rate (word %) | 0.0 | **0.133** (negligible) |
| Whole-word (single-token) rate | 80.46% | 78.08% |
| Avg tokens / word | 1.312 | 1.361 |

Tradeoff:

- **Reuse (chosen):** UNK leakage is 0.13% of words and compression is already equal-or-better on the new text, so a retrain buys almost no tokenization quality. More decisively: the vocabulary stays compatible with all existing checkpoints and results — the embedding table (670k params, ~36% of the 1.86M model) remains meaningful for warm-starting, and every future validation comparison stays on identical tokenization. Cost: zero (BPE retraining costs ~10–15 min for 5,000 merges for no measurable gain here).
- **Retrain (rejected):** would specialize merges to the extra 28% of text, but resets the embedding geometry, invalidates every checkpoint trained so far, and changes the tokenization basis (making val-loss trends across runs incomparable). Justified only if UNK exceeded ~1–2% or compression degraded markedly — neither is true.

**Recommendation: keep the current 5,237-token vocabulary for the next training experiment.**

## 6. Artifacts

- `data/corpus_v2/agri_train_v2.txt` (4,207,588 bytes) · `agri_valid_v2.txt` (264,148 bytes)
- `data/corpus_v2/corpus_v2_stats.json` (machine-readable report incl. full doc id lists)
- `data/corpus_v2/sources_v2.json` + 82 wiki + 7 gutenberg meta/provenance files + raw texts in `raw/`
- `evaluation/corpus_v2/`: audit_v1.json, build & audit scripts

Model training NOT started on v2 — awaiting approval. Tokenized id caches for v2 do not exist yet; they will be created by the data loader on first use (or can be precomputed).
