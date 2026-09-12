# DATA — provenance and purpose

This directory holds all text data. The distinction between uses is documented
per subdirectory (never mix RAG docs with pretraining text without saying so).

## Subdirectories

| dir          | purpose                                              |
|--------------|------------------------------------------------------|
| raw/         | downloaded/collected source text (unchanged)          |
| processed/   | cleaned, tokenized, split, and ready-to-train data    |
| sources/     | one JSON record per external document collected       |

## Agriculture pretraining corpus (System A)

Built by `build_agriculture_corpus.py` on 2026-08-17. **This corpus is for
pretraining the scratch GPT. It is NOT the RAG knowledge base (System B), which
will use a separately recorded document set.**

- Sources (37 documents, full records in `sources/agri_sources.json`):
  - 32 Wikipedia agriculture articles — **CC BY-SA 4.0** (attribution recorded)
  - 5 Project Gutenberg public-domain agriculture books — **public domain (USA)**
  - 1 skipped ("Drought": API error at fetch time)
- ICAR / Govt. of India / FAO material is deliberately NOT included here:
  per-document reuse terms cannot be verified automatically. RAG (System B) may
  later use them with manual per-source license review.

## Final corpus statistics (verified, on disk)

| item | value |
|---|---|
| train | 33 docs, **3,162,028 chars**, 535,320 words |
| valid | 4 docs, 637,605 chars, 111,365 words |
| chars removed by cross-doc dedup | 5,202 |
| BPE | 5,000 merges, **vocab 5,237** |
| train tokens | **897,225** (283.7 tokens / 1000 chars) |
| split | document-level 90/10, seed 0, deterministic |

Files:
- `processed/agri_train.txt`, `processed/agri_valid.txt`
- `processed/agri_bpe_tokenizer.json` (our from-scratch BPE, trained on train
  only; load with `BPETokenizer.load`)
- `processed/agri_corpus_stats.json` (machine-readable statistics)
- `raw/agri/*.txt` + `sources/agri_sources.json` (provenance)

## Educational dataset

- `raw/tiny_shakespeare.txt` (1.1 MB, karpathy/char-rnn, MIT license,
  downloaded 2026-08-16). **Educational only** — NOT part of KrishiGPT and
  NOT mixed with the agriculture corpus.

## Rule (non-negotiable)

Do not invent agricultural facts. If a document is not from a recorded,
trusted public source, it must not be presented as trusted knowledge.
