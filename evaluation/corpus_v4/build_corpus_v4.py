"""Corpus v4 builder — how-to manual expansion on top of v3 (v3 NOT modified).

Corpus v4 = (v3 train docs, byte-reused) + new how-to/practice docs.
Validation = the EXACT v2/v3 validation text (6 docs, never seen by any
model), copied byte-for-byte -> val loss stays directly comparable with
4.0804.

Leak-safety rules (supersets of v3):
  - v1 validation docs stay reserved out entirely.
  - v2 validation docs are NOT retrained; they remain the v4 validation set.
  - new docs whose wiki redirect id collides with a pre-existing doc are
    skipped as corpus duplicates (the existing text stays canonical).
  - exact-line dedupe of new docs against ALL previously trained text.

Tokenization: the EXISTING 5,237-vocab BPE is MEASURED only (no retraining),
via the same verified fast rank-merge encoder as v3.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.build_agriculture_corpus import clean_text  # noqa: E402
from evaluation.corpus_v3.build_corpus_v3 import (    # noqa: E402
    TOPIC_KEYWORDS,
    build_fast_encoder,
    corpus_stats,
    keyword_hits,
    token_stats,
)
from tokenizer import BPETokenizer                       # noqa: E402

V4_DIR = ROOT / "data" / "corpus_v4"
V3_DIR = ROOT / "data" / "corpus_v3"
V2_DIR = ROOT / "data" / "corpus_v2"
V1_DIR = ROOT / "data" / "processed"
TOK_PATH = V1_DIR / "agri_bpe_tokenizer.json"

V1_VALID_TXT = V1_DIR / "agri_valid.txt"
V3_TRAIN_TXT = V3_DIR / "agri_train_v3.txt"
V3_VALID_TXT = V3_DIR / "agri_valid_v3.txt"
V3_STATS = V3_DIR / "corpus_v3_stats.json"

#: Gutenberg books dropped on quality audit (periodical issues are
#: high-noise lists/ads; kept in raw/ with provenance, excluded from corpus).
QUALITY_DROPS = {
    "gutenberg:67813": "periodical issue (ads/lists) — dropped on audit",
    "gutenberg:48753": "periodical issue (ads/lists) — dropped on audit",
    "gutenberg:35696": "periodical issue (ads/lists) — dropped on audit",
}


def main() -> None:
    v3_stats = json.loads(V3_STATS.read_text(encoding="utf-8"))
    v3_new_ids = set(v3_stats["new_docs"]["ids"])
    v2_stats = json.loads(
        (V2_DIR / "corpus_v2_stats.json").read_text(encoding="utf-8"))
    v2_train_ids = set(v2_stats["split"]["train_ids"])
    v2_valid_ids = set(v2_stats["split"]["valid_ids"])
    v1_rec = json.loads(
        (ROOT / "data" / "sources" / "agri_sources.json").read_text(encoding="utf-8"))
    v1_ids = {r["id"] for r in v1_rec}
    reserved_v1_valid = {"gutenberg:12140", "wiki:Harvest", "wiki:Monsoon",
                         "wiki:Organic farming"}
    all_prev_ids = (v1_ids | v2_train_ids | v2_valid_ids | v3_new_ids)

    # ---- load + clean new v4 docs ------------------------------------------
    docs, drops, duplicates = [], [], []
    for meta_path in sorted(V4_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["id"] in QUALITY_DROPS:
            drops.append({"id": meta["id"], "reason": QUALITY_DROPS[meta["id"]]})
            continue
        if meta["id"] in all_prev_ids:
            duplicates.append({"id": meta["id"],
                               "requested": meta.get("requested_title",
                                                     meta["title"]),
                               "reason": "redirect collides with an existing "
                                         "v1/v2/v3 document; existing text "
                                         "stays canonical"})
            continue
        raw = (V4_DIR / "raw" / meta["raw_file"]).read_text(encoding="utf-8")
        if len(raw) < 2000:
            drops.append({"id": meta["id"], "reason": "too short after fetch"})
            continue
        docs.append({**meta,
                     "text": clean_text(
                         raw, cut_sections=meta["provider"] == "Wikipedia")})

    raw_total = corpus_stats(docs)

    # ---- exact-line dedupe vs ALL previously trained text --------------------
    seen: set[str] = set()
    for p in (V3_TRAIN_TXT, V1_VALID_TXT,
              V2_DIR / "agri_valid_v2.txt", V3_VALID_TXT):
        for line in p.read_text(encoding="utf-8").splitlines():
            if len(line) >= 25:
                seen.add(line.lower())
    cleaned: list[dict] = []
    removed_lines = 0
    for doc in docs:
        kept = []
        for line in doc["text"].splitlines():
            if len(line) >= 25 and line.lower() in seen:
                removed_lines += 1
                continue
            kept.append(line)
        cleaned.append({**doc, "text": "\n".join(kept)})
    docs = cleaned
    ded_total = corpus_stats(docs)

    # ---- split: train-only expansion; validation FIXED = v3 valid ------------
    train_docs = docs
    train_new_text = "\n\n".join(d["text"] for d in train_docs)

    v4_train_text = (V3_TRAIN_TXT.read_text(encoding="utf-8") + "\n\n"
                     + train_new_text)
    TRAIN_FILE = V4_DIR / "agri_train_v4.txt"
    VAL_FILE = V4_DIR / "agri_valid_v4.txt"
    TRAIN_FILE.write_text(v4_train_text, encoding="utf-8")
    shutil.copy2(V3_VALID_TXT, VAL_FILE)                # byte-identical eval set

    # ---- leakage assertions ----------------------------------------------------
    new_ids = {d["id"] for d in train_docs}
    assert new_ids.isdisjoint(all_prev_ids), "new doc id collides with history"
    assert new_ids.isdisjoint(v2_valid_ids)
    assert new_ids.isdisjoint(reserved_v1_valid)
    assert VAL_FILE.read_bytes() == V3_VALID_TXT.read_bytes(), "valid not identical"
    leaked_lines = sum(1 for line in train_new_text.splitlines()
                       if len(line) >= 25 and line.lower() in {
                           l.lower() for l in
                           V3_VALID_TXT.read_text(encoding="utf-8").splitlines()
                           if len(l) >= 25})
    assert leaked_lines == 0, leaked_lines

    # ---- tokenizer quality on the EXISTING 5,237-vocab BPE --------------------
    tok = BPETokenizer.load(TOK_PATH)
    encode_word = build_fast_encoder(tok, v4_train_text)
    new_tokens = token_stats(encode_word, train_new_text)
    print("  new-train token stats done", flush=True)
    agg_tokens = token_stats(encode_word, v4_train_text)
    print("  combined-train token stats done", flush=True)
    valid_tokens = token_stats(encode_word, VAL_FILE.read_text(encoding="utf-8"))
    v3_ref = v3_stats["tokens_existing_5237_vocab"]["combined_train"]

    # ---- topic coverage ----------------------------------------------------------
    new_topics = keyword_hits(train_new_text)
    agg_topics = keyword_hits(v4_train_text)

    provider_counts = Counter(d["provider"] for d in train_docs)
    license_counts = Counter(d["license"] for d in train_docs)

    report = {
        "name": "krishigpt_corpus_v4",
        "created_in": "evaluation/corpus_v4/build_corpus_v4.py",
        "policy": {
            "train": "v3 train docs (byte-reused) + v4 new docs",
            "valid": "exact byte copy of v3 valid (6 docs, never trained in "
                     "any run) so token ids/windows and val loss are directly "
                     "comparable with every checkpoint so far",
            "drops": drops,
            "corpus_duplicates_skipped": duplicates,
            "dedupe_rule": "exact-line dedupe of new docs vs all previously "
                           "trained text (lines >=25 chars)",
            "removed_duplicate_lines": removed_lines,
        },
        "total_docs_in_corpus": (v3_stats["total_docs_in_corpus"]
                                 + len(train_docs)),
        "v3_train_docs_reused": v3_stats["total_docs_in_corpus"],
        "new_docs": {
            "docs": len(train_docs),
            "provider_breakdown": dict(provider_counts),
            "license_breakdown": dict(license_counts),
            "ids": sorted(new_ids),
        },
        "validation_docs": {"docs": len(v2_valid_ids),
                            "ids": sorted(v2_valid_ids)},
        "raw_after_clean_new": raw_total,
        "after_dedup_new": ded_total,
        "chars_after_dedup": {
            "v3_train_part": V3_TRAIN_TXT.read_text(encoding="utf-8").__len__(),
            "new_part": ded_total["chars"],
        },
        "split": {
            "train": {"docs": v3_stats["total_docs_in_corpus"] + len(train_docs),
                      "chars": len(v4_train_text),
                      "words": len(v4_train_text.split())},
            "valid": {"docs": len(v2_valid_ids),
                      "chars": len(V3_VALID_TXT.read_text(encoding="utf-8")),
                      "words": len(V3_VALID_TXT.read_text(encoding="utf-8").split())},
        },
        "tokens_existing_5237_vocab": {
            "new_train_part": new_tokens,
            "combined_train": agg_tokens,
            "valid": valid_tokens,
            "reference_v3_train": v3_ref,
        },
        "topic_hits_per_10k_words": {"new_train": new_topics,
                                      "combined_train": agg_topics},
        "artifacts": {
            "train_text": str(TRAIN_FILE.relative_to(ROOT)),
            "valid_text": str(VAL_FILE.relative_to(ROOT)),
            "report": str((V4_DIR / "corpus_v4_stats.json").relative_to(ROOT)),
        },
    }
    (V4_DIR / "corpus_v4_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("CORPUS V4 REPORT")
    print("=" * 72)
    print(f"new docs fetched: {len(train_docs) + len(drops) + len(duplicates)} "
          f"-> kept {len(train_docs)} "
          f"({provider_counts.get('Wikipedia', 0)} wiki, "
          f"{provider_counts.get('Project Gutenberg', 0)} gutenberg)")
    print(f"quality drops: {[d['id'] for d in drops]}")
    print(f"corpus duplicates skipped: {[d['id'] for d in duplicates]}")
    print(f"cross-corpus duplicate lines removed: {removed_lines}")
    tr = report["split"]["train"]
    va = report["split"]["valid"]
    print(f"\nTRAIN: {tr['docs']:>4} docs {tr['chars']:>12,} chars "
          f"{tr['words']:>10,} words   (v3 + {len(train_docs)} new)")
    print(f"VALID: {va['docs']:>4} docs {va['chars']:>12,} chars "
          f"{va['words']:>10,} words   (byte-identical to v3 valid)")
    print(f"\n-- tokenization with EXISTING vocab {tok.vocab_size} (NO retrain) --")
    for name, s in [("v4 new train part", new_tokens),
                    ("v4 combined train", agg_tokens),
                    ("v4 valid (=v3 valid)", valid_tokens)]:
        print(f"  {name:<22} tokens={s['total_tokens']:>10,}  "
              f"tok/kchar={s['tokens_per_1000_chars']:>6}  "
              f"UNK_words={s['unk_word_pct']:>5}%  "
              f"single-tok={s['single_token_word_pct']:>5}%  "
              f"avg_tok/word={s['avg_tokens_per_word']}")
    print(f"  {'v3 train (ref)':<22} tokens={v3_ref['total_tokens']:>10,}  "
          f"tok/kchar={v3_ref['tokens_per_1000_chars']:>6}  "
          f"UNK_words={v3_ref['unk_word_pct']:>5}%  "
          f"single-tok={v3_ref['single_token_word_pct']:>5}%  "
          f"avg_tok/word={v3_ref['avg_tokens_per_word']}")
    print("\n-- topic hits / 10k words (new | combined) --")
    for t in new_topics:
        print(f"  {t:<24} {new_topics[t]:>7} | {agg_topics[t]:>7}")
    print(f"\nartifacts: {TRAIN_FILE.name}, {VAL_FILE.name}, "
          f"corpus_v4_stats.json")


if __name__ == "__main__":
    main()
