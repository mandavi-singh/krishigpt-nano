"""Build corpus v2 from acquired sources (data/corpus_v2/) and compare it
against the existing tokenizer. NO tokenizer retraining here - statistics only,
so we can decide reuse-vs-retrain before the next training experiment.

Split policy: the 4 documents of the v1 validation split (gutenberg:12140,
wiki:Harvest, wiki:Monsoon, wiki:Organic farming) are EXCLUDED from v2
entirely and recorded as reserved - no chance of any doc appearing in both
splits across experiments. v2 then gets its own deterministic 90/10 document
split of the enlarged doc pool.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.build_agriculture_corpus import (  # noqa: E402
    clean_text, dedupe_lines, split_documents,
)
from tokenizer import BPETokenizer  # noqa: E402
from tokenizer.word_tokenizer import word_tokenize  # noqa: E402

V2_DIR = ROOT / "data" / "corpus_v2"
V1_TOK = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
V1_TRAIN_TXT = ROOT / "data" / "processed" / "agri_train.txt"

RESERVED_V1_VALID = {"gutenberg:12140", "wiki:Harvest", "wiki:Monsoon",
                     "wiki:Organic farming"}

#: Documents the warm-start model already SAW during v1 training (re-fetched
#: here as the same articles). They stay in v2 TRAIN (legitimate extra tokens)
#: but are excluded from the v2 VALIDATION pool: otherwise a warm-started
#: model would be evaluated on text it memorized and val loss would read
#: artificially low.
V1_SOURCES = ROOT / "data" / "sources" / "agri_sources.json"

TOPIC_KEYWORDS = {
    "crops/cultivation": "wheat rice maize barley sorghum sugarcane cotton legume crop cultivar sowing harvest potato tomato soybean chickpea mustard vegetable",
    "soil science": "soil loam clay humus salinity salin erod alkal till topsoil",
    "irrigation": "irrigat canal drip water well rainfed watering",
    "fertilizers/nutrients": "fertilis fertiliz manure compost nitrogen phosphor potassium nutrient dung urea npk conditioner",
    "pests/diseases": "pest insect blight rust fung disease weevil locust patholog virus armworm pesticide fungicide herbicide",
    "machinery": "plough plow harrow tractor seed drill thresher reaper implement tillage",
    "plant physiology": "photosynthes pollination stomat germination respiration transpirat breeding grow cultivar seedling",
    "agronomy": "agronom rotation fallow intercrop yield monoculture green revolution tillage agronom",
    "climate/weather": "monsoon drought frost rainfall climate season weather climate-smart",
    "livestock": "cattle cow dairy sheep poultry livestock pig fodder pasture husband",
    "indian agriculture": "india indian punjab bengal madras deccan ryot zamind kharif rabi",
}


def keyword_hits(text: str) -> dict:
    """Cheap topic breakdown: stem-prefix keyword matching per 10k words."""
    words = max(len(text.split()), 1)
    low = text.lower()
    out = {}
    for topic, stems in TOPIC_KEYWORDS.items():
        hits = sum(low.count(s) for s in stems.split())
        out[topic] = round(hits / (words / 10_000), 1)
    return out


def corpus_stats(docs: list[dict]) -> dict:
    chars = sum(len(d["text"]) for d in docs)
    words = sum(len(d["text"].split()) for d in docs)
    return {"docs": len(docs), "chars": chars, "words": words}


def load_v2_docs() -> list[dict]:
    docs, reserved, dropped = [], [], []
    for meta_path in sorted(V2_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = (V2_DIR / "raw" / meta["raw_file"]).read_text(encoding="utf-8")
        if meta["id"] in RESERVED_V1_VALID:
            reserved.append(meta["id"])
            continue
        if len(raw) < 2000:
            dropped.append({"id": meta["id"], "reason": "too short"})
            continue
        cut = meta["provider"] == "Wikipedia"
        docs.append({**meta, "text": clean_text(raw, cut_sections=cut)})
    return docs, reserved, dropped


def token_comparison(tok: BPETokenizer, text: str) -> dict:
    """BPE stats on `text` with the EXISTING tokenizer.

    * tokens / tokens-per-char: compression quality of the old vocab on v2
    * UNK words: v1 vocab coverage gap on new text (word -> [UNK])
    * single-token words: share of whole words already known by the vocab
    """
    freqs: Counter = Counter(word_tokenize(text))
    enc = {w: tok.encode(w) for w in freqs}
    total = sum(len(enc[w]) * freqs[w] for w in freqs)
    unk_words = sum(freqs[w] for w in enc if enc[w] == [1])
    whole = sum(freqs[w] for w in enc if len(enc[w]) == 1)
    n_words = sum(freqs.values())
    return {
        "total_tokens": total,
        "tokens_per_1000_chars": round(total / max(len(text), 1) * 1000, 1),
        "words": n_words,
        "word_types": len(freqs),
        "unk_words": unk_words,
        "unk_word_pct": round(100 * unk_words / max(n_words, 1), 3),
        "single_token_word_pct": round(100 * whole / max(n_words, 1), 2),
        "avg_tokens_per_word": round(total / max(n_words, 1), 3),
    }


def main() -> None:
    import re
    docs, reserved, dropped = load_v2_docs()
    print(f"loaded {len(docs)} candidate v2 docs "
          f"(reserved-out from v1 validation: {len(reserved)}, "
          f"dropped: {len(dropped)})")

    # ---- exact-line dedupe (same rule as v1) --------------------------------
    raw_total = corpus_stats(docs)
    docs = dedupe_lines(docs)
    ded_total = corpus_stats(docs)
    removed_chars = raw_total["chars"] - ded_total["chars"]

    # ---- split (document-level, deterministic, seed 0) ------------------------
    # Validation may ONLY draw from documents unseen by v1 training - the warm
    # start model memorized those. Train keeps them (fresh-revision tokens).
    v1_ids = {r["id"] for r in json.loads(V1_SOURCES.read_text(encoding="utf-8"))}
    v1_train_seen = v1_ids - RESERVED_V1_VALID
    fresh_pool = [d for d in docs if d["id"] not in v1_train_seen]
    seen_pool = [d for d in docs if d["id"] in v1_train_seen]
    _, val_docs = split_documents(fresh_pool, val_fraction=0.10, seed=0)
    val_ids = {d["id"] for d in val_docs}
    train_docs = [d for d in docs if d["id"] not in val_ids]
    assert not val_ids & v1_train_seen, "validation drew a v1-trained doc"
    assert len(seen_pool) + len(fresh_pool) == len(docs)
    print(f"split: valid pool = {len(fresh_pool)} unseen docs "
          f"({len(seen_pool)} v1-seen stay in train)")
    train_text = "\n\n".join(d["text"] for d in train_docs)
    val_text = "\n\n".join(d["text"] for d in val_docs)
    TRAIN_FILE = V2_DIR / "agri_train_v2.txt"
    VAL_FILE = V2_DIR / "agri_valid_v2.txt"
    TRAIN_FILE.write_text(train_text, encoding="utf-8")
    VAL_FILE.write_text(val_text, encoding="utf-8")

    # ---- tokenizer comparison on the existing BPE (no retraining!) ----------
    tok = BPETokenizer.load(V1_TOK)
    train_tokens = token_comparison(tok, train_text)
    val_tokens = token_comparison(tok, val_text)

    # UNK chars on the v1 training text (reference: how often did the old
    # corpus already hit the UNK path with THIS vocab)
    v1_text = V1_TRAIN_TXT.read_text(encoding="utf-8")
    v1_stats = token_comparison(tok, v1_text)

    # ---- topic breakdown ------------------------------------------------------
    train_topics = keyword_hits(train_text)
    val_topics = keyword_hits(val_text)

    # ---- report ---------------------------------------------------------------
    provider_breakdown = Counter(d["provider"] for d in docs)
    license_breakdown = Counter(d["license"] for d in docs)
    topic_docs = Counter(d.get("topic_bucket", "article") for d in docs)

    report = {
        "name": "krishigpt_corpus_v2",
        "created_in": "evaluation/corpus_v2/build_corpus_v2.py",
        "v1_validation_docs_reserved_out": sorted(reserved),
        "dropped": dropped,
        "total_docs": len(docs),
        "provider_breakdown": dict(provider_breakdown),
        "license_breakdown": dict(license_breakdown),
        "topic_doc_buckets": dict(topic_docs),
        "raw_after_clean": raw_total,
        "after_dedup": ded_total,
        "dedupe_removed_chars": removed_chars,
        "dedupe_removed_pct": round(100 * removed_chars / max(raw_total["chars"], 1), 2),
        "split": {
            "val_fraction": 0.10, "seed": 0,
            "valid_pool": ("documents unseen by v1 training only "
                           "(leak-safe for warm-start evaluation)"),
            "n_valid_pool": len(fresh_pool),
            "train_ids": [d["id"] for d in train_docs],
            "valid_ids": [d["id"] for d in val_docs],
            "train": corpus_stats(train_docs),
            "valid": corpus_stats(val_docs),
        },
        "tokens_existing_5237_vocab": {
            "train": train_tokens,
            "valid": val_tokens,
            "reference_v1_train": v1_stats,
        },
        "topic_hits_per_10k_words": {"train": train_topics, "valid": val_topics},
        "artifacts": {
            "train_text": str(TRAIN_FILE.relative_to(ROOT)),
            "valid_text": str(VAL_FILE.relative_to(ROOT)),
            "report": str((V2_DIR / "corpus_v2_stats.json").relative_to(ROOT)),
        },
    }
    (V2_DIR / "corpus_v2_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console summary ------------------------------------------------------
    print("\n" + "=" * 72)
    print("CORPUS V2 REPORT")
    print("=" * 72)
    print(f"docs: {len(docs)}  "
          f"({provider_breakdown.get('Wikipedia',0)} wiki, "
          f"{provider_breakdown.get('Project Gutenberg',0)} gutenberg)")
    print(f"licenses: {dict(license_breakdown)}")
    print(f"chars after clean : {raw_total['chars']:>12,}")
    print(f"chars after dedupe: {ded_total['chars']:>12,}  "
          f"(removed {removed_chars:,}, "
          f"{report['dedupe_removed_pct']}%)")
    tr, va = corpus_stats(train_docs), corpus_stats(val_docs)
    print(f"TRAIN: {tr['docs']:>3} docs {tr['chars']:>11,} chars "
          f"{tr['words']:>9,} words")
    print(f"VALID: {va['docs']:>3} docs {va['chars']:>11,} chars "
          f"{va['words']:>9,} words  ids={[d['id'] for d in val_docs]}")
    print("\n-- tokenization with EXISTING vocab 5237 (no retraining) --")
    for name, s in [("v2 train", train_tokens), ("v2 valid", val_tokens),
                    ("v1 train (ref)", v1_stats)]:
        print(f"  {name:<16} tokens={s['total_tokens']:>10,}  "
              f"tok/kchar={s['tokens_per_1000_chars']:>6}  "
              f"UNK_words={s['unk_word_pct']:>5}%  "
              f"single-tok={s['single_token_word_pct']:>5}%  "
              f"avg_tok/word={s['avg_tokens_per_word']}")
    print("\n-- topic hits / 10k words (train | valid) --")
    for t in train_topics:
        print(f"  {t:<24} {train_topics[t]:>7} | {val_topics[t]:>7}")
    print(f"\nartifacts: {TRAIN_FILE.name}, {VAL_FILE.name}, "
          f"corpus_v2_stats.json")


if __name__ == "__main__":
    main()
