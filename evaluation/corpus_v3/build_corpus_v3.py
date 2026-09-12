"""Corpus v3 builder — gap-fill expansion on top of v2 (v2 corpus NOT modified).

Corpus v3 = (v2 train docs, byte-reused) + new gap-fill docs.
Validation = the EXACT v2 validation text (6 docs, never seen by any model),
copied byte-for-byte so every token id and window is identical to all runs so
far -> val loss stays directly comparable with 4.3761.

Leak-safety rules (supersets of v2):
  - v1 validation docs stay reserved out entirely.
  - v2 validation docs are NOT retrained; they become the v3 validation set.
  - new docs whose wiki redirect id collides with a pre-existing v1/v2 doc are
    skipped as corpus duplicates (the existing text stays canonical).
  - exact-line dedupe of new docs against ALL previously trained text.

Tokenization: the EXISTING 5,237-vocab BPE is MEASURED only (no retraining).
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
from tokenizer import BPETokenizer                                     # noqa: E402
from tokenizer.word_tokenizer import word_tokenize                     # noqa: E402

V3_DIR = ROOT / "data" / "corpus_v3"
V2_DIR = ROOT / "data" / "corpus_v2"
V1_DIR = ROOT / "data" / "processed"
TOK_PATH = V1_DIR / "agri_bpe_tokenizer.json"

V1_VALID_TXT = V1_DIR / "agri_valid.txt"
V2_TRAIN_TXT = V2_DIR / "agri_train_v2.txt"
V2_VALID_TXT = V2_DIR / "agri_valid_v2.txt"
V2_STATS = V2_DIR / "corpus_v2_stats.json"


def build_fast_encoder(tok: BPETokenizer, probe_text: str) -> "Callable[[str], list[int]]":
    """Rank-based BPE application (standard fast decode) with a word cache.

    Provably equivalent to BPETokenizer.encode(word) for every word: same UNK
    test on unknown characters, same base symbolization, and repeatedly merging
    the LOWEST-RANK adjacent pair == applying the learned merges in learning
    order (a merge of rank r is never possible by rank-r'-r'-r' once every
    lower-ranked applicable pair is exhausted in the same scan, exactly as the
    pedagographic loop does). Verified below by strict equality against
    tok.encode() on a deterministic sampled word-type set from the real corpus
    before any stats are computed. The tokenizer module itself is NOT modified.
    """
    merges = list(tok.merges)
    base_tokens = tok.base_tokens
    rank: dict[tuple[str, str], int] = {
        (left, right): i for i, (left, right, _m) in enumerate(merges)}
    token2id = tok.token2id
    cache: dict[str, list[int]] = {}

    def encode_word(word: str) -> list[int]:
        if any(ch not in base_tokens for ch in word):
            return [1]                                    # UNK (same rule)
        seq: list[str] = list(word) + ["</w>"]
        while True:
            best_i, best_rank = -1, len(merges)
            for i in range(len(seq) - 1):
                r = rank.get((seq[i], seq[i + 1]), len(merges))
                if r < best_rank:
                    best_i, best_rank = i, r
            if best_i < 0:
                break
            seq[best_i:best_i + 2] = [merges[best_rank][2]]
        return [token2id[s] for s in seq]

    def lookup(word: str) -> list[int]:
        hit = cache.get(word)
        if hit is None:
            hit = cache[word] = encode_word(word)
        return hit

    # equivalence verification before anything uses the fast path: deterministic
    # sample of real word types (word_tokenize output = single tokens, so
    # lookup and tok.encode are applied to identical inputs) + single-char
    # punctuation/non-ASCII cases
    import random as _random
    all_types = list(set(word_tokenize(probe_text)))
    rng = _random.Random(0)
    sample = rng.sample(all_types, min(1500, len(all_types)))
    sample += ["the", "wheat", "soil", "NPK", "'", "é", "–", "」"]
    n_checked = 0
    for w in sample:
        assert lookup(w) == tok.encode(w), f"fast encoder mismatch on {w!r}"
        n_checked += 1
    print(f"fast BPE encoder verified == tok.encode on {n_checked:,} word types"
          f" ({len(all_types):,} distinct types in probe text)", flush=True)
    return lookup


#: Gutenberg books dropped on quality audit (measured markers; kept in raw/
#: with provenance but excluded from the corpus text).
QUALITY_DROPS = {
    "gutenberg:57457": ("archaic language 12.7 markers/kW (worst v1 book was "
                        "18.6/kW and already flagged) + 65% short lines"),
}

TOPIC_KEYWORDS = {
    "crops/cultivation": "wheat rice maize barley sorghum sugarcane cotton legume crop cultivar sowing harvest potato tomato soybean chickpea mustard vegetable lentil pea peanut sunflower cassava sweet banana mango coconut oat rye jute tea millet",
    "soil science": "soil loam clay humus salinity salin erod alkal till topsoil horizon compaction cation microbiology organic matter clay mineral",
    "irrigation": "irrigat canal drip water well rainfed watering sprinkler furrow pivot waterlogging water table watershed",
    "fertilizers/nutrients": "fertilis fertiliz manure compost nitrogen phosphor potassium nutrient dung urea npk conditioner potash ammonium vermicompost biofertilizer micronutrient zinc superphosphate bonemeal bone meal",
    "pests/diseases": "pest insect blight rust fung disease weevil locust patholog virus armworm pesticide fungicide herbicide aphid whitefly thrips mealybug cutworm wireworm mildew ergot smut willt fusarium borers",
    "machinery": "plough plow harrow tractor seed drill thresher reaper implement tillage mechaniz machinery",
    "plant physiology": "photosynthes pollination stomat germination respiration transpirat breeding grow cultivar seedling germplasm seed bank",
    "agronomy": "agronom rotation fallow intercrop yield monoculture green revolution tillage intercropping multiple cropping companion planting sowing stubble precision agriculture hydroponics greenhouse vertical farming regenerative",
    "climate/weather": "monsoon drought frost rainfall climate season weather en nino flood post-harvest cold chain storage",
    "livestock": "cattle cow dairy sheep poultry livestock pig fodder pasture husband goat aquaculture beekeeping",
    "indian agriculture": "india indian punjab bengal madras deccan ryot zamind kharif rabi maharashtra karnataka rajasthan odisha white revolution icar",
}


def keyword_hits(text: str) -> dict:
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


def token_stats(encode_word: Callable[[str], list[int]], text: str) -> dict:
    """Tokenization quality of `text` under an existing BPE vocab.

    `encode_word` maps one word to its BPE ids and MUST be equality-verified
    against BPETokenizer.encode before use (see build_fast_encoder)."""
    freqs: Counter = Counter(word_tokenize(text))
    enc = {w: encode_word(w) for w in freqs}
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
    v2_stats = json.loads(V2_STATS.read_text(encoding="utf-8"))
    v2_train_ids = set(v2_stats["split"]["train_ids"])
    v2_valid_ids = set(v2_stats["split"]["valid_ids"])
    v1_rec = json.loads(
        (ROOT / "data" / "sources" / "agri_sources.json").read_text(encoding="utf-8"))
    v1_ids = {r["id"] for r in v1_rec}
    reserved_v1_valid = {"gutenberg:12140", "wiki:Harvest", "wiki:Monsoon",
                         "wiki:Organic farming"}
    all_prev_ids = v1_ids | v2_train_ids | v2_valid_ids

    # ---- load + clean new v3 docs ------------------------------------------
    docs, drops, duplicates = [], [], []
    for meta_path in sorted(V3_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["id"] in QUALITY_DROPS:
            drops.append({"id": meta["id"],
                          "reason": f"quality: {QUALITY_DROPS[meta['id']]};"
                                    " file kept with provenance"})
            continue
        if meta["id"] in all_prev_ids:
            duplicates.append({"id": meta["id"],
                               "requested": meta.get("requested_title", meta["title"]),
                               "reason": "redirect collides with an existing "
                                         "v1/v2 document; existing text stays "
                                         "canonical"})
            continue
        raw = (V3_DIR / "raw" / meta["raw_file"]).read_text(encoding="utf-8")
        if len(raw) < 2000:
            drops.append({"id": meta["id"], "reason": "too short after fetch"})
            continue
        docs.append({**meta,
                     "text": clean_text(raw, cut_sections=meta["provider"] == "Wikipedia")})

    raw_total = corpus_stats(docs)

    # ---- exact-line dedupe against ALL previously trained text --------------
    seen: set[str] = set()
    for p in (V2_TRAIN_TXT, V1_VALID_TXT, V2_VALID_TXT):
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
                seen.add(line.lower())
                continue
            kept.append(line)
        cleaned.append({**doc, "text": "\n".join(kept)})
    docs = cleaned
    ded_total = corpus_stats(docs)

    # ---- split: train-only expansion; validation is FIXED = v2 valid --------
    train_docs = docs                                   # all new docs -> train
    train_new_text = "\n\n".join(d["text"] for d in train_docs)

    v3_train_text = (V2_TRAIN_TXT.read_text(encoding="utf-8") + "\n\n"
                     + train_new_text)
    TRAIN_FILE = V3_DIR / "agri_train_v3.txt"
    VAL_FILE = V3_DIR / "agri_valid_v3.txt"
    TRAIN_FILE.write_text(v3_train_text, encoding="utf-8")
    shutil.copy2(V2_VALID_TXT, VAL_FILE)                # byte-identical eval set

    # ---- leakage assertions ---------------------------------------------------
    new_ids = {d["id"] for d in train_docs}
    assert new_ids.isdisjoint(all_prev_ids), "new doc id collides with history"
    assert new_ids.isdisjoint(v2_valid_ids)
    assert new_ids.isdisjoint(reserved_v1_valid)
    assert VAL_FILE.read_bytes() == V2_VALID_TXT.read_bytes(), "valid not identical"
    leaked_lines = sum(1 for line in train_new_text.splitlines()
                       if len(line) >= 25 and line.lower() in {
                           l.lower() for l in
                           V2_VALID_TXT.read_text(encoding="utf-8").splitlines()
                           if len(l) >= 25})
    assert leaked_lines == 0, leaked_lines

    # ---- tokenizer quality on the EXISTING 5,237-vocab BPE -------------------
    # token_stats over 7.1 MB of text through the raw educational encoder is
    # hours slow, so BPE is applied via the verified fast rank-merge encoder
    # (strict-equality checked against tok.encode before use; tokenizer module
    # NOT modified).
    tok = BPETokenizer.load(TOK_PATH)
    encode_word = build_fast_encoder(tok, v3_train_text)
    new_tokens = token_stats(encode_word, train_new_text)
    print("  new-train token stats done", flush=True)
    agg_tokens = token_stats(encode_word, v3_train_text)
    print("  combined-train token stats done", flush=True)
    valid_tokens = token_stats(encode_word, VAL_FILE.read_text(encoding="utf-8"))
    print("  valid token stats done", flush=True)
    v2_ref = v2_stats["tokens_existing_5237_vocab"]["train"]

    # ---- topic coverage --------------------------------------------------------
    new_topics = keyword_hits(train_new_text)
    agg_topics = keyword_hits(v3_train_text)

    provider_counts = Counter(d["provider"] for d in train_docs)
    license_counts = Counter(d["license"] for d in train_docs)

    report = {
        "name": "krishigpt_corpus_v3",
        "created_in": "evaluation/corpus_v3/build_corpus_v3.py",
        "policy": {
            "train": "v2 train docs (byte-reused) + v3 new docs",
            "valid": "exact byte copy of v2 valid (6 docs, never trained in any "
                     "run) so token ids/windows and val loss are directly "
                     "comparable with every checkpoint so far",
            "drops": drops,
            "corpus_duplicates_skipped": duplicates,
            "dedupe_rule": "exact-line dedupe of new docs vs all previously "
                           "trained text (lines >=25 chars)",
            "removed_duplicate_lines": removed_lines,
        },
        "total_docs_in_corpus": len(v2_train_ids) + len(train_docs),
        "v2_train_docs_reused": len(v2_train_ids),
        "new_docs": {
            "docs": len(train_docs),
            "provider_breakdown": dict(provider_counts),
            "license_breakdown": dict(license_counts),
            "ids": sorted(new_ids),
        },
        "validation_docs": {"docs": len(v2_valid_ids), "ids": sorted(v2_valid_ids)},
        "raw_after_clean_new": raw_total,
        "after_dedup_new": ded_total,
        "chars_after_dedup": {
            "v2_train_part": V2_TRAIN_TXT.read_text(encoding="utf-8").__len__(),
            "new_part": ded_total["chars"],
        },
        "split": {
            "train": {"docs": len(v2_train_ids) + len(train_docs),
                      "chars": len(v3_train_text),
                      "words": len(v3_train_text.split())},
            "valid": {"docs": len(v2_valid_ids),
                      "chars": len(V2_VALID_TXT.read_text(encoding="utf-8")),
                      "words": len(V2_VALID_TXT.read_text(encoding="utf-8").split())},
        },
        "tokens_existing_5237_vocab": {
            "new_train_part": new_tokens,
            "combined_train": agg_tokens,
            "valid": valid_tokens,
            "reference_v2_train": v2_ref,
        },
        "topic_hits_per_10k_words": {"new_train": new_topics,
                                     "combined_train": agg_topics},
        "artifacts": {
            "train_text": str(TRAIN_FILE.relative_to(ROOT)),
            "valid_text": str(VAL_FILE.relative_to(ROOT)),
            "report": str((V3_DIR / "corpus_v3_stats.json").relative_to(ROOT)),
        },
    }
    (V3_DIR / "corpus_v3_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("CORPUS V3 REPORT")
    print("=" * 72)
    print(f"new docs fetched: {len(train_docs) + len(drops) + len(duplicates)} "
          f"-> kept {len(train_docs)} "
          f"({provider_counts.get('Wikipedia', 0)} wiki, "
          f"{provider_counts.get('Project Gutenberg', 0)} gutenberg)")
    print(f"quality drops: {[d['id'] for d in drops]}")
    print(f"corpus duplicates skipped: {[d['id'] for d in duplicates]}")
    print(f"cross-corpus duplicate lines removed from new docs: {removed_lines}")
    tr = report["split"]["train"]
    va = report["split"]["valid"]
    print(f"\nTRAIN: {tr['docs']:>4} docs {tr['chars']:>12,} chars "
          f"{tr['words']:>10,} words   (83 v2 + {len(train_docs)} new)")
    print(f"VALID: {va['docs']:>4} docs {va['chars']:>12,} chars "
          f"{va['words']:>10,} words   (byte-identical to v2 valid)")
    print(f"\n-- tokenization with EXISTING vocab {tok.vocab_size} (NO retrain) --")
    for name, s in [("v3 new train part", new_tokens),
                    ("v3 combined train", agg_tokens),
                    ("v3 valid (=v2 valid)", valid_tokens)]:
        print(f"  {name:<20} tokens={s['total_tokens']:>10,}  "
              f"tok/kchar={s['tokens_per_1000_chars']:>6}  "
              f"UNK_words={s['unk_word_pct']:>5}%  "
              f"single-tok={s['single_token_word_pct']:>5}%  "
              f"avg_tok/word={s['avg_tokens_per_word']}")
    print(f"  {'v2 train (ref)':<20} tokens={v2_ref['total_tokens']:>10,}  "
          f"tok/kchar={v2_ref['tokens_per_1000_chars']:>6}  "
          f"UNK_words={v2_ref['unk_word_pct']:>5}%  "
          f"single-tok={v2_ref['single_token_word_pct']:>5}%  "
          f"avg_tok/word={v2_ref['avg_tokens_per_word']}")
    print("\n-- topic hits / 10k words (new | combined) --")
    for t in new_topics:
        print(f"  {t:<24} {new_topics[t]:>7} | {agg_topics[t]:>7}")
    print(f"\nartifacts: {TRAIN_FILE.name}, {VAL_FILE.name}, "
          f"corpus_v3_stats.json")


if __name__ == "__main__":
    main()
