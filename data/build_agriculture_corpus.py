"""KrishiGPT agriculture PRETRAINING corpus builder (System A data only).

PIPELINE (every step recorded; nothing fabricated):
  1. SOURCES    Wikipedia agriculture articles   (CC BY-SA 4.0, attribution
                recorded per article) + Project Gutenberg agriculture books
                (public domain in the USA, per gutendex metadata).
  2. DOWNLOAD   polite sequential HTTPS GETs; per-document metadata written to
                data/sources/agri_sources.json BEFORE the text is used.
  3. CLEAN      Gutenberg banners stripped; Wikipedia citation sections cut;
                whitespace normalized; junk lines dropped.
  4. DEDUPE     cross-doc exact-line deduplication for long (>=25 char) lines.
  5. SPLIT      document-level 90/10 train/valid, fixed seed, deterministic.
  6. TOKENIZE   our from-scratch BPE is TRAINED on the train split only and
                saved for reuse (this is DATA prep, not model training).

This corpus is deliberately separate from model code and is used for
pretraining experiments only. RAG documents (System B) will live elsewhere
with their own source records. ICAR/USDA extension material is NOT included
here because per-document reuse rights cannot be verified automatically.

Run:  python data/build_agriculture_corpus.py   [--merges N] [--no-download]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw" / "agri"
PROCESSED_DIR = DATA_DIR / "processed"
SOURCES_DIR = DATA_DIR / "sources"

sys.path.insert(0, str(DATA_DIR.parent))

from tokenizer import BPETokenizer  # noqa: E402

USER_AGENT = ("myllm-krishigpt-corpus/0.1 (educational LLM project; "
              "run by the repository owner)")

#: Curated, verifiable Wikipedia article list (agriculture domain).
WIKI_TITLES = [
    "Agriculture", "Agriculture in India", "Crop", "Wheat", "Rice", "Maize",
    "Barley", "Sorghum", "Sugarcane", "Cotton", "Legume", "Vegetable", "Soil",
    "Soil fertility", "Soil salinity", "Fertilizer", "Manure", "Compost",
    "Organic farming", "Sustainable agriculture", "Irrigation",
    "Drip irrigation", "Crop rotation", "Monsoon", "Drought", "Weed",
    "Pest (organism)", "Plant pathology", "Pesticide", "Nitrogen fixation",
    "Pollination", "Photosynthesis", "Harvest",
]

#: Section headers that open citation/nav noise in Wikipedia plain extracts.
WIKI_CUT_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
    "sources", "bibliography", "footnotes", "citations",
}

#: Gutenberg new/old body delimiters.
GUT_START_RE = re.compile(r"^\*\*\*\s*START OF .*?EBOOK.*\*\*\*", re.I)
GUT_END_RE = re.compile(r"^\*\*\*\s*END OF .*?EBOOK.*\*\*\*", re.I)

#: Keyword filter for choosing relevant Gutenberg books deterministically.
GUT_TITLE_RE = re.compile(
    r"farm|agricultur|soil|crop|corn|wheat|husband|garden|plant|cotton|"
    r"irrigat|dairy|orchard|country life", re.I)


# --------------------------------------------------------------------- http
def http_get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    """GET with a descriptive User-Agent (Wikimedia policy), retries with
    exponential backoff — the Wikipedia API returns 429 to fast clients."""
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                last_err = e
            else:
                raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last_err})")


# ------------------------------------------------------------- wikipedia
def fetch_wikipedia_article(title: str) -> dict | None:
    """Fetch CLEAN plain text of one article via the MediaWiki extracts API.

    Returns a doc record, or None if the article is missing/errored.
    """
    q = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "format": "json", "redirects": 1, "titles": title,
    })
    try:
        payload = json.loads(http_get(f"https://en.wikipedia.org/w/api.php?{q}"))
        pages = payload.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        if "missing" in page or "extract" not in page:
            return None
        text = page["extract"]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, RuntimeError):
        return None
    return {
        "id": f"wiki:{page.get('title', title)}",
        "provider": "Wikipedia",
        "title": page.get("title", title),
        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page.get('title', title))}",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "accessed": date.today().isoformat(),
        "text": text,
        "use": "System A pretraining (educational corpus, NOT RAG source)",
    }


# ------------------------------------------------------------- gutenberg
def fetch_gutenberg_books(max_books: int, existing: dict[str, dict] | None = None,
                          ) -> list[dict]:
    """Fetch public-domain agriculture books from Project Gutenberg.

    Uses the gutendex JSON API for metadata (title, authors, license info,
    download URLs). Only books explicitly marked no-copyright (public domain
    in the USA) with a plain-text format are accepted. Falls back to an empty
    list if the service is unreachable — the corpus still builds from wiki.
    `existing` caches previously saved records for incremental rebuilds.
    """
    existing = existing or {}
    try:
        payload = json.loads(http_get(
            "https://gutendex.com/books?topic=agriculture&languages=en"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, RuntimeError):
        return []
    books: list[dict] = []
    for b in payload.get("results", []):
        if b.get("copyright", True):
            continue                                    # keep public domain only
        if not GUT_TITLE_RE.search(b.get("title", "")):
            continue
        plain = next((u for k, u in b.get("formats", {}).items()
                      if k.startswith("text/plain")), None)
        if not plain:
            continue
        books.append((b, plain))
    # deterministic: most-downloaded relevant books first
    books.sort(key=lambda t: -t[0].get("download_count", 0))
    docs: list[dict] = []
    for b, plain_url in books[:max_books]:
        doc_id = f"gutenberg:{b['id']}"
        cached_raw = RAW_DIR / _raw_name(doc_id)
        if doc_id in existing and cached_raw.exists():
            rec = dict(existing[doc_id])
            rec["text"] = cached_raw.read_text(encoding="utf-8")
            docs.append(rec)
            print(f"  CACHED {rec['title'][:58]:<58} {len(rec['text']):>9,} chars")
            continue
        try:
            raw = http_get(plain_url, timeout=120).decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                UnicodeDecodeError, RuntimeError):
            continue
        body = gutenberg_body(raw)
        if len(body) < 5000:                            # skip stubs/mostly-scan
            continue
        authors = ", ".join(a.get("name", "?") for a in b.get("authors", []))
        docs.append({
            "id": doc_id,
            "provider": "Project Gutenberg",
            "title": b["title"],
            "authors": authors,
            "url": f"https://www.gutenberg.org/ebooks/{b['id']}",
            "license": "Public domain in the USA (Project Gutenberg)",
            "license_url": "https://www.gutenberg.org/policy/license.html",
            "accessed": date.today().isoformat(),
            "text": body,
            "use": "System A pretraining (educational corpus, NOT RAG source)",
        })
        time.sleep(0.5)                                 # politeness between downloads
    return docs


def gutenberg_body(raw: str) -> str:
    """Strip the license banner/footer between the *** START/END markers."""
    lines = raw.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and GUT_START_RE.match(line.strip()):
            start = i + 1
        if GUT_END_RE.match(line.strip()):
            end = i                                      # last END wins
    if start is None or end is None or end <= start:
        return raw                                       # unexpected layout: keep as-is
    return "\n".join(lines[start:end])


# --------------------------------------------------------------- cleaning
def clean_text(text: str, cut_sections: bool = False) -> str:
    """Normalize whitespace, drop junk lines, optionally cut wiki sections."""
    out: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split())                   # collapse all whitespace
        if not line:
            continue
        if cut_sections and line.lower() in WIKI_CUT_SECTIONS:
            break                                        # citations from here on
        if len(line) < 3:
            continue
        if line.startswith(("ISBN ", "ISSN ", "OCLC ")):
            continue
        out.append(line)
    return "\n".join(out)


def dedupe_lines(docs: list[dict]) -> list[dict]:
    """Cross-document exact-line deduplication.

    Lines >= 25 chars that occur in MORE THAN ONE document are boilerplate
    templates (disambiguation notes, repeated legal/sponsored lines); every
    occurrence after the first one kept is dropped. Short lines are untouched
    (dropping them would gut normal prose like 'Soil matters.').
    """
    seen: set[str] = set()
    cleaned: list[dict] = []
    for doc in docs:
        kept: list[str] = []
        for line in doc["text"].splitlines():
            if len(line) >= 25:
                key = line.lower()
                if key in seen:
                    continue                             # repeated boilerplate
                seen.add(key)
            kept.append(line)
        cleaned.append({**doc, "text": "\n".join(kept)})
    return cleaned


# --------------------------------------------------------------- splitting
def split_documents(docs: list[dict], val_fraction: float = 0.1,
                    seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Document-level split (no leakage between splits).

    n_val = clamp(round(val_fraction * n), 1, n-1): BOTH sides are guaranteed
    non-empty by construction (a character-fill policy can silently empty one
    side on small corpora). Deterministic: docs are sorted by id before the
    seeded shuffle, so the split is exactly reproducible. Because documents
    have unequal sizes, the achieved character fraction approximates
    val_fraction rather than matching it exactly.
    """
    if len(docs) < 2:
        raise ValueError("need at least 2 documents for a train/valid split")
    order = sorted(docs, key=lambda d: d["id"])
    rng = random.Random(seed)
    rng.shuffle(order)
    n_val = max(1, min(len(order) - 1, round(val_fraction * len(order))))
    val_docs = order[:n_val]
    train_docs = order[n_val:]
    return train_docs, val_docs


# ------------------------------------------------------------------ stats
def corpus_stats(docs: list[dict]) -> dict:
    chars = sum(len(d["text"]) for d in docs)
    words = sum(len(d["text"].split()) for d in docs)
    return {"docs": len(docs), "chars": chars, "words": words}


def token_stats(tok: BPETokenizer, text: str) -> dict:
    """Count BPE tokens by encoding each unique WORD type once, weighted by
    frequency (same trick used in Phase 2 — the BPE output of a word depends
    only on its spelling, not on context)."""
    from collections import Counter
    from tokenizer.word_tokenizer import word_tokenize
    freqs: Counter = Counter(word_tokenize(text))
    cache = {w: len(tok.encode(w)) for w in freqs}
    total_tokens = sum(freqs[w] * cache[w] for w in freqs)
    return {"vocab_size": tok.vocab_size,
            "word_types": len(freqs),
            "total_tokens": total_tokens,
            "tokens_per_1000_chars": round(total_tokens / max(len(text), 1) * 1000, 1)}


# ------------------------------------------------------------------- main
def _raw_name(doc_id: str) -> str:
    return doc_id.replace(":", "__").replace(" ", "_").replace("/", "_") + ".txt"


def _load_existing_records() -> dict[str, dict]:
    """Previously saved source records, keyed by doc id (incremental builds)."""
    rec_path = SOURCES_DIR / "agri_sources.json"
    if not rec_path.exists():
        return {}
    try:
        return {r["id"]: r for r in json.loads(rec_path.read_text(encoding="utf-8"))}
    except (json.JSONDecodeError, KeyError):
        return {}


def build(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []
    skipped: list[str] = []
    existing = _load_existing_records()                    # incremental cache

    # ---- 1+2: download with metadata -----------------------------------------
    if not args.no_download:
        print("== Wikipedia (CC BY-SA 4.0) ==")
        for title in WIKI_TITLES:
            doc_id = f"wiki:{title}"
            raw_path = RAW_DIR / _raw_name(doc_id)
            if doc_id in existing and raw_path.exists():
                rec = dict(existing[doc_id])
                rec["text"] = raw_path.read_text(encoding="utf-8")
                docs.append(rec)
                print(f"  CACHED {rec['title']:<26} {len(rec['text']):>8,} chars")
                continue
            doc = fetch_wikipedia_article(title)
            if doc is None:
                skipped.append(doc_id)
                print(f"  SKIP {title} (missing or error)")
            else:
                docs.append(doc)
                print(f"  ok   {doc['title']:<28} {len(doc['text']):>8,} chars")
            time.sleep(1.0)                              # API politeness (avoid 429)

        print("\n== Project Gutenberg (public domain) ==")
        gdocs = fetch_gutenberg_books(max_books=args.max_books, existing=existing)
        if not gdocs:
            skipped.append("gutenberg: (service unreachable or no match)")
            print("  nothing fetched (corpus continues with Wikipedia only)")
        for doc in gdocs:
            docs.append(doc)
            print(f"  ok   {doc['title'][:60]:<60} {len(doc['text']):>9,} chars")
        # persist raw copies + source records
        for doc in docs:
            (RAW_DIR / _raw_name(doc["id"])).write_text(doc["text"], encoding="utf-8")
        records = [{k: v for k, v in d.items() if k != "text"} for d in docs]
        (SOURCES_DIR / "agri_sources.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsource records written -> data/sources/agri_sources.json "
              f"({len(records)} docs, {len(skipped)} skipped)")
    else:
        # offline rebuild: read back previously saved raw files + records
        if not existing:
            raise SystemExit("--no-download needs a previous run's sources file")
        for doc_id, rec in sorted(existing.items()):
            text = (RAW_DIR / _raw_name(doc_id)).read_text(encoding="utf-8")
            docs.append({**rec, "text": text})
        print(f"offline rebuild: {len(docs)} documents loaded from disk")

    if not docs:
        raise SystemExit("no documents fetched - check network access")

    # ---- 3: clean ------------------------------------------------------------
    for doc in docs:
        doc["text"] = clean_text(doc["text"],
                                 cut_sections=doc["provider"] == "Wikipedia")
    raw_total = corpus_stats(docs)

    # ---- 4: dedupe ------------------------------------------------------------
    docs = dedupe_lines(docs)
    deduped_total = corpus_stats(docs)

    # ---- 5: split --------------------------------------------------------------
    train_docs, val_docs = split_documents(docs, val_fraction=args.val_fraction,
                                           seed=args.seed)
    tr = corpus_stats(train_docs)
    va = corpus_stats(val_docs)
    TRAIN_FILE = PROCESSED_DIR / "agri_train.txt"
    VAL_FILE = PROCESSED_DIR / "agri_valid.txt"
    TRAIN_FILE.write_text("\n\n".join(d["text"] for d in train_docs), encoding="utf-8")
    VAL_FILE.write_text("\n\n".join(d["text"] for d in val_docs), encoding="utf-8")

    # ---- 6: BPE on the TRAIN split only ----------------------------------------
    train_text = TRAIN_FILE.read_text(encoding="utf-8")
    print(f"\n== Training from-scratch BPE ({args.merges} merges) on train split ==")
    t0 = time.time()
    tok = BPETokenizer.train(train_text, num_merges=args.merges)
    tok.save(PROCESSED_DIR / "agri_bpe_tokenizer.json")
    tok_time = time.time() - t0

    stats = {
        "created": date.today().isoformat(),
        "sources": {"wikipedia_docs": sum(d["provider"] == "Wikipedia" for d in docs),
                    "gutenberg_docs": sum(d["provider"] == "Project Gutenberg" for d in docs),
                    "skipped": skipped},
        "raw_after_clean": raw_total,
        "after_dedupe": deduped_total,
        "split": {"val_fraction": args.val_fraction, "seed": args.seed,
                  "train": tr, "valid": va,
                  "train_chars_fraction": round(tr["chars"] / max(deduped_total["chars"], 1), 4)},
        "tokens": {
            "train": token_stats(tok, train_text),
            "bpe_merges": len(tok.merges),
            "bpe_train_seconds": round(tok_time, 1),
            "saved": "data/processed/agri_bpe_tokenizer.json",
        },
    }
    (PROCESSED_DIR / "agri_corpus_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    # ---- report ------------------------------------------------------------------
    print("\n================ CORPUS REPORT ================")
    print(f"docs: {stats['sources']['wikipedia_docs']} wiki + "
          f"{stats['sources']['gutenberg_docs']} gutenberg"
          f"  (skipped: {len(skipped)})")
    print(f"chars after clean : {raw_total['chars']:>12,}")
    print(f"chars after dedupe: {deduped_total['chars']:>12,} "
          f"(removed {raw_total['chars'] - deduped_total['chars']:,})")
    print(f"TRAIN: {tr['docs']:>3} docs {tr['chars']:>11,} chars {tr['words']:>9,} words")
    print(f"VALID: {va['docs']:>3} docs {va['chars']:>11,} chars {va['words']:>9,} words")
    tk = stats["tokens"]["train"]
    print(f"BPE: vocab {tk['vocab_size']}  tokens {tk['total_tokens']:,}  "
          f"({tk['tokens_per_1000_chars']}/kchar)  trained in {tok_time:.0f}s")
    print("artifacts: data/processed/agri_train.txt, agri_valid.txt,")
    print("           agri_bpe_tokenizer.json, agri_corpus_stats.json")
    print("           data/sources/agri_sources.json (provenance)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build KrishiGPT pretraining corpus")
    ap.add_argument("--merges", type=int, default=5000)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-books", type=int, default=5)
    ap.add_argument("--no-download", action="store_true",
                    help="rebuild from previously saved raw files")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
