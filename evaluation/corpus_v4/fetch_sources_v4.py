"""Corpus v4 source acquisition — 'how-to' farming-manual expansion.

Reuses the VERIFIED v1/v3 helpers (Wikipedia per-title fetch with CC BY-SA
4.0 attribution; gutendex public-domain shelf with copyright=false enforced).
New raw texts -> data/corpus_v4/raw/, provenance -> data/corpus_v4/sources_v4.json
and per-doc *.meta.json.

Gap targets (per final/FINAL_REPORT.md 'next steps' + funnel L2/L3 needs):
  - human-authored 'how-to' agriculture text (highest-leverage corpus lever)
  - modern clean English, instruction-like phrasing (helps QA tuning)
  - remaining sub-domain breadth: machinery, storage, weeds, seeds, org farming

NOTHING in data/processed, data/raw, data/corpus_v2, data/corpus_v3,
checkpoints/, or colab/ is modified.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.build_agriculture_corpus import (  # noqa: E402
    fetch_wikipedia_article,
    gutenberg_body,
    http_get,
)

V4_DIR = ROOT / "data" / "corpus_v4"
RAW_DIR = V4_DIR / "raw"
SOURCES_OUT = V4_DIR / "sources_v4.json"

NEW_TITLES = [
    # how-to / practice depth (instruction-like modern prose)
    "No-till farming", "Mulching", "Weed control", "Crop protection",
    "Precision seeding", "Seed drill", "Plough", "Harrow (tool)",
    "Manure management", "Composting", "Cover crop", "Crop residue",
    "Land preparation", "Bed (agriculture)", "Ridge till",
    # storage & post-harvest (L3 breadth)
    "Grain storage", "Silo", "Threshing", "Winnowing", "Grain dryer",
    "Postharvest handling", "Food preservation",
    # weeds / seed / propagation (L3 breadth)
    "Weed", "Herbicide", "Seed saving", "Heirloom seed", "Plant propagation",
    "Grafting", "Pruning", "Nursery (agriculture)",
    # organic / sustainability
    "Organic farming", "Biodynamic agriculture", "Agroecology",
    "Integrated pest management", "Biological pest control",
    "Crop diversity", "Seed treatment",
    # water & land depth
    "Drip irrigation", "Irrigation management", "Deficit irrigation",
    "Soil conservation", "Contour plowing", "Windbreak",
    "Terrace farming", "Water use efficiency",
    # mechanization
    "Combine harvester", "Tractor", "Thresher", "Cultivator",
    # economics / extension-style writing
    "Agricultural extension", "Agricultural education", "Agriculture",
]

#: Gutenberg buckets for v4 (pd shelf, copyright=false enforced in code).
#: Deliberately MANUAL/HOW-TO heavy — the corpus gap FINAL_REPORT identified.
GUT_BUCKETS = {
    "how-to manuals": (r"how to|handbook|manual|guide to|for beginners|"
                       r"practical|every ?man|self-?sufficient|home farm", 4),
    "crop manuals": (r"potato|tobacco|onion|asparagus|strawberr|orchard|"
                     r"berry|vegetable|garden|farm crop|field crop", 3),
    "soil & fertilizer": (r"soil|fertiliz|manure|compost|fertilit", 3),
    "pest & disease": (r"pest|insect|spray|fungus|disease|blight|parasite", 3),
    "livestock manuals": (r"cattle|poultry|sheep|swine|horse|dairy|bee", 2),
    "farm management": (r"farm manage|farm econom|farm account|agricultur"
                         r"al ?college|experiment station", 2),
}

# every Gutenberg id already used by v1/v2/v3 — never re-add
USED_GUTENBERG = {
    "gutenberg:12140", "gutenberg:16594", "gutenberg:20772",
    "gutenberg:22973", "gutenberg:51764",
    "gutenberg:16525", "gutenberg:17512", "gutenberg:26975",
    "gutenberg:29665", "gutenberg:56640", "gutenberg:59579",
    "gutenberg:60313",
    "gutenberg:22040", "gutenberg:24080", "gutenberg:38955",
    "gutenberg:57457", "gutenberg:74799",
}
GUT_EXCLUDE_TITLES = re.compile(
    r"food of the gods|gulliver|robinson|alice|cookbook|recipe|periodical",
    re.I)

MAX_GUT_TOTAL = 17
MAX_GUT_BODY_CHARS = 600_000
MIN_GUT_BODY_CHARS = 25_000


def raw_name(doc_id: str) -> str:
    return (doc_id.replace(":", "__").replace(" ", "_")
            .replace("/", "_") + ".txt")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs: list[dict] = []
    skipped: list[str] = []

    # ---- wikipedia ---------------------------------------------------------
    print("== Wikipedia (CC BY-SA 4.0) ==")
    all_titles = sorted(set(NEW_TITLES))
    seen_final: set[str] = set()
    for title in all_titles:
        cache_name = raw_name(f"wiki:{title}")
        cache = RAW_DIR / cache_name
        meta_cache = V4_DIR / cache_name.replace(".txt", ".meta.json")
        if cache.exists() and meta_cache.exists():
            meta = json.loads(meta_cache.read_text(encoding="utf-8"))
            final_id = meta["id"]
            if final_id in seen_final:
                skipped.append(f"wiki:{title} (redirect duplicate)")
                cache.unlink()
                meta_cache.unlink()
                continue
            seen_final.add(final_id)
            docs.append(meta)
            print(f"  CACHED {title:<38} {cache.stat().st_size:>8,}")
            continue
        doc = fetch_wikipedia_article(title)
        if doc is None or len(doc["text"]) < 2500:
            skipped.append(f"wiki:{title}")
            print(f"  SKIP   {title:<38} (missing/short)")
            time.sleep(0.4)
            continue
        final_id = doc["id"]
        if final_id in seen_final:
            skipped.append(f"wiki:{title} (redirect duplicate of {final_id})")
            print(f"  DUP    {title:<38} -> {final_id}")
            time.sleep(0.4)
            continue
        seen_final.add(final_id)
        cache.write_text(doc["text"], encoding="utf-8")
        meta = {k: v for k, v in doc.items() if k != "text"}
        meta["raw_file"] = cache_name
        if final_id != f"wiki:{title}":
            meta["requested_title"] = title
        meta_cache.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        docs.append(meta)
        print(f"  ok     {title:<38} {len(doc['text']):>8,}")
        time.sleep(0.4)

    # ---- gutenberg manual shelf --------------------------------------------
    print("\n== Project Gutenberg (public domain, how-to manuals) ==")
    books: dict[int, dict] = {}
    for page in (1, 2, 3, 4, 5, 6):
        try:
            payload = json.loads(http_get(
                "https://gutendex.com/books?topic=agriculture&languages=en"
                f"&page={page}", timeout=60))
        except Exception as e:                                     # noqa: BLE001
            print(f"  stop at page {page}: {type(e).__name__}")
            break
        for b in payload.get("results", []):
            if not b.get("copyright", True):
                books[b["id"]] = b
        time.sleep(0.6)
    print(f"  pd shelf: {len(books)} books")

    picked: dict[str, str] = {}
    per_bucket: dict[str, int] = {}
    # resumability: previously cached v4 gutenberg books keep their slots
    for meta_path in sorted(V4_DIR.glob("gutenberg__*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        doc_id, bucket = meta["id"], meta.get("topic_bucket", "")
        if doc_id in USED_GUTENBERG or not bucket:
            continue
        if GUT_EXCLUDE_TITLES.search(meta.get("title", "")):
            continue
        picked[doc_id] = bucket
        per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
    for bid in sorted(books, key=lambda i: -books[i].get("download_count", 0)):
        if len(picked) >= MAX_GUT_TOTAL:
            break
        b = books[bid]
        doc_id = f"gutenberg:{bid}"
        if doc_id in USED_GUTENBERG or doc_id in picked:
            continue
        title = b.get("title", "")
        if GUT_EXCLUDE_TITLES.search(title):
            continue
        for bucket, (rx, cap) in GUT_BUCKETS.items():
            if re.search(rx, title, re.I):
                if per_bucket.get(bucket, 0) >= cap:
                    continue
                picked[doc_id] = bucket
                per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
                break
    print(f"  matched: {len(picked)} books; buckets: {sorted(per_bucket.items())}")

    for doc_id, bucket in picked.items():
        bid = int(doc_id.split(":")[1])
        b = books[bid]
        cache = RAW_DIR / raw_name(doc_id)
        meta_cache = V4_DIR / raw_name(doc_id).replace(".txt", ".meta.json")
        if cache.exists() and meta_cache.exists():
            docs.append(json.loads(meta_cache.read_text(encoding="utf-8")))
            print(f"  CACHED {b['title'][:58]:<58}")
            continue
        plain = next((u for k, u in b.get("formats", {}).items()
                      if k.startswith("text/plain")), None)
        if not plain:
            continue
        try:
            raw = http_get(plain, timeout=120).decode("utf-8", errors="replace")
        except Exception as e:                                     # noqa: BLE001
            print(f"  FAIL   {doc_id} {type(e).__name__}")
            continue
        body = gutenberg_body(raw)
        if len(body) < MIN_GUT_BODY_CHARS:
            print(f"  small  {b['title'][:58]:<58} ({len(body):,} chars) skip")
            continue
        if len(body) > MAX_GUT_BODY_CHARS:
            print(f"  huge   {b['title'][:58]:<58} ({len(body):,} chars) skip")
            continue
        authors = ", ".join(a.get("name", "?") for a in b.get("authors", []))
        cache.write_text(body, encoding="utf-8")
        meta = {
            "id": doc_id,
            "provider": "Project Gutenberg",
            "title": b["title"],
            "authors": authors,
            "url": f"https://www.gutenberg.org/ebooks/{bid}",
            "license": "Public domain in the USA (Project Gutenberg)",
            "license_url": "https://www.gutenberg.org/policy/license.html",
            "accessed": date.today().isoformat(),
            "topic_bucket": bucket,
            "download_count": b.get("download_count"),
            "raw_file": raw_name(doc_id),
            "use": "KrishiGPT-nano corpus v4 (System A pretraining only)",
        }
        meta_cache.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        docs.append(meta)
        print(f"  ok     {meta['title'][:58]:<58} {len(body):>9,} [{bucket}]")
        time.sleep(0.5)

    SOURCES_OUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_OUT.write_text(json.dumps(
        {"accessed": date.today().isoformat(), "skipped": skipped,
         "docs": docs}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsources -> {SOURCES_OUT.relative_to(ROOT)}")
    print(f"wiki: {sum(1 for d in docs if d['provider'] == 'Wikipedia')}, "
          f"gutenberg: "
          f"{sum(1 for d in docs if d['provider'] == 'Project Gutenberg')}, "
          f"skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
