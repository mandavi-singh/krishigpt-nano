"""Corpus v2 source acquisition (replaces probe_sources.py after its wiki
batch call was silently truncated).

Reuses the VERIFIED helpers from v1's builder: single-title Wikipedia extract
fetch (CC BY-SA 4.0, attribution per article) and Gutenberg gutendex lookup
(public domain flag enforced). All new raw texts go to data/corpus_v2/raw/,
all provenance to data/corpus_v2/sources_v2.json. Nothing in data/processed
or data/raw is touched.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.build_agriculture_corpus import (  # noqa: E402
    WIKI_CUT_SECTIONS,
    fetch_wikipedia_article,
    gutenberg_body,
    http_get,
)

V2_DIR = ROOT / "data" / "corpus_v2"
RAW_DIR = V2_DIR / "raw"
SOURCES_OUT = V2_DIR / "sources_v2.json"

#: v1 train articles (re-used verbatim, all CC BY-SA; the 4 v1 validation docs
#: stay OUT by construction - they are simply not listed here).
V1_TRAIN_TITLES = [
    "Agriculture", "Agriculture in India", "Crop", "Wheat", "Rice", "Maize",
    "Barley", "Sorghum", "Sugarcane", "Cotton", "Legume", "Vegetable", "Soil",
    "Soil fertility", "Soil salinity", "Fertilizer", "Manure", "Compost",
    "Sustainable agriculture", "Irrigation", "Drip irrigation",
    "Crop rotation", "Weed", "Pest (organism)", "Plant pathology",
    "Pesticide", "Nitrogen fixation", "Pollination", "Photosynthesis",
]

#: NEW articles filling audit gaps (machinery, physiology, climate, livestock,
#: Indian agriculture, nutrient depth, farming systems).
NEW_TITLES = [
    "Plough", "Harrow (tool)", "Tractor", "Combine harvester", "Seed drill",
    "Tillage", "No-till farming",
    "Plant physiology", "Germination", "Plant breeding", "Green Revolution",
    "Soil science", "Soil pH", "Loam", "Topsoil", "Soil erosion",
    "Microirrigation", "Flood irrigation",
    "Integrated pest management", "Locust", "Wheat rust", "Fungicide",
    "Herbicide", "Plant disease complex",
    "Climate change and agriculture", "Climate-smart agriculture",
    "Rainfed agriculture", "Drought mitigation",
    "Dairy farming", "Poultry farming", "Fodder", "Pasture",
    "Animal husbandry", "Silvopasture",
    "Soybean", "Groundnut", "Chickpea", "Mustard plant", "Potato", "Tomato",
    "Green Revolution in India", "Rabi crop", "Kharif crop",
    "Agriculture in Punjab", "Agriculture in Tamil Nadu",
    "Rainwater harvesting", "Indian Council of Agricultural Research",
    "Phosphorus cycle", "NPK fertilizer", "Urea fertilizer",
    "Soil conditioner", "Cover crop",
    "Smallholder", "Subsistence agriculture", "Intensive farming",
    "Agroforestry", "Mixed farming", "Cash crop", "Organic fertilizer",
    "Water scarcity",
]

#: Gutenberg title-regex buckets. Books are matched against the public-domain
#: agriculture shelf (gutendex topic=agriculture, copyright=false enforced).
GUT_BUCKETS = {
    "indian agriculture": r"India|Indian|Punjab|Bengal|Madras|Ceylon|Bombay",
    "irrigation": r"irrigat|canal|water",
    "fertilizers/nutrients": r"fertil|manure|nitrogen|plant food",
    "pests/diseases": r"insect|pest|disease|fungus|blight|rust",
    "machinery": r"machinery|implement|tractor|plow|plough|engine",
    "livestock/dairy": r"dairy|cattle|poultry|swine|horse|sheep|stock",
    "soil science": r"soil|drainage|land",
    "crops/agronomy": r"farm|crop|corn|wheat|husband|orchard|garden|food",
    "climate/general": r"climate|season|weather|country life|rural",
}

#: v1's five Gutenberg books (already in the old corpus; never re-add), PLUS
#: the v1 validation article held out from all training (leakage guard).
OLD_GUTENBERG = {"gutenberg:12140", "gutenberg:16594", "gutenberg:20772",
                 "gutenberg:22973", "gutenberg:51764"}
V1_VALIDATION_IDS = {"gutenberg:12140", "wiki:Harvest", "wiki:Monsoon",
                     "wiki:Organic farming"}

#: Gutenberg titles that match the keyword filter but are fiction/misleading.
GUT_EXCLUDE_TITLES = re.compile(r"food of the gods|gulliver|robinson|alice",
                               re.I)

MAX_GUT_PER_BUCKET = 2          # quality over quantity; scan-noise heavy books
MAX_GUT_TOTAL = 16
MAX_GUT_BODY_CHARS = 600_000    # huge scanned books are mostly noise/ad pages


def raw_name(doc_id: str) -> str:
    return doc_id.replace(":", "__").replace(" ", "_").replace("/", "_") + ".txt"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs: list[dict] = []
    skipped: list[str] = []

    # ---- wikipedia (per-title, proven v1 fetcher) --------------------------
    print("== Wikipedia (CC BY-SA 4.0) ==")
    all_titles = sorted(set(V1_TRAIN_TITLES + NEW_TITLES))
    for title in all_titles:
        doc_id = f"wiki:{title}"
        cache = RAW_DIR / raw_name(doc_id)
        if cache.exists():
            meta_cache = V2_DIR / raw_name(doc_id).replace(".txt", ".meta.json")
            if meta_cache.exists():
                docs.append(json.loads(meta_cache.read_text(encoding="utf-8")))
                print(f"  CACHED {title:<36} {cache.stat().st_size:>8,}")
                continue
        doc = fetch_wikipedia_article(title)
        if doc is None or len(doc["text"]) < 2500:
            skipped.append(doc_id)
            print(f"  SKIP   {title:<36} (missing/short)")
            time.sleep(0.4)
            continue
        cache.write_text(doc["text"], encoding="utf-8")
        meta = {k: v for k, v in doc.items() if k != "text"}
        meta["raw_file"] = raw_name(doc_id)
        (V2_DIR / raw_name(doc_id).replace(".txt", ".meta.json")).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        docs.append(meta)
        print(f"  ok     {title:<36} {len(doc['text']):>8,}")
        time.sleep(0.4)                       # API politeness

    # ---- gutenberg: public-domain agricultural shelf ------------------------
    print("\n== Project Gutenberg (public domain, topic=agriculture) ==")
    books: dict[int, dict] = {}
    for page in (1, 2, 3, 4):
        payload = json.loads(http_get(
            "https://gutendex.com/books?topic=agriculture&languages=en"
            f"&page={page}", timeout=60))
        results = payload.get("results", [])
        if not results:
            break
        for b in results:
            if not b.get("copyright", True):
                books[b["id"]] = b
        time.sleep(0.6)                       # politeness
    print(f"  pd shelf: {len(books)} books")

    picked: dict[str, str] = {}               # doc_id -> bucket
    per_bucket: dict[str, int] = {}
    # resumability: previously cached gutenberg books keep their bucket slots
    for meta_path in sorted(V2_DIR.glob("gutenberg__*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        doc_id, bucket = meta["id"], meta.get("topic_bucket", "")
        if doc_id in OLD_GUTENBERG or not bucket:
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
        if doc_id in OLD_GUTENBERG or doc_id in picked:
            continue
        title = b.get("title", "")
        if GUT_EXCLUDE_TITLES.search(title):
            continue                          # fiction slipped into the shelf
        for bucket, rx in GUT_BUCKETS.items():
            if re.search(rx, title, re.I):
                if per_bucket.get(bucket, 0) >= MAX_GUT_PER_BUCKET:
                    continue
                picked[doc_id] = bucket
                per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
                break
    print(f"  matched: {len(picked)} books total "
          f"(cap {MAX_GUT_PER_BUCKET}/bucket, {MAX_GUT_TOTAL} total); "
          f"buckets: {sorted(per_bucket.items())}")

    for doc_id, bucket in picked.items():
        bid = int(doc_id.split(":")[1])
        b = books[bid]
        cache = RAW_DIR / raw_name(doc_id)
        if cache.exists():
            meta_cache = V2_DIR / raw_name(doc_id).replace(".txt", ".meta.json")
            if meta_cache.exists():
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
        if len(body) < 20000:                   # skip stubs / mostly-scan junk
            print(f"  small  {b['title'][:58]:<58} ({len(body):,} chars) skip")
            continue
        if len(body) > MAX_GUT_BODY_CHARS:      # huge scans are mostly noise
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
            "use": "KrishiGPT-nano corpus v2 (System A pretraining only)",
        }
        (V2_DIR / raw_name(doc_id).replace(".txt", ".meta.json")).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        docs.append(meta)
        print(f"  ok     {meta['title'][:58]:<58} {len(body):>9,} [{bucket}]")
        time.sleep(0.5)

    SOURCES_OUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_OUT.write_text(json.dumps(
        {"accessed": date.today().isoformat(), "skipped": skipped,
         "docs": docs}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsource records -> {SOURCES_OUT.relative_to(ROOT)}")
    print(f"wiki docs: {sum(1 for d in docs if d['provider'] == 'Wikipedia')}, "
          f"gutenberg docs: "
          f"{sum(1 for d in docs if d['provider'] == 'Project Gutenberg')}, "
          f"skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
