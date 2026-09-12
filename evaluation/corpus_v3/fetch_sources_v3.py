"""Corpus v3 source acquisition — gap-fill expansion of the KrishiGPT corpus.

Reuses the VERIFIED helpers from the v1 builder (single-title Wikipedia extract
fetch, CC BY-SA 4.0 attribution per article; gutendex public-domain shelf with
copyright=false enforced). All new raw texts go to data/corpus_v3/raw/, all
provenance to data/corpus_v3/sources_v3.json and per-doc *.meta.json.

NOTHING in data/processed, data/raw, data/corpus_v2, checkpoints/, or colab/
is read-from-or-written here except the read-only helpers imported above.

Gap targets (vs v2 audit + user list): v2 fetch skips, crop diseases, pests,
fertilizer chemistry & nutrient deficiencies, soil science depth, irrigation
methods, farming practices, Indian agriculture (state level + allied crops),
crop-specific knowledge, agriculture terminology, clean modern English.
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

V3_DIR = ROOT / "data" / "corpus_v3"
RAW_DIR = V3_DIR / "raw"
SOURCES_OUT = V3_DIR / "sources_v3.json"

#: Titles v2 tried but skipped (missing/short/rate-limited at the time).
RETRY_V2_SKIPS = [
    "Wheat rust", "Urea fertilizer", "Plant disease complex",
    "Mixed farming", "Groundnut", "Climate change and agriculture",
    "Agriculture in Punjab",
]

NEW_TITLES = [
    # crop diseases (gap: concrete disease biology, modern prose)
    "Late blight", "Rust (fungus)", "Powdery mildew", "Downy mildew",
    "Rice blast", "Corn smut", "Smut (fungus)", "Fusarium wilt",
    "Anthracnose", "Ergot", "Phytophthora infestans",
    # pests (gap: named pest groups, IPM prerequisites)
    "Aphid", "Whitefly", "Thrips", "Mealybug", "Armyworm", "Fall armyworm",
    "Cutworm", "Wireworm", "European corn borer", "Desert locust",
    # fertilizers & nutrients (gap: chemistry + deficiency symptoms)
    "Urea", "Potash", "Ammonium nitrate", "Ammonium sulfate",
    "Superphosphate", "Bone meal", "Blood meal", "Vermicompost",
    "Biofertilizer", "Micronutrient", "Zinc deficiency",
    # soil science depth
    "Soil structure", "Soil compaction", "Soil organic matter",
    "Soil microbiology", "Soil horizon", "Cation exchange capacity",
    "Humus", "Clay minerals", "Soil microorganisms",
    # irrigation methods & water
    "Center pivot irrigation", "Sprinkler irrigation", "Furrow irrigation",
    "Waterlogging (agriculture)", "Groundwater", "Water table",
    "Water conservation", "Watershed management",
    # farming practices (clean modern English, how-to oriented)
    "Intercropping", "Multiple cropping", "Companion planting",
    "Terrace (agriculture)", "Precision agriculture", "Hydroponics",
    "Vertical farming", "Greenhouse", "Conservation agriculture",
    "Regenerative agriculture", "Stubble burning", "Sowing",
    # Indian agriculture (state level + commodity boards)
    "Agriculture in Maharashtra", "Agriculture in Karnataka",
    "Agriculture in Rajasthan", "Agriculture in West Bengal",
    "White Revolution (India)", "Agriculture in Odisha",
    # allied cash & food crops (crop-specific knowledge)
    "Jute", "Tea", "Coffee production in India", "Millet", "Pearl millet",
    "Peanut", "Lentil", "Pea", "Common bean", "Sunflower", "Rapeseed",
    # more crop-specific
    "Oat", "Rye", "Buckwheat", "Cassava", "Sweet potato", "Coconut",
    "Banana", "Mango", "Onion", "Garlic",
    # agriculture terminology
    "Agronomy", "Horticulture", "Cultivar", "Seed bank", "Germplasm",
    "Green manure", "Monoculture", "Crop yield", "Agribusiness",
    "Agricultural machinery", "Food security",
    # climate + post-harvest
    "Flood", "El Niño", "Post-harvest losses", "Cold chain", "Food storage",
    # livestock breadth
    "Goat farming", "Pig farming", "Beekeeping", "Aquaculture",
]

#: Gutenberg buckets for v3 (pd shelf, copyright=false enforced in code).
GUT_BUCKETS = {
    "crop manuals": r"potato|tobacco|onion|asparagus|strawberr|orchard|berry|vegetable garden",
    "irrigation/water": r"irrigat|canal|water suppl|drainage|reservoir",
    "soil science": r"soil|fertility|bacteria of the soil",
    "tropical/plantation": r"tropical|plantation|rubber|cocoa|coffee|sugar|tea",
    "animal husbandry": r"cattle|poultry|bee|sheep|horse|swine|dairy",
    "farm economy": r"farm econom|rural econom|farm manage|marketing of farm",
    "general husbandry": r"husbandry|textbook of agriculture|principles of agriculture",
    "indian agriculture": r"india|punjab|bengal|madras|bombay|ceylon",
}

# every Gutenberg id already used by v1 or v2 — never re-add
USED_GUTENBERG = {
    # v1 (5)
    "gutenberg:12140", "gutenberg:16594", "gutenberg:20772",
    "gutenberg:22973", "gutenberg:51764",
    # v2 (7)
    "gutenberg:16525", "gutenberg:17512", "gutenberg:26975",
    "gutenberg:29665", "gutenberg:56640", "gutenberg:59579",
    "gutenberg:60313",
}

GUT_EXCLUDE_TITLES = re.compile(
    r"food of the gods|gulliver|robinson|alice", re.I)

MAX_GUT_PER_BUCKET = 2
MAX_GUT_TOTAL = 8
MAX_GUT_BODY_CHARS = 600_000


def raw_name(doc_id: str) -> str:
    return (doc_id.replace(":", "__").replace(" ", "_")
            .replace("/", "_") + ".txt")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs: list[dict] = []
    skipped: list[str] = []

    # ---- wikipedia (per-title, proven v1 fetcher) --------------------------
    print("== Wikipedia (CC BY-SA 4.0) ==")
    all_titles = sorted(set(RETRY_V2_SKIPS + NEW_TITLES))
    seen_final: set[str] = set()
    for title in all_titles:
        cache_name = raw_name(f"wiki:{title}")
        cache = RAW_DIR / cache_name
        meta_cache = V3_DIR / cache_name.replace(".txt", ".meta.json")
        if cache.exists() and meta_cache.exists():
            meta = json.loads(meta_cache.read_text(encoding="utf-8"))
            final_id = meta["id"]
            doc_id = final_id
            if final_id in seen_final:
                print(f"  DUP    {title:<36} -> {final_id} (redirect dup) removed")
                skipped.append(f"wiki:{title} (redirect duplicate)")
                cache.unlink()
                meta_cache.unlink()
                continue
            seen_final.add(final_id)
            docs.append(meta)
            print(f"  CACHED {title:<36} {cache.stat().st_size:>8,}"
                  + (f" [{final_id.split(':', 1)[1]}]" if final_id != f"wiki:{title}" else ""))
            continue
        doc = fetch_wikipedia_article(title)
        if doc is None or len(doc["text"]) < 2500:
            skipped.append(f"wiki:{title}")
            print(f"  SKIP   {title:<36} (missing/short)")
            time.sleep(0.4)
            continue
        final_id = doc["id"]
        if final_id in seen_final:              # redirect to an already-fetched article
            skipped.append(f"wiki:{title} (redirect duplicate of {final_id})")
            print(f"  DUP    {title:<36} -> {final_id} — skipped")
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
        suffix = f" -> {final_id.split(':', 1)[1]}" if final_id != f"wiki:{title}" else ""
        print(f"  ok     {title:<36} {len(doc['text']):>8,}{suffix}")
        time.sleep(0.4)

    # ---- gutenberg: additional public-domain shelf --------------------------
    print("\n== Project Gutenberg (public domain, topic=agriculture) ==")
    books: dict[int, dict] = {}
    for page in (1, 2, 3, 4, 5, 6):
        try:
            payload = json.loads(http_get(
                "https://gutendex.com/books?topic=agriculture&languages=en"
                f"&page={page}", timeout=60))
        except Exception as e:                                     # noqa: BLE001
            print(f"  stop at page {page}: {type(e).__name__}")
            break
        results = payload.get("results", [])
        if not results:
            break
        for b in results:
            if not b.get("copyright", True):
                books[b["id"]] = b
        time.sleep(0.6)
    print(f"  pd shelf: {len(books)} books")

    picked: dict[str, str] = {}               # doc_id -> bucket
    per_bucket: dict[str, int] = {}
    # resumability: previously cached v3 gutenberg books keep their bucket slots
    for meta_path in sorted(V3_DIR.glob("gutenberg__*.meta.json")):
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
            meta_cache = V3_DIR / raw_name(doc_id).replace(".txt", ".meta.json")
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
        if len(body) < 20000:
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
            "use": "KrishiGPT-nano corpus v3 (System A pretraining only)",
        }
        (V3_DIR / raw_name(doc_id).replace(".txt", ".meta.json")).write_text(
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
