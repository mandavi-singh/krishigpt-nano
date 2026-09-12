"""Corpus audit + v2 builder for KrishiGPT-nano.

Step 1 of this session: AUDIT the existing corpus (document diversity, topic
coverage, quality markers, duplication, imbalance), offline - no downloads.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.build_agriculture_corpus import _raw_name, split_documents  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REC_PATH = ROOT / "data" / "sources" / "agri_sources.json"
RAW_DIR = ROOT / "data" / "raw" / "agri"
TRAIN_TXT = ROOT / "data" / "processed" / "agri_train.txt"
VALID_TXT = ROOT / "data" / "processed" / "agri_valid.txt"

ARCH = re.compile(
    r"\b(thou|thee|thy|hath|doth|shalt|whereof|thereof|unto|forsooth|"
    r"ye olde|fourscore|husb\b|husb\.)", re.I)

TOPIC_KEYWORDS = {
    "crops/cultivation": re.compile(
        r"\b(wheat|rice|maize|barley|sorghum|sugarcane|cotton|legume|crop|"
        r"cultivar|sowing|plantation|harvest)\b", re.I),
    "soil science": re.compile(
        r"\b(soil|loam|clay|humus|salin|alkal|erosion|till)\b", re.I),
    "irrigation": re.compile(
        r"\b(irrigat|canal|drip|well|water table|watering)\b", re.I),
    "fertilizers/nutrients": re.compile(
        r"\b(fertilis|fertiliz|manure|compost|nitrogen|phosphor|potassium|"
        r"nutrient|dung)\b", re.I),
    "pests/diseases": re.compile(
        r"\b(pest|insect|blight|rust|fung|disease|weevil|locust|patholog)\b", re.I),
    "machinery": re.compile(
        r"\b(plough|plow|harrow|tractor|drill|thresher|reaper|implement|"
        r"ploughing|plowing)\b", re.I),
    "plant physiology": re.compile(
        r"\b(photosynthes|pollination|stomata|germination|chlorophyll|"
        r"respiration|transpiration)\b", re.I),
    "agronomy": re.compile(
        r"\b(agronom|rotation|fallow|intercropping|crop rotation|"
        r"yield|rotation of crops)\b", re.I),
    "climate/weather": re.compile(
        r"\b(monsoon|drought|frost|rainfall|climate|season|weather)\b", re.I),
    "livestock": re.compile(
        r"\b(cattle|cow|dairy|sheep|poultry|livestock|pig|poultry|"
        r"fodder|pasture)\b", re.I),
    "indian agriculture": re.compile(
        r"\b(India|Indian|Punjab|Bengal|Madras|Deccan|ryot|zamind|monsoon)\b",
        re.I),
}

NOISE_MARKER = re.compile(r"=\s*=+\s*=|<ref>|</ref>|\{\{|^Category:", re.I)


def load_docs():
    recs = json.loads(REC_PATH.read_text(encoding="utf-8"))
    docs = []
    for r in recs:
        docs.append({**r,
                     "text": (RAW_DIR / _raw_name(r["id"])).read_text(encoding="utf-8")})
    return docs


def main() -> None:
    docs = load_docs()
    train, val = split_documents(docs, val_fraction=0.10, seed=0)
    val_ids = {d["id"] for d in val}

    print("=" * 78)
    print("CORPUS AUDIT - existing agriculture corpus (v1)")
    print("=" * 78)
    print(f"\n[split] train={len(train)} docs, valid={len(val)} docs "
          f"(seed 0, val_fraction 0.10)")
    print(f"[split] VALID ids: {sorted(val_ids)}")

    print("\n[per-document]")
    hdr = f"{'side':<5} {'id':<26} {'chars':>9} {'words':>8} " \
          f"{'arch/kW':>8} {'short%':>7} {'noise':>6}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for d in sorted(docs, key=lambda x: x["id"]):
        words = max(len(d["text"].split()), 1)
        arch = len(ARCH.findall(d["text"]))
        nlines = max(len(d["text"].splitlines()), 1)
        short = sum(1 for ln in d["text"].splitlines() if len(ln.strip()) < 60)
        noise = len(NOISE_MARKER.findall(d["text"]))
        side = "VAL" if d["id"] in val_ids else "train"
        print(f"{side:<5} {d['id']:<26} {len(d['text']):>9} {words:>8} "
              f"{1000 * arch // words:>8} {100 * short // nlines:>6}% "
              f"{noise:>6}")
        rows.append((d["id"], side, len(d["text"]), words, arch, short,
                     nlines, noise))

    # topic coverage (word-weighted keyword hits)
    print("\n[topic coverage: keyword hits per 10k words]")
    print(f"{'topic':<24} {'all':>8} {'train':>8} {'valid':>8}  imbalance")
    print("-" * 64)
    train_text = TRAIN_TXT.read_text(encoding="utf-8")
    valid_text = VALID_TXT.read_text(encoding="utf-8")
    all_text = train_text + "\n" + valid_text
    for name, pat in TOPIC_KEYWORDS.items():
        ka = len(pat.findall(all_text))
        kt = len(pat.findall(train_text))
        kv = len(pat.findall(valid_text))
        na = round(ka / (len(all_text.split()) / 10000), 1)
        nt = round(kt / (len(train_text.split()) / 10000), 1)
        nv = round(kv / (len(valid_text.split()) / 10000), 1)
        flag = "!!" if nv < 0.2 * nt else ""
        print(f"{name:<24} {na:>8} {nt:>8} {nv:>8}  {flag}")

    print("\n[largest contributors]")
    tot = sum(r[2] for r in rows)
    for i, (doc_id, side, chars, words, *_ ) in enumerate(
            sorted(rows, key=lambda r: -r[2])[:7]):
        print(f"  {chars:>9} chars ({100 * chars / tot:4.1f}%) {side:<5} {doc_id}")

    print("\n[exact-line duplicate audit]")
    seen: Counter = Counter()
    per_doc_lines: dict[str, set[str]] = {}
    for d in docs:
        lines = {ln.lower() for ln in d["text"].splitlines()
                 if len(ln.strip()) >= 25}
        per_doc_lines[d["id"]] = lines
        seen.update(lines)
    cross = [ln for ln, c in seen.items() if c > 1]
    print(f"  long lines (>=25 ch) unique: {sum(len(s) for s in per_doc_lines.values()):,}")
    print(f"  lines duplicated ACROSS documents: {len(cross)}")

    out = ROOT / "evaluation" / "corpus_v2" / "audit_v1.json"
    out.write_text(json.dumps({
        "split": {"train": len(train), "valid": len(val),
                  "valid_ids": sorted(val_ids)},
        "per_doc": [{"id": r[0], "side": r[1], "chars": r[2], "words": r[3]}
                    for r in rows],
        "total_chars": tot,
        "cross_doc_dup_lines": len(cross),
    }, indent=2), encoding="utf-8")
    print(f"\naudit saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
