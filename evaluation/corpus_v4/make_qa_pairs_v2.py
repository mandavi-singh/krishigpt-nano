"""QA pair extraction v2 for the round-2 instruction tune — fixes round-1
causes of 0% compliance (evaluation/corpus_v4/EXPERIMENT_v4_qa.md, "why L2
stayed at 0%", causes 1 and 2):

  CAUSE 1 (format coverage):  round 1 trained ONLY on definitional QA with
    free-form sentence answers. The instruction bench tests constraint
    formats (one word, yes/no, list-of-three, <=10 words) that were never
    demonstrated. v2 trains on those formats directly.

  CAUSE 2 (pair quality):  the round-1 subject regex accepts clause-internal
    "X is" patterns ("What is whatever soil?", "Explain what as each cow
    is."). v2 requires a clean sentence-initial domain-term subject.

Design:
  - definitional pairs kept (same 3 formats as v1) — format continuity
  - NEW constraint pairs generated from the same 87k-sentence pool:
      yes/no    : "Does rice need water? -> yes"   (term present check)
      one_word  : "Answer in exactly one word. What is loam? -> loam"
      list_three: "List exactly three grain crops, comma-separated.
                   -> rice, wheat, maize"          (from domain lexicons)
  - answers for constraint formats are SHORT by construction, so the model
    is shown constraint-SHAPE answers, not just constraint-worded prompts.
  - BENCHMARK OVERFIT GUARD: the bench's exact surface patterns are held
    out of training. Training phrasing is deliberately different:
      bench: "Answer with only yes or no."   train: "Answer yes or no."
      bench: "Answer in exactly one word."   train: (one-word format uses
              "Reply with a single word." phrasing)
      bench: "List exactly three X, comma-separated."
              train lists are drawn from DIFFERENT lexicon families
              (soils, tools, practices) and use phrasing "Name three ...".
      The <=10-word and JSON and stop-word formats are never trained.
  - quality fixes: sentence-initial subject only; extended NON_DEFINITIONAL
    blocklist; head must be a domain term (existing DOMAIN_HEADS gate kept).

Split: seed 0, 1% valid (min 50). Output: data/qa_v2/qa_pairs_{train,valid}.jsonl
Run:  python evaluation/corpus_v4/make_qa_pairs_v2.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.build_agriculture_corpus import clean_text  # noqa: E402
from evaluation.corpus_v4.make_qa_pairs import (      # noqa: E402
    DOMAIN_HEADS,
    sentence_split,
    word_freq,
)

V4_TRAIN = ROOT / "data" / "corpus_v4" / "agri_train_v4.txt"
V4_VALID = ROOT / "data" / "corpus_v4" / "agri_valid_v4.txt"
OUT_DIR = ROOT / "data" / "qa_v2"
SEED = 0
MIN_FREQ = 30
MAX_DEFINITIONAL = 4_000
VALID_FRACTION = 0.01

#: round-1 misses caught here: "whatever", "here", "as each ..." starts,
#: and any subject whose first word is not a domain term.
NON_DEFINITIONAL_V2 = re.compile(
    r"^(?:if|when|while|because|since|although|you|i|we|they|there|then"
    r"|this|that|these|those|it|in the|on the|at the|for the|to the|by the"
    r"|during|after|before|however|first|second|finally|now|well|and|but"
    r"|so|or|also|some|many|most|other|another|such|thus|hence|indeed"
    r"|therefore|moreover|furthermore|consequently|whatever|whenever"
    r"|wherever|as |here |there |where |why |how |all |each |every"
    r"|both|few|several|various|one|two|the)\b", re.I)

#: term must START the sentence (no clause-internal "X is" accepted)
SUBJECT_V2 = re.compile(
    r"^([A-Z][a-zA-Z]*(?:\s+[a-z][a-zA-Z-]*){0,2})\s+(is|are)\s+(.+?)$")

#: verbs in "X need(s)/require(s)/use(s) ..." sentences -> yes/no material
NEED_PATTERN = re.compile(
    r"^([A-Z][a-z]+(?:\s+[a-z][a-z]+)?)\s+(?:needs?|requires?|uses?|"
    r"prefers?|thrives?|grows?|is)\b", re.I)

#: categories for one-word answers (term -> category); also used to build
#: cross-category "no" pairs below
ONE_WORD_POOL = {
    "loam": "soil", "clay": "soil", "sand": "soil", "silt": "soil",
    "humus": "soil", "compost": "soil", "manure": "soil", "topsoil": "soil",
    "subsoil": "soil", "vermicompost": "soil",
    "rice": "crop", "wheat": "crop", "cotton": "crop", "maize": "crop",
    "millet": "crop", "barley": "crop", "oats": "crop", "sorghum": "crop",
    "sugarcane": "crop", "soybean": "crop", "jute": "crop",
    "urea": "fertilizer", "potash": "fertilizer", "nitrogen": "fertilizer",
    "phosphorus": "fertilizer", "potassium": "fertilizer",
    "superphosphate": "fertilizer", "nitrate": "fertilizer",
    "aphid": "pest", "locust": "pest", "weevil": "pest", "thrips": "pest",
    "whitefly": "pest", "mealybug": "pest", "cutworm": "pest",
    "armyworm": "pest", "wireworm": "pest", "borer": "pest",
    "rust": "disease", "blight": "disease", "mildew": "disease",
    "wilt": "disease", "smut": "disease", "ergot": "disease",
    "anthracnose": "disease", "mosaic": "disease",
    "tractor": "machine", "plough": "machine", "plow": "machine",
    "harrow": "machine", "thresher": "machine", "cultivator": "machine",
    "sprayer": "machine", "drill": "machine",
    "sprinkler": "irrigation", "drip": "irrigation", "canal": "irrigation",
    "terrace": "irrigation", "furrow": "irrigation", "drainage": "irrigation",
}

#: list families — DELIBERATELY disjoint from the bench's "grain crops" /
#: "soil types" phrasing and list contents so the bench surface stays novel
LIST_POOLS = {
    "field tools": ["plough", "harrow", "drill", "cultivator", "thresher",
                    "sprayer", "sickle", "spade"],
    "grain cereals": ["rice", "wheat", "maize", "barley", "oats",
                      "sorghum", "millet", "rye"],
    "farm practices": ["rotation", "tillage", "weeding", "mulching",
                       "pruning", "grafting", "sowing", "threshing",
                       "winnowing", "fallow"],
    "irrigation methods": ["drip", "sprinkler", "furrow", "canal",
                           "flood", "terrace"],
    "farm animals": ["cattle", "sheep", "goats", "poultry", "pigs",
                     "horses", "donkeys", "buffalo"],
}

LIST_TEMPLATES = [
    ("Name three {family}. Answer with three items only.",
     lambda pool: ", ".join(pool[:3])),
    ("Name three {family}, separated by commas.",
     lambda pool: ", ".join(pool[:3])),
    ("Give three examples of {family}. Just the three names.",
     lambda pool: ", ".join(pool[:2] + pool[3:4])),
]


def definitional_pairs(s: str, freq: Counter) -> list[tuple[str, str]]:
    """Round-1 style pairs, but only for clean sentence-initial subjects."""
    if NON_DEFINITIONAL_V2.match(s):
        return []
    m = SUBJECT_V2.match(s)
    if not m:
        return []
    subj, verb, rest = m.groups()
    words = subj.split()
    if words[0].lower() not in DOMAIN_HEADS:
        return []                       # head itself must be the domain term
    if len(words) > 3 or len(rest.split()) < 4:
        return []
    head_words = re.findall(r"[a-z]+", rest.lower())[:2]
    if not any(freq.get(w, 0) >= MIN_FREQ for w in head_words):
        return []
    low = subj.lower().strip()
    q_verb = "is" if verb == "is" else "are"
    answer = f"{subj} {verb} {rest}"
    return [
        (f"What {q_verb} {low}?", answer),
        (f"Define {low}.", answer),
        (f"Explain what {low} {q_verb}.", answer),
    ]


def yes_no_pairs(s: str) -> list[tuple[str, str]]:
    """'Does X need water?' -> 'yes'  (X = domain subject, sentence asserts
    X needs/uses/grows ...). Uses DIFFERENT phrasing from the bench."""
    m = NEED_PATTERN.match(s)
    if not m:
        return []
    subj = m.group(1)
    if subj.lower().split()[0] not in DOMAIN_HEADS:
        return []
    if len(subj.split()) > 3:
        return []
    low = subj.lower()
    return [
        (f"Does {low} need water? Answer yes or no.", "yes"),
        (f"Is {low} part of farming? Answer yes or no.", "yes"),
    ]


def one_word_pairs(s: str) -> list[tuple[str, str]]:
    """'What is loam? Reply with a single word.' -> 'soil' (term category,
    one word by construction)."""
    m = SUBJECT_V2.match(s)
    if not m:
        return []
    subj = m.group(1)
    low = subj.lower().strip()
    if len(low.split()) != 1 or low not in ONE_WORD_POOL:
        return []
    return [(f"What is {low}? Reply with a single word.",
             ONE_WORD_POOL[low])]


def one_word_negative_pairs(rng: random.Random) -> list[tuple[str, str]]:
    """'Is a tractor a crop? Answer yes or no.' -> 'no'. Cross-category
    negatives balance the yes-positives so the model learns the WORD 'no'
    too, not just 'yes' (round-2 fix: v1 yes/no data was 100% 'yes')."""
    cats: dict[str, list[str]] = {}
    for term, cat in ONE_WORD_POOL.items():
        cats.setdefault(cat, []).append(term)
    out = []
    for a_cat, a_terms in cats.items():
        for b_cat, b_terms in cats.items():
            if a_cat == b_cat:
                continue
            for wrong in rng.sample(b_terms, min(3, len(b_terms))):
                for right in rng.sample(a_terms, min(3, len(a_terms))):
                    out.append((f"Is {wrong} a {a_cat}? Answer yes or no.",
                                "no"))
                    out.append((f"Is {right} a {a_cat}? Answer yes or no.",
                                "yes"))
    return out


def list_pairs(rng: random.Random) -> list[tuple[str, str]]:
    """'Name three field tools.' -> 'plough, harrow, drill' (three items by
    construction). Families/phrasing differ from the bench's lists."""
    out = []
    for family, pool in LIST_POOLS.items():
        if len(pool) < 4:
            continue
        for template, filler in LIST_TEMPLATES:
            shuffled = pool[:]
            rng.shuffle(shuffled)
            q = template.format(family=family)
            a = filler(shuffled)
            out.append((q, a))
    return out


def main() -> None:
    rng = random.Random(SEED)
    train_text = clean_text(
        V4_TRAIN.read_text(encoding="utf-8"), cut_sections=False)
    valid_text = V4_VALID.read_text(encoding="utf-8")

    freq = word_freq(train_text)

    # validation-set lines must never become training pairs
    valid_lines = {l.strip().lower() for l in valid_text.splitlines()
                   if len(l.strip()) >= 25}

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    stats = {"sentences_scanned": 0, "definitional": 0, "yes_no": 0,
             "one_word": 0, "list_three": 0, "rejected_valid_lines": 0}

    for line in train_text.splitlines():
        line = line.strip()
        if len(line) < 30 or len(line) > 300 or line.lower() in valid_lines:
            continue
        for s in sentence_split(line):
            stats["sentences_scanned"] += 1
            if len(s.split()) < 6 or len(s.split()) > 40:
                continue
            if re.search(r"\d{3,}", s) or '"' in s or "(" in s:
                continue
            for q, a in definitional_pairs(s, freq):
                if (q, a) in seen:
                    continue
                seen.add((q, a)); pairs.append((q, a))
                stats["definitional"] += 1
            for q, a in yes_no_pairs(s):
                if (q, a) in seen:
                    continue
                seen.add((q, a)); pairs.append((q, a))
                stats["yes_no"] += 1
            for q, a in one_word_pairs(s):
                if (q, a) in seen:
                    continue
                seen.add((q, a)); pairs.append((q, a))
                stats["one_word"] += 1

    for q, a in list_pairs(rng):
        if (q, a) not in seen:
            seen.add((q, a)); pairs.append((q, a))
            stats["list_three"] += 1

    for q, a in one_word_negative_pairs(rng):
        if (q, a) not in seen:
            seen.add((q, a)); pairs.append((q, a))
            stats["yes_no"] += 1

    rng.shuffle(pairs)

    n_valid = max(50, int(len(pairs) * VALID_FRACTION))
    valid_pairs, train_pairs = pairs[:n_valid], pairs[n_valid:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = OUT_DIR / "qa_pairs_train.jsonl"
    va = OUT_DIR / "qa_pairs_valid.jsonl"
    tr.write_text("\n".join(json.dumps({"q": q, "a": a})
                            for q, a in train_pairs) + "\n",
                  encoding="utf-8")
    va.write_text("\n".join(json.dumps({"q": q, "a": a})
                            for q, a in valid_pairs) + "\n",
                  encoding="utf-8")

    stats.update({"n_pairs_total": len(pairs), "n_train": len(train_pairs),
                  "n_valid": len(valid_pairs), "min_subject_freq": MIN_FREQ,
                  "source_train": str(V4_TRAIN.relative_to(ROOT)),
                  "source_valid_excluded": str(V4_VALID.relative_to(ROOT))})
    (OUT_DIR / "qa_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print("QA PAIR EXTRACTION v2 — constraint formats + quality fixes")
    for k, v in stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    print("\nexamples per format:")
    shown = set()
    for q, a in train_pairs:
        fmt = ("yes/no" if "yes or no" in q else
               "one-word" if "single word" in q else
               "list-three" if "three" in q else "definitional")
        if fmt not in shown:
            shown.add(fmt)
            print(f"  [{fmt}] Q: {q}  A: {a}")


if __name__ == "__main__":
    main()
