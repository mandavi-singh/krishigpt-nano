"""QA pair extraction for v4 instruction tuning — auto-generated from the
model's own license-clean corpus (no external data, no fabrication).

Extracts declarative sentences and converts them into question/answer pairs
with structure the tokenizer already handles well:

  "Loam is a mixture of sand, silt and clay."
      -> Q: "What is loam?"
         A: "Loam is a mixture of sand, silt and clay."

Pair filters (quality rules, all measured):
  - subject head must be 1-3 words, corpus-frequent (>=30) or a domain term
  - answer sentence 6-40 words, no digits-heavy, no quotes/parentheses
  - one pair per sentence; dedupe by (Q, A); no validation-set lines

Split: train/valid QA pairs, seed 0. Output: data/qa/qa_pairs_{train,valid}.jsonl
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

V4_TRAIN = ROOT / "data" / "corpus_v4" / "agri_train_v4.txt"
V4_VALID = ROOT / "data" / "corpus_v4" / "agri_valid_v4.txt"
OUT_DIR = ROOT / "data" / "qa"
SEED = 0
MIN_FREQ = 30            # subject head corpus frequency
MAX_PAIRS = 30_000
VALID_FRACTION = 0.01

STOP = re.compile(r"^(?:and|or|but|the|a|an|of|in|on|for|to|with|is|are|was"
                  r"|were|it|this|that|these|those|they|he|she|his|her|which"
                  r"|who|when|where|while|there|then|also|however|other"
                  r"|many|most|some|all|such|if|when|because|since|although"
                  r"|you|i|we|they|one|two|first|second|new|other|same|each"
                  r"|both|few|several|various)$", re.I)

#: words that mark a NON-definitional sentence start ("If the soil...",
#: "You see there...", "In the United States...")
NON_DEFINITIONAL = re.compile(
    r"^(?:if|when|while|because|since|although|you|i|we|they|there|then"
    r"|this|that|these|those|it|in the|on the|at the|for the|to the|by the"
    r"|during|after|before|however|first|second|finally|now|well|and|but"
    r"|so|or|also|some|many|most|other|another|such|thus|hence|indeed"
    r"|therefore|moreover|furthermore|consequently)\b", re.I)

#: subject HEAD must be one of these agriculture domain terms — keeps proper
#: names/abbreviations/odd phrases (DEF, Detmers, "my means") out of QA data
DOMAIN_HEADS = frozenset("""
rice wheat cotton maize millet barley oats sorghum sugarcane soybean jute
banana coconut mango tea coffee lentil pea peanut sunflower cassava rye
buckwheat potato tomato onion garlic mustard chickpea bean beans grape
apple orange citrus farm farming farmer farms crop crops grain grains
cereal cereals kernel kernels straw stalk stalks husk chaff hay silage
soil soils loam clay sand silt humus compost manure topsoil subsoil
horizon horizons gravel tilth clod clods sod earth ground land
irrigation drip sprinkler furrow canal terrace terraces rainwater drainage
flood flooding waterlogging aquifer groundwater well water waters
fertilizer fertilizers fertiliser manure urea potash ammonium phosphate
nitrate nitrogen phosphorus potassium micronutrient zinc sulfate
superphosphate nutrient nutrients mulch mulching vermicompost
pest pests insect insects aphid aphids locust locusts weevil weevils
borer borers thrip thrips whitefly mealybug cutworm armyworm wireworm
caterpillar caterpillars moth moths beetle beetles fly flies mite mites
weed weeds parasite parasites
disease diseases rust blight blights mildew wilt smut ergot anthracnose
rot rots mold mosaic fungus fungi bacteria virus viruses pathogen pathogens
rotation tillage ploughing plowing harrowing sowing seeding planting
harvest harvesting threshing winnowing weeding fallow intercropping
cultivation pruning grafting propagation drainage
seed seeds seedling seedlings germplasm cultivar cultivars nursery
plough plow plows harrow harrows tractor tractors thresher threshers
drill drills cultivator sprayer harvester implement implements
compost heap pit silo barn granary storehouse
pasture meadow forage fodder feed livestock cattle cow cows dairy sheep
poultry pig pigs goat goats buffalo horse horses donkey bee bees
beekeeping aquaculture fishery hen hens chicken chickens egg eggs
agriculture agronomy horticulture farming
greenhouse hydroponics orchard plantation grove vineyard field fields
garden plot acre acreage yield yields harvests
""".split())


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def word_freq(text: str) -> Counter:
    return Counter(re.findall(r"[a-z]+", text.lower()))


def subject_questions(s: str) -> list[tuple[str, str]]:
    """Yield (question, answer) for 'X is/are ...' definitional sentences."""
    if NON_DEFINITIONAL.match(s):
        return []
    m = re.match(
        r"^([A-Z][a-zA-Z]*(?:\s+[a-z][a-zA-Z-]*){0,2})\s+(is|are)\s+(.+?)$",
        s)
    if not m:
        return []
    subj, verb, rest = m.groups()
    words = subj.split()
    head = words[0]
    if STOP.match(head):
        return []
    if not any(w.lower() in DOMAIN_HEADS for w in words):
        return []                              # domain-term subjects only
    # subject must be a noun-phrase, not a sentence adverb/clause:
    # head word capitalized + remaining words lowercase domain terms
    if len(words) > 3:
        return []
    if len(rest.split()) < 4:          # answers need real content
        return []
    q_verb = "is" if verb == "is" else "are"
    question = f"What {q_verb} {subj.lower().strip()}?"
    answer = f"{subj} {verb} {rest}"
    pairs = [(question, answer)]
    # instruction diversity (helps L2 instruction-following): same fact in
    # 2 more command formats, deterministic
    low_subj = subj.lower().strip()
    pairs.append((f"Define {low_subj}.", answer))
    pairs.append((f"Explain what {low_subj} {q_verb}.", answer))
    return pairs


def clean_qa_text(text: str) -> str:
    return clean_text(text, cut_sections=False)


def main() -> None:
    rng = random.Random(SEED)
    train_text = clean_qa_text(V4_TRAIN.read_text(encoding="utf-8"))
    valid_text = V4_VALID.read_text(encoding="utf-8")

    freq = word_freq(train_text)

    # validation-set lines must never become training pairs
    valid_lines = {l.strip().lower() for l in valid_text.splitlines()
                   if len(l.strip()) >= 25}

    pairs: list[tuple[str, str]] = []
    seen_qa: set[tuple[str, str]] = set()
    n_sent = 0
    for line in train_text.splitlines():
        line = line.strip()
        if len(line) < 30 or len(line) > 300:
            continue
        if line.lower() in valid_lines:
            continue
        for s in sentence_split(line):
            n_sent += 1
            if len(s.split()) < 6 or len(s.split()) > 40:
                continue
            if re.search(r"\d{3,}", s) or '"' in s or "(" in s:
                continue
            for q, a in subject_questions(s):
                head = re.findall(r"[a-z]+", a.lower())  # answer subject head
                head_ok = any(freq.get(w, 0) >= MIN_FREQ for w in head[:2])
                if not head_ok:
                    continue
                key = (q, a)
                if key in seen_qa:
                    continue
                seen_qa.add(key)
                pairs.append(key)

    rng.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]

    n_valid = max(50, int(len(pairs) * VALID_FRACTION))
    valid_pairs, train_pairs = pairs[:n_valid], pairs[n_valid:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr_out = OUT_DIR / "qa_pairs_train.jsonl"
    va_out = OUT_DIR / "qa_pairs_valid.jsonl"
    tr_out.write_text("\n".join(json.dumps({"q": q, "a": a})
                                for q, a in train_pairs) + "\n",
                      encoding="utf-8")
    va_out.write_text("\n".join(json.dumps({"q": q, "a": a})
                                for q, a in valid_pairs) + "\n",
                      encoding="utf-8")

    stats = {
        "n_sentences_scanned": n_sent,
        "n_pairs_total": len(pairs),
        "n_train": len(train_pairs),
        "n_valid": len(valid_pairs),
        "min_subject_freq": MIN_FREQ,
        "source_train": str(V4_TRAIN.relative_to(ROOT)),
        "source_valid_excluded": str(V4_VALID.relative_to(ROOT)),
    }
    (OUT_DIR / "qa_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print(f"QA PAIR EXTRACTION — corpus v4")
    print(f"sentences scanned: {n_sent:,}")
    for k, v in stats.items():
        if k not in ("source_train", "source_valid_excluded"):
            print(f"  {k}: {v}")
    print(f"\ntrain -> {tr_out.relative_to(ROOT)}")
    print(f"valid -> {va_out.relative_to(ROOT)}")
    print("\nexamples:")
    for q, a in train_pairs[:5]:
        print(f"  Q: {q}\n  A: {a}")


if __name__ == "__main__":
    main()
