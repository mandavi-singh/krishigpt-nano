"""LEVEL 3 — Agriculture domain MCQ benchmark (funnel rubric).

Auto-generates cloze-style MCQs from the model's own corpus (license-clean,
deterministic), tagged by sub-domain and difficulty, and scores the frozen
model by option likelihood (standard base-LM cloze scoring):

  - 7 sub-domains (breadth): crops, soil, irrigation, fertilizer,
    pests, diseases, practices
  - difficulty (depth): Easy/Medium/Hard by term corpus frequency
    (>=100 / 20-100 / <20 occurrences)
  - rubric double weighting:
      within sub-domain:  Easy 20% / Medium 35% / Hard 45%
      domain-wide:        Easy 30% / Medium 40% / Hard 30%
  - option scoring: mean token log-prob of each candidate given the
    sentence prefix (argmax = model's choice)

Run:  python evaluation/mcq_bench.py [run_name]     (default krishigpt_v3)
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.generate import KrishiGenerator  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "krishigpt_v3"
CKPT = ROOT / "checkpoints" / RUN / "best.pt"
TOK = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
CORPUS = ROOT / "data" / "corpus_v3" / "agri_train_v3.txt"
OUT = ROOT / "evaluation" / "results" / RUN / "mcq_bench.json"

N_PER_CELL = 8          # questions per sub-domain x difficulty
SEED = 0

SUBDOMAINS: dict[str, list[str]] = {
    "crops": ["rice", "wheat", "cotton", "maize", "millet", "barley",
              "oats", "sugarcane", "soybean", "sorghum", "jute", "banana",
              "coconut", "mango", "tea", "coffee", "lentil", "pea"],
    "soil": ["loam", "clay", "sand", "silt", "humus", "horizon", "compost",
             "manure", "vermicompost", "topsoil", "subsoil", "gravel"],
    "irrigation": ["drip", "furrow", "sprinkler", "pivot", "canal",
                   "terraces", "rainwater", "flood", "surface", "channel"],
    "fertilizer": ["nitrogen", "phosphorus", "potassium", "urea", "potash",
                   "ammonium", "phosphate", "manure", "nitrate", "zinc"],
    "pests": ["aphid", "locust", "weevil", "borer", "thrips",
              "whitefly", "mealybug", "cutworm", "armyworm", "wireworm"],
    "diseases": ["rust", "blight", "mildew", "wilt", "smut", "ergot",
                 "anthracnose", "rots", "rot", "mold", "mosaic"],
    "practices": ["rotation", "tillage", "fallow", "intercropping", "weeding",
                  "sowing", "harvesting", "ploughing", "mulching", "drainage"],
}

W_SUB = {"easy": 0.20, "medium": 0.35, "hard": 0.45}
W_DOMAIN = {"easy": 0.30, "medium": 0.40, "hard": 0.30}

CORPUS_FILES = [
    ROOT / "data" / "processed" / "agri_train.txt",
    ROOT / "data" / "corpus_v2" / "agri_train_v2.txt",
    ROOT / "data" / "corpus_v3" / "agri_train_v3.txt",
]


def build_freq(text: str) -> dict[str, int]:
    freq: dict[str, int] = {}
    for w in re.findall(r"[a-z]+", text.lower()):
        freq[w] = freq.get(w, 0) + 1
    return freq


def sentences_with_terms(text: str, terms: set[str]) -> list[tuple[str, str]]:
    """(sentence, matched-term) for sentences containing exactly one term."""
    out = []
    for raw in text.split("\n"):
        s = raw.strip()
        if len(s.split()) < 6 or len(s.split()) > 30:
            continue
        words = re.findall(r"[A-Za-z]+", s.lower())
        hits = [t for t in terms if t in words]
        if len(hits) == 1:
            out.append((s, hits[0]))
    return out


def make_mcq(sentence: str, term: str, distractors: list[str],
             freq: dict[str, int]) -> dict | None:
    """Mask the term's first occurrence in the sentence. Difficulty combines
    term rarity with distractor plausibility (band-controlled)."""
    m = re.search(r"\b" + re.escape(term) + r"\b", sentence, re.I)
    if not m:
        return None
    prefix = sentence[:m.start()].rstrip()
    if len(prefix.split()) < 3:
        return None
    correct = term
    options = [correct] + distractors[:3]
    if len(options) < 4:
        return None
    f = freq.get(term, 0)
    if f >= 400:
        difficulty = "easy"      # very common term, distractors same band
    elif f >= 20:
        difficulty = "medium"
    else:
        difficulty = "hard"      # rare term (few corpus examples)
    return {"prefix": prefix + " ", "correct": correct, "options": options,
            "difficulty": difficulty, "frequency": f,
            "full": sentence}


def option_logprob(gen: KrishiGenerator, prefix: str, option: str) -> float:
    """Mean log-prob of the option's tokens given the prefix (cloze score)."""
    tok, model = gen.tok, gen.model
    p_ids = tok.encode(prefix)[-gen.max_len:]      # keep the FULL prefix
    o_ids = tok.encode(option.strip())
    if not o_ids:
        return -1e9
    ids = p_ids + o_ids
    if len(ids) > gen.max_len:
        ids = ids[-gen.max_len:]
        p_len = len(ids) - len(o_ids)
    else:
        p_len = len(p_ids)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    logprobs = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    for i in range(p_len, len(ids)):
        total += logprobs[i - 1, ids[i]].item()
    return total / len(o_ids)


def main() -> None:
    text = "\n".join(f.read_text(encoding="utf-8") for f in CORPUS_FILES)
    freq = build_freq(text)
    rng = random.Random(SEED)

    gen = KrishiGenerator.from_checkpoint(CKPT, TOK)

    questions: list[dict] = []
    for sub, terms in SUBDOMAINS.items():
        terms = [t for t in terms if freq.get(t, 0) >= 5]   # must exist
        if len(terms) < 4:
            continue
        cands = sentences_with_terms(text, set(terms))
        rng.shuffle(cands)
        by_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
        for sent, term in cands:
            if all(len(v) >= N_PER_CELL for v in by_diff.values()):
                break
            f_term = freq.get(term, 0)
            band = "easy" if f_term >= 400 else "medium" if f_term >= 20 \
                else "hard"
            # distractors: same band for easy/medium; for hard (rare term)
            # allow same or more-common band — rare-vs-common is hard by
            # construction only if the model must USE the context, which the
            # likelihood scoring forces.
            pool = [t for t in terms if t != term and freq.get(t, 0) >= 5]
            rng.shuffle(pool)
            if len(pool) < 3:
                continue
            mcq = make_mcq(sent, term, pool, freq)
            if mcq and mcq["difficulty"] == band and \
                    len(by_diff[band]) < N_PER_CELL:
                mcq["subdomain"] = sub
                by_diff[band].append(mcq)
        for d, qs in by_diff.items():
            questions.extend(qs)

    if not questions:
        sys.exit("no MCQs generated — check corpus/lexicon")

    correct_count = 0
    results: list[dict] = []
    for q in questions:
        scores = {o: option_logprob(gen, q["prefix"], o) for o in q["options"]}
        pred = max(scores, key=lambda o: scores[o])
        ok = pred == q["correct"]
        correct_count += ok
        results.append({"subdomain": q["subdomain"],
                        "difficulty": q["difficulty"],
                        "correct": q["correct"], "predicted": pred,
                        "ok": ok, "scores": {k: round(v, 3)
                                             for k, v in scores.items()},
                        "frequency": q["frequency"],
                        "context": q["prefix"] + "____"})

    # ---- aggregate: sub-domain x difficulty --------------------------------
    agg: dict[str, dict[str, dict]] = {}
    for sub in SUBDOMAINS:
        agg[sub] = {}
        for d in ("easy", "medium", "hard"):
            cell = [r for r in results
                    if r["subdomain"] == sub and r["difficulty"] == d]
            n = len(cell)
            acc = sum(1 for r in cell if r["ok"]) / n if n else None
            agg[sub][d] = {"n": n,
                           "acc": round(acc, 4) if acc is not None else None}

    sub_scores = {}
    for sub, cells in agg.items():
        if all(c["acc"] is None for c in cells.values()):
            continue
        sub_scores[sub] = round(sum((cells[d]["acc"] or 0.0) * w
                                    for d, w in W_SUB.items()
                                    if cells[d]["acc"] is not None), 4)

    dom_by_diff = {}
    for d in ("easy", "medium", "hard"):
        cell = [r for r in results if r["difficulty"] == d]
        dom_by_diff[d] = (round(sum(1 for r in cell if r["ok"]) / len(cell), 4)
                          if cell else 0.0)
    present = [d for d in ("easy", "medium", "hard") if dom_by_diff[d] > 0
               or any(r["difficulty"] == d for r in results)]
    wsum = sum(W_DOMAIN[d] for d in present) if present else 1.0
    domain_score = round(sum(dom_by_diff[d] * W_DOMAIN[d]
                             for d in present) / wsum, 4) if present else 0.0

    chance = round(1 / 4, 4)
    out = {
        "run": RUN, "level": 3, "n_questions": len(results),
        "chance_baseline": chance,
        "overall_accuracy": round(correct_count / len(results), 4),
        "by_subdomain": agg,
        "subdomain_scores_weighted": sub_scores,
        "domain_by_difficulty": dom_by_diff,
        "domain_score_weighted": domain_score,
        "weights": {"within_subdomain": W_SUB, "domain_wide": W_DOMAIN},
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"LEVEL 3 — AGRICULTURE DOMAIN MCQ  ({RUN})")
    print("=" * 74)
    print(f"questions: {len(results)}   chance: {chance:.0%}   "
          f"accuracy: {out['overall_accuracy']:.1%}")
    print(f"\n{'sub-domain':<14}{'easy':>8}{'medium':>8}{'hard':>8}"
          f"{'score(20/35/45)':>18}")
    for sub, cells in agg.items():
        fmt = lambda v: f"{v['acc']:.0%}" if v["acc"] is not None else "  -"
        print(f"{sub:<14}{fmt(cells['easy']):>8}{fmt(cells['medium']):>8}"
              f"{fmt(cells['hard']):>8}"
              f"{sub_scores.get(sub, 0):>18.0%}")
    print("-" * 74)
    d = dom_by_diff
    print(f"domain-wide    easy {d['easy']:.0%} / medium {d['medium']:.0%} "
          f"/ hard {d['hard']:.0%}   weighted score: {domain_score:.1%}")
    print(f"saved -> {OUT}")

if __name__ == "__main__":
    main()
