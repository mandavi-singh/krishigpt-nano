"""Word-level fabricated-word evaluation (same metric as the CPU experiment).

Regenerates the exact seeded samples (5 prompts x greedy/temp0.9/topk40/topp0.9,
seeds fixed in the variant kwargs) and checks each reconstructed word against
the FULL v1+v2 corpus via a word-boundary regex. Prints a comparison table
against the two previous checkpoints (v1-extended, v2).

Run:  python evaluation/fabrication_eval.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.generate import KrishiGenerator                  # noqa: E402

TOK_PATH = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"

_POOL_FILES = {
    "v1": [ROOT / "data" / "processed" / "agri_train.txt",
           ROOT / "data" / "processed" / "agri_valid.txt"],
    "v2": [ROOT / "data" / "corpus_v2" / "agri_train_v2.txt",
           ROOT / "data" / "corpus_v2" / "agri_valid_v2.txt"],
    "v3": [ROOT / "data" / "corpus_v3" / "agri_train_v3.txt"],
}
# Each checkpoint is attested against the cumulative text it (and its
# warm-start lineage) has seen: same definition as the previous reports
# (v1 ext -> v1+v2 pool, v2 rows reproduce exactly).
POOLS = {
    "v1 ext": "v1 v2",
    "v2 best": "v1 v2",
    "v2-ext best": "v1 v2",
    "v3 best": "v1 v2 v3",
}
_pool_cache: dict[str, str] = {}


def pool_text(names: str) -> str:
    if names not in _pool_cache:
        _pool_cache[names] = "".join(
            p.read_text(encoding="utf-8")
            for name in names.split() for p in _POOL_FILES[name])
    return _pool_cache[names]


_att_cache: dict[str, bool] = {}


def attested(word: str, pool: str) -> bool:
    key = word.lower()
    if key not in _att_cache:
        _att_cache[key] = re.search(r"(?<![A-Za-z])" + re.escape(key)
                                    + r"(?![A-Za-z])", pool,
                                    re.I) is not None
    return _att_cache[key]


def gen_words(tok, ids: list[int]) -> list[str]:
    words, cur = [], []
    for i in ids:
        s = tok.id2token[i]
        if "</w>" in s:
            cur.append(s.replace("</w>", ""))
            w = "".join(cur)
            if len(w) >= 3 and any(c.isalpha() for c in w) and not w.isdigit():
                words.append(w)
            cur = []
        else:
            cur.append(s)
    if cur:                                    # trailing unfinished word
        w = "".join(cur)
        if len(w) >= 3 and any(c.isalpha() for c in w) and not w.isdigit():
            words.append(w)
    return words


def distinct_n(ids: list[int], n: int) -> float | None:
    if len(ids) < n:
        return None
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


PROMPTS = ["The best fertilizer for wheat", "Soil moisture affects",
           "To control pests in the field", "Rice is grown in",
           "The farmer should irrigate"]
VARIANTS = [("greedy", {}), ("temp0.9", dict(temperature=0.9, seed=0)),
            ("topk40", dict(top_k=40, seed=0)),
            ("topp0.9", dict(top_p=0.9, seed=0))]

RUNS = [
    ("v3 best", ROOT / "checkpoints" / "krishigpt_v3" / "best.pt"),
    ("v2-ext best", ROOT / "checkpoints" / "krishigpt_nano_v2_extended" / "best.pt"),
    ("v2 best", ROOT / "checkpoints" / "krishigpt_nano_v2" / "best.pt"),
    ("v1 ext", ROOT / "checkpoints" / "krishigpt_nano_extended" / "best.pt"),
]


def main() -> None:
    rows = []
    print(f"{'run':<12} {'variant':<9} {'words':>6} {'fab%':>6} {'rep%':>6} "
          f"{'d1':>5} {'d2':>5}   fabricated examples")
    print("-" * 95)
    for label, ckpt in RUNS:
        if not ckpt.exists():
            print(f"{label:<12} -- checkpoint missing, skipped")
            continue
        gen = KrishiGenerator.from_checkpoint(ckpt, TOK_PATH)
        pool = pool_text(POOLS[label])
        _att_cache.clear()
        for vname, kw in VARIANTS:
            tot = fab = rep_seen = n_tok = 0
            d1s, d2s, examples = [], [], []
            for p in PROMPTS:
                r = (gen.greedy(p, 64) if vname == "greedy"
                     else gen.sample(p, 64, **kw))
                n_tok += r.n_generated
                rep_seen += sum(1 for i in range(1, len(r.ids))
                                if r.ids[i] in r.ids[:i])
                d1 = distinct_n(r.ids, 1)
                d2 = distinct_n(r.ids, 2)
                if d1 is not None:
                    d1s.append(d1)
                if d2 is not None:
                    d2s.append(d2)
                for w in gen_words(gen.tok, r.ids):
                    tot += 1
                    if not attested(w, pool):
                        fab += 1
                        if w not in examples:
                            examples.append(w)
            rep = 100 * rep_seen / max(n_tok - len(PROMPTS), 1)
            row = {"run": label, "variant": vname, "words": tot,
                   "fab_pct": round(100 * fab / max(tot, 1), 1),
                   "rep_pct": round(rep, 1),
                   "d1": round(sum(d1s) / max(len(d1s), 1), 2),
                   "d2": round(sum(d2s) / max(len(d2s), 1), 2),
                   "examples": examples[:6]}
            rows.append(row)
            print(f"{label:<12} {vname:<9} {tot:>6} "
                  f"{row['fab_pct']:>5.1f}% {row['rep_pct']:>5.1f}% "
                  f"{row['d1']:>5.2f} {row['d2']:>5.2f}   {examples[:4]}")

    out = ROOT / "evaluation" / "results" / "krishigpt_v3_best"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fabrication.json").write_text(json.dumps(rows, indent=2),
                                          encoding="utf-8")
    print(f"\nsaved -> {out / 'fabrication.json'}")


if __name__ == "__main__":
    main()
