"""LEVEL 2 — General capability: instruction-following benchmark.

Tests constraint-following on the frozen model (funnel rubric, 'I' of the
G/I/E capability triad). The model is a BASE language model with no
instruction tuning, so the honest expectation is near-zero compliance — this
benchmark measures the baseline that QA fine-tuning (v4) must move.

Tests (each a deterministic, auto-checkable constraint):
  T1 one_word      — "Answer in exactly one word."   -> len(words) == 1
  T2 json_format   — "Reply with valid JSON only."   -> json.loads OK
  T3 yes_no        — yes/no question                 -> word in {yes,no}
  T4 list_three    — "List exactly three items, comma-separated."
  T5 short_answer  — "Answer in at most 10 words."
  T6 stop_word     — "End your answer with the word: done"

Scoring: natural + seeded sampling from the frozen checkpoint with the
standard prompt rewrite; post-processing is IDENTICAL to the chatbot (so
the number reflects the deployed system, not a hypothetical).

Run:  python evaluation/instruction_bench.py [run_name]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.app import trim_answer  # noqa: E402
from model.generate import KrishiGenerator  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "krishigpt_v3"
CKPT = ROOT / "checkpoints" / RUN / "best.pt"
TOK = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
OUT = ROOT / "evaluation" / "results" / RUN / "instruction_bench.json"

TESTS = [
    {"id": "one_word",
     "prompt": "Answer in exactly one word. What is the most important crop "
               "for food security? Answer:",
     "check": lambda t: len(t.split()) == 1},
    {"id": "json_format",
     "prompt": "Reply with valid JSON only, like {\"answer\": \"...\"} . "
               "What is loam? JSON:",
     "check": lambda t: _try_json(t)},
    {"id": "yes_no",
     "prompt": "Answer with only yes or no. Does rice need water? Answer:",
     "check": lambda t: t.strip().strip(".").lower() in {"yes", "no"}},
    {"id": "list_three",
     "prompt": "List exactly three grain crops, comma-separated. Answer:",
     "check": lambda t: len([c for c in t.split(",") if c.strip()]) == 3},
    {"id": "short_answer",
     "prompt": "Answer in at most 10 words. What is compost? Answer:",
     "check": lambda t: len(t.split()) <= 10},
    {"id": "stop_word",
     "prompt": "What is irrigation? End your answer with the word done.",
     "check": lambda t: t.rstrip().lower().endswith("done")},
    {"id": "one_word2",
     "prompt": "Answer in exactly one word. What do farmers add to soil to "
               "make it richer? Answer:",
     "check": lambda t: len(t.split()) == 1},
    {"id": "yes_no2",
     "prompt": "Answer with only yes or no. Is nitrogen a plant nutrient? "
               "Answer:",
     "check": lambda t: t.strip().strip(".").lower() in {"yes", "no"}},
    {"id": "short_answer2",
     "prompt": "Answer in at most 10 words. What does irrigation mean? "
               "Answer:",
     "check": lambda t: len(t.split()) <= 10},
    {"id": "list_three2",
     "prompt": "List exactly three soil types, comma-separated. Answer:",
     "check": lambda t: len([c for c in t.split(",") if c.strip()]) == 3},
]


def _try_json(text: str) -> bool:
    t = text.strip()
    for candidate in (t, t[t.find("{"):t.rfind("}") + 1]):
        try:
            json.loads(candidate)
            return True
        except Exception:
            continue
    return False


def main() -> None:
    gen = KrishiGenerator.from_checkpoint(CKPT, TOK)
    results = []
    passed = 0
    for t in TESTS:
        r = gen.sample(t["prompt"], max_tokens=32, top_p=0.9, seed=0)
        text = trim_answer(r.text)
        ok = bool(text) and t["check"](text)
        passed += ok
        results.append({"id": t["id"], "prompt": t["prompt"],
                        "output": text, "compliant": ok})

    score = round(passed / len(TESTS), 4)
    out = {"run": RUN, "level": 2, "n_tests": len(TESTS),
           "compliance": score,
           "note": ("Base LM with no instruction tuning — compliance is "
                    "expected to be near zero; this is the baseline the v4 "
                    "QA fine-tune must improve."),
           "results": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"LEVEL 2 — INSTRUCTION FOLLOWING  ({RUN})")
    print("=" * 74)
    for r in results:
        mark = "PASS" if r["compliant"] else "FAIL"
        print(f"[{mark}] {r['id']:<14} out: {r['output'][:60]!r}")
    print("-" * 74)
    print(f"compliance: {passed}/{len(TESTS)} = {score:.0%}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
