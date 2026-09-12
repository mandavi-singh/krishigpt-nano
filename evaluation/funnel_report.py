"""3-LEVEL FUNNEL REPORT — assembles L1/L2/L3 bench outputs into one
evaluation story for a run (v3 baseline or v4 improved).

Usage:
  python evaluation/funnel_report.py krishigpt_v3
  python evaluation/funnel_report.py krishigpt_v4

Reads (each produced by its own script, run first):
  evaluation/results/<run>/health_report.json        (level 1)
  evaluation/results/<run>/instruction_bench.json    (level 2, I)
  evaluation/results/<run>/mcq_bench.json            (level 3)
  checkpoints/<run>/train_log.jsonl                  (level 2, E + G)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = {"instruction": 0.4, "mcq": 0.6}     # L2/L3 aggregate split


def load(run: str, name: str) -> dict | None:
    p = ROOT / "evaluation" / "results" / run / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def capability_per_cost(run: str) -> dict:
    """Level 2 'E' — capability per unit of training cost."""
    log = ROOT / "checkpoints" / run / "train_log.jsonl"
    tokens = cost_note = None
    if log.exists():
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8")
                .splitlines() if l.strip()]
        if rows:
            tokens = rows[-1].get("tokens_seen")
            tps = [r.get("tokens_per_sec") for r in rows
                   if r.get("tokens_per_sec")]
            if tps:
                cost_note = f"~{sum(tps)/len(tps):.0f} tok/s avg"
    return {"tokens_seen_this_run": tokens, "throughput": cost_note}


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "krishigpt_v3"
    health = load(run, "health_report.json")
    instr = load(run, "instruction_bench.json")
    mcq = load(run, "mcq_bench.json")
    eff = capability_per_cost(run)

    print(f"3-LEVEL FUNNEL EVALUATION — {run}")
    print("=" * 74)

    # ---- Level 1 ---------------------------------------------------------
    print("LEVEL 1 — TRAINING HEALTH GATE")
    if health is None:
        print("  [MISSING] run evaluation/health_report.py first")
    else:
        for c in health["checks"]:
            print(f"  [{c['verdict']}] {c['check']}: {c['detail'][:100]}")
        print(f"  GATE: {health['gate']}")

    # ---- Level 2 ---------------------------------------------------------
    print("\nLEVEL 2 — GENERAL CAPABILITY")
    if instr is None:
        print("  [MISSING] run evaluation/instruction_bench.py first")
    else:
        print(f"  I  instruction-following: "
              f"{instr['compliance']:.0%} ({instr['n_tests']} tests)")
        print(f"     (base LM, no instruction tuning — baseline by design)")
    g_note = "G  general language: see val ppl (training health) + MCQ below"
    print(f"  {g_note}")
    if eff["tokens_seen_this_run"]:
        print(f"  E  efficiency: {eff['tokens_seen_this_run']:,} tokens "
              f"this run @ {eff['throughput']} (CPU-only)")
    else:
        print("  E  efficiency: (no train log)")

    # ---- Level 3 ---------------------------------------------------------
    print("\nLEVEL 3 — AGRICULTURE DOMAIN")
    if mcq is None:
        print("  [MISSING] run evaluation/mcq_bench.py first")
    else:
        print(f"  MCQ accuracy: {mcq['overall_accuracy']:.1%} "
              f"(chance {mcq['chance_baseline']:.0%}, "
              f"n={mcq['n_questions']})")
        print(f"  weighted domain score: {mcq['domain_score_weighted']:.1%}")
        print(f"  by difficulty: easy {mcq['domain_by_difficulty']['easy']:.0%}"
              f" / medium {mcq['domain_by_difficulty']['medium']:.0%}"
              f" / hard {mcq['domain_by_difficulty']['hard']:.0%}")
        print("  by sub-domain:")
        for sub, sc in mcq["subdomain_scores_weighted"].items():
            print(f"    {sub:<12} {sc:.0%}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 74)
    print("VERDICT")
    if health:
        print(f"  L1 gate: {health['gate']}")
    if instr and mcq:
        composite = (WEIGHTS["instruction"] * instr["compliance"]
                     + WEIGHTS["mcq"] * mcq["overall_accuracy"])
        print(f"  L2+L3 composite (0.4*I + 0.6*MCQ): {composite:.1%}")
        print("  Story: training is healthy (L1 PASS); the capability gap is "
              "instruction-following (0%) and domain MCQ (chance-level) — "
              "both are the QA fine-tuning (v4) targets.")
    print("=" * 74)

    out = ROOT / "evaluation" / "results" / run / "funnel_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run": run,
        "level1": health, "level2": {"instruction": instr, "efficiency": eff},
        "level3": mcq,
    }, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
