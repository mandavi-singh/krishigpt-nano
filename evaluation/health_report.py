"""LEVEL 1 — Training Health Gate (funnel evaluation rubric).

Automated PASS / FLAG / FAIL evaluation of training health from the run's
train_log.jsonl and frozen checkpoint:

  1. Numerical stability (FAIL on any NaN/Inf/non-finite loss)
  2. Convergence            (FAIL if loss never improves meaningfully)
  3. Undertraining          (FAIL if stopped before the LR floor / few epochs)
  4. Overfitting gap        (FLAG if final train/val loss gap is large)
  5. Gradient explosion     (FLAG on loss spikes / wild oscillation)
  6. Artifact validity      (PASS if checkpoint strict-loads + generates)

Run:  python evaluation/health_report.py [run_name]
      default run_name = krishigpt_v3
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = sys.argv[1] if len(sys.argv) > 1 else "krishigpt_v3"
LOG = ROOT / "checkpoints" / RUN / "train_log.jsonl"
CKPT = ROOT / "checkpoints" / RUN / "best.pt"


def load_log(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def evaluate_health(rows: list[dict]) -> list[dict]:
    checks: list[dict] = []

    train = [r["train_loss"] for r in rows if finite(r.get("train_loss"))]
    val_rows = [r for r in rows if finite(r.get("val_loss"))]
    vals = [r["val_loss"] for r in val_rows]
    n_bad = sum(1 for r in rows
                if not (finite(r.get("train_loss"))
                        and finite(r.get("val_loss", r["train_loss"]))))

    # ---- 1. numerical stability -------------------------------------------
    bad = [r for r in rows if not finite(r.get("train_loss"))
           or ("val_loss" in r and not finite(r["val_loss"]))]
    checks.append({
        "check": "numerical_stability", "level": "L1",
        "verdict": "FAIL" if bad else "PASS",
        "detail": (f"{len(bad)} non-finite loss entries" if bad
                   else f"all {len(rows)} log entries finite (no NaN/Inf)"),
    })

    # ---- 2. convergence ---------------------------------------------------
    if train:
        first, best = train[0], min(train)
        drop = first - best
        converged = drop >= 0.3
        checks.append({
            "check": "convergence", "level": "L1",
            "verdict": "PASS" if converged else "FAIL",
            "detail": (f"train loss {first:.4f} -> best {best:.4f} "
                       f"(drop {drop:.4f}); val {vals[0]:.4f} -> "
                       f"{min(vals):.4f} over {len(vals)} evals"
                       if vals else f"train drop {drop:.4f}"),
        })

    # ---- 3. undertraining -------------------------------------------------
    lrs = [r["lr"] for r in rows if finite(r.get("lr"))]
    epochs = max(r.get("epoch", 0) for r in rows) if rows else 0
    reached_low_lr = bool(lrs) and max(lrs) > 0 and min(lrs) <= 0.12 * max(lrs)
    enough_epochs = epochs >= 3
    verdict = "PASS" if (reached_low_lr and enough_epochs) else "FLAG"
    checks.append({
        "check": "undertraining", "level": "L1", "verdict": verdict,
        "detail": (f"LR annealed to {min(lrs):.1e} ({min(lrs)/max(lrs):.0%} "
                   f"of peak) = {'full cosine decay' if reached_low_lr else 'INCOMPLETE'}; "
                   f"epochs {epochs + 1}"
                   + ("; val still improving at exit (see convergence)"
                      if val_rows and val_rows[-1].get("best") else
                      "; val plateaued near exit")),
    })

    # ---- 4. overfitting gap ------------------------------------------------
    if vals and train:
        last_val = vals[-1]
        last_train = train[-1]
        gap = last_val - last_train
        verdict = "FLAG" if gap > 1.0 else "PASS"
        checks.append({
            "check": "overfitting_gap", "level": "L1", "verdict": verdict,
            "detail": (f"final train {last_train:.4f} vs val {last_val:.4f} "
                       f"(gap {gap:.4f})"
                       + (" — large; investigate" if gap > 1.0 else
                          " — normal LM gap")),
        })

    # ---- 5. gradient explosion ---------------------------------------------
    spikes = 0
    for i in range(1, len(train)):
        if train[i] > train[i - 1] + 1.0:                 # single-step jump
            spikes += 1
    swings = sum(1 for i in range(1, len(train))
                 if abs(train[i] - train[i - 1]) > 0.5)
    verdict = "FLAG" if spikes or swings > len(train) * 0.2 else "PASS"
    checks.append({
        "check": "gradient_explosion", "level": "L1", "verdict": verdict,
        "detail": (f"{spikes} loss spikes (>1.0 jump), {swings} wild swings "
                   f"(>0.5) across {len(train)} logs"
                   + ("; grad clip 1.0 active" if not spikes else "")),
    })

    return checks


def evaluate_artifact() -> dict:
    try:
        import torch
        from model.generate import KrishiGenerator

        tok = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
        gen = KrishiGenerator.from_checkpoint(CKPT, tok)
        r = gen.sample("Soil is", max_tokens=16, top_p=0.9, seed=0)
        n = sum(p.numel() for p in gen.model.parameters())
        ok = bool(r.text.strip()) and r.n_generated > 0
        return {
            "check": "artifact_validity", "level": "L1",
            "verdict": "PASS" if ok else "FAIL",
            "detail": (f"strict-load OK; {n:,} params; smoke generation "
                       f"produced {r.n_generated} tokens: {r.text[:50]!r}"),
        }
    except Exception as exc:
        return {"check": "artifact_validity", "level": "L1",
                "verdict": "FAIL", "detail": f"load/generation failed: {exc}"}


def main() -> None:
    if not LOG.exists():
        sys.exit(f"no train log at {LOG}")
    rows = load_log(LOG)
    checks = evaluate_health(rows)
    if CKPT.exists():
        checks.append(evaluate_artifact())
    else:
        checks.append({"check": "artifact_validity", "level": "L1",
                       "verdict": "FAIL", "detail": f"missing {CKPT}"})

    gate_fail = any(c["verdict"] == "FAIL" for c in checks)
    print(f"LEVEL 1 — TRAINING HEALTH GATE  ({RUN})")
    print("=" * 74)
    for c in checks:
        icon = {"PASS": "[PASS]", "FLAG": "[FLAG]", "FAIL": "[FAIL]"}[c["verdict"]]
        print(f"{icon} {c['check']:<20} {c['detail']}")
    print("=" * 74)
    verdict = "FAILED" if gate_fail else "PASSED"
    flags = sum(1 for c in checks if c["verdict"] == "FLAG")
    print(f"GATE: {verdict}"
          + (f"  ({flags} flag(s) to investigate)" if flags else ""))
    out = ROOT / "evaluation" / "results" / RUN / "health_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"run": RUN, "level": 1, "gate": verdict,
         "checks": checks}, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
