"""Systematic evaluation of a saved KrishiGPT checkpoint.

For one checkpoint this harness reports:

    * validation CE loss + perplexity (real validation split, cached ids)
    * generation token throughput (generated tokens / wall seconds)
    * per-variant repetition statistics: distinct-1, distinct-2,
      repetition rate (fraction of tokens already seen earlier),
      longest token run
    * full sample texts for every prompt x decoding variant (no cherry-picking)

All artifacts are written under evaluation/results/<name>/, never under
checkpoints/. Checkpoints are opened read-only.

Run:  python evaluation/run_eval.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders  # noqa: E402
from model.generate import KrishiGenerator  # noqa: E402
from training.train import evaluate  # noqa: E402

PROMPTS = [
    "The best fertilizer for wheat",
    "Soil moisture affects",
    "To control pests in the field",
    "Rice is grown in",
    "The farmer should irrigate",
]

MAX_TOKENS = 64

# (name, kind, kwargs) -- sampling variants get a fixed seed for reproducibility
VARIANTS = [
    ("greedy",              "greedy", {}),
    ("greedy_rep1.3",       "greedy", {"repetition_penalty": 1.3}),
    ("temp0.7",             "sample", {"temperature": 0.7, "seed": 0}),
    ("temp0.9",             "sample", {"temperature": 0.9, "seed": 0}),
    ("topk20",              "sample", {"top_k": 20, "seed": 0}),
    ("topk40",              "sample", {"top_k": 40, "seed": 0}),
    ("topp0.9",             "sample", {"top_p": 0.9, "seed": 0}),
    ("t0.9_k40_rep1.15",    "sample", {"temperature": 0.9, "top_k": 40,
                                       "repetition_penalty": 1.15, "seed": 0}),
]

VARIANT_BY_NAME = {name: (kind, kwargs) for name, kind, kwargs in VARIANTS}


# ------------------------------ statistics ---------------------------------

def distinct_n(ids: list[int], n: int) -> float | None:
    if len(ids) < n:
        return None
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_stats(ids: list[int]) -> dict:
    """Repetition diagnostics for one generated sequence."""
    n = len(ids)
    rep = sum(1 for i in range(1, n) if ids[i] in ids[:i]) / max(n - 1, 1)
    run = best = 0
    for i, t in enumerate(ids):
        run = run + 1 if i and ids[i - 1] == t else 1
        best = max(best, run)
    return {"n": n,
            "distinct_1": distinct_n(ids, 1),
            "distinct_2": distinct_n(ids, 2),
            "repetition_rate": rep,
            "max_run": best}


def variant_summary(rows: list[dict]) -> dict:
    """Aggregate one decoding variant over all its samples."""
    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {"samples": len(rows),
            "distinct_1": mean("distinct_1"),
            "distinct_2": mean("distinct_2"),
            "repetition_rate": mean("repetition_rate"),
            "max_run": round(sum(r["max_run"] for r in rows) / len(rows), 2),
            "frac_rep_over_half": round(
                sum(1 for r in rows if r["repetition_rate"] > 0.5) / len(rows),
                3)}


# --------------------------------- main -------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="checkpoints/krishigpt_nano_full/best.pt")
    ap.add_argument("--out-name", default=None,
                    help="dir under evaluation/results/ (default from ckpt path)")
    ap.add_argument("--variant", action="append", default=None,
                    help="limit to these variant names (repeatable)")
    ap.add_argument("--corpus", choices=["v1", "v2", "v3"], default="v1",
                    help="which corpus's validation split to evaluate on")
    args = ap.parse_args()

    ckpt = (ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute() \
        else Path(args.checkpoint)
    tok_path = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
    if args.corpus == "v2":
        train_txt = ROOT / "data" / "corpus_v2" / "agri_train_v2.txt"
        valid_txt = ROOT / "data" / "corpus_v2" / "agri_valid_v2.txt"
    elif args.corpus == "v3":
        train_txt = ROOT / "data" / "corpus_v3" / "agri_train_v3.txt"
        valid_txt = ROOT / "data" / "corpus_v3" / "agri_valid_v3.txt"
    else:
        train_txt = ROOT / "data" / "processed" / "agri_train.txt"
        valid_txt = ROOT / "data" / "processed" / "agri_valid.txt"
    if args.out_name:
        out_dir = ROOT / "evaluation" / "results" / args.out_name
    else:
        out_dir = ROOT / "evaluation" / "results" / \
            f"{ckpt.parent.name}_{ckpt.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [(n, k, kw) for n, k, kw in VARIANTS
                if args.variant is None or n in args.variant]
    if not variants:
        raise SystemExit("no known variants selected")

    print(f"loading {ckpt.parent.name}/{ckpt.name} ...")
    gen = KrishiGenerator.from_checkpoint(ckpt, tok_path)

    # ---- validation ----------------------------------------------------
    cache_dir = valid_txt.parent
    print(f"validation split ({args.corpus} corpus):")
    _, valid_dl, vocab_size = get_dataloaders(
        tok_path, train_txt, valid_txt, cache_dir,
        seq_len=128, batch_size=16, seed=0)
    t0 = time.time()
    val_loss = evaluate(gen.model, valid_dl, "cpu")
    print(f"  val loss {val_loss:.4f}  ppl {math.exp(val_loss):.1f}  "
          f"({time.time() - t0:.1f}s over {len(valid_dl.dataset):,} windows)")

    # ---- generation matrix ------------------------------------------------
    all_results: list[dict] = []
    gen_tokens = 0
    t_gen0 = time.time()
    for name, kind, kwargs in variants:
        for prompt in PROMPTS:
            if kind == "greedy":
                r = gen.greedy(prompt, MAX_TOKENS, **kwargs)
            else:
                r = gen.sample(prompt, MAX_TOKENS, **kwargs)
            stats = repetition_stats(r.ids)
            gen_tokens += r.n_generated
            all_results.append({
                "variant": name, "prompt": prompt,
                "text": r.text, "stop_reason": r.stop_reason,
                **stats})
            d1 = stats["distinct_1"]
            print(f"  {name:<18} | {prompt[:28]:<28} | n={stats['n']:>2} "
                  f"d1={d1:.2f} rep={stats['repetition_rate']:.2f} "
                  f"[{r.stop_reason}]")
    gen_secs = time.time() - t_gen0
    throughput = gen_tokens / gen_secs
    print(f"\ngeneration: {gen_tokens} tokens in {gen_secs:.1f}s "
          f"-> {throughput:.1f} tok/s")

    # ---- aggregate + persist ------------------------------------------------
    summary = {
        "checkpoint": str(ckpt),
        "val_loss": round(val_loss, 4),
        "val_ppl": round(math.exp(val_loss), 1),
        "vocab_size": vocab_size,
        "generation_tps": round(throughput, 1),
        "variants": {name: variant_summary(
            [r for r in all_results if r["variant"] == name])
            for name, _, _ in variants},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "samples.json").write_text(
        json.dumps({"results": all_results}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    lines = [f"# KrishiGPT-nano evaluation - {ckpt.parent.name}/{ckpt.name}",
             "",
             f"- val loss: **{val_loss:.4f}**  (perplexity **{math.exp(val_loss):.1f}**)",
             f"- generation throughput: **{throughput:.0f} tok/s** "
              f"({gen_tokens} tokens over {len(variants) * len(PROMPTS)} samples)",
              "", "## Repetition statistics by decoding variant", "",
              "| variant | distinct-1 | distinct-2 | rep-rate | mean max-run | frac rep>0.5 |",
              "|---|---|---|---|---|---|"]
    for name, _, _ in variants:
        s = summary["variants"][name]
        lines.append(f"| {name} | {s['distinct_1']} | {s['distinct_2']} | "
                     f"{s['repetition_rate']} | {s['max_run']} | "
                     f"{s['frac_rep_over_half']} |")
    lines += ["", "## Samples (all prompts, all variants, no filtering)", ""]
    for name, _, _ in VARIANTS:
        lines.append(f"### {name}")
        for r in all_results:
            if r["variant"] != name:
                continue
            lines.append(f"- **{r['prompt']}** ({r['stop_reason']}, "
                         f"d1={r['distinct_1']}, rep={r['repetition_rate']}): "
                         f"{r['prompt']} {r['text']}")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nartifacts -> {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
