"""FINAL evaluation of the frozen KrishiGPT-nano checkpoint (v3 best).

One authoritative artifact set (final/final_eval.{json,md}):
  - validation loss / perplexity on the byte-fixed validation set
    (identical across v2-ext and v3, 172 windows)
  - parameter count
  - generation with top-p 0.9 and top-k 40 on the 5 standard prompts
    (same seeds as every previous run)
  - repetition metrics (distinct-1/-2, rep-rate, max run)
  - word-level fabrication/unsupported-word metric (attested vs the model's
    own cumulative training text: v1 + v2 + v3)
  - inference throughput (generated tokens / wall seconds; prefill TTFT proxy)

Everything here is READ-ONLY with respect to checkpoints/, data/, tokenizer.

Run:  python final/final_eval.py
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders                 # noqa: E402
from model.generate import KrishiGenerator               # noqa: E402
from training.train import evaluate                      # noqa: E402

CKPT = ROOT / "checkpoints" / "krishigpt_v3" / "best.pt"
TOK = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
OUT = ROOT / "final"

PROMPTS = ["The best fertilizer for wheat", "Soil moisture affects",
           "To control pests in the field", "Rice is grown in",
           "The farmer should irrigate"]
MAX_TOKENS = 64
VARIANTS = [("topp0.9", dict(top_p=0.9, seed=0)),
            ("topk40", dict(top_k=40, seed=0))]

POOL_FILES = [ROOT / "data" / "processed" / "agri_train.txt",
              ROOT / "data" / "processed" / "agri_valid.txt",
              ROOT / "data" / "corpus_v2" / "agri_train_v2.txt",
              ROOT / "data" / "corpus_v2" / "agri_valid_v2.txt",
              ROOT / "data" / "corpus_v3" / "agri_train_v3.txt"]


def attested(w: str, pool: str) -> bool:
    return re.search(r"(?<![A-Za-z])" + re.escape(w.lower()) + r"(?![A-Za-z])",
                     pool, re.I) is not None


def gen_words(tok, ids):
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
    if cur:
        w = "".join(cur)
        if len(w) >= 3 and any(c.isalpha() for c in w) and not w.isdigit():
            words.append(w)
    return words


def distinct_n(ids, n):
    if len(ids) < n:
        return None
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def main() -> None:
    gen = KrishiGenerator.from_checkpoint(CKPT, TOK)
    model = gen.model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {CKPT.parent.name}/{CKPT.name}: {n_params:,} params, "
          f"vocab {model.cfg.vocab_size}, max_len {model.cfg.max_len}")

    # ---- validation -----------------------------------------------------------
    tok_path = TOK
    _, valid_dl, vocab = get_dataloaders(
        tok_path, ROOT / "data" / "corpus_v3" / "agri_train_v3.txt",
        ROOT / "data" / "corpus_v3" / "agri_valid_v3.txt",
        ROOT / "data" / "corpus_v3", seq_len=128, batch_size=16, seed=0)
    t0 = time.time()
    val_loss = evaluate(model, valid_dl, "cpu")
    val_secs = time.time() - t0
    print(f"validation: {len(valid_dl.dataset):,} windows -> "
          f"val_loss {val_loss:.4f}  ppl {math.exp(val_loss):.2f}  ({val_secs:.1f}s)")

    # ---- generation + metrics ---------------------------------------------------
    pool = "".join(p.read_text(encoding="utf-8") for p in POOL_FILES)
    att_cache: dict[str, bool] = {}

    def word_attested(w: str) -> bool:
        k = w.lower()
        if k not in att_cache:
            att_cache[k] = attested(w, pool)
        return att_cache[k]

    results, t_total_tokens = [], 0
    t_gen0 = time.time()
    variant_word_totals = {v: 0 for v, _ in VARIANTS}
    variant_fab_totals = {v: 0 for v, _ in VARIANTS}
    for vname, kw in VARIANTS:
        vfabs, vwords = 0, 0
        reps, d1s, d2s, maxruns = [], [], [], []
        for p in PROMPTS:
            t_start = time.time()
            r = gen.sample(p, MAX_TOKENS, **kw)
            gen_secs = time.time() - t_start
            t_total_tokens += r.n_generated
            ids = r.ids
            rep = sum(1 for i in range(1, len(ids)) if ids[i] in ids[:i]) / \
                max(len(ids) - 1, 1)
            run = best = 0
            for i, t in enumerate(ids):
                run = run + 1 if i and ids[i - 1] == t else 1
                best = max(best, run)
            d1, d2 = distinct_n(ids, 1), distinct_n(ids, 2)
            reps.append(rep)
            if d1 is not None:
                d1s.append(d1)
            if d2 is not None:
                d2s.append(d2)
            maxruns.append(best)
            words = gen_words(gen.tok, ids)
            fabs = [w for w in words if not word_attested(w)]
            vfabs += len(fabs)
            vwords += len(words)
            results.append({
                "variant": vname, "prompt": p, "seed": kw.get("seed"),
                "text": r.text, "stop_reason": r.stop_reason,
                "n_generated": r.n_generated, "wall_secs": round(gen_secs, 2),
                "repetition_rate": round(rep, 4), "max_run": best,
                "distinct_1": round(d1, 4) if d1 is not None else None,
                "distinct_2": round(d2, 4) if d2 is not None else None,
                "fab_words": fabs, "n_words": len(words),
                "label": "MODEL OUTPUT (not human-authored, not from the corpus)",
            })
        variant_word_totals[vname] += vwords
        variant_fab_totals[vname] += vfabs
        print(f"{vname:<8} rep={sum(reps)/len(reps):.3f} "
              f"d1={sum(d1s)/len(d1s):.3f} d2={sum(d2s)/len(d2s):.3f} "
              f"fab={100*vfabs/max(vwords,1):.1f}% ({vfabs}/{vwords} words) "
              f"ctx_tps~{t_total_tokens/(time.time()-t_gen0):.0f}")

    gen_secs_total = time.time() - t_gen0
    tps = t_total_tokens / gen_secs_total

    # model-level single-token decode throughput (pure decode loop, no prompt)
    prompt_ids = gen.tok.encode("Soil moisture affects")
    model.eval()
    with torch.no_grad():
        seq = list(prompt_ids)
        d0 = time.time()
        n_dec = 0
        while len(seq) < gen.max_len and n_dec < 50:
            logits = model(torch.tensor([seq[-gen.max_len:]], dtype=torch.long))
            probs = torch.softmax(logits[0, -1] / 0.9, dim=-1)
            g = torch.Generator().manual_seed(123)
            seq.append(int(torch.multinomial(probs, 1, generator=g).item()))
            n_dec += 1
        decode_secs = time.time() - d0
    decode_tps = n_dec / decode_secs

    summary = {
        "model": "KrishiGPT-nano (final = checkpoints/krishigpt_v3/best.pt)",
        "step": 3800, "params": n_params, "vocab_size": model.cfg.vocab_size,
        "architecture": "6 blocks x d_model 128 x 4 heads, RoPE, pre-norm, "
                        "tied embedding/head, dropout 0.0",
        "val_loss": round(val_loss, 4), "val_ppl": round(math.exp(val_loss), 2),
        "val_windows": len(valid_dl.dataset),
        "generation": {
            "throughput_context_loop_tps": round(tps, 1),
            "decode_only_tps": round(decode_tps, 1),
            "note": "CPU-only (i5-1335U, 10 threads); throughput varies with "
                    "machine load between runs",
            "variants": {
                v: {
                    "avg_rep_rate": round(sum(r["repetition_rate"] for r in results
                                              if r["variant"] == v) / len(PROMPTS), 4),
                    "avg_distinct_1": round(sum(r["distinct_1"] for r in results
                                                if r["variant"] == v) / len(PROMPTS), 4),
                    "avg_distinct_2": round(sum(r["distinct_2"] for r in results
                                                if r["variant"] == v) / len(PROMPTS), 4),
                    "avg_max_run": round(sum(r["max_run"] for r in results
                                             if r["variant"] == v) / len(PROMPTS), 2),
                    "fab_words": variant_fab_totals[v],
                    "words": variant_word_totals[v],
                    "fabrication_pct": round(
                        100 * variant_fab_totals[v] /
                        max(1, variant_word_totals[v]), 1),
                }
                for v, _ in VARIANTS}
        },
    }

    (OUT / "final_eval.json").write_text(json.dumps(
        {"summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"saved -> {OUT / 'final_eval.json'}")


if __name__ == "__main__":
    main()
