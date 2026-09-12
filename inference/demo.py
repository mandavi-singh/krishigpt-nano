"""Generate agriculture continuations from the frozen FINAL KrishiGPT-nano
checkpoint (`checkpoints/krishigpt_v3/best.pt`) using the frozen 5,237-vocab
BPE tokenizer. This is the single-command demo for the submission.

All output is MODEL OUTPUT: next-token prediction, not human-authored, not
verified agricultural advice.

Run:  python inference/demo.py
      python inference/demo.py --prompts "Soil moisture affects" --decode sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.generate import KrishiGenerator  # noqa: E402

CKPT = REPO / "checkpoints" / "krishigpt_v3" / "best.pt"
TOKENIZER = REPO / "data" / "processed" / "agri_bpe_tokenizer.json"

DEFAULT_PROMPTS = [
    "The best fertilizer for wheat",
    "Soil moisture affects",
    "To control pests in the field ,",
    "Rice is grown in",
    "The farmer should irrigate",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate from the frozen final KrishiGPT-nano checkpoint.")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--decode", choices=["greedy", "sample"], default="sample",
                    help="recommended: sample (greedy is degenerate, see model card)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="recommended nucleus setting (final eval used 0.9)")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed => reproducible sample")
    args = ap.parse_args()

    if not CKPT.exists():
        sys.exit(f"final checkpoint missing: {CKPT}")
    gen = KrishiGenerator.from_checkpoint(CKPT, TOKENIZER)
    for p in args.prompts:
        if args.decode == "greedy":
            r = gen.greedy(p, args.max_tokens,
                           repetition_penalty=args.repetition_penalty)
        else:
            r = gen.sample(p, args.max_tokens, temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty,
                           seed=args.seed)
        print(f"prompt : {p!r}")
        print(f"{args.decode:6} : {p} {r.text}")
        print("label  : MODEL OUTPUT (not human-authored, not agricultural advice)\n")


if __name__ == "__main__":
    main()
