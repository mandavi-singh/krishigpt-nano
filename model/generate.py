"""Autoregressive generation for KrishiGPT-nano.

Pipeline: prompt text -> BPE encode -> repeat { forward -> last-position
logits -> pick next id } -> BPE decode:

    greedy:      argmax of the logits (deterministic).
    sample:      logits / temperature, keep top-k, softmax, multinomial.

CONTEXT WINDOW: the model was trained on T=seq_len windows with RoPE
positions 0..T-1. The generation context is therefore always truncated to the
LAST max_len ids (a sliding window that keeps the newest tokens); positions
restart at 0 inside each window, exactly as the model saw in training.

STOPPING: generation ends on EOS (if the model emits one -- our corpus
contains no EOS, so this is a safety guard), when max_tokens new ids were
produced, or when the context holds no room for another token.

Run:  python model/generate.py --checkpoint checkpoints/krishigpt_nano_smoke/best.pt
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.gpt import GPT, GPTConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from tokenizer.special import EOS_ID  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402


@dataclass
class GenerationResult:
    """Everything one generation pass produced, for tests and reporting."""
    prompt: str
    text: str                       # decoded continuation only (prompt excluded)
    ids: list[int]                  # generated ids only
    prompt_ids: list[int]           # encoded prompt after window truncation
    stop_reason: str                # "eos" | "max_tokens" | "context_full"
    n_generated: int


class KrishiGenerator:
    """Greedy + temperature/top-k/top-p/repetition-penalty decoding."""

    def __init__(self, model: nn.Module, tok: BPETokenizer) -> None:
        self.model = model
        self.tok = tok
        self.max_len = int(model.cfg.max_len)

    # ------------------------------------------------------------------ api
    def greedy(self, prompt: str, max_tokens: int = 64,
               repetition_penalty: float = 1.0) -> GenerationResult:
        ids = self.tok.encode(prompt)
        return self._generate(ids, max_tokens, sample=False,
                              repetition_penalty=repetition_penalty)

    def sample(self, prompt: str, max_tokens: int = 64,
               temperature: float = 1.0, top_k: int | None = None,
               top_p: float | None = None, repetition_penalty: float = 1.0,
               seed: int | None = None) -> GenerationResult:
        """Stochastic decoding; pass `seed` for a reproducible trajectory."""
        ids = self.tok.encode(prompt)
        return self._generate(ids, max_tokens, sample=True,
                              temperature=temperature, top_k=top_k, top_p=top_p,
                              repetition_penalty=repetition_penalty, seed=seed)

    def _generate(self, prompt_ids: list[int], max_tokens: int, *,
                  sample: bool, temperature: float = 1.0,
                  top_k: int | None = None, top_p: float | None = None,
                  repetition_penalty: float = 1.0,
                  seed: int | None = None) -> GenerationResult:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        model, max_len = self.model, self.max_len
        if max_len < 2:
            raise ValueError(f"max_len too small to generate: {max_len}")
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        prompt_ids = prompt_ids[-max_len:]        # window keeps newest tokens

        seq: list[int] = list(prompt_ids)
        out: list[int] = []
        stop = "max_tokens"
        model.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                if len(seq) >= max_len:           # no room for one more token
                    stop = "context_full"
                    break
                logits = model(torch.tensor([seq[-max_len:]], dtype=torch.long))
                last = logits[0, -1, :]
                if repetition_penalty > 1.0:
                    last = self._penalize(last, seq, repetition_penalty)
                if not sample:
                    nxt = int(torch.argmax(last).item())    # greedy on penalized
                else:
                    cand = last / temperature
                    if top_k is not None:
                        kth = torch.topk(cand, min(top_k, cand.numel())).values[-1]
                        cand = torch.where(cand >= kth, cand,
                                           torch.full_like(cand, float("-inf")))
                    if top_p is not None:
                        cand = self._nucleus(cand, top_p)
                    nxt = int(torch.multinomial(torch.softmax(cand, dim=-1), 1,
                                                generator=gen).item())
                out.append(nxt)
                if nxt == EOS_ID:
                    stop = "eos"
                    break
                seq.append(nxt)
        return GenerationResult(
            prompt=self.tok.decode(prompt_ids),
            text=self.tok.decode(out),            # skip_special drops a final EOS
            ids=out, prompt_ids=prompt_ids,
            stop_reason=stop, n_generated=len(out))

    # ------------------------------------------------------- logit surgery
    @staticmethod
    def _penalize(logits: torch.Tensor, seen: list[int],
                  penalty: float) -> torch.Tensor:
        """Divide logits of already-seen tokens by `penalty` (multiply when
        negative) so the same token must earn more logit mass to repeat."""
        logits = logits.clone()
        ids = torch.tensor(sorted(set(seen)), dtype=torch.long)
        score = logits[ids]
        logits[ids] = torch.where(score > 0, score / penalty, score * penalty)
        return logits

    @staticmethod
    def _nucleus(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Keep the smallest set of tokens whose cumulative probability
        mass reaches p; mask everything else (top-p / nucleus sampling)."""
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        keep = cum <= p
        keep[0] = True                            # always keep the argmax
        mask = ~keep[order.argsort()]
        return logits.masked_fill(mask, float("-inf"))

    # ------------------------------------------------------------ checkpoint
    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path,
                        tokenizer_path: str | Path,
                        cfg: GPTConfig | None = None,
                        device: str = "cpu") -> "KrishiGenerator":
        """Rebuild the GPT and load weights. `cfg` overrides the config saved
        in the checkpoint; without it the KrishiGPT-nano defaults are used
        (vocab/seq_len/dropout come from the checkpoint config)."""
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = payload.get("config", {})
        tok = BPETokenizer.load(tokenizer_path)
        vocab = payload.get("vocab_size") or cfg_dict.get("vocab_size")
        if vocab is None:
            raise ValueError("checkpoint carries no vocab_size")
        if vocab != tok.vocab_size:
            raise ValueError(
                f"vocab mismatch: checkpoint {vocab}, tokenizer {tok.vocab_size}")
        if cfg is None:
            cfg = GPTConfig(vocab_size=int(vocab),
                            max_len=int(cfg_dict.get("seq_len", 128)),
                            dropout=float(cfg_dict.get("dropout", 0.0)))
        model = GPT(cfg).to(device)
        load_checkpoint(ckpt_path, model, None)
        return cls(model, tok)


def _experiment() -> None:
    """Short generation demo on the saved smoke checkpoint (if present)."""
    root = Path(__file__).resolve().parents[1]
    ckpt = root / "checkpoints" / "krishigpt_nano_smoke" / "best.pt"
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt} -- run the smoke training first")
        return
    g = KrishiGenerator.from_checkpoint(
        ckpt, root / "data" / "processed" / "agri_bpe_tokenizer.json")
    prompts = ["The best fertilizer for wheat",
               "To control pests in the field ,",
               "Soil moisture affects"]
    for p in prompts:
        r = g.greedy(p, max_tokens=80)
        print(f"\nprompt : {p!r}\n"
              f"greedy ({r.stop_reason}, {r.n_generated} tok): {p} {r.text}")
        s = g.sample(p, max_tokens=80, temperature=0.8, top_k=50, seed=7)
        print(f"sample T=0.8 k=50 (seed 7): {p} {s.text}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="checkpoints/krishigpt_nano_smoke/best.pt")
    ap.add_argument("--tokenizer",
                    default="data/processed/agri_bpe_tokenizer.json")
    ap.add_argument("--prompts", nargs="+",
                    default=["The best fertilizer for wheat",
                             "To control pests in the field ,",
                             "Soil moisture affects"])
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--decode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    gen = KrishiGenerator.from_checkpoint(args.checkpoint, args.tokenizer)
    for p in args.prompts:
        if args.decode == "greedy":
            r = gen.greedy(p, args.max_tokens,
                           repetition_penalty=args.repetition_penalty)
        else:
            r = gen.sample(p, args.max_tokens, temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty,
                           seed=args.seed)
        print(f"prompt : {p!r}\n{args.decode:6} : {p} {r.text}\n")
