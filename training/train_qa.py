"""QA fine-tuning for KrishiGPT-nano (v4 instruction phase).

Trains the warm-started model on auto-generated QA pairs (data/qa/) in the
plain 'Q: ... A: ...' text format the corpus BPE already handles. This is
standard LM fine-tuning (CE on shifted targets) — no architecture, tokenizer,
or loss changes.

Design:
  - pairs are serialized as:  "Q: <question>\nA: <answer>" + newline EOS doc
    separator, concatenated into one token stream, windowed like pretraining
  - very small LR (default 1e-4 peak, cosine to 1e-5): the goal is FORMAT
    learning, not knowledge injection (616 pairs cannot teach facts)
  - validation = held-out QA pairs (50) + the byte-fixed corpus val set, both
    logged; best.pt tracked on QA-valid loss
  - resume-safe: step checkpoints + find_latest, same as pretraining

Run:  python training/train_qa.py --config configs/nano_agri_v4_qa.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import SequenceDataset, load_id_stream  # noqa: E402
from model.gpt import GPT, GPTConfig                        # noqa: E402
from training.checkpoint import (                            # noqa: E402
    find_latest,
    load_checkpoint,
    save_checkpoint,
)
from training.loss import next_token_ce, perplexity          # noqa: E402
from training.train import build_param_groups, evaluate      # noqa: E402
from tokenizer import BPETokenizer                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def qa_to_text(path: Path) -> str:
    parts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            parts.append(f"Q: {d['q']}\nA: {d['a']}")
    return "\n\n".join(parts) + "\n"


def encode_cached(tok: BPETokenizer, text: str, cache_path: Path) -> torch.Tensor:
    if cache_path.exists():
        return torch.load(cache_path)
    from tokenizer.word_tokenizer import word_tokenize
    cache: dict[str, list[int]] = {}
    ids: list[int] = []
    for word in word_tokenize(text):
        hit = cache.get(word)
        if hit is None:
            hit = cache[word] = tok.encode(word)
        ids.extend(hit)
    t = torch.tensor(ids, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, cache_path)
    return t


def get_lr(step, warmup, max_lr, min_lr, total):
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= total or total <= warmup:
        return min_lr
    p = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    data_root = ROOT / "data"
    qa_dir = data_root / config.get("qa_dir", "qa")
    run_name = config["run_name"]
    ckpt_dir = ROOT / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    device = config.get("device", "cpu")
    torch.manual_seed(config["seed"])
    if device == "cpu":
        torch.set_num_threads(config.get("torch_threads", 10))

    tok = BPETokenizer.load(data_root / "processed" / "agri_bpe_tokenizer.json")
    train_ids = encode_cached(tok, qa_to_text(qa_dir / "qa_pairs_train.jsonl"),
                              qa_dir / "qa_train_ids.pt")
    valid_ids = encode_cached(tok, qa_to_text(qa_dir / "qa_pairs_valid.jsonl"),
                              qa_dir / "qa_valid_ids.pt")

    seq = config["seq_len"]
    train_ds = SequenceDataset(train_ids, seq)
    valid_ds = SequenceDataset(valid_ids, seq)
    g = torch.Generator().manual_seed(config["seed"])
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        drop_last=True, generator=g, num_workers=0)
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, batch_size=config["batch_size"], shuffle=False,
        drop_last=False, num_workers=0)

    # corpus byte-fixed validation for LM-regression monitoring
    corpus_valid_txt = data_root / "corpus_v4" / "agri_valid_v4.txt"
    corpus_valid_ids = load_id_stream(
        corpus_valid_txt, data_root / "corpus_v4" / "agri_valid_ids.pt", tok)
    corpus_valid_dl = torch.utils.data.DataLoader(
        SequenceDataset(corpus_valid_ids, seq), batch_size=config["batch_size"],
        shuffle=False, num_workers=0)

    max_steps = config["max_steps"]
    cfg = GPTConfig(vocab_size=tok.vocab_size, max_len=seq,
                    dropout=config["dropout"])
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(build_param_groups(model, config["weight_decay"]),
                            lr=config["lr"], betas=tuple(config["betas"]),
                            eps=config["eps"])

    warm_file = ROOT / config["warm_start"] if config.get(
        "warm_start") else None
    latest = find_latest(ckpt_dir)
    start_step, best_qa_loss, tokens_seen = 0, float("inf"), 0
    if latest is not None:
        state = load_checkpoint(latest, model, opt)
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        best_qa_loss = state["best_val_loss"]
        print(f"resumed from {latest.name}: step {start_step}")
    elif warm_file is not None:
        load_checkpoint(warm_file, model, opt)
        print(f"warm start from {warm_file.name}: weights + optimizer loaded")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params | QA train windows {len(train_ds):,} "
          f"({len(train_dl)} steps/epoch) | QA valid windows {len(valid_ds):,}")

    step = start_step
    train_iter = iter(train_dl)
    t0 = time.time()
    while step < max_steps:
        lr = get_lr(step, config["warmup_steps"], config["lr"],
                    config["min_lr"], max_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            xb, yb = next(train_iter)
        opt.zero_grad()
        loss = next_token_ce(model(xb.to(device)), yb.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        opt.step()
        step += 1
        tokens_seen += xb.numel()

        if step % config["log_every"] == 0:
            dt = time.time() - t0
            tps = xb.numel() / max(dt, 1e-9)
            t0 = time.time()
            entry = {"step": step, "lr": round(lr, 8),
                     "train_loss": round(loss.item(), 4),
                     "train_ppl": round(perplexity(loss.item()), 2),
                     "tokens_seen": tokens_seen,
                     "tokens_per_sec": round(tps, 1)}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"step {step:>5} | qa_loss {loss.item():.4f} "
                  f"ppl {perplexity(loss.item()):.2f} | lr {lr:.2e}")

        if step % config["val_every"] == 0:
            model.eval()
            qa_val = evaluate(model, valid_dl, device)
            corpus_val = evaluate(model, corpus_valid_dl, device)
            model.train()
            improved = qa_val < best_qa_loss
            if improved:
                best_qa_loss = qa_val
                save_checkpoint(ckpt_dir / "best.pt", model, opt, step,
                                tokens_seen, best_qa_loss,
                                {**config, "vocab_size": tok.vocab_size}, lr)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step, "qa_val_loss": round(qa_val, 4),
                    "qa_val_ppl": round(perplexity(qa_val), 2),
                    "corpus_val_loss": round(corpus_val, 4),
                    "corpus_val_ppl": round(perplexity(corpus_val), 2),
                    "best": improved}) + "\n")
            print(f"  VALIDATION step {step}: qa_val {qa_val:.4f} "
                  f"(ppl {perplexity(qa_val):.2f}) | corpus_val "
                  f"{corpus_val:.4f} (ppl {perplexity(corpus_val):.2f})"
                  f"{'  *best*' if improved else ''}")

        if (step - start_step) and step % config["save_every"] == 0:
            save_checkpoint(ckpt_dir / f"step_{step:07d}.pt", model, opt, step,
                            tokens_seen, best_qa_loss,
                            {**config, "vocab_size": tok.vocab_size}, lr)

    final_lr = get_lr(step, config["warmup_steps"], config["lr"],
                      config["min_lr"], max_steps)
    save_checkpoint(ckpt_dir / "final.pt", model, opt, step, tokens_seen,
                    best_qa_loss, {**config, "vocab_size": tok.vocab_size},
                    lr=final_lr)
    print(f"\nDONE at step {step}: best QA val loss {best_qa_loss:.4f}; "
          f"checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
