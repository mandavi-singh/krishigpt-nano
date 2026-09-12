"""KrishiGPT-nano training loop (CPU, from-scratch stack).

One iteration of the loop (the same 5-line skeleton as Lab 07):
    zero_grad -> forward -> loss (CE on shifted targets) -> backward -> step

Logging (JSONL, one line per log interval): step, tokens_seen, epoch, lr,
train_loss, train_ppl, tokens_per_sec. Validation adds val_loss/val_ppl and
updates best.pt when val loss improves. Checkpoints: every save_every steps +
resume via find_latest. Learning rate: linear warmup -> cosine decay to
min_lr. Optimizer: AdamW with decay groups (biases and LayerNorm excluded).

Run:  python training/train.py --config configs/nano_agri.json [--smoke]
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

from data.dataset import get_dataloaders  # noqa: E402
from model.gpt import GPT, GPTConfig  # noqa: E402
from training.checkpoint import (  # noqa: E402
    find_latest,
    load_checkpoint,
    save_checkpoint,
)
from training.loss import next_token_ce, perplexity  # noqa: E402


def build_param_groups(model: torch.nn.Module, weight_decay: float
                       ) -> list[dict]:
    """Standard AdamW grouping: decay 2D weights only, never biases/LayerNorm."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_lr(step: int, warmup_steps: int, max_lr: float, min_lr: float,
           total_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps or total_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: GPT, valid_dl, device: str) -> float:
    """Mean CE over the validation split; windows are non-overlapping, so the
    batch mean over the full split equals the corpus mean."""
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in valid_dl:
        loss = next_token_ce(model(xb.to(device)), yb.to(device))
        total += loss.item() * xb.shape[0]
        n += xb.shape[0]
    model.train()
    return total / max(n, 1)


def train(config: dict, smoke: bool = False, root: Path | None = None) -> dict:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    data_root = root / "data"

    max_steps = config.get("max_steps")
    if smoke:
        max_steps = config["smoke"]["max_steps"]
        log_every = config["smoke"]["log_every"]
        val_every = config["smoke"]["val_every"]
        ckpt_dir = root / "checkpoints" / (config["run_name"] + "_smoke")
    else:
        log_every = config["log_every"]
        val_every = config["val_every"]
        ckpt_dir = root / "checkpoints" / config["run_name"]
    # extend_schedule: same cosine formula, horizon stretched to max_steps for
    # a CONTINUATION whose bookkeeping is carried across a warm start
    # (update_warm_start). Has no effect on fresh or resumed runs.
    extend_cfg = config.get("extend_schedule")
    if extend_cfg is not None:
        max_steps = int(extend_cfg["max_steps"])
    log_path = ckpt_dir / "train_log.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if config["device"] == "cpu" else config["device"]
    torch.manual_seed(config["seed"])
    if device == "cpu":
        torch.set_num_threads(config.get("torch_threads", 10))

    # ---- data ---------------------------------------------------------------
    # corpus switch: "v1" (default), "v2" or "v3". Same tokenizer either way;
    # each version has its own text files and id cache dir, so the other
    # versions' artifacts are never touched.
    corpus = config.get("corpus", "v1")
    if corpus == "v2":
        train_txt = data_root / "corpus_v2" / "agri_train_v2.txt"
        valid_txt = data_root / "corpus_v2" / "agri_valid_v2.txt"
        cache_dir = data_root / "corpus_v2"
    elif corpus == "v3":
        train_txt = data_root / "corpus_v3" / "agri_train_v3.txt"
        valid_txt = data_root / "corpus_v3" / "agri_valid_v3.txt"
        cache_dir = data_root / "corpus_v3"
    elif corpus == "v4":
        train_txt = data_root / "corpus_v4" / "agri_train_v4.txt"
        valid_txt = data_root / "corpus_v4" / "agri_valid_v4.txt"
        cache_dir = data_root / "corpus_v4"
    else:
        train_txt = data_root / "processed" / "agri_train.txt"
        valid_txt = data_root / "processed" / "agri_valid.txt"
        cache_dir = data_root / "processed"
    train_dl, valid_dl, vocab_size = get_dataloaders(
        tokenizer_path=data_root / "processed" / "agri_bpe_tokenizer.json",
        train_text_path=train_txt,
        valid_text_path=valid_txt,
        cache_dir=cache_dir,
        seq_len=config["seq_len"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    train_steps_per_epoch = len(train_dl)
    if max_steps is None:
        max_steps = config["epochs"] * train_steps_per_epoch
    print(f"data: train windows {len(train_dl.dataset):,} "
          f"({train_steps_per_epoch} steps/epoch @ bs={config['batch_size']}), "
          f"valid windows {len(valid_dl.dataset):,}, vocab {vocab_size}")

    # ---- model / optimizer ---------------------------------------------------
    cfg = GPTConfig(vocab_size=vocab_size, max_len=config["seq_len"],
                    dropout=config["dropout"])
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(build_param_groups(model, config["weight_decay"]),
                            lr=config["lr"], betas=tuple(config["betas"]),
                            eps=config["eps"])

    # ---- resume / warm start ----------------------------------------------------
    # `warm_start` names a checkpoint whose WEIGHTS + OPTIMIZER we load but whose
    # step/loss bookkeeping we discard: a warm start always pairs with a fresh LR
    # schedule (and usually a fresh corpus), so step/token counters and the
    # best-val baseline restart from zero. A full `resume` (find_latest) keeps
    # everything — the old semantics.
    # `train_stats_only`: DRY-RUN mode. Train for the budget and keep logs, but
    # persist NO weights (the final checkpoint is metadata-only). Used only to
    # verify a schedule/config end-to-end before an expensive run; its run dir
    # must stay disposable (never a resume source).
    stats_only = bool(config.get("train_stats_only", False))
    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    epoch = 0
    warm_path = config.get("warm_start")
    warm_file = None
    if warm_path:
        warm_file = Path(warm_path)
        if not warm_file.is_absolute():
            warm_file = root / warm_path
    latest = None if stats_only else find_latest(ckpt_dir)
    if latest is not None:
        state = load_checkpoint(latest, model, opt)
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        best_val_loss = state["best_val_loss"]
        epoch = state.get("epoch", 0)
        print(f"resumed from {latest.name}: step {start_step}, "
              f"tokens {tokens_seen:,}, best_val_loss {best_val_loss:.4f}")
    elif warm_file is not None:
        state = load_checkpoint(warm_file, model, opt)
        if bool(config.get("update_warm_start")):
            # continuation mode: same fresh load, but the donor's bookkeeping
            # is carried forward (step/tokens/best_val/epoch) and the LR
            # schedule is stretched over the extended horizon.
            start_step = state["step"]
            tokens_seen = state["tokens_seen"]
            best_val_loss = state["best_val_loss"]
            epoch = state.get("epoch", 0)
            assert 0 < start_step < max_steps, (
                "update_warm_start requires extend_schedule.max_steps "
                f"({max_steps}) beyond the warm-start step ({start_step})")
            print(f"warm start CONTINUATION from {warm_file.name}: weights + "
                  f"optimizer loaded; step {start_step}, tokens "
                  f"{tokens_seen:,}, best_val_loss {best_val_loss:.4f} kept; "
                  f"cosine schedule extended to {max_steps} steps")
        else:
            print(f"warm start from {warm_file.name}: weights + optimizer "
                  f"loaded; step/loss counters reset for the fresh schedule")
    total_batches_to_run = config.get("total_batches", float("inf"))

    # ---- loop --------------------------------------------------------------------
    step = start_step
    t_last = time.time()
    tok_last = tokens_seen
    train_iter = iter(train_dl)
    model.train()
    last_log: dict = {}
    while step < max_steps:
        epoch = step // train_steps_per_epoch if train_steps_per_epoch else 0
        for _ in range(1):
            lr = get_lr(step, config["warmup_steps"], config["lr"],
                        config["min_lr"], max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                xb, yb = next(train_iter)
            opt.zero_grad()
            loss = next_token_ce(model(xb.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config["grad_clip"])
            opt.step()
            step += 1
            tokens_seen += xb.numel()
        total_batches_to_run -= 1
        if total_batches_to_run <= 0:
            print("total_batches budget exhausted -> stopping")
            break

        if step % log_every == 0:
            dt = time.time() - t_last
            tps = (tokens_seen - tok_last) / max(dt, 1e-9)
            t_last, tok_last = time.time(), tokens_seen
            last_log = {"step": step, "epoch": epoch, "lr": round(lr, 8),
                        "train_loss": round(loss.item(), 4),
                        "train_ppl": round(perplexity(loss.item()), 2),
                        "tokens_seen": tokens_seen,
                        "tokens_per_sec": round(tps, 1)}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(last_log) + "\n")
            print(f"step {step:>6} | loss {loss.item():.4f} "
                  f"ppl {perplexity(loss.item()):.2f} | lr {lr:.2e} "
                  f"| {tps:,.0f} tok/s")

        if step % val_every == 0:
            val_loss = evaluate(model, valid_dl, device)
            entry = {"step": step, "val_loss": round(val_loss, 4),
                     "val_ppl": round(perplexity(val_loss), 2)}
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, opt, step,
                                tokens_seen, best_val_loss,
                                {**config, "vocab_size": vocab_size}, lr)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({**last_log, **entry,
                                    "best": improved}) + "\n")
            print(f"  VALIDATION step {step}: val_loss {val_loss:.4f} "
                  f"val_ppl {perplexity(val_loss):.2f}"
                  f"{'  *best*' if improved else ''}")

        if (not smoke) and step % config["save_every"] == 0:
            save_checkpoint(ckpt_dir / f"step_{step:07d}.pt", model, opt, step,
                            tokens_seen, best_val_loss,
                            {**config, "vocab_size": vocab_size}, lr)

    # ---- final save + summary ---------------------------------------------------
    final_lr = get_lr(step, config["warmup_steps"], config["lr"],
                      config["min_lr"], max_steps)
    if stats_only:
        save_checkpoint(ckpt_dir / "final.pt", model, opt, step, tokens_seen,
                        best_val_loss, {**config, "vocab_size": vocab_size},
                        lr=final_lr, train_stats_only=True)
        print("DRY RUN: final.pt written as metadata-only "
              "(no weights kept); run dir is disposable")
    else:
        save_checkpoint(ckpt_dir / "final.pt", model, opt, step, tokens_seen,
                        best_val_loss, {**config, "vocab_size": vocab_size},
                        lr=final_lr)
    val_loss = evaluate(model, valid_dl, device)
    print(f"\nDONE at step {step}: final val_loss {val_loss:.4f}, "
          f"best {best_val_loss:.4f}; checkpoints in {ckpt_dir}")
    return {"final_step": step, "start_step": start_step, "val_loss": val_loss,
            "best_val_loss": best_val_loss, "tokens_seen": tokens_seen}


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nano_agri.json")
    ap.add_argument("--smoke", action="store_true",
                    help="short verification run (small step budget)")
    args = ap.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
