"""Checkpointing: safe save / resume for KrishiGPT training.

A checkpoint is ONE file containing everything needed to resume exactly:

    model state_dict, optimizer state_dict, scheduler state (step/lr scalars),
    step, tokens_seen, best_val_loss, run config, RNG state, vocab_size.

Resume policy: `find_latest` picks the highest step_NNNNNNN.pt, falling back
to final.pt (written on any early/interrupted exit); `best.pt` is a copy
written whenever validation loss improves. Writes go to a temp file first
then atomically replace the target, so a mid-write crash cannot corrupt the
last good checkpoint.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    config: dict,
    lr: float,
    epoch: int = 0,
    train_stats_only: bool = False,
) -> Path:
    """Atomic write: tmp file + os.replace. With train_stats_only=True the
    model/optimizer/RNG payloads are omitted (dry-run bookkeeping only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "config": config,
        "lr": lr,
        "vocab_size": config.get("vocab_size"),
    }
    if not train_stats_only:
        payload["model"] = model.state_dict()
        payload["optimizer"] = optimizer.state_dict()
        payload["rng_state"] = torch.get_rng_state()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)                              # atomic on Windows too
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load states into the given model/optimizer; return the scalar state.
    Payloads saved with train_stats_only=True carry no model/optimizer/RNG keys;
    such files are metadata-only and the caller must treat them that way."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(
            f"{Path(path).name} is a stats-only payload (no model weights); "
            "restore the real checkpoint instead")
    if payload.get("vocab_size") is not None:
        model_vocab = getattr(model, "cfg", None)
        if model_vocab is not None and model_vocab.vocab_size != payload["vocab_size"]:
            raise ValueError(
                f"vocab mismatch: checkpoint built with {payload['vocab_size']}, "
                f"model has {model_vocab.vocab_size}")
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint architecture does not match the model "
            "(d_model / n_heads / n_layers / max_len?)") from exc
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload.get("rng_state", torch.get_rng_state()))
    return {k: payload[k] for k in
            ("step", "epoch", "tokens_seen", "best_val_loss", "config", "lr")}


def find_latest(ckpt_dir: str | Path) -> Path | None:
    """Highest-numbered step_NNNNNNN.pt in the directory; if there is none,
    final.pt (written on any early/interrupted exit); else None."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    pat = re.compile(r"step_(\d+)\.pt$")
    found = [(int(m.group(1)), p) for p in ckpt_dir.glob("step_*.pt")
             if (m := pat.search(p.name))]
    if found:
        return max(found)[1]
    # Interrupted / smoke runs write only final.pt; it is a valid resume point.
    final = ckpt_dir / "final.pt"
    return final if final.exists() else None
