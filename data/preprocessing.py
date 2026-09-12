"""Corpus preparation.

Phase 2 uses this module ONLY to fetch Tiny Shakespeare for tokenizer
experiments. Phase 8 extends it into the full training-data pipeline
(splits, sequence windows, DataLoader).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
TINY_SHAKESPEARE_PATH = RAW_DIR / "tiny_shakespeare.txt"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def get_tiny_shakespeare(force: bool = False) -> str:
    """Return the Tiny Shakespeare text, downloading (~1.1 MB) once if needed."""
    if not TINY_SHAKESPEARE_PATH.exists() or force:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, TINY_SHAKESPEARE_PATH)
    return TINY_SHAKESPEARE_PATH.read_text(encoding="utf-8")
