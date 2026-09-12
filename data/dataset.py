"""Corpus -> token windows (B, T) for next-token-prediction training.

Windowing rule (GPT standard): the corpus is a single id stream; window i is
ids[i*seq_len : i*seq_len + seq_len + 1]. The dataset returns

    input  = window[:-1]     (T tokens)
    target = window[1:]      (the SAME window shifted by one)

so position t predicts token t+1 -- exactly the causal setup (every row of
the masked attention matrix is one valid prefix prediction). The trailing
partial window is dropped (fewer than seq_len+1 tokens left).

PERFORMANCE NOTE: our educational BPE encode is O(merges x symbols) per word
(5000 merges => seconds per megabyte in pure Python). We therefore memoize
per word TYPE and persist the full id stream to a .pt cache file under
data/processed/ so re-runs never re-tokenize. This is dataset engineering,
not a change to the tokenizer algorithm.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import BPETokenizer
from tokenizer.word_tokenizer import word_tokenize


def encode_text_cached(tok: BPETokenizer, text: str) -> list[int]:
    """Encode a full text, memoizing the BPE result of each word type."""
    cache: dict[str, list[int]] = {}
    ids: list[int] = []
    for word in word_tokenize(text):
        hit = cache.get(word)
        if hit is None:
            hit = cache[word] = tok.encode(word)
        ids.extend(hit)
    return ids


def load_id_stream(text_path: str | Path, cache_path: str | Path,
                   tok: BPETokenizer) -> torch.Tensor:
    """Token ids of a text file, cached to disk as a long tensor."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path)
    t0 = time.time()
    ids = encode_text_cached(tok, Path(text_path).read_text(encoding="utf-8"))
    tensor = torch.tensor(ids, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, cache_path)
    print(f"tokenized {Path(text_path).name}: {len(ids):,} ids "
          f"(cached -> {cache_path.name}, {time.time() - t0:.0f}s)")
    return tensor


class SequenceDataset(Dataset):
    """Fixed-length windows over one token stream with shifted targets."""

    def __init__(self, ids: torch.Tensor, seq_len: int) -> None:
        self.ids = ids
        self.seq_len = seq_len
        # number of COMPLETE windows of seq_len+1 tokens
        self.n_windows = max(0, (len(ids) - 1) // seq_len)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = i * self.seq_len
        window = self.ids[start : start + self.seq_len + 1]
        return window[:-1], window[1:]                  # input, shifted target


def get_dataloaders(
    tokenizer_path: str | Path,
    train_text_path: str | Path,
    valid_text_path: str | Path,
    cache_dir: str | Path,
    seq_len: int,
    batch_size: int,
    val_batch_size: int | None = None,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, int]:
    """Build train/valid DataLoaders over the agriculture corpus.

    Returns (train_dl, valid_dl, vocab_size). Train is shuffled per epoch
    (seeded generator -> reproducible order); drop_last keeps batches uniform.
    """
    tok = BPETokenizer.load(tokenizer_path)
    cache_dir = Path(cache_dir)
    train_ids = load_id_stream(train_text_path, cache_dir / "agri_train_ids.pt", tok)
    valid_ids = load_id_stream(valid_text_path, cache_dir / "agri_valid_ids.pt", tok)
    gen = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(
        SequenceDataset(train_ids, seq_len),
        batch_size=batch_size, shuffle=True, drop_last=True, generator=gen,
        num_workers=0,
    )
    valid_dl = DataLoader(
        SequenceDataset(valid_ids, seq_len),
        batch_size=val_batch_size or batch_size, shuffle=False, drop_last=False,
        num_workers=0,
    )
    return train_dl, valid_dl, tok.vocab_size
