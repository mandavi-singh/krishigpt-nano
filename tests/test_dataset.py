"""Tests for SequenceDataset windowing, shifted targets, and DataLoaders."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import SequenceDataset, encode_text_cached  # noqa: E402


def test_windows_count_and_coverage() -> None:
    ids = torch.arange(100)
    ds = SequenceDataset(ids, seq_len=10)
    assert len(ds) == 9                     # (100-1)//10 complete windows
    x0, y0 = ds[0]
    assert x0.tolist() == list(range(0, 10))
    assert y0.tolist() == list(range(1, 11))
    x8, y8 = ds[8]
    # window 8 covers ids[80:91]: inputs 80..89, targets 81..90. The 9-token
    # tail (91..99) is shorter than seq_len+1 and is dropped by design.
    assert x8.tolist() == list(range(80, 90))
    assert y8.tolist() == list(range(81, 91))


def test_target_is_input_shifted_by_one() -> None:
    ids = torch.randint(0, 50, (777,))
    ds = SequenceDataset(ids, seq_len=16)
    for i in range(len(ds)):
        x, y = ds[i]
        assert torch.equal(y, ids[i * 16 + 1 : i * 16 + 17])
        assert torch.equal(x, ids[i * 16 : i * 16 + 16])


def test_partial_tail_window_dropped() -> None:
    # 25 tokens, seq_len 10 -> windows need 11 tokens: only 2 fit
    ds = SequenceDataset(torch.arange(25), seq_len=10)
    assert len(ds) == 2
    # 21 tokens, seq_len 10 -> exactly 2 windows of 11 (last ends at 22)... 
    ds2 = SequenceDataset(torch.arange(21), seq_len=10)
    assert len(ds2) == 2                    # (21-1)//10 == 2


def test_dataloader_batch_shapes_no_leak() -> None:
    from torch.utils.data import DataLoader
    ids = torch.arange(500)
    dl = DataLoader(SequenceDataset(ids, seq_len=8), batch_size=5, shuffle=False,
                    drop_last=False)
    it = iter(dl)                            # one iterator, two batches
    xb, yb = next(it)
    assert xb.shape == (5, 8) and yb.shape == (5, 8)
    # every target row is its input row shifted by one *within the same window*
    for i in range(5):
        assert torch.equal(xb[i], ids[i * 8 : i * 8 + 8])
        assert torch.equal(yb[i], ids[i * 8 + 1 : i * 8 + 9])
    # windows tile the stream without overlap: batch 2 starts where batch 1 ended
    xb2, yb2 = next(it)
    assert xb2[0, 0].item() == 40           # 5 windows x 8 = 40


def test_dataloader_no_cross_window_leak_with_shuffle() -> None:
    from torch.utils.data import DataLoader
    ids = torch.arange(0, 200, 2)           # even numbers only
    gen = torch.Generator().manual_seed(1)
    dl = DataLoader(SequenceDataset(ids, seq_len=10), batch_size=4, shuffle=True,
                    drop_last=True, generator=gen)
    xb, yb = next(iter(dl))
    assert xb.shape == (4, 10)
    # contiguity of the original stream must hold in every row: y = x_{+2 step}
    assert torch.equal(yb - xb, torch.full((4, 10), 2))


def test_encode_text_cached_matches_direct() -> None:
    from tokenizer import BPETokenizer
    tok = BPETokenizer.train("soil needs nitrogen wheat wheat crop", num_merges=10)
    text = "soil needs nitrogen" * 3
    fast = encode_text_cached(tok, text)
    slow = tok.encode(text)
    assert fast == slow


def test_empty_stream() -> None:
    ds = SequenceDataset(torch.empty(0, dtype=torch.long), seq_len=8)
    assert len(ds) == 0
    ds2 = SequenceDataset(torch.arange(8), seq_len=8)   # needs 9 tokens
    assert len(ds2) == 0
