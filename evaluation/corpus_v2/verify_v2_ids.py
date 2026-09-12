"""Verify corpus v2 tokenization under the EXISTING 5237-vocab BPE: build the
id-stream caches the training loader uses and assert the token counts match
the build report exactly."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.dataset import load_id_stream          # noqa: E402
from tokenizer import BPETokenizer               # noqa: E402

TOK_PATH = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
V2 = ROOT / "data" / "corpus_v2"
V1 = ROOT / "data" / "processed"

tok = BPETokenizer.load(TOK_PATH)
assert tok.vocab_size == 5237, tok.vocab_size
print(f"tokenizer: vocab {tok.vocab_size}, {len(tok.merges)} merges "
      f"(from {TOK_PATH.name})")

# exact counts reported by the v2 build (after leak-safe split rebuild)
expected = {"train": 1_196_617, "valid": 22_026}

for split, name in (("train", "agri_train_v2"), ("valid", "agri_valid_v2")):
    text_path = V2 / f"{name}.txt"
    cache_path = V2 / f"agri_{split}_ids.pt"
    assert not cache_path.exists(), "unexpected pre-existing cache"
    ids = load_id_stream(text_path, cache_path, tok)
    n = len(ids)
    print(f"{split}: {n:,} ids -> {cache_path.name} "
          f"(expected {expected[split]:,}) "
          f"{'OK' if n == expected[split] else 'MISMATCH!'}")
    assert n == expected[split], (n, expected[split])
    assert int(ids.min()) >= 0 and int(ids.max()) < tok.vocab_size
    # sanity: the stream contains real text tokens (no global UNK collapse)
    assert (ids == 1).sum() / n < 0.01
    torch.save(ids, cache_path)

# windows/steps per epoch for the planned run (seq_len 128, bs 16)
train_ids = torch.load(V2 / "agri_train_ids.pt")
valid_ids = torch.load(V2 / "agri_valid_ids.pt")
seq, bs = 128, 16
tr_win = (len(train_ids) - 1) // seq
va_win = (len(valid_ids) - 1) // seq
print(f"\nsequence windows (seq_len={seq}, bs={bs}):")
print(f"  train: {tr_win:,} windows -> {tr_win // bs} steps/epoch "
      f"(drop_last) -> 4 epochs = {4 * (tr_win // bs)} steps")
print(f"  valid: {va_win:,} windows -> {-(va_win // -bs)} eval batches")
print("\nv1 caches for reference (untouched):")
for f in ("agri_train_ids.pt", "agri_valid_ids.pt"):
    print(f"  {f}: {len(torch.load(V1 / f)):,} ids "
          f"[{Path(V1/f).stat().st_mtime:.0f}]")
