"""Compatibility gate for corpus v3 (read-only on all existing artifacts).

1. v3 valid is byte-identical to v2 valid (assert against on-disk files).
2. The existing 5,237-vocab BPE encodes the v3 valid text into EXACTLY the
   same 22,026-token id stream as the cached v2 valid tensor, so all
   checkpoints' v2-valid losses stay directly comparable with v3.
3. The v3 train text would yield exactly the reported token count.
Uses the verified fast BPE encoder (equality-checked against tok.encode
inside build_corpus_v3).
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders                          # noqa: E402
from evaluation.corpus_v3.build_corpus_v3 import (                # noqa: E402
    TOK_PATH, V2_VALID_TXT, V3_DIR, build_fast_encoder,
)
from tokenizer import BPETokenizer                                # noqa: E402
from tokenizer.word_tokenizer import word_tokenize                # noqa: E402

V2_IDS = torch.load(ROOT / "data" / "corpus_v2" / "agri_valid_ids.pt")

tok = BPETokenizer.load(TOK_PATH)
train_text = (V3_DIR / "agri_train_v3.txt").read_text(encoding="utf-8")
valid_text = (V3_DIR / "agri_valid_v3.txt").read_text(encoding="utf-8")

assert valid_text == V2_VALID_TXT.read_text(encoding="utf-8"), "valid drift!"

enc = build_fast_encoder(tok, train_text + valid_text)

ids: list[int] = []
for w in word_tokenize(valid_text):
    ids.extend(enc(w))
stream = torch.tensor(ids, dtype=torch.long)
print(f"v3 valid fast-encoded: {len(stream):,} ids "
      f"(v2 cached: {len(V2_IDS):,})")
assert torch.equal(stream, V2_IDS), "id stream differs from v2 valid!"

train_ids = []
for w in word_tokenize(train_text):
    train_ids.extend(enc(w))
print(f"v3 train fast-encoded: {len(train_ids):,} ids "
      f"(expected 1,966,105 from token_stats)")
assert len(train_ids) == 1_966_105, len(train_ids)

# loader smoke: dataloaders would produce these windows (seq 128, bs 16)
tr_win, va_win = 0, 0
n_tr = (len(train_ids) - 1) // 128
n_va = (len(stream) - 1) // 128
print(f"windows: train {n_tr:,} -> {n_tr // 16} steps/epoch, "
      f"valid {n_va:,} (drop-last at bs16: {n_va} eval windows)")
print("\nCOMPATIBILITY GATE PASSED")
