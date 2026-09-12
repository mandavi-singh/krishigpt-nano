"""Pre-build the corpus v4 BPE id caches with the VERIFIED fast encoder.

Same trick as corpus v3/v4 builders: rank-merge BPE application with a word
cache, strict-equality verified against BPETokenizer.encode before use.
Writes data/corpus_v4/{agri_train_ids.pt, agri_valid_ids.pt} so training's
load_id_stream finds them and never runs the slow naive encode.

Run:  python evaluation/corpus_v4/build_id_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.corpus_v3.build_corpus_v3 import build_fast_encoder  # noqa: E402
from tokenizer import BPETokenizer                                    # noqa: E402
from tokenizer.word_tokenizer import word_tokenize                    # noqa: E402

V4 = ROOT / "data" / "corpus_v4"
TOK = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"


def encode_text(lookup, text: str) -> list[int]:
    cache_hit = 0
    ids: list[int] = []
    for word in word_tokenize(text):
        ids.extend(lookup(word))
    return ids


def main() -> None:
    tok = BPETokenizer.load(TOK)
    for name in ("agri_train_v4.txt", "agri_valid_v4.txt"):
        text = (V4 / name).read_text(encoding="utf-8")
        lookup = build_fast_encoder(tok, text)
        out = V4 / name.replace("_v4.txt", "_ids.pt").replace(
            "agri_valid_ids", "agri_valid_ids")
        ids = encode_text(lookup, text)
        t = torch.tensor(ids, dtype=torch.long)
        out = V4 / (name.replace("train_v4", "train_ids")
                    .replace("valid_v4", "valid_ids").replace(".txt", ".pt"))
        torch.save(t, out)
        print(f"{name}: {len(ids):,} ids -> {out.name}")


if __name__ == "__main__":
    main()
