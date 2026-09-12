"""Character-level tokenizer: one character == one token.

Simplest possible scheme. The vocabulary is just the sorted set of distinct
characters found in the training corpus.

WHY START HERE
--------------
- Zero ambiguity: encode/decode round-trips EXACTLY for any in-vocab text.
- Tiny vocabulary (~65 for English text) => small embedding table.
- Downside (the reason we move to BPE later): sequences are LONG.
  "understanding" costs 13 tokens, so the model must learn spelling before
  meaning, and a 256-token context window holds only ~256 characters.
"""
from __future__ import annotations

from tokenizer.base import Tokenizer
from tokenizer.special import NUM_RESERVED_IDS, SPECIAL_TOKENS, UNK_ID, is_special_id


class CharTokenizer(Tokenizer):
    """Maps each distinct character in `text` to its own integer id."""

    def __init__(self, text: str) -> None:
        # Sorted => deterministic id assignment for a given corpus.
        self.chars: list[str] = sorted(set(text))
        id2token: dict[int, str] = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}
        for ch in self.chars:                      # real tokens start at id 4
            id2token[len(id2token)] = ch
        super().__init__(id2token)

    # -------------------------------------------------------------- encode
    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        """Each character becomes its id; unseen characters become <UNK>."""
        ids = [self.token2id.get(ch, UNK_ID) for ch in text]
        return self._wrap(ids, add_bos, add_eos)

    # -------------------------------------------------------------- decode
    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        """Join the characters back. Special ids are skipped by default."""
        out: list[str] = []
        for i in ids:
            if skip_special and is_special_id(i):
                continue
            out.append(self.id2token.get(i, ""))
        return "".join(out)

    # -------------------------------------------------------------- info
    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
