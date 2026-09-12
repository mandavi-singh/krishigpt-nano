"""Common interface for every tokenizer in this project.

A tokenizer is two dictionaries plus rules for converting between them:

    token2id : str  -> int     (encode)
    id2token : int  -> str     (decode)

The Transformer never sees text; it sees lists of these integer ids.
"""
from __future__ import annotations

import abc

from tokenizer.special import BOS_ID, EOS_ID


class Tokenizer(abc.ABC):
    """Abstract base class: text <-> token-id conversion with a fixed vocab."""

    def __init__(self, id2token: dict[int, str]) -> None:
        self.id2token = id2token
        self.token2id = {tok: i for i, tok in id2token.items()}
        assert len(self.token2id) == len(id2token), "duplicate tokens in vocab"

    # ------------------------------------------------------------------ vocab
    @property
    def vocab_size(self) -> int:
        """Number of distinct token ids (special tokens included)."""
        return len(self.id2token)

    def get_vocab(self) -> dict[int, str]:
        """Copy of the id -> token-string table."""
        return dict(self.id2token)

    # ------------------------------------------------------------- conversion
    @abc.abstractmethod
    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        """Convert text to a list of token ids."""

    @abc.abstractmethod
    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        """Convert token ids back to text."""

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _wrap(ids: list[int], add_bos: bool, add_eos: bool) -> list[int]:
        """Optionally prepend <BOS> and append <EOS>."""
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids
