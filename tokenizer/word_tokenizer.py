"""Word-level tokenizer: one word (or punctuation mark) == one token.

Splitting rule (regex): consecutive word characters form a token, and each
punctuation character is its own token.

    "Hello, world!"  ->  ["Hello", ",", "world", "!"]

LIMITATIONS OF WORD-LEVEL TOKENIZATION (why real LLMs do not use it)
--------------------------------------------------------------------
1. VOCABULARY SIZE: every distinct surface form is a token. "run", "runs",
   "running", "runner" are 4 unrelated ids. On the web, vocabulary runs into
   the millions and keeps growing with every new proper noun or typo.
2. UNKNOWN WORDS: anything not in the training corpus -> <UNK>. Typos,
   new words and rare names all degrade to one useless token.
3. MORPHOLOGY: the tokenizer sees no structure. "unbelievable" shares nothing
   with "believe"; subword methods fix this by sharing pieces.
4. RARE WORDS: tokens seen once get near-random embeddings (too little
   training signal per token).
5. WHITESPACE INFORMATION: this simple version normalizes whitespace, so
   decode cannot perfectly restore the original spacing; and contractions
   like "It's" split into ["It", "'", "s"] which decode back as "It' s".
"""
from __future__ import annotations

import re
from collections import Counter

from tokenizer.base import Tokenizer
from tokenizer.special import SPECIAL_TOKENS, UNK_ID, is_special_id

# words = runs of word chars; every other non-space char is its own token.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

# Punctuation that glues to the PREVIOUS word:     "Hello ," -> "Hello,"
_CLOSING = set(".,!?;:)]}\"'»’”")
# Punctuation that glues to the NEXT word:         "( quick" -> "(quick"
_OPENING = set("([{“‘«")


def word_tokenize(text: str) -> list[str]:
    """Split text into word/punctuation tokens (whitespace is consumed)."""
    return TOKEN_RE.findall(text)


class WordTokenizer(Tokenizer):
    """Vocabulary = most frequent tokens of the training corpus.

    Args:
        text: training corpus.
        min_freq: tokens seen fewer times are not added to the vocabulary
            (they encode to <UNK>). This is the standard defence against
            the exploding-vocabulary problem of word tokenization.
    """

    def __init__(self, text: str, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        counts = Counter(word_tokenize(text))
        id2token: dict[int, str] = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}
        # sorted() makes the vocabulary deterministic for a given corpus.
        for tok in sorted(counts):
            if counts[tok] >= min_freq:
                id2token[len(id2token)] = tok
        super().__init__(id2token)

    # -------------------------------------------------------------- encode
    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        ids = [self.token2id.get(tok, UNK_ID) for tok in word_tokenize(text)]
        return self._wrap(ids, add_bos, add_eos)

    # -------------------------------------------------------------- decode
    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        """Rejoin tokens with spaces; punctuation re-attaches without a space.

        Note this is lossy for contractions ("It's" -> "It' s") and for
        runs of multiple spaces — a real limitation of word-level tokenizing,
        solved in production by byte-pair tokenizers that encode the space
        itself (GPT-2's `Ġ` prefix).
        """
        tokens: list[str] = []
        for i in ids:
            if skip_special and is_special_id(i):
                continue
            tokens.append(self.id2token[i])
        out = ""
        attach_next = False          # set after an opening bracket/quote
        for tok in tokens:
            if not out:
                out = tok
            elif attach_next or self._is_closing(tok):
                out += tok           # glue without a space
            else:
                out += " " + tok
            attach_next = self._is_opening(tok)
        return out

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _is_closing(tok: str) -> bool:
        return len(tok) == 1 and tok in _CLOSING

    @staticmethod
    def _is_opening(tok: str) -> bool:
        return len(tok) == 1 and tok in _OPENING

    def __repr__(self) -> str:
        return f"WordTokenizer(vocab_size={self.vocab_size})"
