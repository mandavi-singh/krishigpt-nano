"""Byte Pair Encoding (BPE) tokenizer, implemented from scratch.

NO pretrained tokenizer, NO tiktoken, NO sentencepiece — just the classic
Sennrich et al. (2016) algorithm in ~150 readable lines.

==========================================================================
WHY SUBWORDS? (character vs word vs subword)
==========================================================================
Character level:
    + exact round-trip, tiny vocab (~65)            - SEQUENCES ARE LONG
      ("understanding" = 13 tokens), model must learn spelling.
Word level:
    + short, semantic units                          - vocab explodes, <UNK> for
      anything new, no shared structure between related words.
Subword (BPE):
    + frequent words stay whole ("the" -> 1 token), rare words split into
      learned pieces ("uncommon" -> "un" + "common"), NO unknown words for
      in-alphabet text, controllable vocab size (few thousand - ~50k).
    - token boundaries are not linguistically "clean".

==========================================================================
THE ALGORITHM
==========================================================================
TRAIN:
    1. Pre-tokenize the corpus into WORDS and count their frequencies.
    2. Represent every word as its characters + an end-of-word marker </w>:
           "low" -> (l, o, w, </w>)
    3. Count every ADJACENT SYMBOL PAIR across the corpus, weighted by the
       word's frequency.
    4. MERGE the most frequent pair into a new symbol and add it to the
       vocabulary:  (l, o) -> "lo"
    5. Re-count pairs on the updated sequences. Repeat 3-5 `num_merges` times.
    Result: a list of ordered MERGE RULES + the vocabulary built from them.

ENCODE:
    1. Split text into words; turn each word into characters + </w>.
    2. Apply the learned merges IN LEARNING ORDER.
    3. Look up each resulting symbol in the vocabulary.

DECODE:
    Concatenate token strings; </w> becomes a word separator (space).

Example on "low lower lowest" (freq 1 each):
    pairs:   (l,o)x3 (o,w)x3 (w,e)x2 (w,</w>)x1 (e,r)x1 (e,s)x1 (s,t)x1
    merge 1: (l, o)      -> "lo"
    merge 2: (lo, w)     -> "low"      now (low,e)x2 beats (low,</w>)x1
    merge 3: (w, e)      ... wait — (low,e) wins:  -> "lowe"
    merge 4: (lowe, r)   -> "lower"
    ... "lowest" gets its remaining merges while "low" stays whole.
    => common prefixes are compressed first: exactly what we want.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tokenizer.base import Tokenizer
from tokenizer.special import (
    NUM_RESERVED_IDS,
    SPECIAL_TOKENS,
    UNK_ID,
    is_special_id,
)
from tokenizer.word_tokenizer import word_tokenize

#: End-of-word marker: marks where a word ends so that prefixes of different
#: length can coexist ("low</w>" is distinct from the prefix of "lowest").
END_OF_WORD = "</w>"


def _to_symbols(word: str) -> tuple[str, ...]:
    """Initial (pre-merge) representation of a word: characters + </w>."""
    return tuple(word) + (END_OF_WORD,)


def _apply_merge(symbols: tuple[str, ...], pair: tuple[str, str],
                 merged: str) -> tuple[str, ...]:
    """Replace every occurrence of the adjacent `pair` by `merged`.

    Linear scan — deliberately simple. (Production BPE uses priority
    queues or linked lists; we prioritize readability, rule 11/16.)
    """
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPETokenizer(Tokenizer):
    """Educational byte-pair-encoding tokenizer.

    Built by :meth:`train`; vocabulary layout:
        ids 0..3            special tokens (<PAD> <UNK> <BOS> <EOS>)
        ids 4..4+|chars|-1  base alphabet (all training characters + </w>)
        remaining ids       merged symbols, in merge-learning order
    """

    def __init__(self, base_tokens: list[str], merges: list[tuple[str, str, str]]) -> None:
        self.merges = list(merges)
        id2token: dict[int, str] = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok in base_tokens:
            assert tok not in id2token.values(), f"duplicate base token {tok!r}"
            id2token[len(id2token)] = tok
        for _left, _right, merged in merges:
            id2token[len(id2token)] = merged
        super().__init__(id2token)
        # fast membership test for unknown-character detection
        self.base_tokens = set(base_tokens)

    # ---------------------------------------------------------------- train
    @classmethod
    def train(cls, text: str, num_merges: int = 50,
              verbose: bool = False) -> "BPETokenizer":
        """Learn `num_merges` merge rules from `text`.

        Args:
            text: training corpus.
            num_merges: how many pair-merges to learn. vocab_size will be
                4 (special) + |base alphabet| + (merges actually performed).
            verbose: print every merge step (pair, new symbol, frequency).
        """
        word_freqs: Counter = Counter(word_tokenize(text))
        if not word_freqs:
            raise ValueError("empty corpus: nothing to train on")

        # step 1-2: word -> symbol sequence, all characters + </w> as base vocab
        symbols: dict[str, tuple[str, ...]] = {
            w: _to_symbols(w) for w in word_freqs
        }
        base_tokens = sorted({c for seq in symbols.values() for c in seq})

        # steps 3-5: repeated pair counting + merging
        merges: list[tuple[str, str, str]] = []
        for step in range(num_merges):
            pairs: Counter = Counter()
            for word, freq in word_freqs.items():
                seq = symbols[word]
                for i in range(len(seq) - 1):
                    pairs[(seq[i], seq[i + 1])] += freq
            if not pairs:
                break                                # nothing left to merge
            # deterministic tie-break: highest count, then alphabetical
            best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            pair, freq = best
            merged = pair[0] + pair[1]
            merges.append((pair[0], pair[1], merged))
            for word in symbols:                     # apply merge everywhere
                symbols[word] = _apply_merge(symbols[word], pair, merged)
            if verbose:
                print(f"  merge {len(merges):>3}: {pair} -> {merged!r}  (freq {freq})")

        return cls(base_tokens, merges)

    # -------------------------------------------------------------- encode
    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        """Word-split -> characters -> learned merges -> vocab lookup.

        Words containing a character never seen in training cannot be split
        into known symbols, so the whole word maps to <UNK> (classic BPE
        behaviour; byte-level BPE avoids even this — see Phase notes).
        """
        ids: list[int] = []
        for word in word_tokenize(text):
            if any(ch not in self.base_tokens for ch in word):
                ids.append(UNK_ID)
                continue
            seq = _to_symbols(word)
            for left, right, merged in self.merges:   # learning order!
                seq = _apply_merge(seq, (left, right), merged)
            ids.extend(self.token2id[s] for s in seq)
        return self._wrap(ids, add_bos, add_eos)

    # -------------------------------------------------------------- decode
    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        """Glue token strings together; </w> becomes a space between words."""
        pieces: list[str] = []
        for i in ids:
            if skip_special and is_special_id(i):
                continue
            pieces.append(self.id2token[i])
        text = "".join(pieces).replace(END_OF_WORD, " ")
        return text.rstrip()          # last word's </w> leaves a trailing space

    # ------------------------------------------------------------ save/load
    def save(self, path: str | Path) -> None:
        """Persist vocab base + merges as JSON (needed again in Phase 8).

        Explicit UTF-8: on Windows the default locale encoding (cp1252) cannot
        represent all tokens (e.g. Hebrew points that appear in real corpora).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_tokens": [t for t in self.base_tokens],
            "merges": [[l, r, m] for l, r, m in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        # explicit UTF-8: mirrors save(); the Windows cp1252 default cannot
        # decode tokens written with ensure_ascii=False (e.g. u05bb).
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [(l, r, m) for l, r, m in payload["merges"]]
        # base tokens are re-derived in sorted order (same as train())
        base_tokens = sorted(payload["base_tokens"])
        return cls(base_tokens, merges)

    def __repr__(self) -> str:
        return (f"BPETokenizer(vocab_size={self.vocab_size}, "
                f"merges={len(self.merges)})")
