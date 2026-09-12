"""Run every MYLLM tokenizer and print what they do.

Run from the project root:
    python tokenizer/demo.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# allow running as a plain script: put project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.preprocessing import get_tiny_shakespeare  # noqa: E402
from tokenizer import BPETokenizer, CharTokenizer, WordTokenizer  # noqa: E402
from tokenizer.word_tokenizer import word_tokenize  # noqa: E402

SMALL = (
    "First Citizen:\n"
    "Before we proceed any further, hear me speak.\n"
    "\n"
    "All:\n"
    "Speak, speak.\n"
)


def show(name: str, text: str, ids: list[int], decoded: str) -> None:
    print(f"--- {name} ---")
    print(f"  input   : {text!r}")
    print(f"  ids     : {ids}")
    print(f"  decoded : {decoded!r}")
    print(f"  round-trip exact: {decoded == text}")


def special_token_table() -> None:
    print("\n=== 5. SPECIAL TOKENS ===")
    print("  id 0  <PAD>  padding: makes batched rows equal-length; masked in loss")
    print("  id 1  <UNK>  unknown: anything outside the trained vocabulary")
    print("  id 2  <BOS>  begin-of-sequence: marks a fresh sequence")
    print("  id 3  <EOS>  end-of-sequence: generation stops when this is sampled")
    bpe = BPETokenizer.train("low lower", num_merges=20)
    ids = bpe.encode("low", add_bos=True, add_eos=True)
    print("  example: 'low' wrapped ->", ids)
    print("           decoded skip_special=True  :", repr(bpe.decode(ids)))
    print("           decoded skip_special=False :", repr(bpe.decode(ids, skip_special=False)))


def compare_on_shakespeare() -> None:
    print("\n=== 7. COMPARISON ON TINY SHAKESPEARE ===")
    text = get_tiny_shakespeare()
    n_chars = len(text)

    # ---- character ---------------------------------------------------------
    ct = CharTokenizer(text)
    char_tokens = ct.encode(text)

    # ---- word (min_freq=1) ---------------------------------------------------
    wt = WordTokenizer(text, min_freq=1)
    word_tokens = wt.encode(text)

    # ---- BPE -------------------------------------------------------------------
    bpe = BPETokenizer.train(text, num_merges=300, verbose=False)
    # Trick: BPE output depends only on the word TYPE, so encode each distinct
    # word once and weight by frequency instead of encoding 200k tokens.
    freqs = Counter(word_tokenize(text))
    enc_cache = {w: len(bpe.encode(w)) for w in freqs}
    bpe_tokens = sum(freqs[w] * enc_cache[w] for w in freqs)

    def row(name: str, vocab: int, total: int) -> None:
        per_word = total / len(word_tokens)
        per_kilo = total / n_chars * 1000
        print(f"  {name:<9} vocab={vocab:>6}  tokens={total:>8}  "
              f"per word={per_word:5.2f}  per 1000 chars={per_kilo:6.1f}")

    row("char", ct.vocab_size, len(char_tokens))
    row("word", wt.vocab_size, len(word_tokens))
    row("bpe300", bpe.vocab_size, bpe_tokens)
    print(f"  corpus: {n_chars:,} chars, {len(word_tokens):,} word tokens")
    print("  (measured larger BPE budgets: 1000 merges -> ~358/kchar in ~55s,")
    print("   3000 merges -> ~286/kchar in ~165s: compression keeps improving)")
    print("\n  TRADE-OFFS:")
    print("  char  -> tiny vocab (69), exact round-trip, but LONGEST sequences:")
    print("           a 256-token context holds only ~256 characters.")
    print("  word  -> shortest sequences but 13k+ vocab, misses unseen/rare words")
    print("           (<UNK>), and vocab keeps growing with every new text.")
    print("  bpe   -> the compromise IS the knob: 300 merges gives a 368-token vocab")
    print("           at ~2x shorter sequences than characters. It only closes the")
    print("           word-level compression gap once vocab grows (measured above).")
    print("           Reason GPT uses it anyway: no <UNK> for in-alphabet text,")
    print("           shared pieces teach morphology, vocab size is predictable.")


def gpt_connection() -> None:
    print("\n=== 8. GPT CONNECTION ===")
    print("""  Raw text  ->  Tokenizer  ->  Token IDs (ints)  ->  Embedding lookup
       ->  Transformer  ->  Logits  ->  Next-token prediction

  The Transformer ONLY understands vectors/ints. "cat" is not a concept to it,
  it is id 42, which indexes row 42 of a learned embedding matrix, which becomes
  a vector that participates in attention. Meaning is never in the ids - it is
  learned INTO the embedding weights during training. The tokenizer's only job:
  stable, reversible text <-> ints.""")


def main() -> None:
    print("=== 1. CHARACTER TOKENIZER (small corpus) ===")
    ct = CharTokenizer(SMALL)
    print(f"  vocab size: {ct.vocab_size}  (4 special + {ct.vocab_size - 4} chars)")
    print(f"  vocabulary: {ct.chars}")
    sample = "Before we proceed"
    show("char", sample, ct.encode(sample), ct.decode(ct.encode(sample)))

    print("\n=== 2. WORD TOKENIZER ===")
    wt = WordTokenizer(SMALL)
    print(f"  vocab size: {wt.vocab_size}")
    sample = "Speak, speak."
    show("word", sample, wt.encode(sample), wt.decode(wt.encode(sample)))
    print("  limitations: 'speak' and 'speaks' are unrelated ids; typos -> <UNK>;")
    print("  vocab grows with every new word; contractions decode lossily.")

    print("\n=== 3. WHY SUBWORDS? ===")
    print("  char: 'understanding' = 13 tokens.  word: 'misunderstanding' = new token.")
    print("  subword: shares pieces ('un'+'der'+'standing'), survives unseen words,")
    print("  vocab size is a knob (number of merges). GPT-2/3, LLaMA all use BPE.")

    print("\n=== 4. BPE FROM SCRATCH: 'low lower lowest' ===")
    corpus = "low lower lowest"
    bpe = BPETokenizer.train(corpus, num_merges=10, verbose=True)
    print(f"  vocab size: {bpe.vocab_size} = 4 special + base alphabet + {len(bpe.merges)} merges")
    for w in ["low", "lower", "lowest", "wow"]:
        ids = bpe.encode(w)
        toks = [bpe.id2token[i] for i in ids]
        print(f"  encode {w!r:8} -> {toks}   decode -> {bpe.decode(ids)!r}")
    unk_ids = bpe.encode("zzz")
    print(f"  encode 'zzz' (unseen letters) -> {[bpe.id2token[i] for i in unk_ids]}")

    special_token_table()
    compare_on_shakespeare()
    gpt_connection()
    print("\nALL DEMO SECTIONS COMPLETED")


if __name__ == "__main__":
    main()
