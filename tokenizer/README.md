# Phase 2 — Tokenization (from scratch)

Educational tokenizers — no Hugging Face, no tiktoken, no pretrained vocab.

```
tokenizer/
├── special.py         <PAD> <UNK> <BOS> <EOS>, fixed ids 0..3
├── base.py            Tokenizer ABC (vocab, encode/decode contract)
├── char_tokenizer.py  character-level (exact round-trip)
├── word_tokenizer.py  regex word/punct splitting + min_freq
├── bpe_tokenizer.py   byte-pair encoding from scratch (+ save/load)
└── demo.py            run everything, incl. Tiny Shakespeare comparison
```

## Vocabulary layout used by all tokenizers

| ids   | content                                   |
|-------|-------------------------------------------|
| 0     | `<PAD>` padding — batch rows equal length; masked in loss/attn |
| 1     | `<UNK>` unknown — anything outside the trained vocab            |
| 2     | `<BOS>` begin-of-sequence                                        |
| 3     | `<EOS>` end-of-sequence — sampling this stops generation         |
| 4+    | real text tokens                                                 |

## BPE algorithm (bpe_tokenizer.py)

TRAIN:
1. pre-tokenize corpus into words, count frequencies
2. words -> `(char, char, ..., </w>)` symbol sequences
3. count adjacent symbol pairs, weighted by word frequency
4. merge the most frequent pair into a new vocab symbol (deterministic tie-break)
5. repeat `num_merges` times — the ordered merge list IS the learned model

ENCODE: per word, apply learned merges in learning order, look up ids.
DECODE: concatenate token strings, `</w>` -> word boundary, trailing strip.

Encoding a 3000-merge BPE is O(merges × symbols) per word in this naive
implementation: fast enough for this project (165 s to train on Tiny
Shakespeare), and easy to read — production optimizes step 3-4 with priority
queues.

## Measured comparison on Tiny Shakespeare (1.11 MB, 262,927 word tokens)

| tokenizer | vocab  | total tokens | tokens/word | tokens/1000 chars |
|-----------|--------|--------------|-------------|-------------------|
| char      | 69     | 1,115,394    | 4.24        | 1000.0            |
| word      | 13,335 | 262,927      | 1.00        | 235.7             |
| bpe-300   | 368    | 521,904      | 1.98        | 467.9             |
| bpe-1000  | 1,068  | 399,794      | 1.52        | 358.4             |
| bpe-3000  | 3,068  | 319,379      | 1.21        | 286.3             |

Observations (verified by running, not asserted):
- word level compresses best *per token* but the vocab is 36× larger than
  bpe-300 and grows with every unseen word.
- BPE compression improves monotonically with merge budget; it only approaches
  word-level density once vocab gets into the thousands.
- Production BPE (GPT-2: 50,257 tokens) is chosen for the *combination*:
  bounded vocab, zero `<UNK>` for in-alphabet text, morphological sharing.

## Known educational limitations (documented, not hidden)

- word tokenizer decode re-attaches closing punctuation ("barks ." -> "barks.")
  and cannot restore whitespace runs or split contractions.
- BPE decode turns `</w>` into a space and glues, so a *standalone* punctuation
  token between two words can be dropped on decode. Byte-level BPE (GPT-2)
  solves both by encoding the space/rare chars themselves.

## Run

```powershell
python tokenizer/demo.py            # everything + Tiny Shakespeare
pytest tests/test_bpe_tokenizer.py  # or: pytest tests/ for all
```
