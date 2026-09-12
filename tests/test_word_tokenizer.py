"""Tests for the word-level tokenizer."""
import pytest

from tokenizer import WordTokenizer
from tokenizer.special import UNK_ID
from tokenizer.word_tokenizer import word_tokenize

CORPUS = "The quick brown fox jumps over the lazy dog . The dog barks ."


@pytest.fixture(scope="module")
def tok() -> WordTokenizer:
    return WordTokenizer(CORPUS)


def test_splitting_rules() -> None:
    assert word_tokenize("Hello, world!") == ["Hello", ",", "world", "!"]
    assert word_tokenize("It's fine") == ["It", "'", "s", "fine"]
    assert word_tokenize("   spaced   out   ") == ["spaced", "out"]
    assert word_tokenize("") == []


def test_vocab_counts_specials(tok: WordTokenizer) -> None:
    distinct = set(word_tokenize(CORPUS))
    assert tok.vocab_size == 4 + len(distinct)


def test_round_trip_simple_sentence(tok: WordTokenizer) -> None:
    text = "The quick brown fox jumps over the lazy dog"
    assert tok.decode(tok.encode(text)) == text


def test_punctuation_reattaches(tok: WordTokenizer) -> None:
    # closing punctuation glues to the previous word ON DECODE by design:
    # encode("barks .") == [.., "barks", "."]  decode -> "barks."
    # (whitespace around punctuation is not preserved — a word-tokenizer
    # limitation; byte-level BPE solves this by encoding the space itself.)
    ids = tok.encode("The dog barks .")
    assert tok.decode(ids) == "The dog barks."
    assert tok.decode(ids, skip_special=False) == tok.decode(ids)


def test_unknown_word_with_min_freq() -> None:
    # 'zebra' appears once; min_freq=2 must exclude it from the vocab.
    tok = WordTokenizer("cat cat dog zebra", min_freq=2)
    ids = tok.encode("zebra cat")
    assert ids[0] == UNK_ID
    assert tok.decode(ids) == "cat"


def test_encode_case_sensitive(tok: WordTokenizer) -> None:
    # 'The' is in the corpus, 'horse' is not -> <UNK>
    assert tok.encode("horse") == [UNK_ID]
    assert tok.encode("The")[0] != UNK_ID


def test_special_tokens(tok: WordTokenizer) -> None:
    ids = tok.encode("dog", add_bos=True, add_eos=True)
    assert ids[0] == 2 and ids[-1] == 3
    assert tok.decode(ids) == "dog"


def test_empty_input(tok: WordTokenizer) -> None:
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_repeated_tokens(tok: WordTokenizer) -> None:
    ids = tok.encode("dog dog dog")
    assert len(set(ids)) == 1


def test_deterministic() -> None:
    a = WordTokenizer(CORPUS)
    b = WordTokenizer(CORPUS)
    assert a.get_vocab() == b.get_vocab()
    assert a.encode("The dog") == b.encode("The dog")
