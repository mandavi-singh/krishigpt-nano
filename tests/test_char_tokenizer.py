"""Tests for the character-level tokenizer."""
import pytest

from tokenizer import CharTokenizer
from tokenizer.special import UNK_ID

CORPUS = "the cat sat on the mat. the cat purred!"


@pytest.fixture(scope="module")
def tok() -> CharTokenizer:
    return CharTokenizer(CORPUS)


def test_vocab_layout(tok: CharTokenizer) -> None:
    # 4 special + distinct chars of the corpus
    assert tok.vocab_size == 4 + len(set(CORPUS))
    assert tok.id2token[0] == "<PAD>"
    assert tok.id2token[1] == "<UNK>"


def test_round_trip_exact(tok: CharTokenizer) -> None:
    for text in ["the cat", "the cat sat on the mat.", "!!!", "tthhee"]:
        assert tok.decode(tok.encode(text)) == text


def test_encode_shape_and_known_ids(tok: CharTokenizer) -> None:
    ids = tok.encode("the")
    assert isinstance(ids, list) and len(ids) == 3           # char-per-token
    assert all(isinstance(i, int) for i in ids)
    assert ids == [tok.token2id["t"], tok.token2id["h"], tok.token2id["e"]]


def test_unknown_character_maps_to_unk(tok: CharTokenizer) -> None:
    ids = tok.encode("zZ")                                   # not in corpus
    assert ids == [UNK_ID, UNK_ID]
    assert tok.decode(ids) == ""                             # skipped on decode


def test_special_tokens_wrapping(tok: CharTokenizer) -> None:
    ids = tok.encode("cat", add_bos=True, add_eos=True)
    assert ids[0] == 2 and ids[-1] == 3
    assert tok.decode(ids) == "cat"                          # special skipped
    assert "<BOS>" in tok.decode(ids, skip_special=False)


def test_empty_input(tok: CharTokenizer) -> None:
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_repeated_tokens(tok: CharTokenizer) -> None:
    ids = tok.encode("aaa")
    assert len(set(ids)) == 1 and len(ids) == 3


def test_deterministic_vocabulary() -> None:
    a = CharTokenizer(CORPUS)
    b = CharTokenizer(CORPUS)
    assert a.get_vocab() == b.get_vocab()
    assert a.encode("cat") == b.encode("cat")


def test_punctuation_are_tokens(tok: CharTokenizer) -> None:
    ids = tok.encode(".!")
    assert tok.decode(ids) == ".!"
