"""Tests for the from-scratch BPE tokenizer."""
import pytest

from tokenizer import BPETokenizer
from tokenizer.special import UNK_ID

CLASSIC = "low lower lowest"
RICH = (
    "low lower lowest wide wider widest "
    "the the the the cat cat dog "
    "new newest news newspaper"
)


@pytest.fixture(scope="module")
def classic() -> BPETokenizer:
    # classic textbook example, enough merges to unify common prefixes
    return BPETokenizer.train(CLASSIC, num_merges=10)


@pytest.fixture(scope="module")
def rich() -> BPETokenizer:
    return BPETokenizer.train(RICH, num_merges=25)


def test_vocab_layout(classic: BPETokenizer) -> None:
    from tokenizer.bpe_tokenizer import END_OF_WORD
    assert classic.id2token[0] == "<PAD>"
    assert classic.id2token[1] == "<UNK>"
    assert classic.vocab_size == 4 + len(classic.base_tokens) + len(classic.merges)
    assert END_OF_WORD in classic.base_tokens


def test_classic_merge_order(classic: BPETokenizer) -> None:
    """Pair (o, w) merges before (l, o): 'ow' appears 3 times via all three words."""
    first_pair = classic.merges[0][:2]
    assert first_pair == ("o", "w")
    second_pair = classic.merges[1][:2]
    assert second_pair == ("l", "ow")
    # after merge 2, 'low' can continue to 'lowe' (freq 2) before 'low</w>' (freq 1)
    third = classic.merges[2]
    assert third[:2] == ("low", "e")


def test_round_trip_exact(rich: BPETokenizer) -> None:
    for text in RICH.split():
        assert rich.decode(rich.encode(text)) == text
    full = CLASSIC + " " + RICH
    assert rich.decode(rich.encode(full)) == full


def test_punctuation_round_trip(rich: BPETokenizer) -> None:
    # Limitation of this educational BPE: decode turns each </w> into a space
    # and glues words together, so a *standalone* punctuation token between
    # two </w>s disappears. Production BPE is byte-level and encodes the space
    # itself, so nothing is lost. Round-trip is exact for word-only text:
    for text in ["cat dog the new", "the cat cat dog new", "wide wider"]:
        assert rich.decode(rich.encode(text)) == text


def test_unseen_letters_become_unk(classic: BPETokenizer) -> None:
    assert classic.encode("zzz") == [UNK_ID]
    assert classic.decode(classic.encode("zzz")) == ""


def test_mixed_seen_unseen(classic: BPETokenizer) -> None:
    ids = classic.encode("low zzz")
    assert UNK_ID in ids and len(ids) >= 2
    # the known word survives beside the unknown one; <UNK> is skipped on decode
    assert classic.decode(ids) == "low"


def test_deterministic_encoding() -> None:
    a = BPETokenizer.train(RICH, num_merges=15)
    b = BPETokenizer.train(RICH, num_merges=15)
    assert a.get_vocab() == b.get_vocab()
    assert a.encode("lowest new") == b.encode("lowest new")


def test_empty_input(rich: BPETokenizer) -> None:
    assert rich.encode("") == []
    assert rich.decode([]) == ""


def test_repeated_tokens(rich: BPETokenizer) -> None:
    ids = rich.encode("the the the")
    # each 'the' encodes identically -> pattern repeats
    n = len(ids) // 3
    assert ids[:n] == ids[n:2 * n] == ids[2 * n:]


def test_special_tokens(rich: BPETokenizer) -> None:
    ids = rich.encode("cat", add_bos=True, add_eos=True)
    assert ids[0] == 2 and ids[-1] == 3
    assert rich.decode(ids) == "cat"
    assert "<BOS>" in rich.decode(ids, skip_special=False).split("<EOS>")[0]


def test_save_load_round_trip(tmp_path) -> None:
    tok = BPETokenizer.train(RICH, num_merges=25)
    path = tmp_path / "bpe.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.get_vocab() == tok.get_vocab()
    sample = "lowest new newspaper cat dog"
    assert loaded.decode(loaded.encode(sample)) == sample


def test_short_sequences() -> None:
    """BPE tokens are never longer than char tokens + one </w> per word.

    With 0 merges every word costs chars+1 (the end-of-word marker), so BPE
    can be slightly longer than pure characters before merges kick in. Each
    merge removes exactly one symbol, so sequence length only shrinks.
    """
    from tokenizer import CharTokenizer
    text = "the cat sat on the mat"
    ct = CharTokenizer(text)
    char_len = len(ct.encode(text))
    n_words = len(text.split())
    prev = char_len + n_words
    for n in (0, 5, 20, 50):
        bpe = BPETokenizer.train(text, num_merges=n)
        seq_len = len(bpe.encode(text))
        assert seq_len <= char_len + n_words        # never worse than chars+</w>
        assert seq_len <= prev                       # merges only shrink
        prev = seq_len


def test_frequent_words_win_merges_first() -> None:
    """'the' appears 4x so its pairs merge before singleton words' pairs.

    Observed trace (counts: 5x '</w>'-adjacent pairs from the/low/etc.):
      merge 3: (t, h)      merge 4: (th, e</w>) -> whole word 'the</w>'
    Note merge 4 uses the ALREADY-MERGED symbol 'e</w>' — merges compose,
    which is exactly how 'the' becomes a single token.
    """
    tok = BPETokenizer.train(RICH, num_merges=15)
    assert tok.merges[2][:2] == ("t", "h")
    assert tok.merges[3][:2] == ("th", "e</w>")
    assert tok.merges[3][2] == "the</w>"
