"""Offline tests for the agriculture corpus pipeline's pure functions.

No network: cleaning, dedup, Gutenberg banner stripping, and the
document-level split are fully deterministic and testable locally.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_agriculture_corpus import (  # noqa: E402
    clean_text,
    dedupe_lines,
    gutenberg_body,
    split_documents,
)


def doc(id_: str, text: str) -> dict:
    return {"id": id_, "text": text}


# ---------------- clean_text ----------------

def test_clean_collapses_whitespace_and_drops_junk() -> None:
    raw = "  wheat   is   a  crop  \n\n\nab\nISBN 1234567890\nSoil holds water."
    out = clean_text(raw)
    assert out.splitlines() == ["wheat is a crop", "Soil holds water."]


def test_clean_cuts_wikipedia_citation_sections() -> None:
    raw = "Intro text here.\n\n== content ==\nWheat is planted in winter."
    out = clean_text("Facts about wheat.\nSee also\nA list of links.", cut_sections=True)
    assert out == "Facts about wheat."
    assert "links" not in out


def test_clean_cut_is_case_insensitive_and_word_exact() -> None:
    out = clean_text("Body.\nREFERENCES\nCitation line here.", cut_sections=True)
    assert out == "Body."
    # 'references' as part of a longer line must NOT cut
    out2 = clean_text("He cites references often.", cut_sections=True)
    assert out2 == "He cites references often."


# ---------------- gutenberg_body ----------------

def test_gutenberg_body_strips_banners() -> None:
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK A FARM BOOK ***\n"
        "\n"
        "Chapter 1. The wheat field.\n"
        "Chapter 2. Irrigation notes.\n"
        "\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK A FARM BOOK ***\n"
        "license text license text license text\n"
    )
    body = gutenberg_body(raw)
    assert "Chapter 1" in body and "Irrigation" in body
    assert "license text" not in body
    assert "START OF" not in body and "END OF" not in body


def test_gutenberg_body_keeps_everything_without_markers() -> None:
    raw = "some old book without modern markers"
    assert gutenberg_body(raw) == raw


# ---------------- dedupe_lines ----------------

def test_dedupe_removes_cross_doc_repeats_of_long_lines_only() -> None:
    long_line = "This is a long boilerplate line that repeats across documents."
    docs = [
        doc("a", f"Intro A.\n{long_line}\nShort."),
        doc("b", f"Intro B.\n{long_line}\nShort."),
    ]
    out = dedupe_lines(docs)
    assert long_line in out[0]["text"]                   # first doc keeps it
    assert long_line not in out[1]["text"]               # second doc loses it
    # short repeated lines survive
    assert "Short." in out[0]["text"] and "Short." in out[1]["text"]


def test_dedupe_keeps_first_occurrence_case_insensitively() -> None:
    line = "A Long Template Line That Should Deduplicate Properly."
    docs = [doc("a", line), doc("b", line.lower())]
    out = dedupe_lines(docs)
    assert out[0]["text"] == line
    assert out[1]["text"] == ""


# ---------------- split_documents ----------------

def many_docs(n: int = 20, size: int = 500) -> list[dict]:
    return [doc(f"d{i:02d}", f"document number {i} " + "x" * size) for i in range(n)]


def test_split_is_deterministic() -> None:
    a = split_documents(many_docs(), seed=7)
    b = split_documents(many_docs(), seed=7)
    assert [d["id"] for d in a[0]] == [d["id"] for d in b[0]]
    assert [d["id"] for d in a[1]] == [d["id"] for d in b[1]]


def test_split_fraction_approximate_and_no_leakage() -> None:
    docs = many_docs(20)
    train, val = split_documents(docs, val_fraction=0.1, seed=0)
    val_chars = sum(len(d["text"]) for d in val)
    total = sum(len(d["text"]) for d in docs)
    # document-level granularity: allow a loose window around 10%
    assert 0.0 <= val_chars / total <= 0.25
    ids_train = {d["id"] for d in train}
    ids_val = {d["id"] for d in val}
    assert ids_train.isdisjoint(ids_val)                 # no document in both
    assert ids_train | ids_val == {d["id"] for d in docs}


def test_split_never_produces_empty_side() -> None:
    two = [doc("a", "text one"), doc("b", "text two")]
    for frac in (0.1, 0.5, 0.9, 0.99):                   # both sides always exist
        train, val = split_documents(two, val_fraction=frac, seed=0)
        assert train and val
    tiny = [doc("only", "some text")]
    with pytest.raises(ValueError):                      # 1 doc cannot split
        split_documents(tiny, val_fraction=0.5, seed=0)
