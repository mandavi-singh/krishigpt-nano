"""Grounded-answer retrieval for the KrishiGPT-nano chatbot (RAG mode).

The frozen v3 checkpoint cannot store facts (1.86M params, ~17M training
tokens — see final/FINAL_REPORT.md §5), but its TRAINING CORPUS contains the
knowledge verbatim. This module searches that corpus at question time and
returns the best-matching passages, so the chatbot can quote them directly
instead of generating from its compressed (lossy) memory.

Design (deliberately dependency-free, like the rest of the project):
  - BM25-lite ranking over sentence-window "passages", built once at first
    use and cached process-wide (the corpus is ~8 MB of text).
  - Index source = exactly the corpora the frozen v3 model trained on
    (v1+v2+v3 train; POOL definition mirrors api/app.py and
    final/final_eval.py) — grounded answers can never quote documents the
    model never saw, and licensing stays per-document clean.
  - Every hit carries its source document id (from the per-document
    provenance metadata) so answers can cite where a passage came from.
  - No stemmer, no embeddings, no external index — ~15k passages rank in
    milliseconds with plain token overlap scoring.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: exactly the pool the frozen v3 model trained on (mirrors api/app.py)
POOL_FILES = [ROOT / "data" / "processed" / "agri_train.txt",
              ROOT / "data" / "corpus_v2" / "agri_train_v2.txt",
              ROOT / "data" / "corpus_v3" / "agri_train_v3.txt"]

STOPWORDS = frozenset("""
a an the and or but if then than that this these those is are was were be
been being of in on at to for with by from as about into over after before
between during without within along across behind beyond plus under up down
out off again further once here there when where why how all any both each
few more most other some such no nor not only own same so too very can will
just should now do does did doing have has had having i we you he she it
they them his her its our their what which who whom whose
""".split())

TOKEN_RE = re.compile(r"[a-z]+")
MIN_TOKENS = 3            # passages shorter than this are not quotable
PASSAGE_SENTENCES = 2     # rank windows of N consecutive sentences


def _load_passages() -> tuple[list[str], list[str]]:
    """Sentence-pair passages from the training pool, each tagged with the
    document id it came from (via the per-document provenance metadata)."""
    # map each pool file's text to its doc ids: corpus files are built as
    # document texts joined by blank lines, and each doc has a meta file.
    doc_ids: dict[str, str] = {}          # first sentence -> doc id
    for meta_dir in ("data/processed", "data/corpus_v2", "data/corpus_v3"):
        d = ROOT / meta_dir
        if not d.exists():
            continue
        for m in d.glob("*.meta.json"):
            try:
                meta = json.loads(m.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            title = meta.get("title") or meta.get("id") or m.stem
            doc_ids[title.strip().lower()] = m.stem

    passages: list[str] = []
    sources: list[str] = []
    for path in POOL_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for block in text.split("\n\n"):
            block = " ".join(block.split())
            if not block:
                continue
            sents = [s.strip() for s in
                     re.split(r"(?<=[.!?])\s+", block) if s.strip()]
            for i in range(len(sents)):
                window = " ".join(sents[i:i + PASSAGE_SENTENCES])
                if len(TOKEN_RE.findall(window.lower())) >= MIN_TOKENS:
                    passages.append(window)
                    # best-effort source id: match doc title inside the doc's
                    # first line, else the corpus version
                    head = sents[0].lower()[:60]
                    src = next((v for k, v in doc_ids.items()
                                if k and k in head), path.stem)
                    sources.append(src)
    return passages, sources


class _Index:
    """Lazily-built BM25-lite index over the training-pool passages."""

    def __init__(self) -> None:
        self.passages: list[str] = []
        self.sources: list[str] = []
        self.postings: dict[str, list[int]] = {}
        self.doc_len: list[int] = []
        self.avg_len = 1.0
        self.n_docs = 0
        self.idf: dict[str, float] = {}
        self._built = False

    def build(self) -> None:
        self.passages, self.sources = _load_passages()
        self.n_docs = len(self.passages)
        for i, p in enumerate(self.passages):
            toks = [t for t in TOKEN_RE.findall(p.lower())
                    if t not in STOPWORDS]
            self.doc_len.append(max(1, len(toks)))
            for t in set(toks):
                self.postings.setdefault(t, []).append(i)
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 1.0
        for t, docs in self.postings.items():
            self.idf[t] = math.log(1 + (self.n_docs - len(docs) + 0.5)
                                   / (len(docs) + 0.5))
        self._built = True

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self._built:
            self.build()
        q_toks = [t for t in TOKEN_RE.findall(query.lower())
                  if t not in STOPWORDS]
        if not q_toks:
            return []
        k1, b = 1.5, 0.75
        scores: dict[int, float] = {}
        for t in q_toks:
            docs = self.postings.get(t)
            if not docs:
                continue
            idf = self.idf[t]
            for i in docs:
                dl = self.doc_len[i]
                tf = sum(1 for w in TOKEN_RE.findall(
                    self.passages[i].lower()) if w == t)
                denom = tf + k1 * (1 - b + b * dl / self.avg_len)
                scores[i] = scores.get(i, 0.0) + idf * tf * (k1 + 1) / denom
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [{"text": self.passages[i], "source": self.sources[i],
                 "score": round(s, 3)} for i, s in top]


_INDEX = _Index()


def search(query: str, k: int = 3) -> list[dict]:
    """Top-k training-corpus passages for a query (built once, then cached)."""
    return _INDEX.search(query, k)
