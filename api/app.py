"""KrishiGPT-nano chatbot API.

A thin FastAPI layer over the FROZEN final checkpoint
(`checkpoints/krishigpt_v3/best.pt`, SHA-256 pinned in
`final/FREEZE_MANIFEST.json`). The checkpoint and tokenizer are loaded ONCE
at server startup and cached process-wide; every /chat request reuses the
same `KrishiGenerator` from `model/generate.py` — no per-request model
loading and no duplicated generation logic.

Honest framing: the underlying model is a 1.86M-parameter next-token
predictor with a 128-token context window. It is NOT instruction-tuned, has
NO memory and NO internet access. It was pretrained on agriculture
Wikipedia/Gutenberg PROSE, so it performs far better continuing natural
sentences than answering chat-marked questions: questions are therefore
rewritten into declarative seed prompts ("What is soil?" -> "Soil is ...")
before generation, and common greetings get honest canned replies.
Generation uses the validated top-p 0.9 decoding from
final/final_eval.json (greedy decoding is degenerate for this model and is
never the default).

Run:  python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import re
import string
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.generate import KrishiGenerator  # noqa: E402

CKPT = ROOT / "checkpoints" / "krishigpt_v3" / "best.pt"
TOKENIZER = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_CONTEXT = 128
DEFAULT_MAX_TOKENS = 48
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9            # best validated decoding (final/final_eval.json)
N_CANDIDATES = 4               # best-of-N selection, scored by corpus attestation
DEGENERATE_OPENINGS = ("done with", "required with", "such as the", "the term")

POOL_FILES = [ROOT / "data" / "processed" / "agri_train.txt",
              ROOT / "data" / "processed" / "agri_valid.txt",
              ROOT / "data" / "corpus_v2" / "agri_train_v2.txt",
              ROOT / "data" / "corpus_v2" / "agri_valid_v2.txt",
              ROOT / "data" / "corpus_v3" / "agri_train_v3.txt"]

NON_ENGLISH_REPLY = (
    "Sorry — I only understand simple English. My training corpus is "
    "English Wikipedia and public-domain agriculture books, so Hindi/"
    "Hinglish words are unknown to me and I would only produce nonsense. "
    "Please ask in English, for example: \"What are the types of crop "
    "diseases?\""
)

DISCLAIMER = (
    "KrishiGPT-nano is an experimental 1.86M-parameter agriculture language "
    "model. Its outputs are generated text that can be fluent and wrong, may "
    "contain fabricated words, and must NOT be treated as professional "
    "agricultural advice."
)
MEMORY_NOTE = (
    "The model has no memory: history is flattened into one prompt and "
    "truncated to fit the 128-token context window."
)


# --------------------------------------------------------------- singleton
_generator: KrishiGenerator | None = None
_gen_lock = threading.Lock()
_load_count = 0
_load_error: str | None = None
_corpus_words: frozenset[str] | None = None


def get_corpus_words() -> frozenset[str]:
    """Lowercased word set of the model's own training corpora — used to
    score candidate generations by attestation (the project's fabrication
    metric), exactly the pool definition of final/final_eval.py."""
    global _corpus_words
    if _corpus_words is None:
        words: set[str] = set()
        for f in POOL_FILES:
            words.update(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",
                                    f.read_text(encoding="utf-8").lower()))
        _corpus_words = frozenset(words)
    return _corpus_words


def get_generator() -> KrishiGenerator:
    """Load the frozen checkpoint exactly once per process, then cache it."""
    global _generator, _load_count, _load_error
    if _generator is None:
        with _gen_lock:
            if _generator is None:
                if not CKPT.exists():
                    _load_error = f"frozen checkpoint missing: {CKPT}"
                    raise FileNotFoundError(_load_error)
                try:
                    _generator = KrishiGenerator.from_checkpoint(CKPT, TOKENIZER)
                    _load_count += 1
                except Exception as exc:
                    _load_error = f"checkpoint load failed: {exc}"
                    raise
    return _generator


# ----------------------------------------------------------- prompt building
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=MAX_CONTEXT)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0.0, le=5.0)
    top_p: float | None = Field(default=DEFAULT_TOP_P, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = None


class ChatResponse(BaseModel):
    response: str
    disclaimer: str
    stop_reason: str
    n_generated: int
    prompt_tokens: int
    dropped_turns: int
    seed: int | None
    decoding: dict

_FREQ: dict[str, int] | None = None
_FREQ_LIST: list[str] | None = None


def _word_freqs() -> tuple[dict[str, int], list[str]]:
    """Word frequencies over the model's own corpora — used for (a) typo
    correction toward words the model actually knows, (b) seed templates."""
    global _FREQ, _FREQ_LIST
    if _FREQ is None:
        from collections import Counter
        cnt: Counter[str] = Counter()
        for f in POOL_FILES:
            cnt.update(re.findall(r"[a-z]+",
                                  f.read_text(encoding="utf-8").lower()))
        _FREQ = dict(cnt)
        _FREQ_LIST = [w for w, c in cnt.most_common() if c >= 50 and len(w) > 4]
    return _FREQ, _FREQ_LIST or []


def _fix_typos(text: str) -> str:
    """Correct spelling toward corpus words the model actually knows.

    Only replaces words the corpus NEVER contains (like 'diease') — common
    words are left alone. Uses difflib close-match on frequent corpus words,
    picking the most frequent candidate."""
    import difflib
    freq, candidates = _word_freqs()
    out = []
    for tok in text.split():
        stripped = tok.strip(string.punctuation)
        low = stripped.lower()
        if len(low) > 4 and low not in freq:
            close = difflib.get_close_matches(low, candidates, n=1, cutoff=0.75)
            if close:
                fixed = close[0]
                if stripped[0].isupper():
                    fixed = fixed[0].upper() + fixed[1:]
                out.append(fixed)
                continue
        out.append(tok)
    return " ".join(out)


_PLURAL_EXCEPTIONS_SINGULAR = {
    "series", "species", "physics", "economics", "politics", "news"}
_PLURAL_EXCEPTIONS_PLURAL = {
    "bacteria", "data", "crops", "pests", "weeds", "seeds", "tools",
    "grains", "plants", "trees", "roots", "leaves"}


def _verb_for(subject: str) -> str:
    """Pick 'is' or 'are' for a subject phrase (simple heuristic tuned to
    the agriculture domain)."""
    words = subject.lower().split()
    if not words:
        return "is"
    if words[0] in _PLURAL_EXCEPTIONS_SINGULAR:
        return "is"
    if words[0] in _PLURAL_EXCEPTIONS_PLURAL:
        return "are"
    return "are" if words[0].endswith("s") and not words[0].endswith("ss") \
        else "is"


def _to_seed(question: str) -> str:
    """Rewrite a user question into a declarative PROSE seed prompt.

    The model was pretrained on Wikipedia/Gutenberg prose and has never seen
    chat markers, so "Q: ... A:" prompts produce garbage. Turning questions
    into natural sentence starts ("What is soil?" -> "Soil is") conditions the
    model on the distribution it actually learned. Questions also get
    spelling fixed toward corpus words ("diease" -> "disease") and the seed
    verb is chosen for subject agreement ("Pests are", not "Pests is").
    """
    s = _fix_typos(" ".join(question.strip().split()))
    low = s.lower().rstrip("?.! ")

    m = re.match(r"^(what|whats|what's|wat)\s+(is|are|was|were)\s+(.+)$", low)
    if m:
        subj = m.group(3).strip()
        verb = m.group(2)
        if verb in ("is", "are"):
            words = subj.split()
            if words and words[0].lower() in ("the", "a", "an"):
                verb = _verb_for(" ".join(words[1:]))  # head noun after article
            else:
                verb = _verb_for(subj)
        return (subj[0].upper() + subj[1:]) + " " + verb
    m = re.match(r"^what\s+(?:is|are)\s+the\s+(?:role|purpose|use|function)\s+of\s+(.+)$", low)
    if m:
        subj = m.group(1)
        return "The role of " + subj + " is"
    m = re.match(r"^how\s+does\s+(.+?)\s+(affect|help|grow|work|improve|reduce"
                 r"|increase|decrease|damage|prevent|control|benefit|cause"
                 r"|spread|survive|respond|vary)\s+(.*)$", low)
    if m:
        subj, verb, rest = m.groups()
        return (subj[0].upper() + subj[1:]) + " " + verb + "s " + rest
    m = re.match(r"^how\s+does\s+(.+?)\s+([a-z]+)\s+(.+)$", low)
    if m:
        subj, verb, rest = m.groups()
        if re.search(r"(?:s|x|z|ch|sh)$", verb):
            v = verb + "es"
        elif verb.endswith("y") and not verb.endswith(("ay", "ey", "oy", "uy")):
            v = verb[:-1] + "ies"
        else:
            v = verb + "s"
        return (subj[0].upper() + subj[1:]) + " " + v + " " + rest
    m = re.match(r"^how\s+(?:do|can)\s+(?:i|we|farmers|you)\s+(.+)$", low)
    if m:
        return "To " + m.group(1)
    m = re.match(r"^how\s+to\s+(.+)$", low)
    if m:
        return "To " + m.group(1)
    m = re.match(r"^(?:tell me about|describe|define|explain)\s+(.+)$", low)
    if m:
        subj = m.group(1)
        return (subj[0].upper() + subj[1:]) + " " + _verb_for(subj)
    m = re.match(r"^why\s+(?:is|are|do|does)\s+(.+)$", low)
    if m:
        return s[0].upper() + s[1:].rstrip("?") + " because"
    return s.rstrip("?.! ") + " " + (_verb_for(s) if s else "is") if s else s


def _is_greeting(text: str) -> str | None:
    low = " ".join(text.strip().lower().split()).rstrip("!.? ")
    greetings = {"hi", "hello", "hey", "namaste", "hii", "hello!", "yo"}
    if low in greetings:
        return ("Hello! I am KrishiGPT-nano, an experimental 1.86M-parameter "
                "agriculture language model (not a professional advisor). Ask "
                "me about crops, soil, irrigation, fertilizers, pests or "
                "farming practices — e.g. \"What is crop rotation?\"")
    return None


def _is_meta_question(text: str) -> str | None:
    """Honest canned replies for questions the MODEL cannot answer.

    Capability/identity/permission questions have nothing in the training
    corpus (the model has never read anything about itself or about chat
    etiquette), so generating from them produces garbage. These get honest,
    hand-written answers instead — same principle as the greeting.
    """
    low = " ".join(text.strip().lower().split()).rstrip("?.! ")
    if not low:
        return None
    identity = {"who are you", "what are you", "what is your name",
                "tell me about yourself", "who made you", "who created you",
                "who trained you", "who built you", "what are you capable of"}
    if low in identity or re.match(
            r"^(?:who|what)\s+(?:made|created|trained|built)\s+(?:you|u)", low):
        return ("I am KrishiGPT-nano, a 1.86M-parameter agriculture language "
                "model built from scratch (tokenizer, architecture, "
                "training) as a college project. I was trained on English "
                "Wikipedia agriculture articles and public-domain farming "
                "books. I am a next-token predictor, NOT a knowledge "
                "assistant — details in final/MODEL_CARD.md.")
    capability = re.match(
        r"^what\s+(?:can|could|do|does)\s+(?:u|you)\s+(?:do|say|know|tell"
        r"|answer|help)(?:\s+(?:me|us|with\s+.+))?$", low)
    if capability or low in {"what can you do", "what can u do",
                             "what can you do for me", "what can u do for me",
                             "can you help me", "help", "help me",
                             "what do you know", "what do you do"}:
        return ("I can generate short English text continuations about "
                "agriculture topics (crops, soil, irrigation, fertilizers, "
                "pests, farming practices). Important: I am a tiny "
                "experimental next-token predictor, NOT a knowledge assistant "
                "— I do not reliably answer questions, I just continue text, "
                "and my continuations often contain made-up words and wrong "
                "grammar. I have no data about you, no internet access, and I "
                "cannot give personal farming advice. Try a topic prompt "
                "like \"What is crop rotation?\" and judge the output "
                "critically.")
    advice = re.match(
        r"^(?:can|should|would)\s+(?:you|u|i|we)\s+(?:give|recommend|suggest"
        r"|tell)\s+(?:me\s+)?(?:some\s+|any\s+)?(?:advice|suggestion"
        r"|recommendation)", low)
    if advice:
        return ("I cannot give advice — I am a 1.86M-parameter text "
                "continuation model with no factual grounding. For real "
                "agricultural decisions, consult a qualified agronomist or "
                "your local agricultural extension office. You can still ask "
                "me to continue topic sentences, e.g. \"Soil erosion is\".")
    return None


def _render_turns(turns: list[ChatMessage]) -> str:
    """Flatten history into ONE natural-prose prompt (no chat markers).

    Prior assistant answers become plain 'X is <generated text>.' sentences
    and the newest question becomes a declarative seed; the whole prompt is
    what the model continues.
    """
    parts = []
    for m in turns:
        text = " ".join(m.content.split())
        if m.role == "user":
            parts.append(_to_seed(text))
        else:
            parts.append(text)
    return " ".join(parts)


def build_prompt(gen: KrishiGenerator, messages: list[ChatMessage],
                 token_budget: int) -> tuple[str, int, int, str]:
    """Flatten history into ONE bounded natural-prose prompt.

    The model's real context is 128 tokens, so history cannot be unbounded:
    keep the newest turns, drop oldest ones until the prompt fits
    `token_budget` ids; if a single remaining turn is still too long, drop
    its leading words (keep the newest text). Returns
    (prompt, dropped_turns, n_prompt_ids, tail_seed) where tail_seed is the
    declarative seed of the newest user question (to prefix the reply).
    """
    turns = [m for m in messages if m.content.strip()]
    if not turns:
        raise ValueError("no message content to prompt with")
    tail = next((m for m in reversed(turns) if m.role == "user"), turns[-1])
    tail_seed = _to_seed(tail.content)
    start, dropped = 0, 0
    while True:
        prompt = _render_turns(turns[start:])
        n_ids = len(gen.tok.encode(prompt))
        if n_ids <= token_budget:
            return prompt, dropped, n_ids, tail_seed
        if start < len(turns) - 1:
            start += 1
            dropped += 1
            continue
        words = turns[start].content.split()
        if len(words) <= 1:
            return prompt, dropped, n_ids, tail_seed  # window-truncates
        turns[start] = ChatMessage(role=turns[start].role,
                                   content=" ".join(words[1:]))


def trim_answer(text: str) -> str:
    """Keep only coherent prose: stop at fake next questions or wiki section
    markers ('Q:', '= ='), collapse immediate word repeats ('most most'),
    then keep the first complete sentences."""
    kept = []
    for line in text.split("\n"):
        if line.lstrip().startswith("Q:") or " = = " in line:
            break
        kept.append(line)
    t = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if not t:
        return t
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.I)   # most most -> most
    sentences = re.split(r"(?<=[.!?])\s+", t)
    complete = [s.strip() for s in sentences if s.strip()
                and s.strip()[-1] in ".!?:\"'"]
    if not complete:
        return sentences[0].strip()
    out, n = [], 0
    for s in complete[:3]:
        out.append(s)
        n += len(s)
        if n >= 120:
            break
    return " ".join(out)


def _candidate_score(text: str, corpus_words: frozenset[str]) -> float:
    """Rank candidates by corpus attestation + word diversity, penalizing the
    degenerate Wikipedia-style openings this model over-produces and any
    residual immediate word-repeats."""
    if not text:
        return -9.0
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return -9.0
    attested = sum(1 for w in words if w in corpus_words) / len(words)
    diversity = len(set(words)) / len(words)
    penalty = 0.25 if text.lower().startswith(DEGENERATE_OPENINGS) else 0.0
    immediate_repeat = len(words) - len(set(zip(words, words[1:])))
    return attested + 0.15 * diversity - penalty - 0.1 * immediate_repeat


def _looks_non_english(text: str) -> bool:
    """Detect Hindi/Hinglish input the model has no vocabulary for.

    Rule: Devanagari characters always qualify; otherwise count common
    Hinglish function/verb words (many are not English words at all, so even
    one distinctive hit among few words is enough signal).
    """
    if re.search(r"[\u0900-\u097f]", text):        # Devanagari script
        return True
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]
    if not words:
        return False
    strong = {  # essentially never English
        "kya", "kaise", "kyu", "kyun", "kahan", "kab", "kabhi", "ka", "ki",
        "ke", "ko", "se", "me", "mein", "hai", "hain", "ho", "hoti", "hota",
        "hote", "karta", "karti", "kar", "kro", "kare", "karna", "karke",
        "kitna", "kitni", "kitne", "jaldi", "bata", "batao", "bataiya",
        "chahiye", "mera", "meri", "apna", "apni", "tum", "tumhe", "tumse",
        "aap", "aapko", "hoga", "hona", "honi", "hu", "hoon", "tha", "thi",
        "the", "na", "nhi", "nahi", "haan", "acha", "accha", "thik", "theek",
        "konsa", "konsi", "kaisa", "kaisi", "sab", "sabhi", "kuch", "bhi",
        "wala", "wali",         "lagta", "dikhta", "milta", "aata", "aati", "jaati",
        "janta", "janti", "samajh", "samajhte", "samjhe", "batau", "bataiye",
        "pata", "yaar", "bhai", "did", "koi", "ise", "uska", "iska",
        "karu", "karun", "karungi", "sikha", "sikho", "dikhao", "bata",
        "banata", "banti", "lagti", "rakhta", "rakhti", "chahta", "chahti",
        "kaun", "sa", "si", "karo", "baat", "baatein", "hindi", "bhasha",
        "language", "nam", "naam", "tumhara", "tumhara", "liye", "sakte",
        "sakta", "sakti", "mujhe", "hum", "ham", "hamara", "dono", "iska",
    }
    soft = {  # English-valid but frequent in Hinglish questions
        "type", "types", "bolo",
    }
    s_hits = sum(1 for w in words if w in strong)
    w_hits = sum(1 for w in words if w in soft)
    distinctive = s_hits + w_hits
    if s_hits >= 2:
        return True
    if s_hits == 1 and len(words) <= 5 and w_hits:
        return True
    return distinctive / len(words) > 0.34


# -------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        gen = get_generator()
        n = sum(p.numel() for p in gen.model.parameters())
        print(f"KrishiGPT-nano loaded once at startup: {CKPT.name} "
              f"({n:,} params, vocab {gen.model.cfg.vocab_size})")
    except Exception as exc:                     # stay up, report via /health
        print(f"WARNING: model not loaded at startup: {exc}")
    yield


app = FastAPI(
    title="KrishiGPT-nano chatbot API",
    description=DISCLAIMER + " " + MEMORY_NOTE,
    version="1.0-final",
    lifespan=lifespan,
)
_generate_lock = threading.Lock()


@app.get("/health")
def health() -> dict:
    gen = _generator
    info = {
        "status": "ok" if gen is not None else "degraded",
        "model": "KrishiGPT-nano",
        "checkpoint": CKPT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256":
            "137cc83da53dd873cbab22c82e7344c7c84882d79b37017a37c765ff38e14536",
        "loaded": gen is not None,
        "load_error": _load_error,
        "max_context_tokens": MAX_CONTEXT,
        "default_decoding": {"strategy": "top-p (nucleus)",
                             "top_p": DEFAULT_TOP_P,
                             "temperature": DEFAULT_TEMPERATURE,
                             "greedy": "never (degenerate for this model)"},
        "disclaimer": DISCLAIMER,
        "memory_note": MEMORY_NOTE,
    }
    if gen is not None:
        info["params"] = sum(p.numel() for p in gen.model.parameters())
        info["vocab_size"] = gen.model.cfg.vocab_size
    return info


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if req.messages is not None:
        messages = list(req.messages)
    elif req.message is not None:
        messages = [ChatMessage(role="user", content=req.message)]
    else:
        raise HTTPException(status_code=400,
                            detail="provide 'message' or 'messages'")
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if not messages[-1].content.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    greeting = _is_greeting(messages[-1].content)
    if greeting is not None and len(messages) == 1:
        return ChatResponse(
            response=greeting,
            disclaimer=DISCLAIMER,
            stop_reason="greeting",
            n_generated=0,
            prompt_tokens=0,
            dropped_turns=0,
            seed=req.seed,
            decoding={"top_p": req.top_p, "top_k": req.top_k,
                      "temperature": req.temperature},
        )

    meta = _is_meta_question(messages[-1].content)
    if meta is not None:
        return ChatResponse(
            response=meta,
            disclaimer=DISCLAIMER,
            stop_reason="meta_question",
            n_generated=0,
            prompt_tokens=0,
            dropped_turns=0,
            seed=req.seed,
            decoding={"top_p": req.top_p, "top_k": req.top_k,
                      "temperature": req.temperature},
        )

    if _looks_non_english(messages[-1].content):
        return ChatResponse(
            response=NON_ENGLISH_REPLY,
            disclaimer=DISCLAIMER,
            stop_reason="non_english",
            n_generated=0,
            prompt_tokens=0,
            dropped_turns=0,
            seed=req.seed,
            decoding={"top_p": req.top_p, "top_k": req.top_k,
                      "temperature": req.temperature},
        )

    try:
        gen = get_generator()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"model unavailable: {exc}")

    max_new = min(req.max_tokens, MAX_CONTEXT - 2)
    token_budget = max(8, MAX_CONTEXT - max_new - 1)
    try:
        prompt, dropped, n_ids, tail_seed = build_prompt(gen, messages,
                                                         token_budget)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    max_new = min(max_new, MAX_CONTEXT - n_ids - 1)

    try:
        with _generate_lock:
            base_seed = req.seed if req.seed is not None else 0
            candidates = []
            for i in range(N_CANDIDATES):
                r = gen.sample(prompt, max_tokens=max_new,
                               temperature=req.temperature, top_k=req.top_k,
                               top_p=req.top_p, seed=base_seed + i)
                text = trim_answer(r.text)
                if text:
                    candidates.append((text, r))
            if not candidates:
                result = gen.sample(prompt, max_tokens=max_new,
                                    temperature=req.temperature,
                                    top_k=req.top_k, top_p=req.top_p,
                                    seed=base_seed)
                candidates = [(trim_answer(result.text) or
                               result.text.strip(), result)]
    except ValueError as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid decoding parameter: {exc}")

    corpus_words = get_corpus_words()
    text, result = max(candidates,
                       key=lambda c: _candidate_score(c[0], corpus_words))
    if text:
        low = text.lower()
        seed_low = tail_seed.lower()
        starts_like_seed = (low.startswith(seed_low) or
                            (len(seed_low) > 4 and
                             low.startswith(seed_low[:len(seed_low) - 3])))
        if not starts_like_seed:
            text = tail_seed + " " + text      # restore the sentence start
    else:
        text = tail_seed or ("(the model produced no output for this prompt — "
                             "try rephrasing, or note this as a known "
                             "limitation)")
    return ChatResponse(
        response=text,
        disclaimer=DISCLAIMER,
        stop_reason=result.stop_reason,
        n_generated=result.n_generated,
        prompt_tokens=n_ids,
        dropped_turns=dropped,
        seed=req.seed,
        decoding={"top_p": req.top_p, "top_k": req.top_k,
                  "temperature": req.temperature},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
