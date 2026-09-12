"""Tests for the KrishiGPT-nano chatbot API (api/app.py).

These tests run against the REAL frozen final checkpoint
(checkpoints/krishigpt_v3/best.pt): it is loaded ONCE by the app's lifespan
and cached process-wide, and every generated-token call reuses the same
KrishiGenerator. Generation calls keep max_tokens small to bound CPU runtime.
The validated decoding setting from final/final_eval.json (top-p 0.9) is the
default and is exercised throughout.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api.app as api_app                          # noqa: E402
from api.app import (ChatMessage, app,             # noqa: E402
                     build_prompt, get_generator, trim_answer)

AGRI_PROMPT = "What is crop rotation?"
STOP_REASONS = {"eos", "max_tokens", "context_full"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:                     # lifespan loads the model
        yield c


@pytest.fixture(scope="module")
def gen(client):                                   # noqa: ARG001
    return get_generator()


# ------------------------------------------------------------------- health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["loaded"] is True
    assert j["model"] == "KrishiGPT-nano"
    assert j["params"] == 1_860_224
    assert j["vocab_size"] == 5237
    assert j["max_context_tokens"] == 128
    assert j["checkpoint"] == "checkpoints/krishigpt_v3/best.pt"
    assert j["checkpoint_sha256"].startswith("137cc83d")
    assert "experimental" in j["disclaimer"]
    assert j["default_decoding"]["top_p"] == 0.9


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "KrishiGPT-nano" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


# --------------------------------------------------------------------- chat
def test_chat_simple_message(client):
    r = client.post("/chat", json={"message": AGRI_PROMPT,
                                   "seed": 0, "max_tokens": 32})
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j["response"], str) and j["response"].strip()
    assert "must NOT be treated as professional" in j["disclaimer"]
    assert j["stop_reason"] in STOP_REASONS
    assert 1 <= j["n_generated"] <= 32
    assert j["prompt_tokens"] >= 1
    assert j["dropped_turns"] == 0
    assert j["decoding"]["top_p"] == 0.9


def test_chat_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 400
    assert client.post("/chat", json={"message": "   "}).status_code == 400
    assert client.post("/chat", json={"messages": []}).status_code == 400
    assert client.post("/chat", json={}).status_code == 400
    r = client.post("/chat", json={"messages": [
        {"role": "user", "content": "What is loam?"},
        {"role": "assistant", "content": "   "}]})
    assert r.status_code == 400


def test_chat_malformed_requests(client):
    bad_json = client.post("/chat", content=b"{this is not json",
                           headers={"Content-Type": "application/json"})
    assert bad_json.status_code == 422
    cases = [
        {"message": ["not", "a", "string"]},
        {"message": "ok", "max_tokens": 0},
        {"message": "ok", "max_tokens": 10_000},
        {"message": "ok", "top_p": 1.5},
        {"message": "ok", "temperature": 0},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "user"}]},
    ]
    for body in cases:
        assert client.post("/chat", json=body).status_code == 422, body


def test_chat_deterministic_with_seed(client):
    body = {"message": "Soil moisture affects", "seed": 11, "max_tokens": 32}
    r1 = client.post("/chat", json=body)
    r2 = client.post("/chat", json=body)
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["response"] == r2.json()["response"]


def test_chat_meta_questions_get_honest_canned_replies(client):
    for q in ["what can u do for me", "What can you do?",
              "who are you", "can you give me some advice"]:
        r = client.post("/chat", json={"message": q})
        assert r.status_code == 200, q
        j = r.json()
        assert j["stop_reason"] == "meta_question"
        assert "next-token" in j["response"].lower() or \
               "advice" in j["response"].lower()
        assert j["n_generated"] == 0


def test_chat_non_english_gets_honest_reply(client):
    r = client.post("/chat",
                    json={"message": "crop me kitni type ki diease hoti h"})
    assert r.status_code == 200
    j = r.json()
    assert j["stop_reason"] == "non_english"
    assert "English" in j["response"]
    assert j["n_generated"] == 0


def test_chat_conversation_history(client):
    r = client.post("/chat", json={
        "messages": [
            {"role": "user", "content": "What is soil erosion?"},
            {"role": "assistant",
             "content": "Soil erosion is the removal of the top layer of "
                        "soil by water and wind."},
            {"role": "user", "content": "How can farmers prevent it?"},
        ],
        "seed": 0,
        "max_tokens": 32,
    })
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j["response"], str) and j["response"].strip()
    assert j["dropped_turns"] >= 0
    assert j["prompt_tokens"] >= 1


# -------------------------------------------------------- prompt building
def test_build_prompt_keeps_full_history_when_it_fits(gen):
    msgs = [ChatMessage(role="user", content="What is soil erosion?"),
            ChatMessage(role="assistant",
                        content="Soil erosion is the removal of topsoil by "
                                "water and wind."),
            ChatMessage(role="user", content="How can it be prevented?")]
    prompt, dropped, n_ids, tail_seed = build_prompt(gen, msgs, token_budget=500)
    assert dropped == 0
    assert "Soil erosion is" in prompt              # question -> prose seed
    assert "removal of topsoil" in prompt           # prior answer kept as prose
    assert "prevented" in prompt                    # newest question survives
    assert "Q:" not in prompt and "A:" not in prompt   # no chat markers
    assert tail_seed.strip().endswith("is")            # declarative seed form
    assert n_ids == len(gen.tok.encode(prompt))


def test_build_prompt_truncates_history_to_context(gen):
    one = [ChatMessage(role="user",
                       content="Tell me about wheat farming in India. " * 4),
           ChatMessage(role="assistant",
                       content="Wheat is a major cereal crop. " * 8)]
    msgs = one * 6 + [ChatMessage(role="user", content="What is irrigation?")]
    prompt, dropped, n_ids, tail_seed = build_prompt(gen, msgs, token_budget=60)
    assert n_ids <= 60
    assert len(gen.tok.encode(prompt)) <= 60
    assert dropped > 0
    assert "Irrigation is" in prompt               # newest turn as prose seed


def test_trim_answer_stops_at_fake_question():
    assert trim_answer("some answer text\nQ: a fake next question") == \
        "some answer text"
    assert trim_answer("plain answer") == "plain answer"


# -------------------------------------------- model loading + output shape
def test_model_loaded_exactly_once(client, gen):
    assert get_generator() is gen                   # cached singleton
    client.post("/chat", json={"message": "Rice is grown in",
                               "seed": 3, "max_tokens": 16})
    assert get_generator() is gen
    assert api_app._load_count == 1


def test_checkpoint_loads_with_expected_shape(gen):
    model = gen.model
    assert sum(p.numel() for p in model.parameters()) == 1_860_224
    assert model.cfg.n_layers == 6
    assert model.cfg.d_model == 128
    assert model.cfg.n_heads == 4
    assert model.cfg.vocab_size == 5237
    assert model.cfg.max_len == 128
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_generation_result_type_and_shape(gen):
    r = gen.sample("Rice is grown in", max_tokens=16, top_p=0.9, seed=0)
    assert isinstance(r.text, str) and r.text
    assert isinstance(r.ids, list)
    assert all(isinstance(i, int) for i in r.ids)
    assert r.n_generated == len(r.ids) <= 16
    assert r.stop_reason in STOP_REASONS
    r2 = gen.sample("Rice is grown in", max_tokens=16, top_p=0.9, seed=0)
    assert r2.ids == r.ids                          # seeded => reproducible
