"""Tests for autoregressive generation (greedy + sampling + top-p + penalty)."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.generate import KrishiGenerator  # noqa: E402
from model.gpt import GPT, GPTConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from tokenizer.special import EOS_ID  # noqa: E402
from training.checkpoint import save_checkpoint  # noqa: E402


def make_gen(seed: int = 0, max_len: int = 64):
    """Tiny GPT over a toy corpus. 80 merges on a short repeating sentence
    collapse common words to few ids, keeping prompt encodings compact."""
    torch.manual_seed(seed)
    corpus = "the corn grows tall and the wheat grows yellow " * 30
    tok = BPETokenizer.train(corpus, num_merges=80)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, d_model=16, n_heads=2,
                          n_layers=1, max_len=max_len))
    return KrishiGenerator(model, tok)


# ------------------------------ shapes / decode ------------------------------

def test_greedy_shapes_and_roundtrip() -> None:
    g = make_gen()
    r = g.greedy("the corn grows", max_tokens=10)
    assert r.n_generated == len(r.ids) <= 10
    assert all(0 <= i < g.tok.vocab_size for i in r.ids)
    assert r.n_generated > 0 and isinstance(r.text, str)
    # decoding the generated ids reproduces the continuation text exactly
    assert g.tok.decode(r.ids) == r.text


def test_full_window_logit_shape() -> None:
    """The generator relies on the model returning (1, T, V); check a full
    max_len window forward has that shape."""
    g = make_gen(max_len=24)
    logits = g.model(torch.randint(0, g.tok.vocab_size, (1, g.max_len)))
    assert logits.shape == (1, g.max_len, g.tok.vocab_size)
    r = g.greedy("the corn", max_tokens=5)
    assert r.stop_reason in ("max_tokens", "eos", "context_full")


# ------------------------------- eos stopping -------------------------------

class OneShotEOS(torch.nn.Module):
    """Stub emitter: emits <EOS> on the first call, then any fixed id."""

    def __init__(self, vocab_size: int, max_len: int = 32):
        super().__init__()
        self.cfg = GPTConfig(vocab_size=vocab_size, max_len=max_len,
                             d_model=4, n_heads=2, n_layers=1)
        self.calls = 0

    def eval(self):
        return self

    def forward(self, x):
        self.calls += 1
        logits = torch.zeros(1, x.shape[1], self.cfg.vocab_size)
        # first emitted id is EOS; afterwards a neutral non-special id
        next_id = EOS_ID if self.calls == 1 else 4
        logits[0, -1, next_id] = 5.0
        return logits


def test_eos_stops_generation() -> None:
    tok = BPETokenizer.train("ab c cd de", num_merges=6)
    gen = KrishiGenerator(OneShotEOS(tok.vocab_size), tok)
    r = gen.greedy("ab", max_tokens=50)
    assert r.stop_reason == "eos"
    assert r.ids == [EOS_ID] and r.n_generated == 1
    # EOS is special: skipped by decode, so the text is empty
    assert r.text == ""


def test_greedy_is_deterministic() -> None:
    g = make_gen()
    r1 = g.greedy("the wheat grows", max_tokens=30)
    r2 = g.greedy("the wheat grows", max_tokens=30)
    assert r1.ids == r2.ids and r1.text == r2.text
    assert r1.stop_reason == r2.stop_reason
    assert r1.n_generated == 30 and r1.stop_reason == "max_tokens"


def test_greedy_prefix_consistency() -> None:
    """Appending greedy output to the prompt then re-encoding keeps the same
    prefix, and decoding prompt+ids recovers the continuation."""
    g = make_gen()
    prompt = "the corn grows"
    r = g.greedy(prompt, max_tokens=12)
    re_prompt = g.tok.encode(prompt)
    assert re_prompt == r.prompt_ids
    decoded = g.tok.decode(r.prompt_ids + r.ids)
    assert decoded.startswith(g.tok.decode(re_prompt))


# ---------------------------- context-window handling ----------------------------

def make_short_ctx_gen(max_len: int, seed: int = 0):
    torch.manual_seed(seed)
    tok = BPETokenizer.train("the corn grows tall and the wheat grows yellow " * 20,
                             num_merges=60)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, d_model=16, n_heads=2,
                          n_layers=1, max_len=max_len))
    return KrishiGenerator(model, tok)


def test_long_prompt_truncated_without_crash() -> None:
    g = make_short_ctx_gen(max_len=8)
    r = g.greedy("the corn grows tall and the wheat grows yellow on the hill",
                 max_tokens=4)
    assert len(r.prompt_ids) <= 8
    assert r.n_generated <= 4
    assert r.stop_reason in ("max_tokens", "eos", "context_full")
    assert isinstance(r.text, str)


def test_generation_never_exceeds_window() -> None:
    g = make_short_ctx_gen(max_len=12)
    r = g.greedy("the corn grows tall", max_tokens=50)
    # token budget exceeds the room the window offers -> stops at the limit
    assert len(r.prompt_ids) + r.n_generated <= 12
    assert r.stop_reason in ("context_full", "eos")


def test_full_window_prompt_generates_nothing() -> None:
    g = make_short_ctx_gen(max_len=4)
    prompt = "the corn grows tall and the wheat"   # certainly > 4 BPE ids
    r = g.greedy(prompt, max_tokens=10)
    assert r.n_generated == 0 and r.ids == []
    assert r.stop_reason == "context_full"


# ------------------------------- sampling -------------------------------

def test_sampling_ids_in_vocab_and_seeded_reproducible() -> None:
    g = make_gen()
    kwargs = dict(max_tokens=30, temperature=0.8, top_k=5, seed=11)
    r1 = g.sample("the corn grows", **kwargs)
    r2 = g.sample("the corn grows", **kwargs)
    assert r1.n_generated == 30
    assert all(0 <= i < g.tok.vocab_size for i in r1.ids)
    assert r1.ids == r2.ids                     # same seed -> same trajectory
    assert r1.ids != g.sample("the corn grows", **{**kwargs, "seed": 42}).ids


def test_sampling_top1_equals_greedy() -> None:
    g = make_gen()
    rg = g.greedy("the wheat grows", max_tokens=20)
    rs = g.sample("the wheat grows", max_tokens=20, temperature=0.7, top_k=1,
                  seed=3)
    assert rs.ids == rg.ids                     # top-1 sampling == argmax


def test_sampling_rejects_bad_args() -> None:
    g = make_gen()
    with pytest.raises(ValueError):
        g.sample("the corn", temperature=0.0)
    with pytest.raises(ValueError):
        g.sample("the corn", top_k=0)
    with pytest.raises(ValueError):
        g.sample("the corn", top_p=0.0)
    with pytest.raises(ValueError):
        g.sample("the corn", top_p=1.5)
    with pytest.raises(ValueError):
        g.sample("the corn", repetition_penalty=0.5)


# ------------------------------- top-p -------------------------------

def test_top_p_seeded_reproducible_and_in_vocab() -> None:
    g = make_gen()
    kwargs = dict(max_tokens=30, temperature=0.8, top_p=0.9, seed=11)
    r1 = g.sample("the corn grows", **kwargs)
    r2 = g.sample("the corn grows", **kwargs)
    assert r1.n_generated == 30
    assert all(0 <= i < g.tok.vocab_size for i in r1.ids)
    assert r1.ids == r2.ids
    assert r1.ids != g.sample("the corn grows", **{**kwargs, "seed": 42}).ids


def test_top_p_and_top_k_compose() -> None:
    g = make_gen()
    r = g.sample("the wheat grows", max_tokens=30, temperature=0.8,
                 top_k=40, top_p=0.9, seed=9)
    assert r.n_generated == 30
    assert all(0 <= i < g.tok.vocab_size for i in r.ids)


def test_nucleus_mask_matches_definition() -> None:
    """Hand-checked: keep the smallest prefix of the sorted tokens whose
    cumulative probability mass stays <= p, always including the argmax."""
    logits = torch.tensor([math.log(0.5), math.log(0.25), math.log(0.03125),
                           math.log(0.125), math.log(0.0625),
                           math.log(0.03125)])
    masked = KrishiGenerator._nucleus(logits.clone(), 0.9)
    # sorted descending: 0.5, 0.25, 0.125, 0.0625, ... -> cumsum 0.875 at rank3
    kept = {i for i in range(6) if not torch.isinf(masked[i])}
    assert kept == {0, 1, 3}
    # even a tiny p must keep the argmax so the distribution stays valid
    masked2 = KrishiGenerator._nucleus(logits.clone(), 1e-6)
    kept2 = {i for i in range(6) if not torch.isinf(masked2[i])}
    assert kept2 == {0}
    assert torch.softmax(masked2, dim=-1).sum().item() == pytest.approx(1.0)


def test_top_p_extreme_shrinks_to_argmax() -> None:
    g = make_gen()
    rg = g.greedy("the wheat grows", max_tokens=20)
    # with top_p tiny, the nucleus is {argmax}; multinomial then equals greedy
    rs = g.sample("the wheat grows", max_tokens=20, temperature=0.7,
                  top_p=1e-12, seed=3)
    assert rs.ids == rg.ids


# --------------------------- repetition penalty ---------------------------

def test_penalty_only_touches_seen_tokens() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, 3.0])
    out = KrishiGenerator._penalize(logits, [0, 3], 2.0)
    assert out[0].item() == 1.0                   # 2.0 / 2 positive -> divide
    assert out[3].item() == 1.5                   # 3.0 / 2
    assert out[1].item() == -1.0 and out[2].item() == 0.5  # untouched
    out2 = KrishiGenerator._penalize(logits, [1], 2.0)
    assert out2[1].item() == -2.0                 # negative -> multiply
    assert out2[0].item() == 2.0                  # untouched


def test_penalty_dedups_seen_ids() -> None:
    logits = torch.tensor([2.0, 1.0])
    out = KrishiGenerator._penalize(logits, [0, 0, 1, 1, 1], 4.0)
    assert out[0].item() == 0.5 and out[1].item() == 0.25


def test_repetition_penalty_breaks_greedy_loop() -> None:
    """Worst-case looper stub: always shouts the previous token. Without
    penalty greedy loops forever; with a strong penalty it must diversify."""

    class Echo(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.cfg = GPTConfig(vocab_size=vocab_size, max_len=32,
                                 d_model=4, n_heads=2, n_layers=1)

        def eval(self):
            return self

        def forward(self, x):
            logits = torch.zeros(1, x.shape[1], self.cfg.vocab_size)
            logits[0, -1, :] = 1.0                # uniform fallback
            logits[0, -1, int(x[0, -1])] = 10.0   # shout the previous token
            return logits

    tok = BPETokenizer.train("ab c cd de", num_merges=6)
    gen = KrishiGenerator(Echo(tok.vocab_size), tok)
    loop = gen.greedy("ab", max_tokens=8, repetition_penalty=1.0)
    assert len(set(loop.ids)) == 1                # pure repetition loop
    fixed = gen.greedy("ab", max_tokens=8, repetition_penalty=20.0)
    assert len(set(fixed.ids)) > 1                # penalty forces variety
    assert len(set(fixed.ids)) >= len(fixed.ids) - 1


# --------------------------- checkpoint loading ---------------------------

def test_from_checkpoint_roundtrip(tmp_path) -> None:
    g = make_gen(1)
    cfg = {"vocab_size": g.tok.vocab_size, "seq_len": g.max_len, "dropout": 0.0}
    opt = torch.optim.AdamW(g.model.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path / "best.pt", g.model, opt, step=5,
                           tokens_seen=50, best_val_loss=1.0, config=cfg,
                           lr=1e-3)
    tok_path = tmp_path / "tok.json"
    g.tok.save(tok_path)

    # tiny architecture is NOT the nano default, so pass the exact GPTConfig
    model_cfg = GPTConfig(vocab_size=g.tok.vocab_size, d_model=16, n_heads=2,
                          n_layers=1, max_len=g.max_len)
    g2 = KrishiGenerator.from_checkpoint(path, tok_path, cfg=model_cfg)
    assert g2.max_len == g.max_len
    assert g2.tok.vocab_size == g.tok.vocab_size
    for (n1, p1), (n2, p2) in zip(g.model.named_parameters(),
                                  g2.model.named_parameters()):
        assert torch.equal(p1, p2), n1
    # same weights -> same continuation
    assert g.greedy("the corn", 10).ids == g2.greedy("the corn", 10).ids


def test_from_checkpoint_defaults_to_nano_arch(tmp_path) -> None:
    """Omitting cfg builds the standard nano GPT; vocab/seq_len/dropout are
    taken from the saved config."""
    g = make_gen(3)
    payload_cfg = {"vocab_size": g.tok.vocab_size, "seq_len": g.max_len,
                   "dropout": 0.0}
    opt = torch.optim.AdamW(g.model.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path / "t.pt", g.model, opt, 2, 20, 1.0,
                           payload_cfg, 1e-3)
    tok_path = tmp_path / "tok.json"
    g.tok.save(tok_path)
    # weights are for the tiny model, but the default rebuild is the nano
    # architecture -> clear architecture ValueError, not a raw state_dict dump
    with pytest.raises(ValueError, match="architecture"):
        KrishiGenerator.from_checkpoint(path, tok_path)


def test_from_checkpoint_rejects_vocab_mismatch(tmp_path) -> None:
    g = make_gen(2)
    cfg = {"vocab_size": g.tok.vocab_size + 1, "seq_len": 24}
    opt = torch.optim.AdamW(g.model.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path / "bad.pt", g.model, opt, 1, 10, 1.0, cfg,
                           1e-3)
    tok_path = tmp_path / "tok.json"
    g.tok.save(tok_path)
    with pytest.raises(ValueError):
        KrishiGenerator.from_checkpoint(path, tok_path)
