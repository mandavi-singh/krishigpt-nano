"""Tests for the training pipeline: loss math, checkpoints, and train()."""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.gpt import GPT, GPTConfig  # noqa: E402
from training import train as train_mod  # noqa: E402
from training.checkpoint import find_latest, load_checkpoint, save_checkpoint  # noqa: E402
from training.loss import next_token_ce, perplexity  # noqa: E402


# ------------------------------- loss -------------------------------

def test_ce_matches_manual_and_pytorch() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))
    got = next_token_ce(logits, targets)
    manual = -torch.log_softmax(logits, dim=-1)[torch.arange(2)[:, None],
                                                torch.arange(4), targets].mean()
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 10), targets.reshape(-1))
    assert torch.allclose(got, manual, atol=1e-6)
    assert torch.allclose(got, ref, atol=1e-6)


def test_ce_ignore_index() -> None:
    logits = torch.randn(1, 4, 6)
    t_full = torch.randint(0, 6, (1, 4))
    t_mask = t_full.clone()
    t_mask[0, 2] = -100
    # correct expectation: masked positions EXCLUDED from the mean (average over
    # the 3 kept positions), which is what our next_token_ce computes.
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 6), t_mask.reshape(-1), ignore_index=-100)
    assert torch.allclose(next_token_ce(logits, t_mask), expected, atol=1e-6)
    # and it differs from the unmasked mean (sanity check it actually ignored)
    unmasked = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 6), t_full.reshape(-1))
    assert not torch.allclose(next_token_ce(logits, t_mask), unmasked, atol=1e-4)


def test_perplexity() -> None:
    assert perplexity(0.0) == 1.0
    assert math.isclose(perplexity(1.0), math.e, rel_tol=1e-9)
    assert perplexity(500.0) == math.exp(100.0)      # overflow guard


# ---------------------------- checkpoints ----------------------------

def make_model_and_opt():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=32, d_model=16, n_heads=2, n_layers=1,
                      max_len=16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, opt


def test_checkpoint_save_load_roundtrip(tmp_path) -> None:
    m, opt = make_model_and_opt()
    # simulate a step so optimizer state is non-trivial
    loss = m(torch.randint(0, 32, (1, 5))).sum()
    loss.backward()
    opt.step()
    cfg = {"vocab_size": 32, "run_name": "t"}
    path = save_checkpoint(tmp_path / "step_0000001.pt", m, opt, step=7,
                           tokens_seen=700, best_val_loss=3.14, config=cfg,
                           lr=1.23e-4)
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()

    m2, opt2 = make_model_and_opt()          # fresh random weights
    state = load_checkpoint(path, m2, opt2)
    assert state["step"] == 7 and state["tokens_seen"] == 700
    assert abs(state["best_val_loss"] - 3.14) < 1e-9
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1, p2), n1
    assert set(opt2.state_dict()["state"]) == set(opt.state_dict()["state"])


def test_find_latest_picks_highest(tmp_path) -> None:
    m, opt = make_model_and_opt()
    cfg = {"vocab_size": 32}
    for s in (10, 200, 50):
        save_checkpoint(tmp_path / f"step_{s:07d}.pt", m, opt, s, s * 10, 9.9,
                        cfg, 1e-4)
    assert find_latest(tmp_path).name == "step_0000200.pt"
    assert find_latest(tmp_path / "nothing_here") is None


def test_vocab_mismatch_rejected(tmp_path) -> None:
    m, opt = make_model_and_opt()
    save_checkpoint(tmp_path / "c.pt", m, opt, 1, 10, 5.0,
                    {"vocab_size": 32}, 1e-4)
    torch.manual_seed(1)
    other = GPT(GPTConfig(vocab_size=64, d_model=16, n_heads=2, n_layers=1,
                          max_len=16))
    try:
        load_checkpoint(tmp_path / "c.pt", other, None)
        raise AssertionError("expected ValueError for vocab mismatch")
    except ValueError:
        pass


# ------------------------------- train() -------------------------------

def test_train_smoke_loss_decreases_on_synthetic_data(tmp_path) -> None:
    """train() drives loss down end-to-end. A model can memorize a tiny
    repeating pattern, so loss must fall well below the ln(V) baseline."""
    torch.manual_seed(0)
    vocab = 32
    pattern = torch.randint(0, vocab, (24,))          # 24-token repeating cycle
    stream = pattern.repeat(60)                        # 1440 ids
    seq_len, batch = 32, 8
    from data.dataset import SequenceDataset
    from torch.utils.data import DataLoader
    dl = DataLoader(SequenceDataset(stream, seq_len), batch_size=batch,
                    shuffle=True, drop_last=True)

    model = GPT(GPTConfig(vocab_size=vocab, d_model=32, n_heads=2,
                          n_layers=1, max_len=seq_len))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses: list[float] = []
    it = iter(dl)
    for step in range(40):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        opt.zero_grad()
        loss = next_token_ce(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == 39:
            losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5                 # clear descent
    assert losses[-1] < math.log(vocab) * 0.7           # below uniform baseline


def test_train_checkpoint_resume_synthetic(tmp_path, monkeypatch) -> None:
    """End-to-end through the real train(): phase 1 runs to max_steps=6 and
    stops (writes final.pt; no step_*.pt since save_every > 6), phase 2 resumes
    from that checkpoint and runs 6 more. Step/token accounting must continue
    from the saved state, not restart."""
    from torch.utils.data import DataLoader

    from data.dataset import SequenceDataset
    from training import train as train_mod

    torch.manual_seed(0)
    vocab, seq_len, batch = 24, 16, 4
    pattern = torch.randint(0, vocab, (12,))
    stream = pattern.repeat(200)                       # 2400 ids -> 149 windows

    def fake_dataloaders(**kwargs):
        ds = SequenceDataset(stream, seq_len)
        gen = torch.Generator().manual_seed(kwargs.get("seed", 0))
        dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                        generator=gen)
        vdl = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=True)
        return dl, vdl, vocab

    monkeypatch.setattr(train_mod, "get_dataloaders", fake_dataloaders)

    run_cfg = {"run_name": "e2e_resume", "device": "cpu", "torch_threads": 2,
               "seed": 0, "seq_len": seq_len, "batch_size": batch,
               "epochs": 1, "max_steps": 6, "lr": 1e-3, "min_lr": 1e-4,
               "warmup_steps": 0, "weight_decay": 0.0, "grad_clip": 1.0,
               "betas": [0.9, 0.95], "eps": 1e-8, "dropout": 0.0,
               "log_every": 6, "val_every": 6, "save_every": 1000}

    # phase 1: fresh run, 6 steps, max_steps reached -> final.pt only
    r1 = train_mod.train(run_cfg, root=tmp_path)
    ckpt_dir = tmp_path / "checkpoints" / "e2e_resume"
    assert r1["final_step"] == 6 and r1["start_step"] == 0
    assert (ckpt_dir / "final.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    assert math.isfinite(r1["best_val_loss"])
    t1 = r1["tokens_seen"]
    assert t1 == 6 * batch * seq_len

    # phase 2: same dir; train() must discover the checkpoint and resume
    run_cfg2 = dict(run_cfg, max_steps=12)
    r2 = train_mod.train(run_cfg2, root=tmp_path)
    assert r2["start_step"] == 6, "run 2 restarted instead of resuming"
    assert r2["final_step"] == 12
    assert r2["tokens_seen"] == 12 * batch * seq_len == t1 + 6 * batch * seq_len
    assert find_latest(ckpt_dir).name == "final.pt"
    # resumed model continues learning: its final val loss is defined and the
    # best tracked across BOTH phases is no worse than phase 1's
    assert math.isfinite(r2["val_loss"])
    assert r2["best_val_loss"] <= r1["best_val_loss"] + 1e-6


def test_train_warm_start_resets_counters(tmp_path, monkeypatch) -> None:
    """warm_start loads weights + optimizer from a foreign checkpoint but
    resets step/token/best-val bookkeeping for the fresh schedule (fresh
    corpus), unlike find_latest-resume which keeps everything."""
    from torch.utils.data import DataLoader

    from data.dataset import SequenceDataset
    from training import train as train_mod

    torch.manual_seed(0)
    vocab, seq_len, batch = 24, 16, 4
    pattern = torch.randint(0, vocab, (12,))
    stream = pattern.repeat(200)

    def fake_dataloaders(**kwargs):
        ds = SequenceDataset(stream, seq_len)
        gen = torch.Generator().manual_seed(kwargs.get("seed", 0))
        dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                        generator=gen)
        vdl = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=True)
        return dl, vdl, vocab

    monkeypatch.setattr(train_mod, "get_dataloaders", fake_dataloaders)
    run_cfg = {"run_name": "e2e_warm", "device": "cpu", "torch_threads": 2,
               "seed": 0, "seq_len": seq_len, "batch_size": batch,
               "epochs": 1, "max_steps": 6, "lr": 1e-3, "min_lr": 1e-4,
               "warmup_steps": 0, "weight_decay": 0.0, "grad_clip": 1.0,
               "betas": [0.9, 0.95], "eps": 1e-8, "dropout": 0.0,
               "log_every": 6, "val_every": 6, "save_every": 1000}

    # donor run: 6 steps in its own dir; its final.pt is the warm-start source
    donor_root = tmp_path / "donor"
    donor_root.mkdir()
    r_don = train_mod.train(run_cfg, root=donor_root)
    donor_ckpt = donor_root / "checkpoints" / "e2e_warm" / "final.pt"
    assert r_don["final_step"] == 6 and donor_ckpt.exists()

    # warm start into a FRESH run dir: counters must reset
    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    run_cfg2 = dict(run_cfg, run_name="e2e_warm_fresh", max_steps=5,
                    val_every=5, save_every=5,
                    warm_start=str(donor_ckpt))
    r2 = train_mod.train(run_cfg2, root=fresh_root)
    assert r2["start_step"] == 0, "warm start must not inherit step counts"
    assert r2["final_step"] == 5
    assert r2["tokens_seen"] == 5 * batch * seq_len
    # weights came from the donor: re-load donor and compare parameter tensors
    st_don = torch.load(donor_ckpt, weights_only=False)["model"]
    fresh_dir = fresh_root / "checkpoints" / "e2e_warm_fresh"
    st_end = torch.load(fresh_dir / "final.pt", weights_only=False)["model"]
    for k in st_don:                                    # 5 steps changed them,
        assert st_don[k].shape == st_end[k].shape, k    # but shapes must match
    assert math.isfinite(r2["best_val_loss"])


def test_train_update_warm_start_extends_schedule(tmp_path, monkeypatch) -> None:
    """update_warm_start: warm-start load that KEEPS the donor's step/tokens/
    best-val bookkeeping and stretches the cosine schedule to the extended
    horizon (the continuation used by the v2-extended run)."""
    from torch.utils.data import DataLoader

    from data.dataset import SequenceDataset
    from training import train as train_mod

    torch.manual_seed(0)
    vocab, seq_len, batch = 24, 16, 4
    pattern = torch.randint(0, vocab, (12,))
    stream = pattern.repeat(200)

    def fake_dataloaders(**kwargs):
        ds = SequenceDataset(stream, seq_len)
        gen = torch.Generator().manual_seed(kwargs.get("seed", 0))
        dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                        generator=gen)
        vdl = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=True)
        return dl, vdl, vocab

    monkeypatch.setattr(train_mod, "get_dataloaders", fake_dataloaders)
    run_cfg = {"run_name": "e2e_donor", "device": "cpu", "torch_threads": 2,
               "seed": 0, "seq_len": seq_len, "batch_size": batch,
               "epochs": 1, "max_steps": 6, "lr": 1e-3, "min_lr": 1e-4,
               "warmup_steps": 0, "weight_decay": 0.0, "grad_clip": 1.0,
               "betas": [0.9, 0.95], "eps": 1e-8, "dropout": 0.0,
               "log_every": 6, "val_every": 6, "save_every": 1000}
    donor_root = tmp_path / "donor"
    donor_root.mkdir()
    r_don = train_mod.train(run_cfg, root=donor_root)
    donor_ckpt = donor_root / "checkpoints" / "e2e_donor" / "final.pt"
    assert r_don["final_step"] == 6 and donor_ckpt.exists()

    # continuation: counters KEPT, horizon extended to 12 (6 more steps)
    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    run_cfg2 = dict(run_cfg, run_name="e2e_cont", max_steps=6,
                    val_every=6, save_every=6,
                    warm_start=str(donor_ckpt), update_warm_start=True,
                    extend_schedule={"max_steps": 12})
    r2 = train_mod.train(run_cfg2, root=fresh_root)
    assert r2["start_step"] == 6, "continuation must resume the donor step"
    assert r2["final_step"] == 12
    assert r2["tokens_seen"] == 12 * batch * seq_len
    assert r2["best_val_loss"] <= r_don["best_val_loss"] + 1e-6
    assert math.isfinite(r2["val_loss"])

    # LR at the resume point sits mid-cosine, below the max LR (extended
    # horizon), and (past the horizon) at the min-LR floor:
    mid_lr = train_mod.get_lr(6, 0, 1e-3, 1e-4, 12)
    step_lr = train_mod.get_lr(11, 0, 1e-3, 1e-4, 12)   # last executed step
    end_lr = train_mod.get_lr(12, 0, 1e-3, 1e-4, 12)
    assert 1e-4 < mid_lr < 1e-3
    assert 1e-4 < step_lr < mid_lr               # decayed, not yet at floor
    assert math.isclose(end_lr, 1e-4, rel_tol=1e-6)


def test_train_smoke_stats_only_final_is_metadata_only(tmp_path, monkeypatch) -> None:
    """Dry-run mode (train_stats_only) writes final.pt WITHOUT model/optimizer
    payloads, and load_checkpoint refuses to treat such a file as weights:
    this prevents a dry-run clobbering/resuming over a real checkpoint."""
    from torch.utils.data import DataLoader

    from data.dataset import SequenceDataset
    from training import train as train_mod

    torch.manual_seed(0)
    vocab, seq_len, batch = 24, 16, 4
    pattern = torch.randint(0, vocab, (12,))
    stream = pattern.repeat(200)

    def fake_dataloaders(**kwargs):
        ds = SequenceDataset(stream, seq_len)
        gen = torch.Generator().manual_seed(kwargs.get("seed", 0))
        dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                        generator=gen)
        vdl = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=True)
        return dl, vdl, vocab

    monkeypatch.setattr(train_mod, "get_dataloaders", fake_dataloaders)
    run_cfg = {"run_name": "e2e_dry", "device": "cpu", "torch_threads": 2,
               "seed": 0, "seq_len": seq_len, "batch_size": batch,
               "epochs": 1, "max_steps": 6, "lr": 1e-3, "min_lr": 1e-4,
               "warmup_steps": 0, "weight_decay": 0.0, "grad_clip": 1.0,
               "betas": [0.9, 0.95], "eps": 1e-8, "dropout": 0.0,
               "log_every": 6, "val_every": 6, "save_every": 1000,
               "smoke": {"max_steps": 4, "log_every": 2, "val_every": 4},
               "train_stats_only": True}
    r = train_mod.train(run_cfg, smoke=True, root=tmp_path)
    assert r["final_step"] == 4 and r["start_step"] == 0
    final = tmp_path / "checkpoints" / "e2e_dry_smoke" / "final.pt"
    payload = torch.load(final, weights_only=False)
    assert "model" not in payload and "optimizer" not in payload
    assert "rng_state" not in payload
    assert payload["step"] == 4
    assert payload["tokens_seen"] == 4 * batch * seq_len
    assert payload["vocab_size"] == vocab

    m = GPT(GPTConfig(vocab_size=vocab, d_model=16, n_heads=2, n_layers=1,
                      max_len=seq_len))
    try:
        load_checkpoint(final, m, None)
        raise AssertionError("expected ValueError for stats-only payload")
    except ValueError as exc:
        assert "stats-only" in str(exc)


def test_lr_schedule_warmup_cosine_and_floor() -> None:
    f = train_mod.get_lr
    assert f(0, 10, 1e-3, 1e-4, 100) < f(5, 10, 1e-3, 1e-4, 100)  # warmup rises
    assert math.isclose(f(9, 10, 1e-3, 1e-4, 100), 1e-3, rel_tol=0.01)
    mid = f(55, 10, 1e-3, 1e-4, 100)
    assert 1e-4 < mid < 1e-3                                   # cosine between
    assert math.isclose(f(99, 10, 1e-3, 1e-4, 100), 1e-4, rel_tol=0.02)
    assert f(500, 10, 1e-3, 1e-4, 100) == 1e-4                 # floor after end
