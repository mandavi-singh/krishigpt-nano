"""Pre-flight for the corpus-v3 warm-start run (read-only on all v1/v2
artifacts; the only writes are the v3 token-id caches the training loader
would create on first use anyway).

Verifies before launch:
  1. krishigpt_nano_v2_extended/best.pt strict-loads (weights + AdamW state);
  2. architecture 6/128/4/tied, 1,860,224 params;
  3. tokenizer vocab == checkpoint vocab == 5237;
  4. v3 train token count == 1,966,105 (verified corpus report);
  5. v3 validation token ids byte/exact identical to the v2 validation cache;
  6. baseline validation loss on the byte-identical valid set reproduces
     4.3761 (so the whole v3 run stays comparable with the v2-extended row).

The id caches are produced by the equality-verified fast BPE encoder
(checked against tok.encode inside build_corpus_v3 before use); the tokenizer
module itself is not modified.
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders          # noqa: E402
from evaluation.corpus_v3.build_corpus_v3 import (  # noqa: E402
    build_fast_encoder,
)
from model.gpt import GPT, GPTConfig              # noqa: E402
from tokenizer import BPETokenizer                # noqa: E402
from tokenizer.word_tokenizer import word_tokenize  # noqa: E402
from training.checkpoint import load_checkpoint   # noqa: E402
from training.train import build_param_groups, evaluate  # noqa: E402

CKPT = ROOT / "checkpoints" / "krishigpt_nano_v2_extended" / "best.pt"
TOK_PATH = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"
V3 = ROOT / "data" / "corpus_v3"
V2_VALID_IDS = torch.load(ROOT / "data" / "corpus_v2" / "agri_valid_ids.pt")

payload = torch.load(CKPT, weights_only=False)
cfg_dict = payload.get("config", {})
print(f"checkpoint step={payload['step']} tokens_seen={payload['tokens_seen']:,} "
      f"best_val_loss={payload['best_val_loss']:.4f} lr={payload['lr']:.2e}")
assert payload["step"] == 4600, payload["step"]
assert payload["tokens_seen"] == 9_420_800, payload["tokens_seen"]
assert abs(payload["best_val_loss"] - 4.3761) < 1e-3, payload["best_val_loss"]
print(f"config: vocab={payload['vocab_size']} seq_len={cfg_dict.get('seq_len')} "
      f"dropout={cfg_dict.get('dropout')} corpus={cfg_dict.get('corpus')}")

tok = BPETokenizer.load(TOK_PATH)
assert payload["vocab_size"] == tok.vocab_size == 5237, "vocab mismatch"
print(f"tokenizer vocab {tok.vocab_size} == checkpoint vocab "
      f"{payload['vocab_size']}  (no retraining)")

cfg = GPTConfig(vocab_size=5237, max_len=128, dropout=0.0)
model = GPT(cfg)
opt = torch.optim.AdamW(build_param_groups(model, 0.1), lr=1e-3,
                        betas=(0.9, 0.95), eps=1e-8)
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 1_860_224, n_params
assert model.cfg.n_layers == 6 and model.cfg.d_model == 128 and model.cfg.n_heads == 4
assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), "tying lost"
print(f"architecture OK: 6 layers, d_model=128, 4 heads, RoPE, tied head; "
      f"{n_params:,} params")

state = load_checkpoint(CKPT, model, opt)          # strict by default
adam_states = len(opt.state_dict()["state"])
assert adam_states == 99, adam_states
print("strict load_checkpoint: weights + optimizer state restored "
      f"(AdamW states: {adam_states}; RNG state restored too)")

logits = model(torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long))
assert tuple(logits.shape) == (1, 5, 5237)
assert torch.isfinite(logits).all()
print(f"forward sanity: {tuple(logits.shape)} finite logits OK")

# ---- v3 id caches (loader-compatible; built with the verified fast encoder) --
train_text = (V3 / "agri_train_v3.txt").read_text(encoding="utf-8")
valid_text = (V3 / "agri_valid_v3.txt").read_text(encoding="utf-8")
encode_word = build_fast_encoder(tok, train_text + valid_text)


def encode_stream(text: str) -> torch.Tensor:
    ids: list[int] = []
    last = None
    for w in word_tokenize(text):
        last = encode_word(w)
        ids.extend(last)
    return torch.tensor(ids, dtype=torch.long)


for split, text, expect in (("train", train_text, 1_966_105),
                            ("valid", valid_text, 22_026)):
    cache = V3 / f"agri_{split}_ids.pt"
    if cache.exists():
        stream = torch.load(cache)
        print(f"{split}: cache present, verifying {len(stream):,} ids ...")
    else:
        t0 = time.time()
        stream = encode_stream(text)
        torch.save(stream, cache)
        print(f"{split}: encoded {len(stream):,} ids -> {cache.name} "
              f"({time.time() - t0:.0f}s)")
    assert len(stream) == expect, (len(stream), expect)
    assert int(stream.min()) >= 0 and int(stream.max()) < 5237
assert torch.equal(torch.load(V3 / "agri_valid_ids.pt"), V2_VALID_IDS), (
    "v3 valid id stream is NOT identical to the v2 cache")
print("v3 valid id stream == v2 validation cache EXACTLY (byte-level ids)")

# ---- loader + baseline eval ---------------------------------------------------
train_dl, valid_dl, vocab_size = get_dataloaders(
    tokenizer_path=TOK_PATH,
    train_text_path=V3 / "agri_train_v3.txt",
    valid_text_path=V3 / "agri_valid_v3.txt",
    cache_dir=V3,
    seq_len=128, batch_size=16, seed=0)
assert vocab_size == 5237, vocab_size
assert len(train_dl) == 960, len(train_dl)
xb, yb = next(iter(train_dl))
assert tuple(xb.shape) == (16, 128) and tuple(yb.shape) == (16, 128)
print(f"dataloaders OK: {len(train_dl):,} train batches (960/epoch @ bs16), "
      f"{len(valid_dl)} valid eval windows, batch shape {tuple(xb.shape)}")

val_loss = evaluate(model, valid_dl, "cpu")
print(f"baseline v3 validation (= v2 validation, byte-identical): "
      f"{len(valid_dl.dataset):,} windows -> val_loss {val_loss:.4f} "
      f"(v2-extended best recorded 4.3761, rel err "
      f"{abs(val_loss - 4.3761) / 4.3761:.6f})")
assert abs(val_loss - 4.3761) < 0.005, val_loss

print("\nALL PRE-FLIGHT CHECKS PASSED - safe to warm-start the v3 run")
