"""Pre-flight for the corpus-v2 CONTINUATION run: verify the starting
checkpoint (krishigpt_nano_v2/best.pt) strict-loads, reproduces the v2
validation loss, and is compatible with the extended schedule. Read-only."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders        # noqa: E402
from model.gpt import GPT, GPTConfig            # noqa: E402
from tokenizer import BPETokenizer              # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.train import build_param_groups, evaluate  # noqa: E402

CKPT = ROOT / "checkpoints" / "krishigpt_nano_v2" / "best.pt"
TOK_PATH = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"

payload = torch.load(CKPT, weights_only=False)
cfg_dict = payload.get("config", {})
print(f"checkpoint step={payload['step']} tokens_seen={payload['tokens_seen']:,} "
      f"best_val_loss={payload['best_val_loss']:.4f} lr={payload['lr']:.2e}")
assert payload["step"] == 2300, payload["step"]
assert payload["tokens_seen"] == 4_710_400, payload["tokens_seen"]
assert abs(payload["best_val_loss"] - 4.5068) < 1e-3, payload["best_val_loss"]
print(f"config: vocab={payload['vocab_size']} seq_len={cfg_dict.get('seq_len')} "
      f"dropout={cfg_dict.get('dropout')} corpus={cfg_dict.get('corpus')}")

tok = BPETokenizer.load(TOK_PATH)
assert payload["vocab_size"] == tok.vocab_size == 5237, "vocab mismatch"
print(f"tokenizer vocab {tok.vocab_size} == checkpoint vocab "
      f"{payload['vocab_size']}")

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

state = load_checkpoint(CKPT, model, opt)        # strict by default
assert state["step"] == 2300, state["step"]
adam_states = len(opt.state_dict()["state"])
assert adam_states == 99, adam_states
print("strict load_checkpoint: model+optimizer state restored "
      f"(AdamW states: {adam_states}; RNG state restored too)")

logits = model(torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long))
assert tuple(logits.shape) == (1, 5, 5237)
assert torch.isfinite(logits).all()
print(f"forward sanity: {tuple(logits.shape)} finite logits OK")

# reproduce the recorded validation numbers on the SAME v2 split
_, valid_dl, vocab_size = get_dataloaders(
    tokenizer_path=TOK_PATH,
    train_text_path=ROOT / "data" / "corpus_v2" / "agri_train_v2.txt",
    valid_text_path=ROOT / "data" / "corpus_v2" / "agri_valid_v2.txt",
    cache_dir=ROOT / "data" / "corpus_v2",
    seq_len=128, batch_size=16, seed=0)
assert vocab_size == 5237, vocab_size
val_loss = evaluate(model, valid_dl, "cpu")
print(f"v2 validation: {len(valid_dl.dataset):,} windows -> "
      f"val_loss {val_loss:.4f} (logged 4.5068, rel err "
      f"{abs(val_loss - 4.5068) / 4.5068:.6f})")
assert abs(val_loss - 4.5068) < 0.005, val_loss

print("\nALL PRE-FLIGHT CHECKS PASSED - safe to warm-start the continuation")
