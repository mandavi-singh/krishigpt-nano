"""Pre-flight: verify warm-start checkpoint, model architecture, vocab, and
tokenizer compatibility for the corpus-v2 run. Read-only."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, GPTConfig            # noqa: E402
from tokenizer import BPETokenizer              # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.train import build_param_groups  # noqa: E402

CKPT = ROOT / "checkpoints" / "krishigpt_nano_v2" / "warm_start.pt"
TOK_PATH = ROOT / "data" / "processed" / "agri_bpe_tokenizer.json"

payload = torch.load(CKPT, weights_only=False)
cfg_dict = payload.get("config", {})
print(f"checkpoint step={payload['step']} tokens_seen={payload['tokens_seen']:,} "
      f"best_val_loss={payload['best_val_loss']:.4f} lr={payload['lr']:.2e}")
print(f"config: vocab={payload['vocab_size']} seq_len={cfg_dict.get('seq_len')} "
      f"dropout={cfg_dict.get('dropout')}")

tok = BPETokenizer.load(TOK_PATH)
assert payload["vocab_size"] == tok.vocab_size == 5237, "vocab mismatch"
print(f"tokenizer vocab {tok.vocab_size} == checkpoint vocab "
      f"{payload['vocab_size']}  (5000 merges, {ROOT.name}/processed)")

cfg = GPTConfig(vocab_size=5237, max_len=128, dropout=0.0)
model = GPT(cfg)
# must match train(): 2 param groups (decay / no-decay) - a 1-group AdamW
# cannot load this checkpoint's optimizer state
opt = torch.optim.AdamW(build_param_groups(model, 0.1), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 1_860_224, n_params
assert model.cfg.n_layers == 6 and model.cfg.d_model == 128 and model.cfg.n_heads == 4
assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), "tying lost"
print(f"architecture OK: 6 layers, d_model=128, 4 heads, RoPE, tied head; "
      f"{n_params:,} params")

state = load_checkpoint(CKPT, model, opt)         # strict by default
assert state["step"] == 2400, state["step"]
print("strict load_checkpoint: model+optimizer state restored "
      f"(AdamW states: {len(opt.state_dict()['state'])})")

logits = model(torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long))
assert tuple(logits.shape) == (1, 5, 5237)
assert torch.isfinite(logits).all()
print(f"forward sanity: {tuple(logits.shape)} finite logits OK")
print("\nALL PRE-FLIGHT CHECKS PASSED - safe to warm-start")
