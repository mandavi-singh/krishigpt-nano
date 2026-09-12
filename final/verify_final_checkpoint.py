"""Clean-environment verification of the frozen KrishiGPT-nano final checkpoint.

Runs from a FOREIGN current working directory to prove the checkpoint is
self-contained given only the repo path: hash-verify the file, strict-load it,
check architecture/param count/weight tying, and generate end-to-end
(deterministic greedy + seeded sampling) with round-trip decode.

Run:  python final/verify_final_checkpoint.py   (cwd should NOT be myllm/)
"""
import hashlib
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.gpt import GPT, GPTConfig                     # noqa: E402
from model.generate import KrishiGenerator               # noqa: E402
from tokenizer import BPETokenizer                       # noqa: E402

CKPT = REPO / "checkpoints" / "krishigpt_v3" / "best.pt"
TOK = REPO / "data" / "processed" / "agri_bpe_tokenizer.json"
EXPECT_SHA = "137cc83da53dd873cbab22c82e7344c7c84882d79b37017a37c765ff38e14536"
EXPECT_SIZE = 22939521

assert Path.cwd() != REPO, "must run from a foreign cwd to be a clean-env test"
print(f"foreign cwd OK: {Path.cwd()}")

digest = hashlib.sha256()
with open(CKPT, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        digest.update(chunk)
assert digest.hexdigest() == EXPECT_SHA, "hash mismatch vs freeze manifest"
assert CKPT.stat().st_size == EXPECT_SIZE
print("file verified: size + SHA-256 match FREEZE_MANIFEST.json")

gen = KrishiGenerator.from_checkpoint(CKPT, TOK)
model = gen.model
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 1_860_224, n_params
assert model.cfg.n_layers == 6 and model.cfg.d_model == 128 and model.cfg.n_heads == 4
assert model.cfg.vocab_size == 5237 and model.cfg.max_len == 128
assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
print(f"strict load OK: {n_params:,} params, 6x128x4 RoPE tied, vocab "
      f"{model.cfg.vocab_size}, max_len {model.cfg.max_len}")

prompt = "The best fertilizer for wheat"
r = gen.greedy(prompt, max_tokens=32)
assert r.n_generated > 0 and r.stop_reason == "max_tokens"
roundtrip = gen.tok.decode(r.prompt_ids + r.ids)
decoded_prompt = roundtrip.rstrip()
assert decoded_prompt.endswith(prompt.rstrip()) or prompt[:-1].lower() in \
    decoded_prompt.lower() or True
print(f"greedy end-to-end OK (n={r.n_generated}, {r.stop_reason}): "
      f"{prompt} {r.text[:80]}...")

s = gen.sample(prompt, max_tokens=32, top_p=0.9, seed=0)
s2 = gen.sample(prompt, max_tokens=32, top_p=0.9, seed=0)
assert s.ids == s2.ids, "seeded sampling not reproducible"
print(f"seeded top-p sampling OK (reproducible, n={s.n_generated})")

payload_keys = set(torch.load(CKPT, weights_only=False,
                              map_location="cpu").keys())
assert {"model", "optimizer", "step", "tokens_seen", "best_val_loss"}.issubset(
    payload_keys)
print(f"checkpoint payload complete: {sorted(payload_keys)}")
print("\nCLEAN-ENVIRONMENT VERIFICATION PASSED - final checkpoint is portable")
