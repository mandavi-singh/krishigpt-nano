"""Builds colab/KrishiGPT_nano_v2_extended.ipynb - the controlled continuation
experiment notebook, generated from the verified project state (hashes, paths,
and config values are embedded verbatim from the CPU-verified run)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # project root (myllm/)

def md(text):
    lines = text.split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

def code(text):
    lines = text.split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

CELLS = []

CELLS.append(md(r"""# KrishiGPT-nano: corpus-v2 extension on GPU (Colab)

**Controlled continuation experiment** - this notebook trains the *existing*
KrishiGPT-nano for a bounded 4 epochs on corpus v2, resuming from the verified
`checkpoints/krishigpt_nano_v2/best.pt`. It changes **nothing** about the
model (6 layers, d_model 128, 4 heads, RoPE, pre-norm, tied embeddings,
1,860,224 params), the 5,237-vocab BPE tokenizer, corpus v2, the AdamW setup,
the CE loss, or the training algorithm.

Workflow:
1. Mount Google Drive (persistent checkpoints live in Drive; training runs on
   a scratch copy).
2. Install exact dependencies (torch 2.9.1, pytest).
3. Verify every required artifact by SHA-256 before anything runs.
4. Detect CUDA, benchmark real training throughput, print runtime estimate.
5. Create `checkpoints/krishigpt_nano_v2_extended/` and seed it from the
   fixed best.pt (no random reinitialization, optimizer state preserved).
6. Train the bounded 4-epoch budget (steps 2300 -> 2336). Restart-safe: re-run
   the training cell after any Colab disconnect to resume.
7. Evaluate (harness + fabricated-word metric) and compare against the v2
   best.pt and v1-extended best.pt.
8. Run the full test suite (175 tests).

**Before first use:** upload the `myllm` project folder to
`MyDrive/KrishiGPT/myllm` in Google Drive (the notebook trains from a copy of
it and never writes back into your Drive copy except by explicit sync cells)."""))

CELLS.append(code(r"""# 1. Mount Google Drive and load the project onto the scratch disk.
import os
import shutil
import sys
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    SRC = Path('/content/drive/MyDrive/KrishiGPT/myllm')
    if not (SRC / 'training' / 'train.py').exists():
        raise SystemExit(
            f'project not found under {SRC} - upload the "myllm" folder to '
            f'MyDrive/KrishiGPT/ first (keep data/corpus_v2, data/processed, '
            f'checkpoints/ included).')
except ImportError:                      # non-Colab fallback (local testing)
    SRC = Path.cwd() if Path('training/train.py').exists() else Path('myllm')
    if not (SRC / 'training' / 'train.py').exists():
        raise SystemExit('no project found: run inside myllm/ or mount Drive')

WORK = Path('/content/myllm')
if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SRC, WORK,
                ignore=shutil.ignore_patterns(
                    '.venv', '__pycache__', '.git', '.pytest_cache'))
os.chdir(WORK)                       # (notebook cells run from here onward)
sys.path.insert(0, str(WORK))
print('project source :', SRC)
print('working copy   :', WORK)
print('drive-backed dirs:',
      [p.name for p in (WORK / 'checkpoints').iterdir() if p.is_dir()])"""))

CELLS.append(md(r"""## 2. Environment

Exact, minimal dependencies of the verified project (imports audited: the
codebase uses only `torch`; tests use `pytest`). torch 2.9.1 matches the CPU
verification machine."""))

CELLS.append(code(r"""# 2. Install exact dependencies and verify versions.
import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                       'torch==2.9.1', 'pytest'])
import pytest
import torch
print('python :', sys.version.split()[0])
print('torch  :', torch.__version__)
print('pytest :', pytest.__version__)
assert torch.__version__.startswith('2.9.1'), torch.__version__"""))

CELLS.append(md(r"""## 3. Device detection

Use CUDA when available; otherwise fall back to CPU (the notebook still runs,
just slowly)."""))

CELLS.append(code(r"""# 3. Detect CUDA.
import torch

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.benchmark = True
    print('device        :', DEVICE)
    print('GPU           :', torch.cuda.get_device_name(0))
    print('capability    :', torch.cuda.get_device_capability(0))
    print('memory        :',
          round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
else:
    DEVICE = 'cpu'
    print('device: cpu (no CUDA runtime - training will be CPU-bound)')"""))

CELLS.append(md(r"""## 4. Artifact integrity (SHA-256)

Every input of this experiment is verified against the hashes recorded on the
CPU verification machine. If ANY hash mismatches, the notebook stops before
touching anything - the Drive copy is out of sync with the verified state.
Files in the "must-not-change" set are opened read-only throughout."""))

CELLS.append(code(r"""# 4. Verify all frozen inputs by SHA-256 (recorded on the verified CPU run).
import hashlib
import os
from pathlib import Path

# safe re-run in a fresh session without re-executing the mount cell
if Path('/content/myllm').exists():
    os.chdir('/content/myllm')

EXPECTED_SHA256 = {
    'data/processed/agri_bpe_tokenizer.json':
        'F92DA6E195BFFC18E70514344B4E99C480FB768739E8C0EEA136C3494B9ADBCD',
    'data/corpus_v2/agri_train_v2.txt':
        '813498EF984F81E08FC09C04ED9A56FBFFDC8D54635E4F78AB0ACA17124F07B5',
    'data/corpus_v2/agri_valid_v2.txt':
        'DA9910D0F4C41907608489BB0614C949AF638BD534DFAF28956CAA79FD3B5091',
    'data/corpus_v2/agri_train_ids.pt':
        '2EB0276CB2D7CC0C377A23B54D1013903605C0B91FDA2053A34BEC1A74423ADF',
    'data/corpus_v2/agri_valid_ids.pt':
        '55C60FF405BC626EDF8FE4431289639E460613355014A140BE65860CDCE78848',
    'checkpoints/krishigpt_nano_v2/best.pt':
        'F87C21C734FE61397FBD2FFE750711C8FD52AB8EE928C2C672C40141CD631A6C',
    'checkpoints/krishigpt_nano_v2/warm_start.pt':
        'F9D1076955CC4CE5349CC276EDC2491F6A3B1DC9C1B6BFB2E58BA43AC17B8B43',
    'checkpoints/krishigpt_nano_extended/best.pt':
        'F9D1076955CC4CE5349CC276EDC2491F6A3B1DC9C1B6BFB2E58BA43AC17B8B43',
    'configs/nano_agri_v2.json':
        'D4DD17B66020D01E8F75AB9CD96652EAEB8E2EF20343CF4A42F95155A22D5D2D',
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

fails = []
for rel, want in EXPECTED_SHA256.items():
    p = Path(rel)
    if not p.exists():
        fails.append(f'MISSING {rel}')
        continue
    got = sha256(p)
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'} {rel}  {got[:16]}...")
    if not ok:
        fails.append(rel)
if fails:
    raise SystemExit('integrity check failed for: ' + ', '.join(fails))
print('\nall frozen inputs verified - safe to proceed')"""))

CELLS.append(md(r"""## 5. Architecture / tokenizer / checkpoint compatibility (pre-train)

Reproduces the local `training/precheck_v2.py` verification: strict state-dict
load of `krishigpt_nano_v2/best.pt` into a freshly built nano GPT plus the
run's exact 2-group AdamW. This checks vocab (5,237), architecture
(6/128/4, RoPE, tied head, 1,860,224 params), and optimizer-state shape
BEFORE any training cell runs."""))

CELLS.append(code(r"""# 5. Pre-train compatibility check: best.pt must load strictly.
import sys
from pathlib import Path

sys.path.insert(0, '/content/myllm')
import torch
from model.gpt import GPT, GPTConfig
from tokenizer import BPETokenizer
from training.checkpoint import load_checkpoint
from training.train import build_param_groups

CKPT = Path('checkpoints/krishigpt_nano_v2/best.pt')
payload = torch.load(CKPT, map_location='cpu', weights_only=False)
tok = BPETokenizer.load('data/processed/agri_bpe_tokenizer.json')
assert payload['vocab_size'] == tok.vocab_size == 5237, 'vocab mismatch'
assert len(tok.merges) == 5000

cfg = GPTConfig(vocab_size=5237, max_len=128, dropout=0.0)
assert cfg.n_layers == 6 and cfg.d_model == 128 and cfg.n_heads == 4
model = GPT(cfg)
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 1_860_224, n_params
assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), \
    'weight tying lost'

opt = torch.optim.AdamW(build_param_groups(model, 0.1), lr=6e-4,
                        betas=(0.9, 0.95), eps=1e-8)
state = load_checkpoint(CKPT, model, opt)          # strict model+optimizer load
assert state['step'] == 2300, state['step']
assert torch.isfinite(model(torch.tensor([[4, 5, 6]], dtype=torch.long))).all()
print(f"checkpoint : {CKPT.name} -> step {state['step']}, "
      f"tokens {state['tokens_seen']:,}, best_val_loss {state['best_val_loss']:.4f}")
print(f"model      : {cfg.n_layers}L/{cfg.d_model}d/{cfg.n_heads}H, tied, "
      f"{n_params:,} params")
print(f"tokenizer  : vocab {tok.vocab_size}, {len(tok.merges)} merges")
print(f"optimizer  : {len(opt.state_dict()['param_groups'])} param groups, "
      f"{len(opt.state_dict()['state'])} states restored")
print('ALL PRE-TRAIN CHECKS PASSED')"""))

CELLS.append(md(r"""## 6. GPU benchmark + runtime estimate

Runs the exact training step shape (batch 16 x seq 128, forward + backward +
AdamW) on the detected device. The benchmark uses a throwaway model - it never
touches the verified checkpoints or their optimizer states. Estimate covers the
bounded budget: resume at step 2,300, train 4 additional epochs (584
steps/epoch on corpus v2) -> 2,336 remaining steps out of 4,636 total."""))

CELLS.append(code(r"""# 6. Benchmark on real step geometry; estimate runtime for the bounded budget.
import time

import torch
from model.gpt import GPT, GPTConfig
from training.train import build_param_groups

B, T, V = 16, 128, 5237
STEPS_PER_EPOCH, TOTAL_STEPS, RESUME_STEP = 584, 4636, 2300
remaining = TOTAL_STEPS - RESUME_STEP

bench = GPT(GPTConfig(vocab_size=V, max_len=T, dropout=0.0)).to(DEVICE)
bopt = torch.optim.AdamW(build_param_groups(bench, 0.1), lr=6e-4,
                         betas=(0.9, 0.95), eps=1e-8)
xb = torch.randint(0, V, (B, T), device=DEVICE)
tb = torch.randint(0, V, (B, T), device=DEVICE)
bench.train()
lossf = torch.nn.functional.cross_entropy

def step():
    bopt.zero_grad()
    loss = lossf(bench(xb).reshape(B * T, V), tb.reshape(B * T))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(bench.parameters(), 1.0)
    bopt.step()

for _ in range(2):                       # warmup (allocator / kernels)
    step()
if DEVICE == 'cuda':
    torch.cuda.synchronize()
t0 = time.time()
N = 20
for _ in range(N):
    step()
if DEVICE == 'cuda':
    torch.cuda.synchronize()
dt = time.time() - t0
tps = N * B * T / dt
eta_min = remaining * (B * T) / tps / 60
del bench, bopt, xb, tb
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
print(f'measured throughput : {tps:,.0f} tokens/sec ({dt / N:.3f}s/step)')
print(f'bounded budget      : steps {RESUME_STEP} -> {TOTAL_STEPS} '
      f'({remaining} steps = 4 epochs x {STEPS_PER_EPOCH} steps/epoch)')
print(f'estimated runtime   : {eta_min:.1f} min (+ ~1 min eval overhead)')"""))

CELLS.append(md(r"""## 7. Training (bounded, restart-safe)

Creates `checkpoints/krishigpt_nano_v2_extended/` and trains **4 additional
epochs** (2,336 steps, total 2300 -> 4636) **from** the verified
`krishigpt_nano_v2/best.pt`:

- **No random reinit**: on first launch the run dir is seeded with an exact
  copy of best.pt named `step_0002300.pt`; the pipeline's resume path picks it
  up and continues from step 2300 (weights **and** optimizer state).
- **LR schedule preserved**: same schedule machinery as every prior run -
  linear warmup (done) then cosine decay from 6e-4 toward the 6e-5 floor,
  re-anchored over the extended horizon (exactly the same mechanism the
  accepted `krishigpt_nano_extended` run used: warm weights + fresh schedule).
- **Restart-safe**: after any Colab disconnect, re-run this cell. It detects
  existing checkpoints in the run dir and resumes instead of re-seeding.
- Saves: periodic `step_*.pt`, `best.pt` (on val improvement below 4.5068),
  `final.pt`, `train_log.jsonl`.

Config is identical to the verified v2 run except: `run_name`,
`device` (auto), `max_steps` 4636 (4 more epochs), no `warm_start` (resume is
used instead), and denser logging/saving granularity."""))

CELLS.append(code(r"""# 7. Bounded 4-epoch continuation. Restart-safe: run this cell after any
# disconnect to resume from the latest checkpoint.
import json
import shutil
import time
from pathlib import Path

import torch

RUN_NAME = 'krishigpt_nano_v2_extended'
RUN_DIR = Path('checkpoints') / RUN_NAME
SEED_CKPT = Path('checkpoints/krishigpt_nano_v2/best.pt')   # frozen input

CFG = {
    "run_name": RUN_NAME,
    "corpus": "v2",
    "device": DEVICE,
    "torch_threads": 10,
    "seed": 0,
    "seq_len": 128,
    "batch_size": 16,
    "epochs": 8,
    "max_steps": 4636,
    "lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_steps": 200,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "dropout": 0.0,
    "log_every": 25,
    "val_every": 100,
    "save_every": 100,
    "smoke": {"max_steps": 30, "log_every": 5, "val_every": 15},
    "notes": "GPU continuation of the verified v2 run: resume "
             "krishigpt_nano_v2/best.pt (step 2300) for 4 ADDITIONAL epochs "
             "(2336 steps -> max_steps 4636); same architecture/tokenizer/"
             "corpus/optimizer; cosine LR re-anchored over the extended "
             "horizon, same mechanism as krishigpt_nano_extended.",
}
CFG_PATH = Path('configs/nano_agri_v2_extended.json')
CFG_PATH.write_text(json.dumps(CFG, indent=2), encoding='utf-8')

RUN_DIR.mkdir(exist_ok=True)

def is_this_run(payload):
    return payload.get('config', {}).get('run_name') == RUN_NAME

if any(RUN_DIR.glob('step_*.pt')):
    action = 'RESUME from latest step checkpoint found in run dir'
elif (RUN_DIR / 'final.pt').exists() and is_this_run(
        torch.load(RUN_DIR / 'final.pt', map_location='cpu',
                   weights_only=False)):
    action = 'RESUME from final.pt of the previous completed/interrupted run'
else:
    # first launch: seed the run dir with the frozen best.pt (no reinit)
    seed = torch.load(SEED_CKPT, map_location='cpu', weights_only=False)
    shutil.copy(SEED_CKPT, RUN_DIR / f"step_{seed['step']:07d}.pt")
    action = (f"FIRST RUN: seeded {RUN_DIR.name}/step_{seed['step']:07d}.pt "
              f"from {SEED_CKPT} (SHA-verified)")
print(action)

from concurrent.futures import ThreadPoolExecutor            # noqa: E402

from training.train import train                            # noqa: E402

cfg = json.loads(CFG_PATH.read_text(encoding='utf-8'))
log_path = RUN_DIR / 'train_log.jsonl'
B, T = 16, 128
RESUME, TOTAL = 2300, 4636

def watch():
    # tail train_log.jsonl while training runs: show epoch/step/loss/val/
    # lr/tokens-per-sec/ETA (train() itself also streams its progress lines)
    try:
        if not log_path.exists():
            return
        lines = [l for l in log_path.read_text(encoding='utf-8').splitlines()
                 if l.strip()]
        if not lines:
            return
        last = json.loads(lines[-1])
        step = last['step']
        tps = last.get('tokens_per_sec', 0)
        frac = (step - RESUME) / (TOTAL - RESUME)
        eta_min = (TOTAL - step) * B * T / tps / 60 if tps > 0 else float('nan')
        val = f" | val_loss {last['val_loss']:.4f}{' *best*' if last.get('best') else ''}" \
            if 'val_loss' in last else ''
        print(f"[watch] epoch {last['epoch']} | step {step}/{TOTAL} "
              f"({100 * frac:.1f}% of the 2,336-step budget) | "
              f"train_loss {last['train_loss']:.4f}{val} | "
              f"lr {last['lr']:.2e} | {tps:,.0f} tok/s | ETA {eta_min:.1f} min",
              flush=True)
    except Exception:                                        # noqa: BLE001
        pass                                                 # watchdog never kills the run

t0 = time.time()
with ThreadPoolExecutor(max_workers=1) as ex:
    fut = ex.submit(train, cfg)
    while not fut.done():
        time.sleep(30)
        watch()
    watch()
    result = fut.result()                   # propagates training exceptions
elapsed = time.time() - t0
steps_per_epoch = 584
print(f"\ntrain() finished in {elapsed / 60:.1f} min")
print(f"step {result['start_step']} -> {result['final_step']} "
      f"(schedule-epochs {result['start_step'] / steps_per_epoch:.2f} -> "
      f"{result['final_step'] / steps_per_epoch:.2f}; 4 additional epochs "
      f"beyond the v2 run), tokens_seen {result['tokens_seen']:,}")
print(f"final val_loss {result['val_loss']:.4f}, "
      f"best_val_loss {result['best_val_loss']:.4f}")"""))

CELLS.append(md(r"""## 8. Persist new artifacts to Drive

The training cells wrote to the scratch copy; this copies only the NEW run
directory (plus the generated extended config) into the Drive-backed project
folder. The frozen inputs are never overwritten."""))

CELLS.append(code(r"""# 8. Sync the new run's checkpoints + logs (+ eval artifacts) back to Drive.
import shutil
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/KrishiGPT/myllm')
for src, dst in [
    ('checkpoints/krishigpt_nano_v2_extended',
     DRIVE_ROOT / 'checkpoints' / 'krishigpt_nano_v2_extended'),
    ('evaluation/results/krishigpt_nano_v2_extended_best',
     DRIVE_ROOT / 'evaluation' / 'results' / 'krishigpt_nano_v2_extended_best'),
]:
    src = Path(src)
    if not src.exists():
        print(f'skip (not present yet): {src}')
        continue
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    n = len(list(dst.rglob('*')))
    mb = sum(f.stat().st_size for f in dst.rglob('*') if f.is_file()) / 1e6
    print(f'synced {src} -> {dst} ({n} files, {mb:.0f} MB)')
shutil.copy('configs/nano_agri_v2_extended.json', DRIVE_ROOT / 'configs')
print('synced configs/nano_agri_v2_extended.json')"""))

CELLS.append(md(r"""## 9. Evaluation

Runs the **existing evaluation harness** (`evaluation/run_eval.py`) on the new
checkpoint with the exact same 5 prompts and decoding variants
(greedy / temperature 0.9 / top-k 40 / top-p 0.9), plus repetition rate and
distinct-1/-2. If `best.pt` was not rewritten (validation never improved over
4.5068), it falls back to `final.pt` and says so.

Note: if you skip the sync cell, best.pt may equal the seeded step_0002300
weights; the report prints which file was evaluated."""))

CELLS.append(code(r"""# 9a. Existing evaluation harness on the new checkpoint (same prompts/variants).
import subprocess
import sys
from pathlib import Path

import torch

run_dir = Path('checkpoints/krishigpt_nano_v2_extended')
eval_ckpt = run_dir / 'best.pt'
if not eval_ckpt.exists():
    eval_ckpt = run_dir / 'final.pt'
seed_same = torch.equal(
    torch.load(eval_ckpt, map_location='cpu', weights_only=False)['model']['tok_emb.weight'],
    torch.load('checkpoints/krishigpt_nano_v2/best.pt', map_location='cpu',
               weights_only=False)['model']['tok_emb.weight'])
print(f'evaluating: {eval_ckpt} '
      f'{"(weights identical to v2 best.pt - no improvement saved yet)" if seed_same else ""}')
subprocess.check_call([sys.executable, 'evaluation/run_eval.py',
                       '--checkpoint', str(eval_ckpt),
                       '--corpus', 'v2',
                       '--out-name', 'krishigpt_nano_v2_extended_best',
                       '--variant', 'greedy', '--variant', 'temp0.9',
                       '--variant', 'topk40', '--variant', 'topp0.9'])"""))

CELLS.append(code(r"""# 9b. Word-level fabricated-word rate + repetition/distinctness comparison.
# Same metric as the CPU experiment: regenerate the exact seeded samples and
# check each reconstructed word against the full v1+v2 corpus via word-boundary
# regex. Reference rows are the verified CPU measurements.
import re
import sys
from pathlib import Path

sys.path.insert(0, '/content/myllm')
from model.generate import KrishiGenerator                  # noqa: E402

corpora = (Path('data/processed/agri_train.txt').read_text(encoding='utf-8')
           + Path('data/processed/agri_valid.txt').read_text(encoding='utf-8')
           + Path('data/corpus_v2/agri_train_v2.txt').read_text(encoding='utf-8')
           + Path('data/corpus_v2/agri_valid_v2.txt').read_text(encoding='utf-8'))
_cache = {}

def attested(word):
    key = word.lower()
    if key not in _cache:
        _cache[key] = re.search(r"(?<![A-Za-z])" + re.escape(key)
                                + r"(?![A-Za-z])", corpora, re.I) is not None
    return _cache[key]

def gen_words(tok, ids):
    words, cur = [], []
    for i in ids:
        s = tok.id2token[i]
        if '</w>' in s:
            cur.append(s.replace('</w>', ''))
            w = ''.join(cur)
            if len(w) >= 3 and any(c.isalpha() for c in w) and not w.isdigit():
                words.append(w)
            cur = []
        else:
            cur.append(s)
    return words

PROMPTS = ['The best fertilizer for wheat', 'Soil moisture affects',
           'To control pests in the field', 'Rice is grown in',
           'The farmer should irrigate']
VARIANTS = [('greedy', {}), ('temp0.9', dict(temperature=0.9, seed=0)),
            ('topk40', dict(top_k=40, seed=0)), ('topp0.9', dict(top_p=0.9, seed=0))]

run_dir = Path('checkpoints/krishigpt_nano_v2_extended')
new_ckpt = run_dir / 'best.pt' if (run_dir / 'best.pt').exists() else run_dir / 'final.pt'
RUNS = [('NEW v2-ext', str(new_ckpt)),
        ('v2 best', 'checkpoints/krishigpt_nano_v2/best.pt'),
        ('v1 ext', 'checkpoints/krishigpt_nano_extended/best.pt')]

def distinct_n(ids, n):
    if len(ids) < n:
        return None
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)

print(f"{'run':<12} {'variant':<9} {'words':>6} {'fab%':>6} {'rep%':>6} "
      f"{'d1':>5} {'d2':>5}   fabricated examples")
print('-' * 92)
for label, ckpt in RUNS:
    gen = KrishiGenerator.from_checkpoint(ckpt,
                                          'data/processed/agri_bpe_tokenizer.json')
    for vname, kw in VARIANTS:
        tot = fab = rep_seen = n_tok = 0
        d1s, d2s, examples = [], [], []
        for p in PROMPTS:
            r = gen.greedy(p, 64) if vname == 'greedy' else gen.sample(p, 64, **kw)
            n_tok += r.n_generated
            rep_seen += sum(1 for i in range(1, len(r.ids))
                            if r.ids[i] in r.ids[:i])
            d1 = distinct_n(r.ids, 1); d2 = distinct_n(r.ids, 2)
            if d1 is not None: d1s.append(d1)
            if d2 is not None: d2s.append(d2)
            for w in gen_words(gen.tok, r.ids):
                tot += 1
                if not attested(w):
                    fab += 1
                    if w not in examples:
                        examples.append(w)
        rep = 100 * rep_seen / max(n_tok - len(PROMPTS), 1)
        print(f"{label:<12} {vname:<9} {tot:>6} {100 * fab / max(tot, 1):>5.1f}% "
              f"{rep:>5.1f}% {sum(d1s) / max(len(d1s), 1):>5.2f} "
              f"{sum(d2s) / max(len(d2s), 1):>5.2f}   {examples[:4]}")"""))

CELLS.append(md(r"""## 10. Test suite

Runs the project's complete pytest suite (175 tests on the verified state)."""))

CELLS.append(code(r"""# 10. Full test suite.
import subprocess
import sys

r = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'],
                   check=False)
print('\ntest suite exit code:', r.returncode)
assert r.returncode == 0, 'test suite failed'"""))

CELLS.append(md(r"""## Notes and reference numbers

Verified CPU measurements (for comparison; v1 and v2 validation losses use
different doc sets and are not directly comparable):

| checkpoint | eval set | val loss | val ppl | fab% temp0.9 / topk40 / topp0.9 |
|---|---|---|---|---|
| v1 extended best.pt | v1 valid | 5.2720 | 194.8 | 14.8 / 5.8 / 14.6 |
| v2 best.pt (step 2300) | v2 valid | 4.5068 | 90.6 | 8.1 / 3.7 / 15.5 |
| NEW v2-extended | v2 valid | ? (this run) | ? | ? |

What to watch for in the comparison:
- **Overfitting signal**: train loss keeps falling while v2 val loss rises
  above 4.5068.
- **Helpful signal**: val loss < 4.5068 and fabricated-word % at or below the
  v2 best row.
- This run trains 2,336 additional steps (about 5x the v2 run's remaining
  budget at the time of its best checkpoint), so a val-loss move is plausible,
  but the corpus is only 1.2M tokens - overfitting must be watched closely."""))


nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.5"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": CELLS,
}

out = Path(__file__).resolve().parents[1] / "colab" / \
    "KrishiGPT_nano_v2_extended.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out.relative_to(Path(__file__).resolve().parents[1]),
      f"({len(CELLS)} cells)")
