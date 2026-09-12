"""Builds colab/KrishiGPT_v4_qa2_gpu.ipynb — the ROUND-2 QA fine-tune
notebook (constraint-format instruction tuning). SELF-PROVISIONING build:
no Drive uploads are required beyond what is already there.

Round 1 (krishigpt_v4_qa) taught Q/A surface format but scored 0% on the
instruction bench because constraint formats were never demonstrated in
the data (see evaluation/corpus_v4/EXPERIMENT_v4_qa.md). Round 2 trains
on data/qa_v2 (yes/no, one-word, list-of-three + cleaned definitional).

The notebook does everything in Colab at runtime:
  - embeds (base64, byte-identical) the two locally-verified generator
    scripts and writes them into the workspace, then runs the generator
    against the corpus_v4 text already on Drive (seed 0, deterministic);
    pair counts are hard-verified against the local run (851/801/50)
  - patches training/train_qa.py in the WORKSPACE copy only (adds the
    qa_dir config key; no-op if already patched; Drive file untouched)
  - writes configs/nano_agri_v4_qa2.json with the runtime device
  - trains from the round-1 donor, runs the frozen funnel bench, syncs
    everything (including data/qa_v2 for the record) back to Drive
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # myllm/


def md(text):
    lines = text.split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


# ---- embed the locally-verified artifacts (byte-identical) ------------------
b64_pairs = base64.b64encode(
    (ROOT / "evaluation/corpus_v4/make_qa_pairs.py").read_bytes()).decode()
b64_pairs_v2 = base64.b64encode(
    (ROOT / "evaluation/corpus_v4/make_qa_pairs_v2.py").read_bytes()).decode()
b64_train_qa = base64.b64encode(
    (ROOT / "training/train_qa.py").read_bytes()).decode()
cfg_json = json.dumps(json.loads(
    (ROOT / "configs/nano_agri_v4_qa2.json").read_text(encoding="utf-8")))

CELLS = []

CELLS.append(md(r"""# KrishiGPT v4 QA ROUND 2: constraint-format instruction tune (GPU)

Round 1 (`krishigpt_v4_qa`) learned the Q/A surface format but scored **0%**
instruction compliance — the fine-tune data never demonstrated constraint
formats (one word / yes-no / list-of-three / <=10 words).

Round 2 fixes the two identified causes:
- **format coverage:** constraint formats trained directly (yes/no balanced
  251 yes / 118 no, one-word, list-of-three + definitional)
- **pair quality:** sentence-initial domain-term subjects only (round-1
  noise like "What is whatever soil?" eliminated)

**SELF-PROVISIONING — nothing to upload to Drive.** This notebook generates
`data/qa_v2/` inside Colab from the corpus-v4 text already on Drive (same
deterministic generator, byte-identical scripts embedded below, pair counts
verified against the local run), patches the trainer in the workspace copy,
writes the config, trains, evaluates on the frozen bench, and syncs results
back to Drive. Frozen v3 is NEVER touched (SHA-verified before and after).

**Benchmark-overfit guard:** training phrasing deliberately differs from the
bench's; <=10-words / JSON / stop-word formats never trained.
"""))

CELLS.append(code(r"""# 1. Mount Drive, LOCATE the project, prepare workspace.
import os, shutil, subprocess, sys
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except ImportError:
    raise SystemExit('This notebook must run on Google Colab.')

DRIVE = Path('/content/drive/MyDrive')
WORK = Path('/content/myllm')
MARKER = 'training/train.py'


def is_project(p: Path) -> bool:
    return (p / MARKER).exists() and (p / 'model' / 'gpt.py').exists()


def find_project(root: Path) -> Path | None:
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            for child in d.iterdir():
                if child.name.startswith('.') or child.name in (
                        '.shortcut-targets-by-id', '.Trash'):
                    continue
                if child.is_dir():
                    if is_project(child):
                        return child
                    stack.append(child)
        except (PermissionError, OSError):
            continue
    return None


SRC = None
for cand in (DRIVE / 'KrishiGPT' / 'myllm', DRIVE / 'myllm'):
    if is_project(cand):
        SRC = cand
        break
if SRC is None:
    print('searching Drive for the myllm project folder (one-time)...')
    SRC = find_project(DRIVE)

if SRC is not None:
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK, ignore=shutil.ignore_patterns(
        '.venv', '__pycache__', '.pytest_cache', '*.pyc', 'colab'))
    print('project folder found at:', SRC)
else:
    raise SystemExit(
        'myllm project folder not found in Drive — it must contain the v4 '
        'checkpoints and corpus (from the previous Colab runs).')

os.chdir(WORK)
sys.path.insert(0, str(WORK))
print('workspace:', WORK)

# ---- verify what round 2 needs (all already on Drive) -----------------------
required = [
    'checkpoints/krishigpt_v3/best.pt',
    'checkpoints/krishigpt_v4_qa/best.pt',      # round-2 warm-start donor
    'data/processed/agri_bpe_tokenizer.json',
    'data/corpus_v4/agri_train_v4.txt',         # generator source
    'data/corpus_v4/agri_valid_v4.txt',
    'training/train_qa.py',
    'evaluation/health_report.py',
    'evaluation/mcq_bench.py',
    'evaluation/instruction_bench.py',
    'evaluation/funnel_report.py',
]
missing = [r for r in required if not (WORK / r).exists()]
if missing:
    print('MISSING FILES (these should already be on Drive from earlier '
          'runs — check the myllm folder):')
    for m in missing:
        print('  -', m)
    raise SystemExit(f'{len(missing)} required files missing.')
print('all round-2 prerequisites present')

r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'torch==2.9.1', 'numpy==2.5.2'],
                   capture_output=True, text=True)
print('deps ok' if r.returncode == 0 else r.stderr[-500:])
"""))

CELLS.append(code(r"""# 2. SHA-verify frozen v3 (never modified) + record the round-2 donor.
import hashlib

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

expect_v3 = '137cc83da53dd873cbab22c82e7344c7c84882d79b37017a37c765ff38e14536'
got = sha(WORK / 'checkpoints' / 'krishigpt_v3' / 'best.pt')
assert got == expect_v3, f'v3 checkpoint hash mismatch: {got}'
print('frozen v3 best.pt SHA-256 verified (submission artifact untouched)')

donor = sha(WORK / 'checkpoints' / 'krishigpt_v4_qa' / 'best.pt')
print(f'round-2 donor krishigpt_v4_qa/best.pt SHA-256: {donor}')
"""))

CELLS.append(code(r"""# 3. SELF-PROVISION: write the locally-verified scripts (byte-identical),
#    generate data/qa_v2 in Colab, verify counts, write config + device.
import base64, json, subprocess, sys
from pathlib import Path

B64_MAKE_QA_PAIRS = '''%s'''
B64_MAKE_QA_PAIRS_V2 = '''%s'''
B64_TRAIN_QA = '''%s'''

gen_dir = WORK / 'evaluation' / 'corpus_v4'
gen_dir.mkdir(parents=True, exist_ok=True)
(gen_dir / 'make_qa_pairs.py').write_text(
    base64.b64decode(B64_MAKE_QA_PAIRS).decode('utf-8'), encoding='utf-8')
(gen_dir / 'make_qa_pairs_v2.py').write_text(
    base64.b64decode(B64_MAKE_QA_PAIRS_V2).decode('utf-8'), encoding='utf-8')
(WORK / 'training' / 'train_qa.py').write_text(
    base64.b64decode(B64_TRAIN_QA).decode('utf-8'), encoding='utf-8')  # qa_dir key included
print('generator + trainer scripts written (byte-identical to the '
      'locally-verified copies; trainer now supports qa_dir)')

# ---- generate data/qa_v2 from the corpus already on Drive -------------------
r = subprocess.run([sys.executable, 'evaluation/corpus_v4/make_qa_pairs_v2.py'],
                   cwd=str(WORK))
assert r.returncode == 0, 'pair generation failed — see log above'

stats = json.loads((WORK / 'data' / 'qa_v2' / 'qa_stats.json').read_text())
print('pair stats:', {k: stats[k] for k in
                      ('n_pairs_total', 'n_train', 'n_valid')})
assert (stats['n_pairs_total'], stats['n_train'], stats['n_valid']) == \
       (851, 801, 50), (
    'pair counts differ from the verified local generation (851/801/50) — '
    'the Drive corpus_v4 text may differ from the local one. STOP and '
    'paste this output back before training.')

# drop any stale id caches so the trainer re-tokenizes the new data
for f in (WORK / 'data' / 'qa_v2').glob('*.pt'):
    f.unlink()

# ---- write the round-2 config with the runtime device ------------------------
import torch
print('CUDA available:', torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print('device:', torch.cuda.get_device_name(0))
else:
    print('no GPU — CPU fallback (much slower)')
cfg = json.loads('''%s''')
cfg['device'] = DEVICE
if DEVICE == 'cpu':
    cfg['torch_threads'] = 8
(WORK / 'configs' / 'nano_agri_v4_qa2.json').write_text(
    json.dumps(cfg, indent=2))
print(f"config written: device={DEVICE} | max_steps={cfg['max_steps']} "
      f"(3 epochs x 21 steps) | warm_start={cfg['warm_start']}")
""" % (b64_pairs, b64_pairs_v2, b64_train_qa, cfg_json)))

CELLS.append(code(r"""# 4. ROUND 2 — constraint-format QA fine-tune (63 steps, ~2-3 min on T4).
# Restart-safe: re-run after a disconnect to resume from the latest step_*.pt.
# Fresh-run semantics: if a previous qa2 dir exists, archive best.pt and
# clear resume points so training starts clean from the round-1 donor.
import shutil
from pathlib import Path

qa2_dir = WORK / 'checkpoints' / 'krishigpt_v4_qa2'
if qa2_dir.exists():
    for f in qa2_dir.iterdir():
        if f.name.startswith('step_') or f.name in ('final.pt',):
            f.unlink()
    for f in qa2_dir.glob('*.tmp'):
        f.unlink()
    old = qa2_dir / 'best.pt'
    if old.exists():
        old.rename(qa2_dir / 'best_prev.pt')
    (qa2_dir / 'train_log.jsonl').unlink(missing_ok=True)
    print('cleared previous qa2 run dir (old best kept as best_prev.pt)')

import subprocess, sys
r = subprocess.run([sys.executable, 'training/train_qa.py',
                    '--config', 'configs/nano_agri_v4_qa2.json'],
                   cwd=str(WORK))
assert r.returncode == 0, 'round-2 QA tuning failed — see log above'

# verify the run actually trained on qa_v2 with the corrected schedule:
# best.pt carries the run config; check the distinctive fields
import torch as _torch
state = _torch.load(WORK / 'checkpoints' / 'krishigpt_v4_qa2' / 'best.pt',
                    map_location='cpu', weights_only=False)
cfg_in_ckpt = state['config']
assert cfg_in_ckpt.get('qa_dir') == 'qa_v2', \
    f"trainer used qa_dir={cfg_in_ckpt.get('qa_dir')!r}, expected 'qa_v2'"
assert cfg_in_ckpt['max_steps'] == 63, cfg_in_ckpt['max_steps']
assert cfg_in_ckpt['warm_start'] == 'checkpoints/krishigpt_v4_qa/best.pt'
n_epochs_done = state['step'] / 21
print(f"verified: qa_dir=qa_v2 | donor=v4_qa best | stopped at step "
      f"{state['step']} ({n_epochs_done:.2f} epochs) | best QA val loss "
      f"{state['best_val_loss']:.4f}")
"""))

CELLS.append(code(r"""# 5. FROZEN FUNNEL EVALUATION on the round-2 checkpoint.
import subprocess, sys

for script, arg in (('evaluation/health_report.py', 'krishigpt_v4_qa2'),
                    ('evaluation/instruction_bench.py', 'krishigpt_v4_qa2'),
                    ('evaluation/mcq_bench.py', 'krishigpt_v4_qa2'),
                    ('evaluation/funnel_report.py', 'krishigpt_v4_qa2')):
    print(f'== {script} {arg}')
    r = subprocess.run([sys.executable, script, arg], cwd=str(WORK))
    print()
"""))

CELLS.append(code(r"""# 6. Three-way comparison: v3 (frozen baseline) vs v4_qa (round 1) vs qa2.
import json

res = lambda run, name: json.loads(
    (WORK / 'evaluation' / 'results' / run / name).read_text())

for name in ('instruction_bench.json', 'mcq_bench.json'):
    print(f'== {name}')
    for run in ('krishigpt_v3', 'krishigpt_v4_qa', 'krishigpt_v4_qa2'):
        try:
            d = res(run, name)
            if name.startswith('instruction'):
                print(f'  {run}: compliance {d["compliance"]:.0%}')
            else:
                print(f'  {run}: MCQ acc {d["overall_accuracy"]:.1%} '
                      f'(weighted {d["domain_score_weighted"]:.1%})')
        except FileNotFoundError:
            print(f'  {run}: (missing)')

# per-test compliance detail for round 2
d = res('krishigpt_v4_qa2', 'instruction_bench.json')
print('\nround-2 per-test:')
for t in d['results']:
    mark = 'PASS' if t['compliant'] else 'FAIL'
    print(f'  [{mark}] {t["id"]:<14} out: {t["output"][:60]!r}')
"""))

CELLS.append(code(r"""# 7. Smoke-test constraint formats from the round-2 model.
import sys
sys.path.insert(0, str(WORK))
from model.generate import KrishiGenerator

gen = KrishiGenerator.from_checkpoint(
    WORK / 'checkpoints' / 'krishigpt_v4_qa2' / 'best.pt',
    WORK / 'data' / 'processed' / 'agri_bpe_tokenizer.json')
prompts = (
    'Q: Does rice need water? Answer yes or no.\nA:',
    'Q: What is loam? Reply with a single word.\nA:',
    'Q: Name three farm practices, separated by commas.\nA:',
    'Q: What is compost?\nA:',
)
for p in prompts:
    r = gen.sample(p, max_tokens=32, top_p=0.9, seed=0)
    print(f'{p}\n-> {r.text[:160]}\n')
"""))

CELLS.append(code(r"""# 8. Sync round-2 artifacts back to Drive (v3 untouched), final verify.
import hashlib, shutil
from pathlib import Path

run = 'krishigpt_v4_qa2'
src = WORK / 'checkpoints' / run
dst = SRC / 'checkpoints' / run
if src.exists():
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f'synced {run} checkpoints -> Drive')
src = WORK / 'evaluation' / 'results' / run
dst = SRC / 'evaluation' / 'results' / run
if src.exists():
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f'synced {run} results -> Drive')
# keep the generated data on Drive too (record + no regeneration needed)
src = WORK / 'data' / 'qa_v2'
dst = SRC / 'data' / 'qa_v2'
if src.exists():
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print('synced data/qa_v2 -> Drive')

h = hashlib.sha256()
with open(WORK / 'checkpoints' / 'krishigpt_v3' / 'best.pt', 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
assert h.hexdigest() == ('137cc83da53dd873cbab22c82e7344c7c84882d79b'
                         '37017a37c765ff38e14536')
print('done — v3 checkpoint still untouched (post-run verify)')
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": CELLS,
}
out = ROOT / "colab" / "KrishiGPT_v4_qa2_gpu.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"notebook written -> {out}")
