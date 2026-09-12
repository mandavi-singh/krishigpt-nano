"""Builds colab/KrishiGPT_v4_gpu.ipynb — the v4 lineage notebook.

Generated notebook trains: (1) corpus-v4 how-to pretraining warm-started
from the frozen v3 best.pt, then (2) QA fine-tuning on the auto-generated
QA pairs, then (3) runs the 3-level funnel evaluation (health, instruction,
MCQ) and syncs results back to Drive. SHA-verified, restart-safe, GPU-aware.
"""
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


CELLS = []

CELLS.append(md(r"""# KrishiGPT v4: how-to pretraining + QA fine-tuning (GPU)

New v4 lineage — the frozen submitted v3 checkpoint is NEVER modified.
Warm start: v3 best.pt (step 3800, val 4.0804) → corpus v4 (2.27M tokens,
217 docs, +33 how-to manuals) → QA fine-tune (616 auto-generated pairs).

After training, the 3-level funnel evaluation runs:
- L1 training health gate
- L2 instruction-following bench
- L3 agriculture domain MCQ bench

**Before first use:** upload the updated `myllm` folder to
`MyDrive/KrishiGPT/myllm` (must contain `data/corpus_v4/`, `data/qa/`,
`evaluation/mcq_bench.py`, `evaluation/health_report.py`,
`evaluation/instruction_bench.py`, `training/train_qa.py`).
"""))

CELLS.append(code(r"""# 1. Mount Drive, LOCATE the project (folder OR zip), prepare workspace.
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
    # BFS the whole Drive for the project folder.
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


# ---- locate: fast paths first, then full-Drive search ----------------------
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
    # no folder: maybe a myllm zip was uploaded — find and extract the largest
    zips = [p for p in DRIVE.rglob('*.zip')
            if 'myllm' in p.name.lower() and not p.name.startswith('.')]
    if not zips:
        raise SystemExit(
            'myllm project folder (or myllm zip) not found in Drive.\n'
            'Fix: right-click the myllm folder on your PC -> "Download"\n'
            '(makes myllm.zip) -> upload that zip into Google Drive ->\n'
            're-run this cell. The zip must contain training/, model/,\n'
            'data/, checkpoints/, configs/, evaluation/.')
    zips.sort(key=lambda z: z.stat().st_size, reverse=True)
    print('extracting', zips[0].name, f'({zips[0].stat().st_size/1e6:.0f} MB)...')
    shutil.unpack_archive(zips[0], WORK.parent)
    if not is_project(WORK):
        hits = [p for p in WORK.parent.rglob('train.py')
                if p.as_posix().endswith('/training/train.py')]
        if hits:
            inner = hits[0].parents[1]
            if inner != WORK:
                shutil.rmtree(WORK, ignore_errors=True)
                shutil.move(str(inner), str(WORK))
    if not is_project(WORK):
        raise SystemExit('zip extracted but no training/train.py found — '
                         'check the zip structure (it should contain the '
                         'myllm folder).')
    print('project extracted from zip')

os.chdir(WORK)
sys.path.insert(0, str(WORK))
print('workspace:', WORK)

# ---- verify every v4-critical artifact BEFORE training ---------------------
required = [
    'checkpoints/krishigpt_v3/best.pt',
    'data/processed/agri_bpe_tokenizer.json',
    'data/corpus_v4/agri_train_v4.txt',
    'data/corpus_v4/agri_valid_v4.txt',
    'data/corpus_v4/agri_train_ids.pt',
    'data/corpus_v4/agri_valid_ids.pt',
    'data/qa/qa_pairs_train.jsonl',
    'data/qa/qa_pairs_valid.jsonl',
    'data/qa/qa_train_ids.pt',
    'data/qa/qa_valid_ids.pt',
    'training/train_qa.py',
    'evaluation/health_report.py',
    'evaluation/mcq_bench.py',
    'evaluation/instruction_bench.py',
    'evaluation/funnel_report.py',
    'configs/nano_agri_v4.json',
    'configs/nano_agri_v4_qa.json',
]
missing = [r for r in required if not (WORK / r).exists()]
if missing:
    print('MISSING FILES (re-upload the FULL myllm folder):')
    for m in missing:
        print('  -', m)
    raise SystemExit(f'{len(missing)} required files missing — upload them '
                     'and re-run this cell.')
print('all v4-critical artifacts present')

r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'torch==2.9.1', 'numpy==2.5.2'],
                   capture_output=True, text=True)
print('deps ok' if r.returncode == 0 else r.stderr[-500:])
"""))

CELLS.append(code(r"""# 2. SHA-verify the frozen v3 source-of-truth checkpoint (read-only).
import hashlib

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

expect = '137cc83da53dd873cbab22c82e7344c7c84882d79b37017a37c765ff38e14536'
got = sha(WORK / 'checkpoints' / 'krishigpt_v3' / 'best.pt')
assert got == expect, f'v3 checkpoint hash mismatch: {got}'
print('frozen v3 best.pt SHA-256 verified (source of truth, never modified)')
"""))

CELLS.append(code(r"""# 3. GPU check + config switch to cuda + CORRECTED QA schedule.
import torch, json

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    DEVICE = 'cuda'
else:
    print('no GPU — CPU fallback (much slower)')
    DEVICE = 'cpu'

for cfg_path in ('configs/nano_agri_v4.json', 'configs/nano_agri_v4_qa.json'):
    p = WORK / cfg_path
    cfg = json.loads(p.read_text())
    cfg['device'] = DEVICE
    if DEVICE == 'cpu':
        cfg['torch_threads'] = 8
    if cfg_path.endswith('_qa.json'):
        # CORRECTED QA schedule (the first Colab run used max_steps=240 =
        # ~11 epochs on 21 steps/epoch -> train loss collapsed 3.84 -> 1.31 =
        # overfit). 3 real epochs = 63 steps; validate/save every epoch.
        cfg['max_steps'] = 63
        cfg['val_every'] = 21
        cfg['save_every'] = 21
        cfg['log_every'] = 5
    p.write_text(json.dumps(cfg, indent=2))
    print(f"{cfg_path} -> device={DEVICE}", end='')
    if cfg_path.endswith('_qa.json'):
        print(f" | max_steps={cfg['max_steps']} (corrected)")
    else:
        print()
"""))

CELLS.append(code(r"""# 4. PHASE 1 — corpus v4 pretraining (warm start from frozen v3).
# Restart-safe: re-run after a Colab disconnect to resume from the latest
# step checkpoint. ~4,440 steps on a T4 at bs=16: well under an hour.
import subprocess, sys

r = subprocess.run([sys.executable, 'training/train.py',
                    '--config', 'configs/nano_agri_v4.json'],
                   cwd=str(WORK))
assert r.returncode == 0, 'phase 1 (v4 pretraining) failed — see log above'
"""))

CELLS.append(code(r"""# 5. PHASE 2 — QA fine-tuning (warm start from v4 best).
# Fresh re-run semantics: the previous krishigpt_v4_qa dir (if any, e.g. from
# an earlier over-long schedule) is archived so training starts clean from
# the v4 pretraining best.pt with the corrected 63-step schedule.
import shutil
from pathlib import Path
qa_dir = WORK / 'checkpoints' / 'krishigpt_v4_qa'
if qa_dir.exists():
    for f in qa_dir.iterdir():                     # drop resume points
        if f.name.startswith('step_') or f.name in ('final.pt',):
            f.unlink()
    for f in qa_dir.glob('*.tmp'):
        f.unlink()
    # keep old best.pt as best_overfit_v1.pt for the record, then remove it
    old = qa_dir / 'best.pt'
    if old.exists():
        old.rename(qa_dir / 'best_overfit_v1.pt')
    (qa_dir / 'train_log.jsonl').unlink(missing_ok=True)
    print('cleared previous QA-tune run dir (old best kept as '
          'best_overfit_v1.pt)')

r = subprocess.run([sys.executable, 'training/train_qa.py',
                    '--config', 'configs/nano_agri_v4_qa.json'],
                   cwd=str(WORK))
assert r.returncode == 0, 'phase 2 (QA tuning) failed — see log above'
"""))

CELLS.append(code(r"""# 6. FUNNEL EVALUATION on the QA-tuned checkpoint (v4_qa).
import subprocess, sys

for script, arg in (('evaluation/health_report.py', 'krishigpt_v4'),
                    ('evaluation/health_report.py', 'krishigpt_v4_qa'),
                    ('evaluation/instruction_bench.py', 'krishigpt_v4_qa'),
                    ('evaluation/mcq_bench.py', 'krishigpt_v4_qa'),
                    ('evaluation/funnel_report.py', 'krishigpt_v4_qa')):
    print(f'== {script} {arg}')
    r = subprocess.run([sys.executable, script, arg], cwd=str(WORK))
    print()
"""))

CELLS.append(code(r"""# 7. Compare v3 (frozen baseline) vs v4_qa side by side.
import json
from pathlib import Path

res = lambda run, name: json.loads(
    (WORK / 'evaluation' / 'results' / run / name).read_text())

for name in ('instruction_bench.json', 'mcq_bench.json'):
    print(f'== {name}')
    for run in ('krishigpt_v3', 'krishigpt_v4_qa'):
        try:
            d = res(run, name)
            if name.startswith('instruction'):
                print(f'  {run}: compliance {d["compliance"]:.0%}')
            else:
                print(f'  {run}: MCQ acc {d["overall_accuracy"]:.1%} '
                      f'(weighted {d["domain_score_weighted"]:.1%})')
        except FileNotFoundError:
            print(f'  {run}: (missing)')
"""))

CELLS.append(code(r"""# 8. Smoke-test generation from the QA-tuned model.
import sys
sys.path.insert(0, str(WORK))
from model.generate import KrishiGenerator

gen = KrishiGenerator.from_checkpoint(
    WORK / 'checkpoints' / 'krishigpt_v4_qa' / 'best.pt',
    WORK / 'data' / 'processed' / 'agri_bpe_tokenizer.json')
for p in ('Q: What is loam?\nA:', 'Q: Define compost.\nA:',
          'Q: What is drip irrigation?\nA:'):
    r = gen.sample(p, max_tokens=48, top_p=0.9, seed=0)
    print(f'{p}\n-> {r.text[:160]}\n')
"""))

CELLS.append(code(r"""# 9. Sync checkpoints + results back to Drive (v3 untouched).
import shutil
from pathlib import Path

for run in ('krishigpt_v4', 'krishigpt_v4_qa'):
    src = WORK / 'checkpoints' / run
    dst = SRC / 'checkpoints' / run
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f'synced {run} checkpoints -> Drive')
for run in ('krishigpt_v4', 'krishigpt_v4_qa'):
    src = WORK / 'evaluation' / 'results' / run
    dst = SRC / 'evaluation' / 'results' / run
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f'synced {run} results -> Drive')
print('done — v3 checkpoint untouched (verify cell 2 anytime)')
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
out = ROOT / "colab" / "KrishiGPT_v4_gpu.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"notebook written -> {out}")
