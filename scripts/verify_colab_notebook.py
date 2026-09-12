"""Dry-run verification of the Colab notebook's critical cells against a
scratch COPY of the project. Executes: integrity cell, compatibility cell,
and the training cell's seed/resume state machine (up to - but not including -
train()). The real project and checkpoints are never modified."""
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "colab" / "KrishiGPT_nano_v2_extended.ipynb"
PY = str(Path(sys.executable))

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
print(f"notebook: {len(cells)} code cells")

scratch = Path(tempfile.mkdtemp(prefix="krishi_colab_dryrun_"))
target = scratch / "myllm"
shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns(
    ".venv", "__pycache__", ".git", ".pytest_cache"))
print(f"scratch copy -> {target}")

# ---- cell 4 (SHA integrity), cwd = scratch/myllm ---------------------------
code = "\n".join(lines for lines in cells[3].splitlines()
                 if not lines.startswith("import os"))
snippet = Path(scratch) / "cell4.py"
snippet.write_text(code, encoding="utf-8")
r = subprocess.run([PY, str(snippet)], cwd=target, capture_output=True, text=True)
print("\n=== cell 4 (integrity) ===")
print(r.stdout[-700:] if r.stdout else "")
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    raise SystemExit("cell 4 failed on verified state")

# ---- cell 5 (compatibility), cwd = scratch/myllm ---------------------------
snippet = Path(scratch) / "cell5.py"
snippet.write_text(cells[4], encoding="utf-8")
r = subprocess.run([PY, str(snippet)], cwd=target, capture_output=True, text=True)
print("=== cell 5 (compatibility) ===")
print(r.stdout[-800:] if r.stdout else "")
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    raise SystemExit("cell 5 failed on verified state")

# ---- cell 7 (training cell): run everything BEFORE the train() call --------
src = cells[6]
cut = src.index("with ThreadPoolExecutor")
pre = src[:cut] + "\nprint('DRY RUN: stopping right before train()')"
snippet = Path(scratch) / "cell7_pre.py"
snippet.write_text(pre, encoding="utf-8")
r = subprocess.run([PY, str(snippet)], cwd=target, capture_output=True, text=True)
print("=== cell 7 (seed state machine, FIRST launch) ===")
print(r.stdout[-800:] if r.stdout else "")
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    raise SystemExit("cell 7 seed phase failed")

run_dir = target / "checkpoints" / "krishigpt_nano_v2_extended"
seeded = sorted(run_dir.glob("step_*.pt"))
assert seeded == [run_dir / "step_0002300.pt"], seeded
cfg_written = json.loads((target / "configs" / "nano_agri_v2_extended.json")
                         .read_text(encoding="utf-8"))
assert cfg_written["max_steps"] == 4636 and cfg_written["corpus"] == "v2"
assert cfg_written["device"] == "cuda"          # DEVICE was never set in this
                                                # harness... check below
print("seeded checkpoint:", seeded[0].name)
print("written config max_steps:", cfg_written["max_steps"],
      "corpus:", cfg_written["corpus"])
print("\nNOTE: cell 7 references DEVICE (set by cell 3); this harness skipped")
