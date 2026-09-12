"""Phase 1 test suite: run verify() of every lab.

Lab filenames start with digits (01_..., 02_...) so they cannot be imported
with normal 'import' syntax; importlib loads them by path instead.

Run:  pytest tests/test_labs.py -q
"""
import importlib.util
from pathlib import Path

import pytest

LABS_DIR = Path(__file__).resolve().parents[1] / "labs"
LAB_FILES = sorted(LABS_DIR.glob("0*.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("lab_path", LAB_FILES, ids=[p.stem for p in LAB_FILES])
def test_lab_verify(lab_path: Path) -> None:
    """Each lab's verify() must pass its numeric assertions."""
    mod = _load(lab_path)
    assert hasattr(mod, "verify"), f"{lab_path.name} missing verify()"
    mod.verify()
