"""Load code from each Part folder without copying it.

Every Part keeps its own src/ folder so it can still be run and graded on its own.
The pipeline imports those files in place. Two Parts both have a file named run_part3.py,
so each file is loaded under a unique name (for example "MI_Part3_Neural.run_part3").
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

AI1, AI2, AI3 = REPO / "AI_Part1_Search_CSP", REPO / "AI_Part2_Perception", REPO / "AI_Part3_NLP_KR"
MI1, MI2, MI3 = REPO / "MI_Part1_EDA_Regression", REPO / "MI_Part2_Classification", REPO / "MI_Part3_Neural"


def load(part, name):
    """Import part/src/name.py. The Part's src folder is put first on the path while it loads,
    so its own 'from kb import ...' lines find the Part's own files."""
    src = part / "src"
    key = f"{part.name}.{name}"
    if key in sys.modules:
        return sys.modules[key]
    sys.path.insert(0, str(src))
    try:
        spec = importlib.util.spec_from_file_location(key, src / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(src))
    return mod
