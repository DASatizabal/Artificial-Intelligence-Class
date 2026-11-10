"""Sync check: every Part keeps its own copy of the shared data and shared code so it can run alone
in Colab. This script makes sure no copy has drifted. It stops with an error if any copy differs.

Run:  python src/sync_check.py
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parts import REPO, AI1, AI2, AI3, MI1, MI2, MI3

# file (relative to a Part folder) -> Parts that must hold an identical copy
SHARED = {
    "data/kb/county_grid.csv": [AI1, AI3, MI1, MI2, MI3],
    "data/kb/zip_cells.csv": [AI1, AI3, MI1, MI2, MI3],
    "data/kb/providers.csv": [AI1, AI3, MI1, MI2, MI3],
    "data/kb/access_standards.csv": [AI1, AI3, MI1, MI2, MI3],
    "data/kb/provider_slots.csv": [AI1, MI1, MI2, MI3],
    "data/members_part1.csv": [MI1, MI2, MI3],
    "data/member_months.csv": [MI1, MI2, MI3],
    "data/labels.csv": [AI2, MI3],
    "src/kb.py": [AI1, AI3, MI1, MI2, MI3],
    "src/gen_claims.py": [MI1, MI2, MI3],
    "src/gen_docs.py": [AI2, MI3],
    "src/dataset.py": [AI2, MI3],
}
# a hand-off file and the place it was copied to
HANDOFFS = [(MI2 / "outputs/predicted_high_cost_flags.csv", AI1 / "data/predicted_high_cost_flags.csv"),
            (AI1 / "data/members.csv", MI1 / "data/members_part1.csv")]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def image_fingerprint(part):
    """One hash over every document image, in file-name order."""
    h = hashlib.md5()
    for f in sorted((part / "data/images").glob("*.png")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def check(verbose=True):
    rows, bad = [], []
    for rel, parts in SHARED.items():
        hashes = {p.name: md5(p / rel) for p in parts}
        ok = len(set(hashes.values())) == 1
        rows.append({"file": rel, "copies": len(parts), "identical": ok})
        if not ok:
            bad.append((rel, hashes))
    imgs = {p.name: image_fingerprint(p) for p in (AI2, MI3)}
    rows.append({"file": "data/images/*.png", "copies": 2, "identical": len(set(imgs.values())) == 1})
    if len(set(imgs.values())) != 1:
        bad.append(("data/images", imgs))
    for src, dst in HANDOFFS:
        ok = md5(src) == md5(dst)
        rows.append({"file": f"{src.relative_to(REPO)} -> {dst.relative_to(REPO)}", "copies": 2, "identical": ok})
        if not ok:
            bad.append((str(dst.relative_to(REPO)), {"source": md5(src), "copy": md5(dst)}))
    if verbose:
        for r in rows:
            print(f"{'OK   ' if r['identical'] else 'DRIFT'} {r['copies']} copies  {r['file']}")
    if bad:
        raise SystemExit(f"Sync check failed: {len(bad)} shared file(s) differ between Parts: {[b[0] for b in bad]}")
    return rows


if __name__ == "__main__":
    rows = check()
    print(f"All {len(rows)} shared files match.")
