"""AI Part 1 integration (healthcare): A* distances -> CSP assignment -> stress test -> metrics.
Run from the project root:  python src/run_part1.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb import KnowledgeBase
from planner import Planner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
CONFIGS = {"Plain backtracking": dict(mrv=False, fc=False),
           "Backtracking + forward checking": dict(mrv=False, fc=True),
           "MRV + forward checking": dict(mrv=True, fc=True),
           "MRV + FC + LCV": dict(mrv=True, fc=True, lcv=True),
           "MRV + FC + look-ahead": dict(mrv=True, fc=True, lookahead=True)}
FINAL = "MRV + FC + look-ahead"
NODE_LIMIT = 200_000


def solve_all(pl, label=FINAL):
    """Solve every specialty. Returns final assignment, per-specialty search stats, capacity gaps."""
    final, rows, cap_gaps = {}, [], []
    mem = pl.members
    for spec in sorted(mem["need_specialty"].unique()):
        mids = [m for m in mem.index[mem["need_specialty"] == spec] if m in pl.domains]
        best, short = pl.max_placeable(mids)
        cap_gaps += sorted(short)
        solvable = [m for m in mids if m not in short]
        for name, cfg in (CONFIGS.items() if label is None else [(label, CONFIGS[label])]):
            a, st = pl.solve(solvable, node_limit=NODE_LIMIT, **cfg)
            rows.append({"specialty": spec, "method": name, "members": len(solvable), **st})
            if name == FINAL:
                final.update(a)
    return final, pd.DataFrame(rows), cap_gaps


def summarize(pl, assign, gaps, cap_gaps):
    mem = pl.members
    miles = [pl.miles[(m, p)] for m, p in assign.items() if p]
    used = pd.Series([p for p in assign.values() if p]).value_counts()
    open_slots = pl.prov.loc[pl.prov["new_slots"] > 0, "new_slots"]
    hc_share = {p: sum(mem.at[m, "high_cost"] for m, q in assign.items() if q == p) for p in used.index}
    by_spec = []
    for spec, g in mem.groupby("need_specialty"):
        placed = sum(1 for m in g.index if assign.get(m))
        by_spec.append({"specialty": spec, "members": len(g), "placed_within_standard": placed,
                        "share": round(placed / len(g), 4), "meets_90pct": placed / len(g) >= 0.90,
                        "distance_gaps": sum(1 for m in g.index if m in gaps),
                        "capacity_gaps": sum(1 for m in g.index if m in cap_gaps)})
    return {
        "members": len(mem), "placed": len(miles), "share_placed_within_standard": round(len(miles) / len(mem), 4),
        "distance_gaps": len(gaps), "capacity_gaps": len(cap_gaps),
        "avg_miles": round(float(np.mean(miles)), 2), "max_miles": int(max(miles)),
        "slot_use": round(float(used.sum() / open_slots.sum()), 4),
        "high_cost_cap_violations": int(sum(hc_share[p] > pl.cap[p] for p in hc_share)),
        "capacity_violations": int(sum(used[p] > pl.slots[p] for p in used.index)),
        "by_specialty": by_spec,
    }


def draw_map(pl, assign, gaps, cap_gaps, removed_group, path):
    g = pl.kb.grid; mem = pl.members; prov = pl.prov
    fig, ax = plt.subplots(figsize=(7, 8.5))
    ax.imshow(g, cmap="Blues", vmin=0, vmax=1.6, alpha=0.6)
    for m, p in list(assign.items()):
        if p:
            ax.plot([mem.at[m, "col"], prov.at[p, "col"]], [mem.at[m, "row"], prov.at[p, "row"]], color="#888", lw=0.3, alpha=0.5)
    ok = mem.index.isin([m for m, p in assign.items() if p])
    ax.scatter(mem.loc[ok, "col"], mem.loc[ok, "row"], s=6, c="#1f5fbf", label="member placed")
    ax.scatter(mem.loc[mem.index.isin(gaps), "col"], mem.loc[mem.index.isin(gaps), "row"], s=28, marker="x", c="#b42318", label="distance gap (no doctor inside the standard)")
    ax.scatter(mem.loc[mem.index.isin(cap_gaps), "col"], mem.loc[mem.index.isin(cap_gaps), "row"], s=22, marker="^", c="#b85c00", label="capacity gap (nearby doctors full)")
    act = prov[prov["new_slots"] > 0]
    ax.scatter(act["col"], act["row"], s=40, marker="s", c="#0e7c86", edgecolor="white", label="provider with open slots")
    grp = act[act["group"] == removed_group]
    ax.scatter(grp["col"], grp["row"], s=70, marker="s", facecolor="none", edgecolor="#b85c00", lw=2, label=f"{removed_group} (stress test)")
    ax.set_title("Member assignments on the county grid (1 cell = 1 mile)\nGray lines connect each member to their doctor"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    kb = KnowledgeBase(ROOT / "data" / "kb")
    members = pd.read_csv(ROOT / "data" / "members.csv", dtype={"zip": str})
    slots = pd.read_csv(ROOT / "data" / "kb" / "provider_slots.csv")
    pl = Planner(kb, members, slots).build_domains()
    gaps = list(pl.gaps)

    # 1) compare the three search strategies on every specialty
    final, search, cap_gaps = solve_all(pl, label=None)
    base = summarize(pl, final, gaps, cap_gaps)

    # 2) stress test: the provider group serving the most members leaves the network
    used_groups = pl.prov.loc[[p for p in final.values()], "group"].value_counts()
    group = used_groups.index[0]
    removed = list(pl.prov.index[pl.prov["group"] == group])
    pl.build_domains(removed=removed)
    gaps2 = list(pl.gaps)
    repaired, mc = pl.min_conflicts(final, removed)
    # compare with solving everything again from scratch
    resolved, _, cap_gaps2 = solve_all(pl)
    others = [m for m in final if final[m] not in removed]
    full_moved = sum(1 for m in others if resolved.get(m) != final[m])
    clean_repair = {m: p for m, p in repaired.items() if p}
    after = summarize(pl, clean_repair, gaps2, [m for m, p in repaired.items() if p is None and m not in gaps2])
    after_full = summarize(pl, resolved, gaps2, cap_gaps2)

    stress = {"group_removed": group, "providers_removed": len(removed), "min_conflicts": mc,
              "full_resolve": {"other_members_moved": full_moved, "placed": after_full["placed"]},
              "after_min_conflicts": {k: after[k] for k in ("placed", "share_placed_within_standard", "avg_miles", "capacity_violations")},
              "after_full_resolve": {k: after_full[k] for k in ("placed", "share_placed_within_standard", "avg_miles", "capacity_violations")}}

    draw_map(Planner(kb, members, slots).build_domains(), final, gaps, cap_gaps, group, OUT / "assignment_map.png")
    pd.DataFrame([{"member_id": m, "provider_id": p, "miles": pl.miles.get((m, p))} for m, p in final.items()]).to_csv(OUT / "assignments.csv", index=False)
    search.to_csv(OUT / "search_comparison.csv", index=False)
    totals = search.groupby("method")[["nodes", "backtracks", "seconds"]].sum()
    totals["specialties_solved"] = search.groupby("method")["solved"].sum()
    metrics = {"baseline": base, "search_totals": totals.reset_index().to_dict("records"), "stress_test": stress,
               "astar": {"calls": kb.astar_calls, "node_expansions": kb.astar_expansions}}
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in base.items() if k != "by_specialty"}, indent=1))
    print(pd.DataFrame(base["by_specialty"]).to_string(index=False))
    print(totals.to_string())
    print(search.to_string(index=False))
    print(json.dumps(stress, indent=1))


if __name__ == "__main__":
    main()
