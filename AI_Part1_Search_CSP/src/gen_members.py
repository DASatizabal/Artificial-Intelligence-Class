"""Generate the members and the open panel slots for AI Part 1 (healthcare version).

Synthetic and seeded (seed 42). Reuses the Part 3 knowledge base in data/kb/.
Outputs:
  data/members.csv           600 members: home cell, ZIP, needed specialty, age, placeholder high-cost flag
  data/kb/provider_slots.csv new-patient slots this month, high-cost cap, and provider group for each provider
The high-cost flag is a placeholder until MI Part 2 builds the real model.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb"
N_MEMBERS = 600
NEED = {"primary care": 0.50, "cardiology": 0.10, "dermatology": 0.07, "endocrinology": 0.08,
        "ob-gyn": 0.09, "orthopedic surgery": 0.08, "psychiatry": 0.08}
SLACK = 1.15          # total open slots per specialty = 115% of demand, so the puzzle is tight but solvable
HIGH_COST_SHARE = 0.35  # a doctor can take high-cost members for at most 35% of their open slots


def main():
    rng = np.random.default_rng(SEED)
    grid = pd.read_csv(KB / "county_grid.csv", header=None).values
    zips = pd.read_csv(KB / "zip_cells.csv", dtype={"zip": str})
    prov = pd.read_csv(KB / "providers.csv")
    open_cells = np.argwhere(grid == 0)

    # members live within 3 miles of a ZIP center (more people in the bigger areas)
    w = np.ones(len(zips)); w[zips["area"].isin(["Hialeah", "Hollywood", "Kendall", "Miami Gardens"])] = 2.0
    rows = []
    for i in range(N_MEMBERS):
        z = zips.iloc[rng.choice(len(zips), p=w / w.sum())]
        near = open_cells[(np.abs(open_cells[:, 0] - z["row"]) <= 3) & (np.abs(open_cells[:, 1] - z["col"]) <= 3)]
        r, c = near[rng.integers(len(near))]
        age = int(np.clip(rng.normal(58, 17), 18, 95))
        p_high = 0.06 + 0.10 * (age >= 65) + 0.08 * (age >= 80)      # older members more likely high-cost
        rows.append({"member_id": f"M{i + 1:04d}", "zip": z["zip"], "row": int(r), "col": int(c), "age": age,
                     "need_specialty": rng.choice(list(NEED), p=list(NEED.values())),
                     "high_cost": int(rng.random() < p_high)})
    mem = pd.DataFrame(rows)
    mem.to_csv(ROOT / "data" / "members.csv", index=False)

    # open slots: only in-network providers accepting new patients get any
    prov["new_slots"] = 0
    ok = (prov["in_network"] == 1) & (prov["accepting_new"] == 1) & (prov["panel_size"] < prov["panel_capacity"])
    demand = mem["need_specialty"].value_counts()
    for spec, g in prov[ok].groupby("specialty"):
        raw = rng.uniform(0.5, 1.5, len(g))                         # some offices have more room than others
        target = int(np.ceil(demand.get(spec, 0) * SLACK))
        slots = np.floor(raw / raw.sum() * target).astype(int)
        slots[np.argsort(-raw)[: target - slots.sum()]] += 1        # hand out the remainder
        prov.loc[g.index, "new_slots"] = slots
    prov["high_cost_cap"] = np.ceil(prov["new_slots"] * HIGH_COST_SHARE).astype(int)

    # provider group = the nearest ZIP area (used for the "big group leaves" stress test)
    zr, zc = zips["row"].to_numpy(), zips["col"].to_numpy()
    prov["group"] = [zips["area"].iloc[int(np.argmin(np.abs(zr - r) + np.abs(zc - c)))] + " Medical Group"
                     for r, c in zip(prov["row"], prov["col"])]
    prov[["provider_id", "new_slots", "high_cost_cap", "group"]].to_csv(KB / "provider_slots.csv", index=False)
    print(f"{len(mem)} members ({mem.high_cost.mean():.1%} high-cost) | open slots by specialty:")
    print(pd.DataFrame({"demand": demand, "slots": prov.groupby("specialty")["new_slots"].sum()}).to_string())


if __name__ == "__main__":
    main()
