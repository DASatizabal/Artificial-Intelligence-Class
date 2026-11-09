"""Generate the member-month claims table for the Machine Intelligence side of the project.

Synthetic and seeded (seed 42). No real people, providers or claims. Licensed for course use.

  data/member_months.csv   1,500 members x 21 months (Jan 2025 to Sep 2026) = about 31,500 rows
                           plus a few planted data-quality problems for the audit

How it is built
  * Members M0001-M0600 are the same members as AI Part 1; M0601-M1500 are added the same way.
  * Each member's distance to the nearest in-network primary care doctor comes from A* on the county grid.
  * 2024 is simulated as a warm-up year so every 2025 row has a full 12 months of history.
  * Every month: the member may see their PCP (less likely when the doctor is far), may go to the ER
    (more likely when sick, far from a doctor, or no recent checkup), may be admitted, and fills prescriptions.
  * Planned change on 2026-06-01: Pinecrest Medical Group leaves the network. Its members now live farther
    from a PCP, see their doctor less, and missing checkups now pushes ER use up more (data + concept drift).
Assumptions are mine, not measured: each extra mile to the PCP raises ER visits by 10%, and a member with no
PCP visit in the last 3 months has 1.8x the ER rate (2.6x after the drift date).
Prediction target: cost_next_3m = total paid in the NEXT three months. Every other column is known at the
end of the row's month, so nothing comes from the future.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb import KnowledgeBase

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
N_TOTAL = 1500
START, PUBLISH, END = "2024-01-01", "2025-01-01", "2026-12-01"   # simulate through Dec 2026 so Sep 2026 has a target
LAST_ROW = "2026-09-01"
DRIFT = pd.Timestamp("2026-06-01")
LEAVING_GROUP = "Pinecrest Medical Group"
BROWARD = {"33309", "33317", "33024", "33025", "33027", "33029"}


def add_members(rng, kb, base):
    """Keep the 600 AI Part 1 members and add more, placed the same way (near a ZIP center)."""
    zips = kb.zips.reset_index()
    open_cells = np.argwhere(kb.grid == 0)
    w = np.ones(len(zips)); w[zips["area"].isin(["Hialeah", "Hollywood", "Kendall", "Miami Gardens"])] = 2.0
    rows = []
    for i in range(len(base), N_TOTAL):
        z = zips.iloc[rng.choice(len(zips), p=w / w.sum())]
        near = open_cells[(np.abs(open_cells[:, 0] - z["row"]) <= 3) & (np.abs(open_cells[:, 1] - z["col"]) <= 3)]
        r, c = near[rng.integers(len(near))]
        rows.append({"member_id": f"M{i + 1:04d}", "zip": z["zip"], "row": int(r), "col": int(c),
                     "age": int(np.clip(rng.normal(58, 17), 18, 95))})
    mem = pd.concat([base[["member_id", "zip", "row", "col", "age"]], pd.DataFrame(rows)], ignore_index=True)
    mem["sex"] = rng.choice(["F", "M"], len(mem), p=[0.54, 0.46])
    mem["chronic_conditions"] = np.minimum(rng.poisson(0.3 + 0.035 * np.clip(mem["age"] - 30, 0, None)), 8)
    over65 = mem["age"] >= 65
    mem["plan_type"] = np.where(over65, "Medicare Advantage",
                                rng.choice(["Medicaid", "Marketplace"], len(mem), p=[0.55, 0.45]))
    mem["county"] = np.where(mem["zip"].isin(BROWARD), "Broward", "Miami-Dade")
    mem["frailty"] = np.exp(rng.normal(0, 0.35, len(mem)))          # hidden: how sick beyond what's recorded
    return mem


def pcp_miles(kb, mem, exclude_group=None, groups=None):
    """A* miles from each member to the nearest in-network PCP (optionally without one provider group)."""
    p = kb.providers[(kb.providers["specialty"] == "primary care") & (kb.providers["in_network"] == 1)]
    if exclude_group is not None:
        p = p[~p["provider_id"].isin(groups.index[groups == exclude_group])]
    out = []
    for r, c in zip(mem["row"], mem["col"]):
        cand = sorted((abs(pr - r) + abs(pc - c), pr, pc) for pr, pc in zip(p["row"], p["col"]))
        best = None
        for lb, pr, pc in cand:                  # Manhattan lower bound lets us stop early
            if best is not None and lb >= best:
                break
            d = kb.astar((int(r), int(c)), (int(pr), int(pc)))
            if d is not None and (best is None or d < best):
                best = d
        out.append(best)
    return np.array(out, dtype=float)


def main():
    rng = np.random.default_rng(SEED)
    kb = KnowledgeBase(ROOT / "data" / "kb")
    base = pd.read_csv(ROOT / "data" / "members_part1.csv", dtype={"zip": str})
    mem = add_members(rng, kb, base)
    groups = pd.read_csv(ROOT / "data" / "kb" / "provider_slots.csv").set_index("provider_id")["group"]
    miles_before = pcp_miles(kb, mem)
    miles_after = pcp_miles(kb, mem, LEAVING_GROUP, groups)
    months = pd.date_range(START, END, freq="MS")
    n, T = len(mem), len(months)

    age = mem["age"].to_numpy(); chronic = mem["chronic_conditions"].to_numpy(); frail = mem["frailty"].to_numpy()
    price = mem["plan_type"].map({"Medicaid": 0.8, "Medicare Advantage": 1.0, "Marketplace": 1.1}).to_numpy()
    risk = np.exp(0.025 * (age - 60) + 0.30 * chronic) * frail       # how much care this member tends to need

    pcp = np.zeros((n, T), int); er = np.zeros((n, T), int); adm = np.zeros((n, T), int); cost = np.zeros((n, T))
    miles_hist = np.zeros((n, T))
    for t, mth in enumerate(months):
        after = mth >= DRIFT
        miles = miles_after if after else miles_before
        miles_hist[:, t] = miles
        recent_pcp = pcp[:, max(0, t - 3):t].sum(axis=1) > 0
        p_pcp = np.clip(0.30 + 0.06 * chronic - 0.035 * miles, 0.03, 0.85)
        pcp[:, t] = rng.random(n) < p_pcp
        flu = 1.3 if mth.month in (12, 1, 2) else 1.0
        missed = np.where(recent_pcp, 1.0, 2.6 if after else 1.8)     # concept drift: missing checkups hurts more
        er[:, t] = rng.poisson(0.035 * risk * (1 + 0.10 * miles) * missed * flu)
        adm[:, t] = rng.poisson(0.008 * risk * (1 + 0.8 * er[:, t]))
        rx = 35 * chronic * rng.lognormal(0, 0.4, n)
        specialist = rng.poisson(0.15 * risk) * 220 * rng.lognormal(0, 0.3, n)
        cost[:, t] = price * (rx + pcp[:, t] * 160 + specialist
                              + er[:, t] * 1800 * rng.lognormal(0, 0.5, n)
                              + adm[:, t] * 14000 * rng.lognormal(0, 0.6, n))

    rows = []
    for t, mth in enumerate(months):
        if mth < pd.Timestamp(PUBLISH) or mth > pd.Timestamp(LAST_ROW):
            continue
        rows.append(pd.DataFrame({
            "member_id": mem["member_id"], "month": mth.strftime("%Y-%m"), "age": age, "sex": mem["sex"],
            "zip": mem["zip"], "county": mem["county"], "plan_type": mem["plan_type"],
            "chronic_conditions": chronic, "miles_to_pcp": miles_hist[:, t],
            "saw_pcp_last_3m": (pcp[:, t - 2:t + 1].sum(axis=1) > 0).astype(int),
            "er_visits_6m": er[:, t - 5:t + 1].sum(axis=1), "admits_12m": adm[:, t - 11:t + 1].sum(axis=1),
            "cost_month": cost[:, t].round(2), "cost_prior_3m": cost[:, t - 2:t + 1].sum(axis=1).round(2),
            "cost_prior_12m": cost[:, t - 11:t + 1].sum(axis=1).round(2),
            "cost_next_3m": cost[:, t + 1:t + 4].sum(axis=1).round(2)}))
    df = pd.concat(rows, ignore_index=True)

    # planted data-quality problems for the audit
    idx = rng.choice(len(df), 40, replace=False)
    df.loc[idx, "cost_month"] = -(df.loc[idx, "cost_month"].abs() + rng.uniform(50, 400, 40)).round(2)  # claim reversals
    miss = rng.choice(len(df), int(0.015 * len(df)), replace=False)
    df.loc[miss, "plan_type"] = np.nan                                  # enrollment feed gaps
    dup = df.sample(63, random_state=7)                                 # double-loaded rows
    df = pd.concat([df, dup]).sort_values(["month", "member_id"], kind="stable").reset_index(drop=True)
    df.to_csv(ROOT / "data" / "member_months.csv", index=False)
    print(f"{len(df):,} rows | {df.member_id.nunique()} members | {df.month.min()} to {df.month.max()}")
    print(f"miles to PCP: mean {miles_before.mean():.2f} before, {miles_after.mean():.2f} after Pinecrest leaves; "
          f"{(miles_after > miles_before).sum()} members farther")


if __name__ == "__main__":
    main()
