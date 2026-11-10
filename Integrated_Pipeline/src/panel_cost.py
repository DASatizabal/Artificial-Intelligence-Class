"""Stage 2: run the AI Part 1 planner on the scored members and put dollars on every doctor's panel.

The planner itself is unchanged (MRV + forward checking + max-flow look-ahead). What is new:
  - each placed member brings their MI Part 1 predicted cost to the doctor they are seated with;
  - the run is repeated with the high-cost cap switched off, to measure what the cap buys:
    does spreading MI Part 2's flagged members also spread the predicted dollars?
  - the seats each doctor still has open are handed to the front door (Stage 3), so a chat request
    is matched to a doctor who can actually take a new patient.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parts import load, AI1


def build(scores):
    kbm = load(AI1, "kb"); plm = load(AI1, "planner")
    kb = kbm.KnowledgeBase(AI1 / "data/kb")
    members = pd.read_csv(AI1 / "data/members.csv", dtype={"zip": str}).rename(columns={"high_cost": "high_cost_placeholder"})
    members = members.merge(scores[["member_id", "high_cost_flag", "pred_cost_next_3m"]], on="member_id", how="left")
    assert members["high_cost_flag"].notna().all()
    members["high_cost"] = members["high_cost_flag"].astype(int)
    slots = pd.read_csv(AI1 / "data/kb/provider_slots.csv")
    return kb, members, slots, plm


def panel_table(pl, assign, members):
    cost = members.set_index("member_id")["pred_cost_next_3m"]
    hc = members.set_index("member_id")["high_cost"]
    rows = []
    for pid in sorted(set(p for p in assign.values() if p)):
        mids = [m for m, p in assign.items() if p == pid]
        rows.append({"provider_id": pid, "specialty": pl.prov.at[pid, "specialty"], "group": pl.prov.at[pid, "group"],
                     "new_slots": int(pl.slots[pid]), "high_cost_cap": int(pl.cap[pid]), "members_seated": len(mids),
                     "high_cost_seated": int(hc[mids].sum()), "predicted_panel_cost": round(float(cost[mids].sum()), 2),
                     "predicted_cost_per_member": round(float(cost[mids].mean()), 2)})
    return pd.DataFrame(rows)


def spread(tbl):
    """How evenly the predicted dollars are spread. Each doctor is compared with others of the same
    specialty (a cardiologist's panel is not compared with a dermatologist's)."""
    per = tbl["predicted_cost_per_member"]
    within = tbl.groupby("specialty")["predicted_cost_per_member"].agg(lambda s: s.max() / s.mean() if len(s) > 1 else 1.0)
    return {"panels": len(tbl), "max_high_cost_in_one_panel": int(tbl["high_cost_seated"].max()),
            "highest_cost_per_member": round(float(per.max()), 2), "lowest_cost_per_member": round(float(per.min()), 2),
            "cost_per_member_cv": round(float(per.std() / per.mean()), 4),
            "worst_panel_vs_specialty_average": round(float(within.max()), 3)}


def run(scores):
    kb, members, slots, plm = build(scores)
    r1 = load(AI1, "run_part1")
    pl = plm.Planner(kb, members, slots).build_domains()
    gaps = list(pl.gaps)
    assign, _, cap_gaps = r1.solve_all(pl)
    base = r1.summarize(pl, assign, gaps, cap_gaps)
    reported = json.load(open(AI1 / "outputs/metrics.json"))["baseline"]
    assert (base["placed"], base["avg_miles"]) == (reported["placed"], reported["avg_miles"]), "planner did not reproduce AI Part 1"
    with_cap = panel_table(pl, assign, members)

    # same planner, high-cost cap switched off (cap = all open seats)
    pl_nc = plm.Planner(kb, members, slots)
    pl_nc.cap = dict(pl_nc.slots)
    pl_nc.build_domains()
    assign_nc, _, cap_gaps_nc = r1.solve_all(pl_nc)
    base_nc = r1.summarize(pl_nc, assign_nc, list(pl_nc.gaps), cap_gaps_nc)
    no_cap = panel_table(pl_nc, assign_nc, members)

    # seats each doctor still has after the plan (handed to the front door)
    used = pd.Series([p for p in assign.values() if p]).value_counts()
    hc_used = pd.Series([p for m, p in assign.items() if p and pl.members.at[m, "high_cost"] == 1]).value_counts()
    seats = pd.DataFrame({"provider_id": list(pl.slots)})
    seats["seats_left"] = [int(pl.slots[p] - used.get(p, 0)) for p in seats.provider_id]
    seats["high_cost_seats_left"] = [int(pl.cap[p] - hc_used.get(p, 0)) for p in seats.provider_id]

    # sensitivity: flag a bigger share of members (by MI Part 2 probability) and see when the cap starts to bite
    sens = []
    rank = scores.set_index("member_id")["p_high_cost"].rank(ascending=False, method="first")
    for share in (0.10, 0.20, 0.30, 0.40):
        m2 = members.copy()
        m2["high_cost"] = (m2["member_id"].map(rank) <= round(share * len(m2))).astype(int)
        row = {"flagged_share": share, "flagged": int(m2["high_cost"].sum())}
        for tag, relax in (("cap", False), ("no_cap", True)):
            p2 = plm.Planner(kb, m2, slots)
            if relax:
                p2.cap = dict(p2.slots)
            p2.build_domains()
            a2, _, cg2 = r1.solve_all(p2)
            row[f"placed_{tag}"] = sum(1 for p in a2.values() if p)
            hc_by = pd.Series([p for m, p in a2.items() if p and p2.members.at[m, "high_cost"] == 1]).value_counts()
            row[f"max_high_cost_in_one_panel_{tag}"] = int(hc_by.max()) if len(hc_by) else 0
            row[f"panels_over_real_cap_{tag}"] = int(sum(hc_by[p] > pl.cap[p] for p in hc_by.index))
        sens.append(row)

    placed = members[members.member_id.isin([m for m, p in assign.items() if p])]
    info = {"placed": base["placed"], "avg_miles": base["avg_miles"], "distance_gaps": base["distance_gaps"],
            "capacity_gaps": base["capacity_gaps"], "high_cost_cap_violations": base["high_cost_cap_violations"],
            "predicted_cost_placed_members": round(float(placed["pred_cost_next_3m"].sum()), 2),
            "predicted_cost_unplaced_members": round(float(members.loc[~members.member_id.isin(placed.member_id), "pred_cost_next_3m"].sum()), 2),
            "with_high_cost_cap": {"placed": base["placed"], "avg_miles": base["avg_miles"], **spread(with_cap)},
            "without_high_cost_cap": {"placed": base_nc["placed"], "avg_miles": base_nc["avg_miles"], **spread(no_cap)},
            "doctors_with_seats_left": int((seats.seats_left > 0).sum()), "seats_left_total": int(seats.seats_left.sum()),
            "cap_sensitivity": sens}
    return {"assign": assign, "planner": pl, "kb": kb, "members": members, "seats": seats,
            "panels": with_cap, "panels_no_cap": no_cap, "info": info}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from member_scores import score_members
    s, _ = score_members()
    out = run(s)
    print(json.dumps(out["info"], indent=1))
