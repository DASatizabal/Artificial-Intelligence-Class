"""Stage 1: score every planner member with all three MI Parts.

For the 600 members the AI Part 1 planner seats, using their latest test month (2026-02, the same
month MI Part 2 used for its hand-off):
  MI Part 1  predicted next-quarter cost in dollars (linear regression, the MI Part 1 champion).
             Linear is used, not the log model, because the planner ADDS costs up per doctor, and
             a model of the typical (median) member undercounts totals.
  MI Part 2  P(high-cost) and the top-10% flag (read from its hand-off file, no refit).
  MI Part 3  MC Dropout spread (how unsure the MLP is) -> "needs review" list.
             HMM chance of a high-cost spell 3 months out -> "watch list".
Each model is retrained exactly the way its Part trains it (same code, same seed), and the script
checks that the retrained model reproduces the number its Part reported.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parts import load, MI1, MI2, MI3

MONTH = "2026-02"
PLANNER_MEMBERS = 600          # M0001 to M0600 are the AI Part 1 members
REVIEW_SHARE = 0.10            # the 10% of members the MLP is least sure about go to a person


def mi1_cost(df_all):
    """MI Part 1 linear regression, trained on the same rows and features."""
    m1 = load(MI1, "run_mi_part1")
    df, _ = m1.audit(df_all)
    train = df[(df.month >= m1.TRAIN[0]) & (df.month <= m1.TRAIN[1])]
    test = df[(df.month >= m1.TEST[0]) & (df.month <= m1.TEST[1])]
    pre = ColumnTransformer([("num", StandardScaler(), m1.NUM),
                             ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Unknown"),
                                                   OneHotEncoder(handle_unknown="ignore")), m1.CAT)])
    lin = make_pipeline(clone(pre), LinearRegression()).fit(train[m1.NUM + m1.CAT], train[m1.TARGET])
    mae = mean_absolute_error(test[m1.TARGET], lin.predict(test[m1.NUM + m1.CAT]))
    reported = next(r["MAE"] for r in json.load(open(MI1 / "outputs/metrics.json"))["scores"] if r["model"] == "Linear regression")
    assert abs(mae - reported) < 0.01, f"MI Part 1 linear model did not reproduce: {mae} vs {reported}"
    last = df[df.month == MONTH]
    return pd.Series(lin.predict(last[m1.NUM + m1.CAT]), index=last["member_id"].to_numpy()), round(mae, 3)


def mi3_scores():
    """MI Part 3 MLP + MC Dropout and HMM, trained exactly as run_part3.py trains them."""
    common = load(MI3, "common"); mlpm = load(MI3, "mlp_baseline"); bh = load(MI3, "bayes_hmm")
    common.set_seed()
    X, y, frames, cutoff, _ = common.tabular()
    counts = np.bincount(y["fit"]); cw = (len(y["fit"]) / (2 * counts)).tolist()
    common.set_seed()
    mlp = mlpm.MLP(X["fit"].shape[1], 2, 128, 64, 0.2)
    mlp, _, _ = mlpm.train(mlp, X["fit"], y["fit"], X["val"], y["val"], class_weight=cw)
    auc = roc_auc_score(y["test"], mlpm.predict_proba(mlp, X["test"])[:, 1])
    reported = next(r["test_roc_auc"] for r in json.load(open(MI3 / "outputs/metrics.json"))["tabular"]
                    if r["model"].startswith("MLP (128-64"))
    assert abs(auc - reported) < 1e-3, f"MI Part 3 MLP did not reproduce: {auc} vs {reported}"
    mc_mean, mc_var = mlpm.mc_dropout(mlp, X["test"])          # all test rows, same as MI Part 3
    t = frames["test"][["member_id", "month"]].copy()
    t["p_high_mlp_mc"] = mc_mean[:, 1]; t["sd_mlp_mc"] = np.sqrt(mc_var[:, 1])

    # HMM: same training sequences and same spell line as MI Part 3
    df = pd.read_csv(MI3 / "data/member_months.csv", dtype={"zip": str}).drop_duplicates().sort_values(["member_id", "month"])
    spell_line = float(df.loc[(df.month >= common.TRAIN[0]) & (df.month <= common.TRAIN[1]), "cost_prior_3m"].quantile(0.90))
    fit_ids = set(frames["fit"]["member_id"])
    tr = df[(df.month <= common.TRAIN[1]) & df.member_id.isin(fit_ids)]
    seqs = [(g["cost_prior_3m"].ge(spell_line).astype(int).to_numpy(), bh.band(g["cost_month"].to_numpy())) for _, g in tr.groupby("member_id")]
    hmm = bh.CostHMM().fit([s for s, _ in seqs], [o for _, o in seqs])
    flag_line = json.load(open(MI3 / "outputs/metrics.json"))["hmm"]["flag_line"]
    last = t[t.month == MONTH].copy()
    hist = {mid: g for mid, g in df.groupby("member_id")}
    last["p_spell_3m_hmm"] = [hmm.risk_ahead(bh.band(hist[m].loc[hist[m].month <= MONTH, "cost_month"].to_numpy())) for m in last["member_id"]]
    return last.set_index("member_id"), {"mlp_test_roc_auc": round(auc, 4), "hmm_flag_line": flag_line}


def score_members():
    df_all = pd.read_csv(MI1 / "data/member_months.csv", dtype={"zip": str})
    cost, mi1_mae = mi1_cost(df_all)
    flags = pd.read_csv(MI2 / "outputs/predicted_high_cost_flags.csv").set_index("member_id")
    mi3, mi3_check = mi3_scores()
    ids = sorted(flags.index)
    assert len(ids) == PLANNER_MEMBERS
    s = pd.DataFrame(index=pd.Index(ids, name="member_id"))
    s["pred_cost_next_3m_raw"] = cost.reindex(ids).to_numpy()
    s["pred_cost_next_3m"] = s["pred_cost_next_3m_raw"].clip(lower=0)   # a negative bill is not possible
    s["p_high_cost"] = flags.loc[ids, "p_high_cost"]
    s["high_cost_flag"] = flags.loc[ids, "high_cost_flag"].astype(int)
    s["p_high_mlp_mc"] = mi3.loc[ids, "p_high_mlp_mc"]
    s["sd_mlp_mc"] = mi3.loc[ids, "sd_mlp_mc"]
    s["p_spell_3m_hmm"] = mi3.loc[ids, "p_spell_3m_hmm"]
    review_line = float(s["sd_mlp_mc"].quantile(1 - REVIEW_SHARE))
    s["needs_review"] = (s["sd_mlp_mc"] >= review_line).astype(int)
    # watch list: HMM says a spell is likely, but the main model did not flag the member
    s["watch_list"] = ((s["p_spell_3m_hmm"] >= mi3_check["hmm_flag_line"]) & (s["high_cost_flag"] == 0)).astype(int)
    s = s.round({"pred_cost_next_3m_raw": 2, "pred_cost_next_3m": 2, "p_high_mlp_mc": 4, "sd_mlp_mc": 4, "p_spell_3m_hmm": 4})
    info = {"month": MONTH, "members": len(s), "mi1_linear_test_mae_check": mi1_mae, **mi3_check,
            "negative_cost_predictions_clipped": int((s["pred_cost_next_3m_raw"] < 0).sum()),
            "high_cost_flagged": int(s["high_cost_flag"].sum()),
            "needs_review": int(s["needs_review"].sum()), "review_sd_line": round(review_line, 4),
            "needs_review_and_flagged": int(((s["needs_review"] == 1) & (s["high_cost_flag"] == 1)).sum()),
            "watch_list": int(s["watch_list"].sum()),
            "predicted_cost_total": round(float(s["pred_cost_next_3m"].sum()), 2),
            "flagged_share_of_predicted_cost": round(float(s.loc[s.high_cost_flag == 1, "pred_cost_next_3m"].sum() / s["pred_cost_next_3m"].sum()), 4)}
    return s.reset_index(), info


if __name__ == "__main__":
    s, info = score_members()
    print(json.dumps(info, indent=1)); print(s.head())
