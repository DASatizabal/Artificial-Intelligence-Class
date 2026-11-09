"""MI Part 1: data audit, exploratory data analysis (EDA) and a baseline regression for next-quarter cost.
Run from the project folder:  python src/run_mi_part1.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
TARGET = "cost_next_3m"
NUM = ["age", "chronic_conditions", "miles_to_pcp", "saw_pcp_last_3m", "er_visits_6m", "admits_12m",
       "cost_month", "cost_prior_3m", "cost_prior_12m"]
CAT = ["sex", "county", "plan_type"]
# time split with a 3-month gap: a training row's answer covers the next 3 months, so training rows must end
# 3 months before the test period starts, or the training answers would overlap the test months
TRAIN = ("2025-01", "2025-09"); GAP = ("2025-10", "2025-12"); TEST = ("2026-01", "2026-02")


def audit(raw):
    df = raw.drop_duplicates().reset_index(drop=True)
    rows = [
        ("Shape", f"{raw.shape[0]:,} rows x {raw.shape[1]} columns", "None"),
        ("Duplicate rows", f"{raw.duplicated().sum()}", "Removed (double-loaded rows)"),
        ("Members / months", f"{df.member_id.nunique():,} members, {df.month.nunique()} months ({df.month.min()} to {df.month.max()})", "Split by month, never randomly"),
        ("Missing plan_type", f"{df.plan_type.isna().sum()} ({df.plan_type.isna().mean():.1%})", "Fill with 'Unknown' inside the pipeline"),
        ("Other missing values", f"{int(df.drop(columns='plan_type').isna().sum().sum())}", "None"),
        ("Negative cost_month", f"{(df.cost_month < 0).sum()} rows, min ${df.cost_month.min():,.0f}", "Kept: claim reversals are real refunds"),
        ("Age range", f"{df.age.min()} to {df.age.max()}", "Valid"),
        ("Miles to PCP", f"{df.miles_to_pcp.min():.0f} to {df.miles_to_pcp.max():.0f}", "Valid (CMS limit for PCP is 5)"),
        ("Target cost_next_3m", f"median ${df[TARGET].median():,.0f}, mean ${df[TARGET].mean():,.0f}, max ${df[TARGET].max():,.0f}", "Very skewed: also model log(cost)"),
        ("Top 10% share of cost", f"{df[TARGET].nlargest(int(len(df) * 0.1)).sum() / df[TARGET].sum():.1%}", "The few drive most spend (sets up MI Part 2)"),
    ]
    return df, pd.DataFrame(rows, columns=["Check", "Finding", "Action"])


def eda_figures(df):
    d = df.copy(); d["month_dt"] = pd.to_datetime(d["month"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(d[TARGET].clip(upper=20000), bins=60, color="#1f5fbf")
    ax[0, 0].set_title("Next-quarter cost (capped at $20k for display)"); ax[0, 0].set_xlabel("dollars"); ax[0, 0].set_ylabel("rows")
    ax[0, 1].hist(np.log1p(d[TARGET]), bins=60, color="#0e7c86")
    ax[0, 1].set_title("Same cost after log(1 + cost)"); ax[0, 1].set_xlabel("log dollars")
    by_cc = d.groupby(d["chronic_conditions"].clip(upper=5))[TARGET].mean()
    ax[1, 0].bar([str(i) if i < 5 else "5+" for i in by_cc.index], by_cc.values, color="#b85c00")
    ax[1, 0].set_title("Average next-quarter cost by chronic conditions"); ax[1, 0].set_xlabel("chronic conditions"); ax[1, 0].set_ylabel("dollars")
    m = d.groupby("month_dt").agg(cost=(TARGET, "mean"), miles=("miles_to_pcp", "mean"))
    ax[1, 1].plot(m.index, m["cost"], "o-", color="#b42318"); ax[1, 1].set_ylabel("avg next-quarter cost ($)", color="#b42318")
    a2 = ax[1, 1].twinx(); a2.plot(m.index, m["miles"], "s--", color="#555"); a2.set_ylabel("avg miles to PCP", color="#555")
    ax[1, 1].axvline(pd.Timestamp("2026-06-01"), color="#0b2a5b", ls=":"); ax[1, 1].set_title("Monthly trend (dotted line = Pinecrest leaves)")
    ax[1, 1].tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(OUT / "eda_overview.png", dpi=140); plt.close()

    corr = d[NUM + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig, axc = plt.subplots(figsize=(6.5, 4))
    axc.barh(corr.index, corr.values, color=["#b42318" if v < 0 else "#1f5fbf" for v in corr.values])
    axc.set_title("Correlation with next-quarter cost"); axc.set_xlabel("Pearson r")
    plt.tight_layout(); plt.savefig(OUT / "correlations.png", dpi=140); plt.close()

    band = pd.cut(d["miles_to_pcp"], [-1, 2, 5, 8, 20], labels=["0-2", "3-5", "6-8", "9+"])
    access = d.groupby(band, observed=True).agg(rows=(TARGET, "size"), saw_pcp=("saw_pcp_last_3m", "mean"),
                                                er_6m=("er_visits_6m", "mean"), next_q_cost=(TARGET, "mean"))
    return corr.round(3), access.round(3)


def main():
    raw = pd.read_csv(ROOT / "data" / "member_months.csv", dtype={"zip": str})
    df, audit_tbl = audit(raw)
    audit_tbl.to_csv(OUT / "audit.csv", index=False)
    corr, access = eda_figures(df)

    train = df[(df.month >= TRAIN[0]) & (df.month <= TRAIN[1])]
    test = df[(df.month >= TEST[0]) & (df.month <= TEST[1])]
    Xtr, ytr, Xte, yte = train[NUM + CAT], train[TARGET], test[NUM + CAT], test[TARGET]

    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Unknown"),
                                                   OneHotEncoder(handle_unknown="ignore")), CAT)])
    lin = make_pipeline(clone(pre), LinearRegression()).fit(Xtr, ytr)     # clone: each model gets its own copy
    # log model: dollars are squeezed with log(1 + x) on BOTH sides, so huge bills don't dominate the fit
    COST = ["cost_month", "cost_prior_3m", "cost_prior_12m"]
    logx = FunctionTransformer(lambda X: X.assign(**{c: np.log1p(X[c].clip(lower=0)) for c in COST}))
    loglin = make_pipeline(logx, clone(pre), LinearRegression()).fit(Xtr, np.log1p(ytr))
    preds = {
        "Mean of training rows": np.full(len(yte), ytr.mean()),
        "Last quarter repeats": test["cost_prior_3m"].to_numpy(),
        "Linear regression": lin.predict(Xte),
        "Log model, typical member (median)": np.expm1(loglin.predict(Xte)),
    }
    scores = pd.DataFrame([{"model": k, "MAE": mean_absolute_error(yte, p), "RMSE": mean_squared_error(yte, p) ** 0.5,
                            "R2": r2_score(yte, p)} for k, p in preds.items()]).round(3)

    names = NUM + list(lin[0].named_transformers_["cat"][-1].get_feature_names_out(CAT))
    coefs = pd.Series(lin[-1].coef_, index=names).sort_values(key=abs, ascending=False).round(1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(yte, preds["Linear regression"], s=5, alpha=0.3, color="#1f5fbf")
    lim = [0, 30000]; ax.plot(lim, lim, "--", color="gray"); ax.set_xlim(lim); ax.set_ylim(-3000, 30000)
    ax.set_xlabel("actual next-quarter cost ($)"); ax.set_ylabel("predicted ($)"); ax.set_title("Linear regression on the test months")
    plt.tight_layout(); plt.savefig(OUT / "pred_vs_actual.png", dpi=140); plt.close()

    metrics = {"rows_after_dedup": len(df), "train_rows": len(train), "test_rows": len(test),
               "split": {"train": TRAIN, "gap": GAP, "test": TEST}, "scores": scores.to_dict("records"),
               "linear_coefficients_per_1sd": coefs.to_dict(), "correlations": corr.to_dict(),
               "access_bands": access.reset_index().astype({"miles_to_pcp": str}).to_dict("records"),
               "negative_predictions_linear": int((preds["Linear regression"] < 0).sum())}
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(audit_tbl.to_string(index=False)); print(access); print(corr)
    print(scores.to_string(index=False)); print(coefs.head(10)); print("negative linear predictions:", metrics["negative_predictions_linear"])
    print(f"train {len(train):,} | test {len(test):,}")


if __name__ == "__main__":
    main()
