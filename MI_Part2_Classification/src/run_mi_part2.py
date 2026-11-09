"""MI Part 2: classification pipeline for the high-cost flag.

Question: will this member be HIGH-COST next quarter (top 10% of next-quarter cost)? Yes / no.
The flag feeds the AI Part 1 planner (spread heavy patients, place the sickest first).
Run from the project folder:  python src/run_mi_part2.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
NUM = ["age", "chronic_conditions", "miles_to_pcp", "saw_pcp_last_3m", "er_visits_6m", "admits_12m",
       "cost_month", "cost_prior_3m", "cost_prior_12m"]
CAT = ["sex", "county", "plan_type"]
TRAIN = ("2025-01", "2025-09"); TEST = ("2026-01", "2026-02")      # same split and 3-month gap as MI Part 1


def preprocessor():
    """The assembly line: scale the numbers, turn categories into 0/1 columns (missing plan -> 'Unknown')."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT)])


MODELS = {
    "Majority baseline (always 'no')": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=2000),
    "Logistic regression, balanced": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision tree, balanced": DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=SEED),
}


def metrics(y, pred, prob):
    out = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}
    out["roc_auc"] = roc_auc_score(y, prob) if len(np.unique(prob)) > 1 else 0.5
    out["pr_auc"] = average_precision_score(y, prob)
    return {k: round(float(v), 4) for k, v in out.items()}


def pipeline_diagram(path):
    steps = [("Member-month\nrows", "#c8d3e6"), ("Remove 63\nduplicates", "#c8d3e6"), ("Time split\ntrain / gap / test", "#c8d3e6"),
             ("Numbers:\nStandardScaler", "#d7ece9"), ("Categories:\nfill 'Unknown' +\nOneHotEncoder", "#d7ece9"),
             ("Classifier\n(LR or tree)", "#f3e3c8"), ("Evaluate:\nCV by member +\ntest months", "#f2d4d0")]
    fig, ax = plt.subplots(figsize=(12, 2.6)); ax.axis("off")
    x = 0.01
    for i, (t, c) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 0.25), 0.12, 0.5, boxstyle="round,pad=0.01", fc=c, ec="#333"))
        ax.text(x + 0.06, 0.5, t, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.142, 0.5), xytext=(x + 0.122, 0.5), arrowprops=dict(arrowstyle="->"))
        x += 0.142
    ax.plot([0.01 + 3 * 0.142, 0.01 + 4 * 0.142 + 0.12], [0.2, 0.2], color="#0e7c86", lw=1.5)
    ax.text(0.01 + 3 * 0.142 + 0.131, 0.1, "ColumnTransformer (fit on training rows only)", ha="center", fontsize=8.5, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("MI Part 2 pipeline: preprocessing -> training -> evaluation", fontsize=11)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def main():
    df = pd.read_csv(ROOT / "data" / "member_months.csv", dtype={"zip": str}).drop_duplicates().reset_index(drop=True)
    train = df[(df.month >= TRAIN[0]) & (df.month <= TRAIN[1])].reset_index(drop=True)
    test = df[(df.month >= TEST[0]) & (df.month <= TEST[1])].reset_index(drop=True)
    cutoff = float(train["cost_next_3m"].quantile(0.90))          # "high-cost" line is set from TRAINING rows only
    ytr = (train["cost_next_3m"] >= cutoff).astype(int); yte = (test["cost_next_3m"] >= cutoff).astype(int)
    Xtr, Xte = train[NUM + CAT], test[NUM + CAT]

    # cross-validation: 5 folds, same high-cost mix in each, and every member's months stay in ONE fold
    # (otherwise the model sees a member's March row in training and is graded on their April row)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(Xtr, ytr, groups=train["member_id"]))
    cv_rows, test_rows, fitted, probs = [], [], {}, {}
    for name, est in MODELS.items():
        per_fold = []
        for a, b in folds:
            pipe = Pipeline([("pre", preprocessor()), ("model", clone(est))]).fit(Xtr.iloc[a], ytr.iloc[a])
            per_fold.append(metrics(ytr.iloc[b], pipe.predict(Xtr.iloc[b]), pipe.predict_proba(Xtr.iloc[b])[:, 1]))
        f = pd.DataFrame(per_fold)
        cv_rows.append({"model": name, **{f"{k}_mean": round(f[k].mean(), 4) for k in f}, **{f"{k}_sd": round(f[k].std(), 4) for k in ("recall", "roc_auc")}})
        pipe = Pipeline([("pre", preprocessor()), ("model", clone(est))]).fit(Xtr, ytr)
        fitted[name] = pipe; probs[name] = pipe.predict_proba(Xte)[:, 1]
        test_rows.append({"model": name, **metrics(yte, pipe.predict(Xte), probs[name])})
    cv_tbl, test_tbl = pd.DataFrame(cv_rows), pd.DataFrame(test_rows)

    # confusion matrices on the test months
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        cm = confusion_matrix(yte, pipe.predict(Xte), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["not high", "high"]); ax.set_yticks([0, 1], ["not high", "high"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(name, fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrices.png", dpi=140); plt.close()

    # ROC and precision-recall curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, p in probs.items():
        if name.startswith("Majority"):
            continue
        fpr, tpr, _ = roc_curve(yte, p); ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
        pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(yte, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="gray", label="coin flip"); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate (recall)")
    ax[0].set_title("ROC curve (test months)"); ax[0].legend(fontsize=7)
    ax[1].axhline(yte.mean(), ls="--", color="gray", label=f"base rate {yte.mean():.3f}"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-recall curve (test months)"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "roc_pr_curves.png", dpi=140); plt.close()

    pipeline_diagram(OUT / "pipeline_diagram.png")

    # what drives the predictions
    lr = fitted["Logistic regression, balanced"]
    names = list(lr["pre"].get_feature_names_out())
    coefs = pd.Series(lr["model"].coef_[0], index=[n.split("__")[1] for n in names]).sort_values(key=abs, ascending=False).round(3)
    tree = fitted["Decision tree, balanced"]
    imp = pd.Series(tree["model"].feature_importances_, index=[n.split("__")[1] for n in names]).sort_values(ascending=False).round(3)

    # hand-off to the AI planner: predicted flag for the 600 AI Part 1 members (latest test month)
    last = test[test.month == TEST[1]]
    flags = pd.DataFrame({"member_id": last["member_id"], "month": last["month"],
                          "p_high_cost": lr.predict_proba(last[NUM + CAT])[:, 1].round(4)})
    flags = flags[flags.member_id <= "M0600"].copy()
    # the planner needs a short list: flag the riskiest 10% (same share as the high-cost definition), ranked by probability
    flags["rank"] = flags["p_high_cost"].rank(ascending=False, method="first").astype(int)
    flags["high_cost_flag"] = (flags["rank"] <= round(0.10 * len(flags))).astype(int)
    flags.sort_values("rank").to_csv(OUT / "predicted_high_cost_flags.csv", index=False)

    out = {"high_cost_cutoff_dollars": round(cutoff, 2), "train_rows": len(train), "test_rows": len(test),
           "train_positive_rate": round(float(ytr.mean()), 4), "test_positive_rate": round(float(yte.mean()), 4),
           "cv": cv_tbl.to_dict("records"), "test": test_tbl.to_dict("records"),
           "lr_balanced_coefficients": coefs.head(10).to_dict(), "tree_importances": imp.head(8).to_dict(),
           "flags_for_ai_planner": {"members": len(flags), "flagged": int(flags["high_cost_flag"].sum()),
                                    "would_be_flagged_at_0.5": int((flags["p_high_cost"] >= 0.5).sum())}}
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    pd.set_option("display.width", 200)
    print(f"cutoff ${cutoff:,.0f} | train {len(train):,} ({ytr.mean():.3f} high) | test {len(test):,} ({yte.mean():.3f} high)")
    print(cv_tbl.to_string(index=False)); print(test_tbl.to_string(index=False))
    print(coefs.head(10)); print(imp.head(8)); print(out["flags_for_ai_planner"])


if __name__ == "__main__":
    main()
