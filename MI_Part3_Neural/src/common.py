"""Shared pieces for MI Part 3: seeds, the tabular high-cost data (same as MI Part 2), metrics and ECE."""
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
NUM = ["age", "chronic_conditions", "miles_to_pcp", "saw_pcp_last_3m", "er_visits_6m", "admits_12m",
       "cost_month", "cost_prior_3m", "cost_prior_12m"]
CAT = ["sex", "county", "plan_type"]
TRAIN = ("2025-01", "2025-09"); TEST = ("2026-01", "2026-02")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def tabular():
    """Same rows, split, high-cost line and preprocessing as MI Part 2, plus a validation set
    (20% of training MEMBERS) used only for early stopping."""
    df = pd.read_csv(ROOT / "data" / "member_months.csv", dtype={"zip": str}).drop_duplicates().reset_index(drop=True)
    train = df[(df.month >= TRAIN[0]) & (df.month <= TRAIN[1])].reset_index(drop=True)
    test = df[(df.month >= TEST[0]) & (df.month <= TEST[1])].reset_index(drop=True)
    cutoff = float(train["cost_next_3m"].quantile(0.90))
    rng = np.random.default_rng(SEED)
    members = np.array(sorted(train["member_id"].unique()))
    val_members = set(rng.choice(members, int(0.2 * len(members)), replace=False))
    is_val = train["member_id"].isin(val_members)
    fit, val = train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])
    pre.fit(fit[NUM + CAT])
    X = {k: pre.transform(d[NUM + CAT]).astype(np.float32) for k, d in (("fit", fit), ("val", val), ("test", test))}
    y = {k: (d["cost_next_3m"] >= cutoff).astype(int).to_numpy() for k, d in (("fit", fit), ("val", val), ("test", test))}
    names = [n.split("__")[1] for n in pre.get_feature_names_out()]
    return X, y, {"fit": fit, "val": val, "test": test}, cutoff, names


def ece(probs, labels, n_bins=10):
    """Expected calibration error: average gap between confidence and accuracy across confidence bins."""
    conf, preds = probs.max(1), probs.argmax(1)
    acc = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def scores(y, probs, threshold=None):
    """probs: n x 2 (binary) or n x k. Returns the metrics used in every table.
    threshold: optional yes/no line for P(class 1); default is the most likely class."""
    pred = probs.argmax(1) if threshold is None else (probs[:, 1] >= threshold).astype(int)
    out = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"), "ece": ece(probs, y)}
    if probs.shape[1] == 2:
        out.update({"recall_high": recall_score(y, pred), "roc_auc": roc_auc_score(y, probs[:, 1]),
                    "pr_auc": average_precision_score(y, probs[:, 1])})
    return {k: round(float(v), 4) for k, v in out.items()}
