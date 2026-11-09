"""MI Part 3: neural models, a probabilistic model, uncertainty, ablations and error analysis.

Run from the project folder (seed 42 everywhere):
  python src/gen_claims.py && python src/gen_docs.py && python src/run_part3.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import SEED, ROOT, scores, set_seed
import mlp_baseline as mlpm
import cnn_vision as cnnm
from bayes_hmm import CostHMM, band
from dataset import load, CLASSES

OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)


def tabular_track():
    X, y, frames, cutoff, names = common.tabular()
    counts = np.bincount(y["fit"]); cw = (len(y["fit"]) / (2 * counts)).tolist()   # "balanced" weights
    rows, logs, P = [], {}, {}

    # MI Part 2 champion, refit on the same rows so the comparison is fair
    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X["fit"], y["fit"])
    t_lr = round(time.perf_counter() - t0, 2)
    P["Project 2 best: logistic regression, balanced"] = (lr.predict_proba(X["val"]), lr.predict_proba(X["test"]),
                                                         X["fit"].shape[1] + 1, t_lr, "Traditional baseline")
    t0 = time.perf_counter()
    nb = GaussianNB().fit(X["fit"], y["fit"])
    P["Gaussian Naive Bayes"] = (nb.predict_proba(X["val"]), nb.predict_proba(X["test"]), 2 * 2 * X["fit"].shape[1],
                                 round(time.perf_counter() - t0, 2), "Fast; assumes features are independent")

    variants = {"MLP (128-64, dropout 0.2, class weights)": dict(h1=128, h2=64, dropout=0.2, w=cw),
                "Ablation: no dropout": dict(h1=128, h2=64, dropout=0.0, w=cw),
                "Ablation: half the hidden units (64-32)": dict(h1=64, h2=32, dropout=0.2, w=cw),
                "Ablation: no class weights": dict(h1=128, h2=64, dropout=0.2, w=None)}
    models = {}
    for name, v in variants.items():
        set_seed()
        m = mlpm.MLP(X["fit"].shape[1], 2, v["h1"], v["h2"], v["dropout"])
        m, log, secs = mlpm.train(m, X["fit"], y["fit"], X["val"], y["val"], class_weight=v["w"])
        models[name] = m; logs[name] = log
        P[name] = (mlpm.predict_proba(m, X["val"]), mlpm.predict_proba(m, X["test"]), mlpm.n_params(m), secs,
                   f"stopped at epoch {len(log)} (kept epoch {int(np.argmax([r['val_auc'] for r in log])) + 1})")

    main = models["MLP (128-64, dropout 0.2, class weights)"]
    mc_mean, mc_var = mlpm.mc_dropout(main, X["test"])
    mc_val, _ = mlpm.mc_dropout(main, X["val"])
    P["MLP + MC Dropout (20 passes)"] = (mc_val, mc_mean, mlpm.n_params(main), P["MLP (128-64, dropout 0.2, class weights)"][3],
                                         "Average of 20 dropout passes; variance = uncertainty")

    # HMM on monthly cost sequences
    t0 = time.perf_counter()
    df = pd.read_csv(ROOT / "data" / "member_months.csv", dtype={"zip": str}).drop_duplicates().sort_values(["member_id", "month"])
    spell_line = float(df.loc[(df.month >= common.TRAIN[0]) & (df.month <= common.TRAIN[1]), "cost_prior_3m"].quantile(0.90))
    fit_ids = set(frames["fit"]["member_id"])
    tr = df[(df.month <= common.TRAIN[1]) & df.member_id.isin(fit_ids)]
    seqs = [(g["cost_prior_3m"].ge(spell_line).astype(int).to_numpy(), band(g["cost_month"].to_numpy())) for _, g in tr.groupby("member_id")]
    hmm = CostHMM().fit([s for s, _ in seqs], [o for _, o in seqs])
    hist = {mid: g for mid, g in df.groupby("member_id")}

    def hmm_probs(frame):
        p = np.array([hmm.risk_ahead(band(hist[mid].loc[hist[mid].month <= mo, "cost_month"].to_numpy()))
                      for mid, mo in zip(frame["member_id"], frame["month"])])
        return np.c_[1 - p, p]
    # the HMM's raw probabilities are on a different scale, so pick its yes/no line to flag the same 10% share
    hv, ht = hmm_probs(frames["val"]), hmm_probs(frames["test"])
    line = float(np.quantile(hv[:, 1], 0.90))     # flag the riskiest 10% of validation rows, same share as "high-cost"
    P["HMM (2 states, cost bands)"] = (hv, ht, 2 + 4 + 8, round(time.perf_counter() - t0, 2),
                                       f"Interpretable tables; yes/no line P >= {line:.3f}")
    thresholds = {"HMM (2 states, cost bands)": line}
    hmm_info = {"spell_line_dollars": round(spell_line, 2), "start": hmm.pi.round(4).tolist(),
                "transition": hmm.A.round(4).tolist(), "emission": hmm.B.round(4).tolist(),
                "flag_line": round(line, 4)}
    # Viterbi example: the test member whose most likely path switches state the most
    switches = {mid: int(np.abs(np.diff(hmm.viterbi(band(hist[mid].loc[hist[mid].month <= common.TEST[1], "cost_month"].to_numpy())))).sum())
                for mid in sorted(frames["test"]["member_id"].unique())}
    ex = hist[max(switches, key=switches.get)]
    ex = ex[ex.month <= common.TEST[1]]
    hmm_info["viterbi_example"] = {"member_id": ex["member_id"].iloc[0], "months": ex["month"].tolist(),
                                   "cost_band": band(ex["cost_month"]).tolist(), "viterbi_state": hmm.viterbi(band(ex["cost_month"].to_numpy()))}

    feats = {"Project 2 best: logistic regression, balanced": "Tabular (17 features)", "Gaussian Naive Bayes": "Tabular",
             "HMM (2 states, cost bands)": "Monthly cost sequence"}
    for name, (pv, pt, params, secs, note) in P.items():
        th = thresholds.get(name)
        rows.append({"model": name, "features": feats.get(name, "Tabular"), "val_accuracy": scores(y["val"], pv, th)["accuracy"],
                     **{f"test_{k}": v for k, v in scores(y["test"], pt, th).items()}, "params": int(params), "train_seconds": secs, "notes": note})
    table = pd.DataFrame(rows)

    # reliability diagram for P(high-cost)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for name in ["Project 2 best: logistic regression, balanced", "MLP (128-64, dropout 0.2, class weights)",
                 "MLP + MC Dropout (20 passes)", "Ablation: no class weights"]:
        fp, mp = calibration_curve(y["test"], P[name][1][:, 1], n_bins=10, strategy="quantile")
        ax.plot(mp, fp, "o-", ms=3, label=name.replace("Project 2 best: ", ""))
    ax.set_xlabel("predicted chance of high-cost (bin average)"); ax.set_ylabel("share that really was high-cost")
    ax.set_title("Reliability diagram (test months)"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "reliability_diagram.png", dpi=140); plt.close()

    # uncertainty examples: biggest MC Dropout spread
    t = frames["test"].copy()
    t["p_high_mc"] = mc_mean[:, 1].round(3); t["sd_mc"] = np.sqrt(mc_var[:, 1]).round(3); t["truly_high"] = y["test"]
    uncertain = t.sort_values("sd_mc", ascending=False).head(4)[["member_id", "month", "age", "chronic_conditions", "er_visits_6m",
                                                                "cost_prior_12m", "p_high_mc", "sd_mc", "truly_high", "cost_next_3m"]]
    # error analysis on the main MLP
    pm = P["MLP (128-64, dropout 0.2, class weights)"][1]
    t["pred"] = pm.argmax(1); t["p_high"] = pm[:, 1].round(3)
    med_prior = float(frames["fit"]["cost_prior_12m"].median()); hi_prior = float(frames["fit"]["cost_prior_12m"].quantile(0.9))
    def cause(r):
        if r.truly_high and not r.pred:
            return "Missed: little history, then a big quarter" if r.cost_prior_12m < med_prior else "Missed: warning signs, but under the line"
        if r.pred and not r.truly_high:
            return "False alarm: expensive last year, quiet next quarter" if r.cost_prior_12m >= hi_prior else "False alarm: risky profile (age, conditions), stayed low"
        return None
    t["error_type"] = t.apply(cause, axis=1)
    err_counts = t["error_type"].value_counts().to_dict()
    err_p = t.dropna(subset=["error_type"]).groupby("error_type")["p_high"].describe()[["min", "50%", "max"]].round(3)
    err_p.columns = ["p_min", "p_median", "p_max"]
    # confusion matrix for the main MLP on the test months
    cm = confusion_matrix(y["test"], t["pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["not high", "high"]); ax.set_yticks([0, 1], ["not high", "high"]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("MLP (class weights) on the test months", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "mlp_confusion_test.png", dpi=140); plt.close()
    err_examples = pd.concat([g.head(2) for _, g in t.dropna(subset=["error_type"]).groupby("error_type")])[
        ["error_type", "age", "chronic_conditions", "er_visits_6m", "cost_prior_12m", "p_high", "cost_next_3m"]]
    # subgroup check (fairness)
    t["age_band"] = pd.cut(t["age"], [0, 44, 64, 120], labels=["18-44", "45-64", "65+"])
    t["plan_type"] = t["plan_type"].fillna("Unknown")
    sub = []
    for col in ["age_band", "plan_type", "county"]:
        for k, g in t.groupby(col, observed=True):
            if g["truly_high"].sum() >= 5:
                sub.append({"group": f"{col} = {k}", "rows": len(g), "high_cost_rate": round(g["truly_high"].mean(), 3),
                            "flag_rate": round(g["pred"].mean(), 3), "recall": round(recall_score(g["truly_high"], g["pred"]), 3)})
    sub = pd.DataFrame(sub)
    return table, logs, hmm_info, uncertain, err_counts, err_examples, sub, err_p, cm


def augment(X, rng, copies=2):
    out = []
    for _ in range(copies):
        for a in X:
            img = Image.fromarray(a.astype(np.uint8)).rotate(rng.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=255,
                                                             translate=tuple(int(v) for v in rng.integers(-4, 5, 2)))
            b = np.asarray(img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0))), dtype=float)
            grad = np.linspace(rng.uniform(-50, 0), rng.uniform(0, 50), b.shape[0])
            out.append(np.clip(b + (grad[None, :] if rng.random() < 0.5 else grad[:, None]) + rng.normal(0, 18, b.shape), 0, 255))
    return np.stack(out).astype(np.float32)


def vision_track():
    Xtr, ytr, _ = load("train"); Xva, yva, _ = load("val"); Xte, yte, _ = load("test"); Xms, yms, fms = load("messy")
    Xaug = np.concatenate([Xtr, augment(Xtr, np.random.default_rng(SEED))]); yaug = np.concatenate([ytr] * 3)
    rows, logs, models = [], {}, {}
    for name, (Xa, ya) in {"CNN, clean training": (Xtr, ytr), "CNN + augmentation": (Xaug, yaug)}.items():
        set_seed()
        m, log, secs = cnnm.train(cnnm.TinyCNN(), Xa, ya, Xva, yva)
        models[name] = m; logs[name] = log
        rows.append({"model": name, "training_images": len(Xa), "val_accuracy": scores(yva, cnnm.predict_proba(m, Xva))["accuracy"],
                     **{f"test_{k}": v for k, v in scores(yte, cnnm.predict_proba(m, Xte)).items()},
                     **{f"messy_{k}": v for k, v in scores(yms, cnnm.predict_proba(m, Xms)).items()},
                     "params": mlpm.n_params(m), "train_seconds": secs})
    vis = pd.DataFrame(rows)

    m = models["CNN, clean training"]
    pm = cnnm.predict_proba(m, Xms); pred = pm.argmax(1)
    cm = confusion_matrix(yms, pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    short = ["claim", "referral", "enroll"]
    ax.set_xticks(range(3), short); ax.set_yticks(range(3), short); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"CNN (clean training) on messy scans\naccuracy {(pred == yms).mean():.3f}", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "cnn_confusion_messy.png", dpi=140); plt.close()

    right = int(np.where(pred == yms)[0][0]); wrong = np.where(pred != yms)[0]
    picks = [(right, "correct")] + ([(int(wrong[0]), "wrong")] if len(wrong) else [(int(np.where(pred == yms)[0][1]), "correct")])
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 6.4))
    for r, (i, tag) in enumerate(picks):
        ax[r, 0].imshow(Xms[i], cmap="gray"); ax[r, 0].set_title(f"messy {CLASSES[yms[i]]}\npredicted {CLASSES[pred[i]]} ({tag})", fontsize=8)
        ax[r, 1].imshow(cnnm.saliency(m, Xms[i]), cmap="hot"); ax[r, 1].set_title("saliency: pixels that moved the score", fontsize=8)
        for a in ax[r]: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(OUT / "cnn_saliency.png", dpi=140); plt.close()
    return vis, logs


def main():
    set_seed()
    table, tab_logs, hmm_info, uncertain, err_counts, err_examples, sub, err_p, mlp_cm = tabular_track()
    vis, vis_logs = vision_track()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, log in tab_logs.items():
        d = pd.DataFrame(log); ax[0].plot(d.epoch, d.val_loss, label=(name.split(" (")[0] if name.startswith("MLP") else name) + " val")
    d = pd.DataFrame(tab_logs["MLP (128-64, dropout 0.2, class weights)"])
    ax[0].plot(d.epoch, d.train_loss, "k--", label="MLP train")
    ax[0].set_title("Tabular MLPs: training vs validation loss", fontsize=10); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=7)
    for name, log in vis_logs.items():
        d = pd.DataFrame(log); ax[1].plot(d.epoch, d.train_loss, "--", label=f"{name} train"); ax[1].plot(d.epoch, d.val_loss, label=f"{name} val")
    ax[1].set_title("Document CNNs: training and validation loss"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=140); plt.close()

    pd.concat([pd.DataFrame(l).assign(model=n) for n, l in {**tab_logs, **vis_logs}.items()]).to_csv(OUT / "training_logs.csv", index=False)
    table.to_csv(OUT / "comparison_tabular.csv", index=False); vis.to_csv(OUT / "comparison_vision.csv", index=False)
    with open(OUT / "metrics.json", "w") as fh:
        json.dump({"tabular": table.to_dict("records"), "vision": vis.to_dict("records"), "hmm": hmm_info,
                   "mc_dropout_most_uncertain": uncertain.to_dict("records"), "error_counts": err_counts,
                   "error_examples": err_examples.to_dict("records"), "error_probability_ranges": err_p.reset_index().to_dict("records"),
                   "mlp_confusion_test": mlp_cm.tolist(), "subgroups": sub.to_dict("records")}, fh, indent=2, default=str)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(table.drop(columns=["notes"]).to_string(index=False)); print(vis.to_string(index=False))
    print(json.dumps(hmm_info, indent=1)[:1500]); print(uncertain.to_string(index=False)); print(err_counts)
    print(err_examples.to_string(index=False)); print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
