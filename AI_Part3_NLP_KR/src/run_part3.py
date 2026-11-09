"""AI Part 3 integration script: intent model -> slot parser -> rule base -> structured JSON.

Fixes from the starter run_part3.py: it trained and predicted on the SAME texts (so accuracy meant nothing),
had no metrics, and wrote JSON without confidence, rule traces or provider results.
Run from the project root:  python src/run_part3.py
"""
import json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intent_baseline import IntentClassifier
from slots import parse_slots, apply_schema, SCHEMA
from rules import check_rules, detect_injection
from kb import KnowledgeBase

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "part3_nlp_interface"
OUT.mkdir(parents=True, exist_ok=True)
GOLD = ["specialty", "zip", "max_miles", "language", "provider_name", "time_hour"]


def gold_slots(row):
    g = {}
    for s in GOLD:
        v = row.get(f"gold_{s}")
        if pd.isna(v) or v == "":
            continue
        g[s] = int(float(v)) if s in ("max_miles", "time_hour") else str(v)
    return g


def slot_scores(df, intents_used):
    """Per-slot accuracy over the slots each request's GOLD intent can carry, plus whole-frame exact match."""
    per, frames = {s: [0, 0] for s in GOLD}, []
    for (_, row), it in zip(df.iterrows(), intents_used):
        pred, gold = apply_schema(it, parse_slots(row["utterance"])), gold_slots(row)
        ok_all = True
        for s in SCHEMA[row["intent"]]:
            hit = pred.get(s) == gold.get(s)
            per[s][0] += hit; per[s][1] += 1; ok_all &= hit
        frames.append(ok_all)
    return {s: round(a / n, 4) for s, (a, n) in per.items() if n}, round(float(np.mean(frames)), 4)


def main():
    t0 = time.perf_counter()
    req = pd.read_csv(ROOT / "data" / "nlp" / "requests.csv", dtype={"gold_zip": str})
    chal = pd.read_csv(ROOT / "data" / "nlp" / "challenge_requests.csv", dtype={"gold_zip": str})
    kb = KnowledgeBase()

    # 1) leakage-free split: the model never sees the test sentences
    train, test = train_test_split(req, test_size=0.30, stratify=req["intent"], random_state=SEED)

    # 2) choose the model on the TRAINING data only (5-fold CV), never on test
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for kind in ("bow", "tfidf"):
        fold = []
        for a, b in cv.split(train["utterance"], train["intent"]):
            m = IntentClassifier(kind).fit(train["utterance"].iloc[a], train["intent"].iloc[a])
            fold.append(f1_score(train["intent"].iloc[b], m.predict(train["utterance"].iloc[b]), average="macro"))
        cv_scores[kind] = {"mean_macro_f1": round(float(np.mean(fold)), 4), "std": round(float(np.std(fold)), 4)}
    champion = max(cv_scores, key=lambda k: (cv_scores[k]["mean_macro_f1"], k == "bow"))  # tie -> simpler model

    # 3) fit both on all training rows, evaluate on held-out test and on the hand-written challenge set
    models = {k: IntentClassifier(k).fit(train["utterance"], train["intent"]) for k in ("bow", "tfidf")}
    evals = {}
    for k, m in models.items():
        for name, d in (("test", test), ("challenge", chal)):
            p = m.predict(d["utterance"])
            evals[f"{k}_{name}"] = {"accuracy": round(accuracy_score(d["intent"], p), 4),
                                    "macro_f1": round(f1_score(d["intent"], p, average="macro"), 4)}
    model = models[champion]
    labels = list(model.classes_)

    # confusion matrices for the champion
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (name, d) in zip(axes, (("Held-out test (n=%d)" % len(test), test), ("Challenge set (n=%d)" % len(chal), chal))):
        cm = confusion_matrix(d["intent"], model.predict(d["utterance"]), labels=labels)
        ax.imshow(cm, cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        short = [l.replace("_", "\n") for l in labels]
        ax.set_xticks(range(len(labels)), short, fontsize=8); ax.set_yticks(range(len(labels)), short, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(name)
    fig.suptitle(f"Intent confusion matrices, champion = {champion.upper()} + logistic regression")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

    # 4) slot parser: with the true intent (parser alone) and with the predicted intent (full pipeline)
    slots_eval = {}
    for name, d in (("test", test), ("challenge", chal)):
        oracle, oracle_frame = slot_scores(d, d["intent"])
        piped, piped_frame = slot_scores(d, model.predict(d["utterance"]))
        slots_eval[name] = {"per_slot_true_intent": oracle, "frame_match_true_intent": oracle_frame,
                            "per_slot_predicted_intent": piped, "frame_match_predicted_intent": piped_frame}

    # 5) injection detector (pure rules, so it is scored on every request we have)
    alltxt = pd.concat([req[["utterance", "injection"]], chal[["utterance", "injection"]]])
    flag = alltxt["utterance"].map(lambda t: bool(detect_injection(t)))
    tp = int((flag & (alltxt["injection"] == 1)).sum()); fp = int((flag & (alltxt["injection"] == 0)).sum())
    fn = int((~flag & (alltxt["injection"] == 1)).sum())
    inj = {"true_pos": tp, "false_pos": fp, "false_neg": fn,
           "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4)}

    # 6) full pipeline -> structured JSON for every held-out and challenge request
    outputs, actions, fired = [], Counter(), Counter()
    for split, d in (("test", test), ("challenge", chal)):
        probs = model.predict_proba(d["utterance"])
        for (_, row), pr in zip(d.iterrows(), probs):
            intent, conf = labels[int(pr.argmax())], float(pr.max())
            raw_slots = parse_slots(row["utterance"])
            slots = apply_schema(intent, raw_slots)
            valid, msg, res = check_rules(intent, slots, kb, row["utterance"], conf)
            actions[(split, res["action"])] += 1
            fired.update(t["rule"] for t in res["proof_trace"] if t["then"].startswith("action:"))
            outputs.append({"request_id": row["request_id"], "split": split, "text": row["utterance"],
                            "intent": intent, "confidence": round(conf, 4), "true_intent": row["intent"],
                            "slots": slots, "valid": valid, "action": res["action"], "msg": msg,
                            **{k: v for k, v in res.items() if k not in ("action",)}})
    with open(OUT / "nlp_outputs.json", "w") as fh:
        json.dump(outputs, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    metrics = {
        "data": {"requests": len(req), "train": len(train), "test": len(test), "challenge": len(chal),
                 "intent_counts": req["intent"].value_counts().to_dict(), "providers": len(kb.providers)},
        "model_selection_cv_on_train": cv_scores, "champion": champion,
        "intent_eval": evals, "top_terms": model.top_terms(5),
        "slots": slots_eval, "injection_detector": inj,
        "actions": {f"{s}:{a}": n for (s, a), n in sorted(actions.items())},
        "rules_fired": dict(fired.most_common()),
        "astar": {"calls": kb.astar_calls, "node_expansions": kb.astar_expansions,
                  "providers_considered_without_pruning_note": "Manhattan lower bound stops the search once the top 3 are certain"},
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps({k: metrics[k] for k in ("model_selection_cv_on_train", "champion", "intent_eval", "injection_detector", "actions", "rules_fired", "astar")}, indent=1))
    print(json.dumps(metrics["slots"], indent=1))


if __name__ == "__main__":
    main()
