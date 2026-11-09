"""AI Part 2 pipeline: load -> flatten -> softmax baseline -> augmentation -> evaluation -> routing.
Run from the project folder:  python src/run_part2.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load, flatten, Scaler, CLASSES
from simple_linear import SoftmaxClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
NEXT_STEP = {"claim_form": "send to claims (feeds MI cost data)",
             "referral": "start a provider search (AI Part 1 planner)",
             "enrollment_form": "add or update the provider directory"}


def augment(X, rng, copies=2):
    """Make messy copies of the TRAINING images only: tilt, shift, blur, noise, uneven light."""
    out = []
    for _ in range(copies):
        for a in X:
            img = Image.fromarray(a.astype(np.uint8))
            img = img.rotate(rng.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=255,
                             translate=tuple(int(v) for v in rng.integers(-4, 5, 2)))
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
            b = np.asarray(img, dtype=float)
            grad = np.linspace(rng.uniform(-50, 0), rng.uniform(0, 50), b.shape[0])
            b = b + (grad[None, :] if rng.random() < 0.5 else grad[:, None]) + rng.normal(0, 18, b.shape)
            out.append(np.clip(b, 0, 255))
    return np.stack(out).astype(np.float32)


def scores(y, p):
    return {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4)}


def main():
    rng = np.random.default_rng(SEED)
    Xtr, ytr, _ = load("train"); Xva, yva, _ = load("val"); Xte, yte, _ = load("test"); Xms, yms, fms = load("messy")

    Xaug = augment(Xtr, rng)
    yaug = np.concatenate([ytr] * 2)
    # augmented samples sheet (deliverable)
    fig, ax = plt.subplots(3, 6, figsize=(10, 5.4))
    for r, c in enumerate(CLASSES):
        i = int(np.where(ytr == r)[0][0])
        for j, (img, title) in enumerate([(Xtr[i], "original"), (Xaug[i], "augmented 1"), (Xaug[i + len(Xtr)], "augmented 2")] +
                                        [(Xms[int(np.where(yms == r)[0][k])], f"messy scan {k + 1}") for k in range(3)]):
            ax[r, j].imshow(img, cmap="gray", vmin=0, vmax=255); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if r == 0: ax[r, j].set_title(title, fontsize=8)
        ax[r, 0].set_ylabel(c.replace("_", "\n"), fontsize=8)
    plt.suptitle("Training image, two augmented copies, and three messy scans (never trained on)")
    plt.tight_layout(); plt.savefig(OUT / "augmented_samples.png", dpi=140); plt.close()

    results, models = {}, {}
    for name, (Xa, ya) in {"Softmax, clean training": (Xtr, ytr),
                           "Softmax + augmentation": (np.concatenate([Xtr, Xaug]), np.concatenate([ytr, yaug]))}.items():
        sc = Scaler().fit(flatten(Xa))
        m = SoftmaxClassifier(seed=SEED).fit(sc.transform(flatten(Xa)), ya, sc.transform(flatten(Xva)), yva)
        models[name] = (m, sc)
        results[name] = {"training_images": len(Xa), "best_epoch": m.best_epoch,
                         "test": scores(yte, m.predict(sc.transform(flatten(Xte)))),
                         "messy": scores(yms, m.predict(sc.transform(flatten(Xms))))}

    # confusion matrices: rows = model, columns = test set
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    short = ["claim", "referral", "enroll"]
    for r, (name, (m, sc)) in enumerate(models.items()):
        for c, (sname, X, y) in enumerate([("clean test", Xte, yte), ("messy scans", Xms, yms)]):
            yp = m.predict(sc.transform(flatten(X)))
            cm = confusion_matrix(y, yp, labels=[0, 1, 2])
            ax = axes[r, c]; ax.imshow(cm, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks(range(3), short); ax.set_yticks(range(3), short); ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"{name}\non {sname} (accuracy {accuracy_score(y, yp):.3f})", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrices.png", dpi=140); plt.close()

    # what the augmented model looks at: one weight image per class (blue = pushes toward, red = pushes away)
    m, sc = models["Softmax + augmentation"]
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
    for k, c in enumerate(CLASSES):
        w = m.W[:, k].reshape(64, 64); v = np.abs(w).max()
        ax[k].imshow(w, cmap="RdBu", vmin=-v, vmax=v); ax[k].set_title(c); ax[k].set_xticks([]); ax[k].set_yticks([])
    plt.suptitle("Softmax weights as images: where ink pushes toward (blue) or away from (red) each class")
    plt.tight_layout(); plt.savefig(OUT / "weight_images.png", dpi=140); plt.close()

    # training curve
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, (mm, _) in models.items():
        h = np.array(mm.history); ax.plot(h[:, 0], h[:, 2], label=f"{name} (validation)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Validation loss (training stops when it stops improving)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=140); plt.close()

    # routing: auto-route confident predictions, send unsure ones to a person
    P = m.predict_proba(sc.transform(flatten(Xms)))
    conf, pred = P.max(1), P.argmax(1)
    routing = []
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        auto = conf >= t
        routing.append({"threshold": t, "auto_routed_share": round(float(auto.mean()), 4),
                        "accuracy_of_auto_routed": round(float((pred[auto] == yms[auto]).mean()), 4) if auto.any() else None,
                        "sent_to_human": int((~auto).sum()), "wrong_auto_routes": int((pred[auto] != yms[auto]).sum())})
    examples = [{"file": fms[i], "true": CLASSES[yms[i]], "predicted": CLASSES[pred[i]], "confidence": round(float(conf[i]), 3),
                 "next_step": NEXT_STEP[CLASSES[pred[i]]] if conf[i] >= 0.7 else "human review"} for i in rng.choice(len(fms), 8, replace=False)]

    metrics = {"data": {"train": len(Xtr), "val": len(Xva), "test": len(Xte), "messy": len(Xms), "augmented_copies": len(Xaug),
                        "pixels_per_image": 64 * 64},
               "models": results, "routing_messy": routing, "routing_examples": examples}
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(results, indent=1)); print(pd.DataFrame(routing).to_string(index=False))


if __name__ == "__main__":
    main()
