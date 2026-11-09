"""Dataset loader: read the PNGs, flatten each 64 x 64 image into 4,096 numbers, and scale them."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["claim_form", "referral", "enrollment_form"]


def load(set_name, root=ROOT):
    lab = pd.read_csv(root / "data" / "labels.csv")
    lab = lab[lab["set"] == set_name].reset_index(drop=True)
    X = np.stack([np.asarray(Image.open(root / "data" / "images" / f), dtype=np.float32) for f in lab["filename"]])
    y = lab["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return X, y, lab["filename"].tolist()


def flatten(X):
    """Ink = 1, paper = 0, then one long row of 4,096 numbers per image."""
    return (1.0 - X / 255.0).reshape(len(X), -1)


class Scaler:
    """Standardize each pixel with the TRAINING mean and spread only (no peeking at test images)."""
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd
