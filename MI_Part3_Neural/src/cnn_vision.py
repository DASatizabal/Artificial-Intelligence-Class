"""Tiny CNN for the AI Part 2 document images (PyTorch).

Architecture (assignment spec): Conv(8, 3x3) -> ReLU -> MaxPool(2) -> Conv(16, 3x3) -> ReLU -> MaxPool(2)
                                -> Flatten -> Dense(32) -> ReLU -> Dense(3)
Input: 64 x 64 grayscale, ink = 1, paper = 0. Adam (lr 1e-3), batch 32, up to 50 epochs, early stopping.
Saliency: the gradient of the winning class score with respect to each pixel shows which pixels mattered.
"""
import time
import numpy as np
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16 * 16, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def to_tensor(X):
    """uint8-style images (n, 64, 64) -> float tensor (n, 1, 64, 64), ink = 1."""
    return torch.tensor((1.0 - X / 255.0)[:, None, :, :], dtype=torch.float32)


def train(model, Xtr, ytr, Xva, yva, epochs=50, lr=1e-3, batch=32, patience=6, seed=42):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Xtr_t, ytr_t, Xva_t, yva_t = to_tensor(Xtr), torch.tensor(ytr), to_tensor(Xva), torch.tensor(yva)
    loss_fn, opt = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, wait, log = np.inf, None, 0, []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xtr_t), generator=g)
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            opt.zero_grad(); loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            va = loss_fn(model(Xva_t), yva_t).item()
        log.append({"epoch": ep, "train_loss": total / len(Xtr_t), "val_loss": va})
        if va < best - 1e-4:
            best, best_state, wait = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, log, round(time.perf_counter() - t0, 2)


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(to_tensor(X)), 1).numpy()


def saliency(model, img):
    """|d score_winner / d pixel| for one image."""
    model.eval()
    x = to_tensor(img[None]).requires_grad_(True)
    out = model(x)
    out[0, out.argmax(1).item()].backward()
    return x.grad.abs()[0, 0].numpy()
