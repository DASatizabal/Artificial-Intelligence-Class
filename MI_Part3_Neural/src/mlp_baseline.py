"""Baseline MLP for the tabular high-cost flag (PyTorch).

Architecture (assignment spec): Input -> Dense(128) -> ReLU -> Dropout(0.2) -> Dense(64) -> ReLU -> Dense(2)
Training: Adam (lr 1e-3, small weight decay), batch 64, up to 50 epochs, early stopping on validation ROC-AUC
(patience 8): the weights from the epoch that ranked validation members best are kept.
Class weights make a missed high-cost member count more, the same idea as "balanced" in MI Part 2.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class MLP(nn.Module):
    def __init__(self, d_in, d_out=2, h1=128, h2=64, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(d_in, h1), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, Xtr, ytr, Xva, yva, class_weight=None, epochs=50, lr=1e-3, batch=64, patience=8, seed=42):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    w = torch.tensor(class_weight, dtype=torch.float32) if class_weight is not None else None
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, wait, log = -np.inf, None, 0, []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xtr_t), generator=g)
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            out = model(Xva_t)
            va = loss_fn(out, yva_t).item()
            auc = roc_auc_score(yva, torch.softmax(out, 1)[:, 1].numpy())
        log.append({"epoch": ep, "train_loss": total / len(Xtr_t), "val_loss": va, "val_auc": auc})
        if auc > best + 1e-4:
            best, best_state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, log, round(time.perf_counter() - t0, 2)


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X)), 1).numpy()


def mc_dropout(model, X, passes=20, seed=42):
    """Keep dropout ON at prediction time and ask the same question 20 times.
    The spread of the answers is the model's uncertainty."""
    torch.manual_seed(seed)
    model.train()
    with torch.no_grad():
        P = np.stack([torch.softmax(model(torch.tensor(X)), 1).numpy() for _ in range(passes)])
    model.eval()
    return P.mean(0), P.var(0)
