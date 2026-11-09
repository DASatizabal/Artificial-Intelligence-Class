"""Softmax (multinomial logistic) regression written from scratch with NumPy.

score_k = x . w_k + b_k  for each class k
p_k     = exp(score_k) / sum_j exp(score_j)        (softmax turns scores into probabilities that add to 1)
loss    = -log p_true + lambda * ||W||^2           (cross-entropy plus L2 to keep weights small)
Trained with mini-batch gradient descent; stops when the validation loss has not improved for `patience` epochs.
"""
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # subtract the max so exp() never overflows
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    def __init__(self, n_classes=3, lr=0.02, l2=1e-3, epochs=300, batch=64, patience=15, seed=42):
        self.k, self.lr, self.l2, self.epochs, self.batch, self.patience = n_classes, lr, l2, epochs, batch, patience
        self.rng = np.random.default_rng(seed)
        self.history = []

    def _loss(self, X, y):
        p = softmax(X @ self.W + self.b)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean() + self.l2 * (self.W ** 2).sum())

    def fit(self, X, y, Xval, yval):
        n, d = X.shape
        self.W = self.rng.normal(0, 0.01, (d, self.k)); self.b = np.zeros(self.k)
        Y = np.eye(self.k)[y]
        best, best_state, wait = np.inf, None, 0
        for ep in range(self.epochs):
            order = self.rng.permutation(n)
            for i in range(0, n, self.batch):
                idx = order[i:i + self.batch]
                p = softmax(X[idx] @ self.W + self.b)
                g = (p - Y[idx]) / len(idx)                  # gradient of cross-entropy w.r.t. the scores
                self.W -= self.lr * (X[idx].T @ g + 2 * self.l2 * self.W)
                self.b -= self.lr * g.sum(0)
            tr, va = self._loss(X, y), self._loss(Xval, yval)
            self.history.append((ep + 1, tr, va))
            if va < best - 1e-4:
                best, best_state, wait = va, (self.W.copy(), self.b.copy()), 0
            else:
                wait += 1
                if wait >= self.patience:                    # early stopping
                    break
        self.W, self.b = best_state
        self.best_epoch = int(np.argmin([h[2] for h in self.history]) + 1)
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
