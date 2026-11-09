"""Probabilistic models for MI Part 3.

HMM (the main probabilistic model)
  Hidden state each month: 0 = normal, 1 = in a high-cost spell (last-3-month cost in the training top 10%).
  What we observe each month: the cost band of that month's claims (0: under $100, 1: $100-999, 2: $1,000-4,999, 3: $5,000+).
  Start, transition and emission probabilities are COUNTED from the training months, with +1 pseudo-counts
  so nothing is ever exactly 0 (Laplace smoothing).
  Viterbi finds the most likely state path for a member. The forward filter gives P(in a spell now), and
  pushing that through the transition matrix three times gives P(in a spell three months from now).
Naive Bayes (comparison)
  Gaussian Naive Bayes on the same tabular features as the MLP: fast, and its assumptions are easy to state.
"""
import numpy as np

BANDS = [100, 1000, 5000]


def band(cost):
    return np.digitize(np.asarray(cost), BANDS)      # negatives (refunds) fall in band 0


class CostHMM:
    def __init__(self, n_states=2, n_obs=4, pseudo=1.0):
        self.S, self.O, self.pseudo = n_states, n_obs, pseudo

    def fit(self, state_seqs, obs_seqs):
        pi = np.full(self.S, self.pseudo); A = np.full((self.S, self.S), self.pseudo); B = np.full((self.S, self.O), self.pseudo)
        for s, o in zip(state_seqs, obs_seqs):
            pi[s[0]] += 1
            for t in range(len(s)):
                B[s[t], o[t]] += 1
                if t:
                    A[s[t - 1], s[t]] += 1
        self.pi, self.A, self.B = pi / pi.sum(), A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True)
        return self

    def filter(self, obs):
        """Forward algorithm, normalized each step: P(state_t | observations up to t)."""
        a = self.pi * self.B[:, obs[0]]; a /= a.sum(); out = [a]
        for o in obs[1:]:
            a = (a @ self.A) * self.B[:, o]; a /= a.sum(); out.append(a)
        return np.array(out)

    def viterbi(self, obs):
        """Most likely hidden state path (in log space so small numbers don't vanish)."""
        lA, lB = np.log(self.A), np.log(self.B)
        d = np.log(self.pi) + lB[:, obs[0]]; back = []
        for o in obs[1:]:
            cand = d[:, None] + lA
            back.append(cand.argmax(0)); d = cand.max(0) + lB[:, o]
        path = [int(d.argmax())]
        for bp in reversed(back):
            path.append(int(bp[path[-1]]))
        return path[::-1]

    def risk_ahead(self, obs, steps=3):
        """P(high-cost spell `steps` months from now), from everything observed so far."""
        return (self.filter(obs)[-1] @ np.linalg.matrix_power(self.A, steps))[1]
