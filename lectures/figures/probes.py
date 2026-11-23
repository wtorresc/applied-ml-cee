"""Linear probes on frozen features, with the inspector's recall target (Lectures 15 and 17).

A probe is Lecture 6's logistic regression (L2 penalty, standardized inputs) fitted on frozen
features of a pretrained network; its strength C is chosen on validation by average precision.
The decision threshold is set on validation so that a target share of the cracked tiles is
flagged, and the test set is scored once at that threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

C_GRID: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)  # inverse L2 strengths tried
RECALL: float = 0.90  # the inspector's target: flag 9 of every 10 cracked tiles


@dataclass(frozen=True)
class Probe:
    """Logistic regression with an L2 penalty on standardized features (Lecture 6)."""

    scaler: StandardScaler
    model: LogisticRegression
    C: float
    val_ap: float

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Probability of "cracked" for each row of ``feats``."""
        return self.model.predict_proba(self.scaler.transform(feats))[:, 1]

    def logit(self, feats: np.ndarray) -> np.ndarray:
        """Log-odds of "cracked" (exact, where a probability would round to 0 or 1)."""
        return self.model.decision_function(self.scaler.transform(feats))


def fit_probe(feats: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> Probe:
    """Fit on ``train`` for each C in ``C_GRID``; keep the best average precision on ``val``.

    Args:
        feats: Features, one row per tile.
        y: 1 if cracked.
        train: Boolean mask of training rows.
        val: Boolean mask of validation rows.

    Returns:
        The chosen probe.
    """
    scaler = StandardScaler().fit(feats[train])
    zt, zv = scaler.transform(feats[train]), scaler.transform(feats[val])
    best: tuple[float, float, LogisticRegression] | None = None
    for c in C_GRID:
        # l1_ratio=0 is the L2 penalty (scikit-learn 1.8 deprecates penalty="l2")
        m = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
        m.fit(zt, y[train])
        ap = average_precision_score(y[val], m.predict_proba(zv)[:, 1])
        if best is None or ap > best[0]:
            best = (ap, c, m)
    assert best is not None
    return Probe(scaler, best[2], best[1], float(best[0]))


def threshold_for(p: np.ndarray, y: np.ndarray, recall: float = RECALL) -> float:
    """The highest threshold that flags at least ``recall`` of the cracked tiles (p >= t).

    Flag the top ceil(recall n) of the n cracked tiles' scores. (floor((1 - 0.9) n) is one too
    small whenever n is a multiple of 10: 1 - 0.9 is 0.09999... in floating point.)
    """
    pos = np.sort(p[y == 1])
    return float(pos[len(pos) - math.ceil(recall * len(pos) - 1e-9)])


def scores(p: np.ndarray, y: np.ndarray, t: float) -> dict[str, float]:
    """Ranking and decision metrics of probabilities ``p`` for labels ``y`` at threshold ``t``."""
    flag = p >= t
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "recall": float(flag[y == 1].mean()),
        "review": float(flag.mean()),
        "acc": float(((p >= 0.5) == y).mean()),
        "prevalence": float(y.mean()),
        "n": int(len(y)),
    }
