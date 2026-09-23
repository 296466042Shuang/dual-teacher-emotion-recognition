"""Small evaluation helpers for multi-label affect prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_roc_auc(y_true, y_score) -> float:
    """Macro ROC-AUC over labels with both positive and negative examples."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    per_label = []
    for index in range(y_true.shape[1]):
        target = y_true[:, index]
        if np.unique(target).size < 2:
            continue
        per_label.append(roc_auc_score(target, y_score[:, index]))

    if not per_label:
        return float("nan")
    return float(np.mean(per_label))
