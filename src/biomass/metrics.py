from __future__ import annotations

import numpy as np


def weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: list[float],
) -> tuple[float, np.ndarray]:
    r2_scores: list[float] = []
    for idx in range(y_true.shape[1]):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    per_target = np.asarray(r2_scores, dtype=np.float64)
    weighted = float(np.sum(per_target * np.asarray(weights)) / np.sum(weights))
    return weighted, per_target
