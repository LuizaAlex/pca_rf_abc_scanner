# strategies/baselines.py
import numpy as np

def select_random(avail: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Pick B random nodes from available."""
    if len(avail) <= B:
        return avail
    return rng.choice(avail, size=B, replace=False)

def select_greedy_topk(avail, scores, k: int):
    """
    Select top-k indices from avail by score.

    Works with:
    - scores shaped (N,) where N = total samples -> uses scores[avail]
    - scores shaped (len(avail),) -> already aligned to avail
    """
    avail = np.asarray(avail)
    scores = np.asarray(scores)

    if len(avail) == 0 or k <= 0:
        return np.array([], dtype=int)

    k = min(int(k), len(avail))

    # If scores already correspond to avail order, use directly.
    if scores.shape[0] == len(avail):
        avail_scores = scores
    else:
        avail_scores = scores[avail]

    # Top-k (descending)
    if k == len(avail):
        top_idx = np.argsort(-avail_scores)
    else:
        top_idx = np.argpartition(-avail_scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-avail_scores[top_idx])]

    return avail[top_idx]
