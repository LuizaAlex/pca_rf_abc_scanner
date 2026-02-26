# evaluation/stats.py
import numpy as np

def permutation_test_paired(a, b, n_perm=5000, seed=0):
    """
    Paired permutation test (non-parametric).
    Tests whether mean(a - b) is significantly > 0 or != 0 (two-sided).
    Returns two-sided p-value.

    a, b: arrays of same length (per-run metric for two strategies)
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    assert len(a) == len(b)

    rng = np.random.default_rng(seed)
    diff = a - b
    observed = abs(diff.mean())

    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_stat = abs((diff * signs).mean())
        if perm_stat >= observed:
            count += 1

    return (count + 1) / (n_perm + 1)
