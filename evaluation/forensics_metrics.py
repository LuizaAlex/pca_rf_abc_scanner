import numpy as np

def confirmed_coverage(observed_y: np.ndarray, benign_class_id: int, m_confirm: int) -> int:
    """How many attack classes have >= m_confirm samples in observed_y."""
    if observed_y.size == 0:
        return 0
    y = observed_y.astype(int)
    attack = y[y != benign_class_id]
    if attack.size == 0:
        return 0
    classes, counts = np.unique(attack, return_counts=True)
    return int(np.sum(counts >= m_confirm))

def evidence_balance(observed_y: np.ndarray, benign_class_id: int, n_attack_classes: int, eps: float = 1e-12) -> float:
    """
    Normalized entropy of attack evidence distribution, in [0,1].
    Uses only attack samples from observed_y.
    """
    if n_attack_classes <= 1:
        return 0.0
    y = observed_y.astype(int)
    attack = y[y != benign_class_id]
    if attack.size == 0:
        return 0.0
    classes, counts = np.unique(attack, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + eps))
    return float(H / np.log(n_attack_classes))