import numpy as np

def b3_hybrid_score(
    proba: np.ndarray,
    benign_class_id: int,
    seen_attack_class_ids: set[int],
    alpha: float,
    delta: float
) -> np.ndarray:
    """
    B3 forensic score:
      alpha * (1 - P(benign))  +  delta * sum_{unseen attack classes} P(class)
    """
    K = proba.shape[1]
    attack_mass = 1.0 - proba[:, benign_class_id]

    unseen = [k for k in range(K) if k != benign_class_id and k not in seen_attack_class_ids]
    if len(unseen) == 0:
        novelty_mass = np.zeros(proba.shape[0], dtype=float)
    else:
        novelty_mass = proba[:, unseen].sum(axis=1)

    return alpha * attack_mass + delta * novelty_mass