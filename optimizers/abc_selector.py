# optimizers/abc_selector.py
import numpy as np


def redundancy_penalty(indices: np.ndarray, X_norm_or_gram) -> float:
    """
    Average pairwise similarity inside the batch (off-diagonal of cosine-sim matrix).
    Accepts:
      - X_norm_or_gram = None -> returns 0.0
      - X_norm_or_gram = X_norm with shape (N, D)
      - X_norm_or_gram = Gram matrix with shape (N, N) (NOT recommended for large N)
    """
    if X_norm_or_gram is None:
        return 0.0

    if indices is None or len(indices) <= 1:
        return 0.0

    # If it's an NxN square matrix, treat it as a precomputed Gram matrix
    if getattr(X_norm_or_gram, "ndim", 0) == 2 and X_norm_or_gram.shape[0] == X_norm_or_gram.shape[1]:
        sim = X_norm_or_gram[np.ix_(indices, indices)]
    else:
        M = X_norm_or_gram[indices]
        sim = M @ M.T

    off_diag = sim.sum() - np.trace(sim)
    denom = (len(indices) * (len(indices) - 1))
    return float(off_diag / denom) if denom > 0 else 0.0

def expected_new_class_gain(
    indices: np.ndarray,
    proba: np.ndarray,
    unseen_class_ids: list[int],
) -> float:
    """
    Batch-level reward for discovering *new* attack classes.

    For each unseen class k, reward the probability that the batch contains
    at least one sample of class k:

        gain_k = 1 - Π_{i in batch} (1 - P(y=k | x_i))

    Summing over unseen classes encourages *coverage* of multiple families.
    """
    if proba is None or unseen_class_ids is None or len(unseen_class_ids) == 0:
        return 0.0

    P = proba[indices][:, unseen_class_ids]   # (B, |unseen|)
    P = np.clip(P, 0.0, 1.0)                  # numeric safety

    prod = np.prod(1.0 - P, axis=0)           # Π(1-p)
    gain = 1.0 - prod                         # 1 - Π(1-p)
    return float(np.sum(gain))


def class_redundancy_penalty(indices: np.ndarray, proba: np.ndarray) -> float:
    """
    Optional redundancy penalty in *class-probability* space.

    If selected samples have very similar probability vectors, batches can
    collapse onto one likely class. We penalize average cosine similarity.
    """
    if proba is None or len(indices) <= 1:
        return 0.0

    M = proba[indices]
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sim = M @ M.T
    off_diag = sim.sum() - np.trace(sim)
    return float(off_diag / (len(indices) * (len(indices) - 1)))

def fitness(
    indices: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    cost: np.ndarray,
    X_norm: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    lamb: float,
    proba: np.ndarray | None = None,
    unseen_class_ids: list[int] | None = None,
    kappa: float = 0.0,
    eta: float = 0.0
) -> float:

    red = 0.0
    if lamb != 0.0 and X_norm is not None:
        red = redundancy_penalty(indices, X_norm)

    base = (
        alpha * p[indices].sum()
        + beta * u[indices].sum()
        - gamma * cost[indices].sum()
        - lamb * red
    )

    if kappa != 0.0:
        base += kappa * expected_new_class_gain(indices, proba, unseen_class_ids or [])

    if eta != 0.0:
        base -= eta * class_redundancy_penalty(indices, proba)

    return float(base)

def abc_select_batch(
    avail: np.ndarray,
    X_norm: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    cost: np.ndarray,
    B: int,
    rng: np.random.Generator,
    colony_size: int,
    iters: int,
    limit: int,
    alpha: float,
    beta: float,
    gamma: float,
    lamb: float,
    # --- Forensics upgrades (optional)
    proba: np.ndarray | None = None,
    unseen_class_ids: list[int] | None = None,
    kappa: float = 0.0,
    eta: float = 0.0,
    init_foods: list[np.ndarray] | None = None
) -> np.ndarray:
    """
    Discrete ABC selector (with warm-start support).
    """
    if len(avail) <= B:
        return avail

    avail = np.asarray(avail)
    n_avail = len(avail)

    # Work in "position space" [0..n_avail-1] so we can index p/u/cost/proba safely.
    pos_pool = np.arange(n_avail, dtype=int)

    def dedup_valid_batch(batch: np.ndarray) -> np.ndarray:
        """
        Candidate batch in POSITION space:
          - 1D int numpy array
          - values in [0, n_avail)
          - unique
          - length exactly B (fill/trim as needed)
        """
        if batch is None:
            batch = np.array([], dtype=int)

        batch = np.asarray(batch, dtype=int).ravel()

        # keep only valid positions
        batch = batch[(batch >= 0) & (batch < n_avail)]

        # deduplicate while preserving order
        seen = set()
        uniq = []
        for x in batch:
            xi = int(x)
            if xi not in seen:
                uniq.append(xi)
                seen.add(xi)
        batch = np.array(uniq, dtype=int)

        # fill to length B if needed
        if len(batch) < B:
            remaining = np.array([i for i in range(n_avail) if i not in seen], dtype=int)
            if len(remaining) > 0:
                fill = rng.choice(remaining, size=min(B - len(batch), len(remaining)), replace=False)
                batch = np.concatenate([batch, fill.astype(int)])

        # trim if too long
        if len(batch) > B:
            batch = batch[:B]

        # final safety
        if len(batch) < B:
            return np.array(list(dict.fromkeys(pos_pool.tolist()))[:B], dtype=int)

        return batch

    def random_batch() -> np.ndarray:
        return rng.choice(pos_pool, size=B, replace=False)

    def neighbor(batch: np.ndarray) -> np.ndarray:
        """
        Generate a neighbor solution by replacing 1 element of the batch
        with a random index from avail that is not already in the batch.
        """
        batch = dedup_valid_batch(batch)
        if batch is None or len(batch) == 0:
            return random_batch()

        cand = batch.copy()

        # choose a position to mutate
        pos = int(rng.integers(0, B))
        used = set(int(x) for x in cand)

        # candidate replacements not already used
        choices = np.array([i for i in range(n_avail) if i not in used], dtype=int)

        # If no alternative exists, return unchanged candidate
        if len(choices) == 0:
            return cand

        cand[pos] = int(rng.choice(choices))
        return dedup_valid_batch(cand)

    # Initialize food sources (warm-start first, then random)
    foods = []
    if init_foods:
        # map global ids -> positions
        pos_map = {int(g): i for i, g in enumerate(avail)}
        for f in init_foods:
            f = np.asarray(f, dtype=int).ravel()
            f_pos = np.array([pos_map[int(x)] for x in f if int(x) in pos_map], dtype=int)
            foods.append(dedup_valid_batch(f_pos))
            if len(foods) >= colony_size:
                break

    while len(foods) < colony_size:
        foods.append(random_batch())

    trials = np.zeros(colony_size, dtype=int)

    best = foods[0].copy()
    best_fit = -np.inf

    for _ in range(iters):
        fits = np.array([
            fitness(
                f, p, u, cost, X_norm,
                alpha, beta, gamma, lamb,
                proba=proba,
                unseen_class_ids=unseen_class_ids,
                kappa=kappa,
                eta=eta,
            )
            for f in foods
        ])

        j = int(np.argmax(fits))
        if fits[j] > best_fit:
            best_fit = float(fits[j])
            best = foods[j].copy()

        # Employed bee phase
        for i in range(colony_size):
            cand = neighbor(foods[i])
            cand_fit = fitness(
                cand, p, u, cost, X_norm,
                alpha, beta, gamma, lamb,
                proba=proba,
                unseen_class_ids=unseen_class_ids,
                kappa=kappa,
                eta=eta,
            )
            if cand_fit > fits[i]:
                foods[i] = cand
                fits[i] = cand_fit
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker bee phase
        min_f = fits.min()
        probs = fits - min_f + 1e-9
        probs = probs / probs.sum()

        for _ in range(colony_size):
            i = int(rng.choice(np.arange(colony_size), p=probs))
            cand = neighbor(foods[i])
            cand_fit = fitness(
                cand, p, u, cost, X_norm,
                alpha, beta, gamma, lamb,
                proba=proba,
                unseen_class_ids=unseen_class_ids,
                kappa=kappa,
                eta=eta,
            )
            if cand_fit > fits[i]:
                foods[i] = cand
                fits[i] = cand_fit
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout bee phase (unchanged)
        for i in range(colony_size):
            if trials[i] >= limit:
                foods[i] = random_batch()
                trials[i] = 0

    return avail[best]