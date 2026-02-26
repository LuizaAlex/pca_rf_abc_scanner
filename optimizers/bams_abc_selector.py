import numpy as np
from optimizers.abc_selector import abc_select_batch


def bams_abc_select_batch(
    avail: np.ndarray,
    X_norm,
    p: np.ndarray,        # avail-aligned (len(avail),)
    u: np.ndarray,        # avail-aligned
    cost: np.ndarray,     # avail-aligned
    proba: np.ndarray,    # avail-aligned (len(avail), K)
    observed_y: np.ndarray,
    benign_class_id: int,
    B: int,
    rng: np.random.Generator,
    rho_g: float = 0.30,
    rho_f: float = 0.30,
    colony_size: int = 25,
    iters: int = 15,
    limit: int = 10,
    alpha: float = 1.0,
    beta: float = 0.15,
    gamma: float = 0.0,
    lamb: float = 0.1,
) -> np.ndarray:
    """
    BAMS-ABC: pick Bg greedy anchors + Bf forensic anchors + Ba ABC fill.

    CONTRACT:
      - avail contains GLOBAL indices (shape n_avail)
      - p/u/cost are aligned to avail (shape n_avail)
      - proba is aligned to avail (shape n_avail, K)
    RETURNS:
      - GLOBAL indices (shape B)
    """
    avail = np.asarray(avail, dtype=int)
    n_avail = len(avail)

    if n_avail == 0:
        return np.array([], dtype=int)
    if n_avail <= B:
        return avail[:B].astype(int)

    # Defensive: ensure alignment
    if len(p) != n_avail or len(u) != n_avail or len(cost) != n_avail:
        raise ValueError(
            f"bams_abc_select_batch expects p/u/cost aligned to avail. "
            f"len(avail)={n_avail}, len(p)={len(p)}, len(u)={len(u)}, len(cost)={len(cost)}"
        )
    if proba is not None and proba.shape[0] != n_avail:
        raise ValueError(f"proba must be aligned to avail: proba.shape[0]={proba.shape[0]} vs {n_avail}")

    B = int(B)
    rho_g = float(max(0.0, min(1.0, rho_g)))
    rho_f = float(max(0.0, min(1.0 - rho_g, rho_f)))

    Bg = min(int(np.ceil(rho_g * B)), B)
    Bf = min(int(np.ceil(rho_f * B)), B - Bg)
    Ba = B - Bg - Bf

    chosen_pos = []
    chosen_set = set()

    # --------------------------
    # 1) Greedy anchors (pos-space) by p
    # --------------------------
    if Bg > 0:
        scores = np.asarray(p, dtype=float)
        if Bg >= n_avail:
            top_pos = np.argsort(-scores)
        else:
            top_pos = np.argpartition(-scores, Bg - 1)[:Bg]
            top_pos = top_pos[np.argsort(-scores[top_pos])]

        for pos in top_pos:
            ip = int(pos)
            if ip not in chosen_set:
                chosen_pos.append(ip)
                chosen_set.add(ip)

    # remaining positions
    remaining_pos = np.array([i for i in range(n_avail) if i not in chosen_set], dtype=int)

    # --------------------------
    # 2) Forensic/balance anchors (pos-space) using proba
    # --------------------------
    if (
        Bf > 0 and proba is not None and proba.ndim == 2 and proba.shape[1] > 2
        and len(remaining_pos) > 0
    ):
        K = int(proba.shape[1])
        benign = int(benign_class_id)

        y = observed_y.astype(int)
        attack = y[y != benign]

        counts = {k: 0 for k in range(K) if k != benign}
        if attack.size > 0:
            cls, cnt = np.unique(attack, return_counts=True)
            for k, c in zip(cls, cnt):
                kk = int(k)
                if kk != benign:
                    counts[kk] = int(c)

        classes_sorted = sorted(counts.keys(), key=lambda k: (counts[k] > 0, counts[k]))

        for k in classes_sorted:
            if len(chosen_pos) >= Bg + Bf or len(remaining_pos) == 0:
                break

            scores_k = proba[remaining_pos, int(k)]
            if scores_k.size == 0 or float(np.max(scores_k)) <= 1e-12:
                continue

            order = np.argsort(scores_k)[::-1]
            picked = None
            for j in order:
                pos = int(remaining_pos[int(j)])
                if pos not in chosen_set:
                    picked = pos
                    break

            if picked is not None:
                chosen_pos.append(picked)
                chosen_set.add(picked)
                remaining_pos = np.array([i for i in range(n_avail) if i not in chosen_set], dtype=int)

    # --------------------------
    # 3) ABC fill on the remaining pool
    # --------------------------
    if Ba > 0 and len(remaining_pos) > 0:
        remaining_global = avail[remaining_pos]

        # Slice arrays so they are aligned to remaining_global
        p_r = p[remaining_pos]
        u_r = u[remaining_pos]
        c_r = cost[remaining_pos]
        proba_r = proba[remaining_pos] if proba is not None else None

        picked_global = abc_select_batch(
            avail=remaining_global,
            X_norm=X_norm,    # can be None safely
            p=p_r,
            u=u_r,
            cost=c_r,
            B=min(Ba, len(remaining_global)),
            rng=rng,
            colony_size=colony_size,
            iters=iters,
            limit=limit,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lamb=lamb,
            proba=proba_r,     # keep aligned
        )

        # Mark picked as chosen
        global_to_pos = {int(g): i for i, g in enumerate(avail)}
        for g in np.asarray(picked_global, dtype=int).ravel():
            if int(g) in global_to_pos:
                pos = int(global_to_pos[int(g)])
                if pos not in chosen_set:
                    chosen_pos.append(pos)
                    chosen_set.add(pos)

    # Final fill if short
    if len(chosen_pos) < B:
        left = np.array([i for i in range(n_avail) if i not in chosen_set], dtype=int)
        if len(left) > 0:
            fill = rng.choice(left, size=min(B - len(chosen_pos), len(left)), replace=False)
            for pos in fill:
                chosen_pos.append(int(pos))
                chosen_set.add(int(pos))

    chosen_pos = np.array(chosen_pos[:B], dtype=int)
    return avail[chosen_pos]