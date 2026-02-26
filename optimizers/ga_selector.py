# optimizers/ga_selector.py
import numpy as np

from optimizers.abc_selector import fitness


def ga_select_batch(
    avail: np.ndarray,
    X_norm,
    p: np.ndarray,      # aligned to avail
    u: np.ndarray,      # aligned to avail
    cost: np.ndarray,   # aligned to avail
    B: int,
    rng: np.random.Generator,
    pop_size: int = 40,
    generations: int = 40,
    elite_frac: float = 0.2,
    tournament_k: int = 3,
    mutation_rate: float = 0.2,
    mutation_swaps: int = 2,
    alpha: float = 1.0,
    beta: float = 0.35,
    gamma: float = 0.2,
    lamb: float = 0.5,
) -> np.ndarray:
    """
    Genetic Algorithm selector for discrete batch selection.

    CONTRACT:
      - avail: GLOBAL ids, shape (n_avail,)
      - p/u/cost: aligned to avail, shape (n_avail,)

    Internally we evolve individuals in POSITION space [0..n_avail-1],
    then return GLOBAL ids at the end.
    """
    avail = np.asarray(avail, dtype=int)
    n_avail = len(avail)

    if n_avail == 0:
        return np.array([], dtype=int)
    if n_avail <= B:
        return avail[:B].astype(int)

    # Defensive alignment checks
    if len(p) != n_avail or len(u) != n_avail or len(cost) != n_avail:
        raise ValueError(
            f"ga_select_batch expects p/u/cost aligned to avail. "
            f"len(avail)={n_avail}, len(p)={len(p)}, len(u)={len(u)}, len(cost)={len(cost)}"
        )

    B = min(int(B), n_avail)
    pos_pool = np.arange(n_avail, dtype=int)

    def random_individual():
        return rng.choice(pos_pool, size=B, replace=False)

    def eval_fit(ind_pos):
        # fitness expects indices to index p/u/cost arrays directly (avail-aligned)
        return fitness(ind_pos, p, u, cost, X_norm, alpha, beta, gamma, lamb)

    def tournament_select(pop, fits):
        idxs = rng.choice(len(pop), size=min(tournament_k, len(pop)), replace=False)
        best_i = idxs[np.argmax(fits[idxs])]
        return pop[best_i]

    def crossover(parent1, parent2):
        cut = B // 2
        child = list(rng.choice(parent1, size=cut, replace=False))

        for g in rng.permutation(parent2):
            gg = int(g)
            if gg not in child:
                child.append(gg)
            if len(child) == B:
                break

        if len(child) < B:
            current = set(child)
            candidates = [a for a in pos_pool if int(a) not in current]
            if candidates:
                extra = rng.choice(candidates, size=B - len(child), replace=False)
                child.extend([int(x) for x in extra])

        return np.array(child, dtype=int)

    def mutate(ind):
        if rng.random() > mutation_rate:
            return ind

        ind = ind.copy()
        current = set(ind.tolist())

        for _ in range(mutation_swaps):
            pos = int(rng.integers(0, B))
            candidates = [a for a in pos_pool if int(a) not in current]
            if not candidates:
                break
            new_gene = int(rng.choice(candidates))

            current.remove(int(ind[pos]))
            ind[pos] = new_gene
            current.add(new_gene)

        return ind

    # Init population
    population = [random_individual() for _ in range(pop_size)]
    elite_n = max(1, int(pop_size * elite_frac))

    best = population[0]
    best_fit = -np.inf

    for _ in range(generations):
        fits = np.array([eval_fit(ind) for ind in population], dtype=float)

        i_best = int(np.argmax(fits))
        if fits[i_best] > best_fit:
            best_fit = float(fits[i_best])
            best = population[i_best].copy()

        elite_idxs = np.argsort(-fits)[:elite_n]
        elites = [population[i].copy() for i in elite_idxs]

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop

    # Convert best positions -> global ids
    return avail[best]