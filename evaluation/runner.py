# evaluation/runner.py
import numpy as np
import config

from models.pca_rf import make_pca_rf_model, uncertainty_entropy
from optimizers.bams_abc_selector import bams_abc_select_batch
from strategies.baselines import select_random, select_greedy_topk
from optimizers.abc_selector import abc_select_batch
from optimizers.ga_selector import ga_select_batch
from optimizers.greedy_abc_selector import greedy_shortlist_abc_select_batch

from evaluation.seed_cache import cache_path, save_seed_cache, load_seed_cache
from evaluation.model_cache import model_cache_path, save_model, load_model

from sklearn.metrics import f1_score
from evaluation.forensics_metrics import confirmed_coverage, evidence_balance


def count_attacks(y: np.ndarray) -> int:
    """
    Count how many samples are "attack".

    - Binary case: y in {0,1} where 1 = attack
    - Multiclass case: assume class 0 = benign, anything else = attack
    """
    if len(y) == 0:
        return 0

    if y.max() <= 1:
        return int((y == 1).sum())
    return int((y != 0).sum())


def b3_hybrid_score(
    proba: np.ndarray,
    benign_class_id: int,
    seen_attack_class_ids: set,
    alpha: float,
    delta: float
) -> np.ndarray:
    """
    B3 forensic score:
      alpha * (1 - P(benign)) + delta * sum_{unseen attack classes} P(class)

    - benign_class_id: index of benign class (recommended 0)
    - seen_attack_class_ids: set of attack class ids already discovered (excluding benign)
    """
    K = proba.shape[1]
    attack_mass = 1.0 - proba[:, benign_class_id]




    unseen = [k for k in range(K) if k != benign_class_id and k not in seen_attack_class_ids]
    if len(unseen) == 0:
        novelty_mass = np.zeros(proba.shape[0], dtype=float)
    else:
        novelty_mass = proba[:, unseen].sum(axis=1)

    return alpha * attack_mass + delta * novelty_mass


def run_experiment(
    world,
    rng,
    strategy: str,
    rounds: int,
    batch_size: int,
    initial_seed_scans: int,
    pca_components: int,
    seed: int,
    abc_params: dict,
    ga_params: dict,
    hybrid_params: dict,
    fitness_weights: dict,
):
    """
    Runs an active scanning experiment for one strategy:
    - random
    - greedy
    - abc
    - greedy_shortlist_abc
    - ga

    Optional forensic multiclass scoring:
    - config.FORENSICS_MODE = "b3" enables hybrid detection+coverage scoring.

    Adds forensic success curves:
    - confirmed_coverage_curve (>= m samples per attack class)
    - evidence_balance_curve (normalized entropy across attack class evidence)
    - macro_f1_curve (macro-F1 on scanned set)
    """

    X_norm = None
    need_xnorm = (fitness_weights.get("lamb", 0.0) > 0.0) and (
            strategy in {"abc", "ga", "bams_abc", "greedy_shortlist_abc"}
    )

    # Pre-normalize only if we will actually use redundancy penalty.
    # IMPORTANT: Do NOT build an NxN Gram matrix for large N.
    if need_xnorm:
        norms = np.linalg.norm(world.X, axis=1, keepdims=True).astype(np.float32)
        X_norm = world.X / (norms + 1e-9)
        X_norm = X_norm.astype(np.float32, copy=False)
    dataset_name = getattr(config, "DATASET_NAME", "unknown").lower()

    # -----------------------------
    # Seed cache: reuse the same initial seed batch/labels per run seed
    # -----------------------------
    use_seed_cache = getattr(config, "USE_SEED_CACHE", True)
    seed_cache_dir = getattr(config, "SEED_CACHE_DIR", "cache")

    avail = world.available_indices()
    S = min(initial_seed_scans, len(avail))
    seed_file = cache_path(dataset_name, seed, cache_dir=seed_cache_dir)

    if use_seed_cache and seed_file.exists():
        init_batch, y_init, c_init = load_seed_cache(seed_file)

        #  Validate cache against current dataset size
        n = len(world.y)
        init_batch = np.asarray(init_batch, dtype=int)

        if init_batch.size == 0 or init_batch.min() < 0 or init_batch.max() >= n:
            print(f"[seed-cache] Cache incompatible with current dataset size (n={n}). Rebuilding seed cache.")
            init_batch = select_random(avail, S, rng)
            y_init, c_init = world.scan(init_batch)
            save_seed_cache(seed_file, init_batch, y_init, c_init)
        else:
            world.scan(init_batch)  # update inspected state
    else:
        init_batch = select_random(avail, S, rng)
        y_init, c_init = world.scan(init_batch)
        if use_seed_cache:
            save_seed_cache(seed_file, init_batch, y_init, c_init)
    # Observed training set starts from the seed batch
    observed_X = world.X[init_batch]
    observed_y = y_init

    # -----------------------------
    # Forensics config/state
    # -----------------------------
    forensic_mode = getattr(config, "FORENSICS_MODE", "off")
    benign_class_id = int(getattr(config, "BENIGN_CLASS_ID", 0))
    forensic_alpha = float(getattr(config, "FORENSICS_ALPHA", 1.0))
    forensic_delta = float(getattr(config, "FORENSICS_DELTA", 0.5))
    m_confirm = int(getattr(config, "FORENSICS_CONFIRM_M", 3))

    # Number of attack classes in the dataset (excluding benign)
    all_classes = np.unique(world.y.astype(int))
    attack_classes = [c for c in all_classes if int(c) != benign_class_id]
    n_attack_classes = len(attack_classes)

    # Track which attack classes have already been discovered in this run (excluding benign)
    seen_attack_class_ids = set(int(cls) for cls in np.unique(y_init) if int(cls) != benign_class_id)


    # -----------------------------
    # Model cache: load/save the seed-fitted PCA+RF model
    # -----------------------------
    use_model_cache = getattr(config, "USE_MODEL_CACHE", True)
    model_cache_dir = getattr(config, "MODEL_CACHE_DIR", "cache")

    pca_components_effective = min(pca_components, world.X.shape[1])

    m_path = model_cache_path(
        dataset=dataset_name,
        seed=seed,
        pca_components=pca_components_effective,
        cache_dir=model_cache_dir
    )

    if use_model_cache and m_path.exists():
        model = load_model(m_path)
        seed_model_is_fitted = True
    else:
        model = make_pca_rf_model(pca_components_effective, seed)
        model.fit(observed_X, observed_y)
        seed_model_is_fitted = True
        if use_model_cache:
            save_model(m_path, model)

    # -----------------------------
    # History + curves
    # -----------------------------
    history = {
        "vulns_curve": [],
        "cost_curve": [],
        "scans_curve": [],
        "total_vulns": 0,
        "total_cost": float(np.sum(c_init)),
        "total_scans": len(init_batch),

        # Existing forensic curve: number of distinct attack classes discovered so far (first-hit)
        "attack_class_coverage_curve": [],

        # New forensic success curves:
        "confirmed_coverage_curve": [],
        "evidence_balance_curve": [],
        "macro_f1_curve": [],

        "tp_scanned": 0,
        "tn_scanned": 0,
        "fp_scanned": 0,
        "fn_scanned": 0,
    }

    history["total_vulns"] += count_attacks(y_init)

    # Helper to append forensic success curves (called after model is fitted)
    def append_forensics_success_curves():
        """
        Append CCC/EBC/AQC once per round (aligned with cost_curve points).
        For binary tasks, we append 0.0 so shapes remain consistent.
        """
        # Determine K safely from current model probabilities
        K_local = len(getattr(model, "classes_", []))

        if K_local > 2:
            history["confirmed_coverage_curve"].append(
                float(confirmed_coverage(observed_y, benign_class_id, m_confirm))
            )
            history["evidence_balance_curve"].append(
                float(evidence_balance(observed_y, benign_class_id, n_attack_classes))
            )
            # Macro-F1 on scanned set (consistent across strategies).
            y_pred_obs = model.predict(observed_X)
            history["macro_f1_curve"].append(
                float(f1_score(observed_y, y_pred_obs, average="macro"))
            )
        else:
            history["confirmed_coverage_curve"].append(0.0)
            history["evidence_balance_curve"].append(0.0)
            history["macro_f1_curve"].append(0.0)

    # -----------------------------
    # Main scanning loop
    # -----------------------------
    for round_idx in range(rounds):
        avail = world.available_indices()
        if len(avail) == 0:
            break

        # Refit the model after round 0 (round 0 can reuse cached seed-fitted model)
        if round_idx > 0 or not seed_model_is_fitted:
            model.fit(observed_X, observed_y)

        # ---- Build avail-aligned arrays ONCE ----
        avail = np.asarray(avail, dtype=int)
        X_avail = world.X[avail]
        cost_avail = world.cost[avail]
        proba_raw = model.predict_proba(X_avail)

        classes = np.asarray(getattr(model, "classes_", np.arange(proba_raw.shape[1])), dtype=int)

        # Build proba where column index == true class label (0..max_label)
        max_label = int(np.max(world.y.astype(int)))
        K_full = max_label + 1

        proba = np.zeros((proba_raw.shape[0], K_full), dtype=float)
        for j, cls in enumerate(classes):
            c = int(cls)
            if 0 <= c < K_full:
                proba[:, c] = proba_raw[:, j]

        K = proba.shape[1]

        # --- Compute attack score p (avail-aligned) ---
        if K == 2:
            p = proba[:, 1]
        else:
            if forensic_mode == "b3":
                p = b3_hybrid_score(
                    proba=proba,
                    benign_class_id=benign_class_id,
                    seen_attack_class_ids=seen_attack_class_ids,
                    alpha=forensic_alpha,
                    delta=forensic_delta
                )
            else:
                p = 1.0 - proba[:, benign_class_id]

        # Uncertainty for exploration (entropy) (avail-aligned)
        u = uncertainty_entropy(proba)

        # --- Choose batch according to strategy ---
        if strategy == "random":
            batch = select_random(avail, min(batch_size, len(avail)), rng)

        elif strategy == "greedy":
            batch = select_greedy_topk(avail, p, min(batch_size, len(avail)))


        elif strategy == "abc":

            batch = abc_select_batch(

                avail=avail,
                X_norm=X_norm,
                p=p,
                u=u,
                cost=cost_avail,
                B=min(batch_size, len(avail)),
                rng=rng,
                colony_size=abc_params["colony_size"],
                iters=abc_params["iters"],
                limit=abc_params["limit"],
                alpha=fitness_weights["alpha"],
                beta=fitness_weights["beta"],
                gamma=fitness_weights["gamma"],
                lamb=fitness_weights["lamb"],
                proba=proba,

            )

        elif strategy == "bams_abc":

            bp = hybrid_params.get("bams_abc", {})

            batch = bams_abc_select_batch(

                avail=avail,
                X_norm=X_norm,
                p=p,
                u=u,
                cost=cost_avail,
                proba=proba,
                observed_y=observed_y,
                benign_class_id=benign_class_id,
                B=min(batch_size, len(avail)),
                rng=rng,
                rho_g=float(bp.get("rho_g", getattr(config, "BAMS_RHO_G", 0.30))),
                rho_f=float(bp.get("rho_f", getattr(config, "BAMS_RHO_F", 0.30))),
                colony_size=abc_params["colony_size"],
                iters=abc_params["iters"],
                limit=abc_params["limit"],
                alpha=fitness_weights["alpha"],
                beta=fitness_weights["beta"],
                gamma=fitness_weights["gamma"],
                lamb=fitness_weights["lamb"],

            )



        elif strategy == "ga":

            batch = ga_select_batch(
                avail=avail,
                X_norm=X_norm,
                p=p,
                u=u,
                cost=cost_avail,
                B=min(batch_size, len(avail)),
                rng=rng,
                pop_size=ga_params["pop_size"],
                generations=ga_params["generations"],
                elite_frac=ga_params["elite_frac"],
                tournament_k=ga_params["tournament_k"],
                mutation_rate=ga_params["mutation_rate"],
                mutation_swaps=ga_params["mutation_swaps"],
                alpha=fitness_weights["alpha"],
                beta=fitness_weights["beta"],
                gamma=fitness_weights["gamma"],
                lamb=fitness_weights["lamb"],

            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # --- Confusion counts on scanned items (simple, per-scan) ---
        # Predict labels for the chosen batch BEFORE scanning (so it's a real "online" prediction).
        y_pred_batch = model.predict(world.X[batch]).astype(int)

        # Binary view: benign=0, attack=1 (y != benign_class_id)
        y_pred_bin = (y_pred_batch != benign_class_id).astype(int)
        # --- Scan selected batch (reveal labels, pay costs) ---
        y_new, c_new = world.scan(batch)

        y_true_batch = y_new.astype(int)
        y_true_bin = (y_true_batch != benign_class_id).astype(int)

        history["tp_scanned"] += int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
        history["tn_scanned"] += int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
        history["fp_scanned"] += int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
        history["fn_scanned"] += int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

        # Update observed training set
        observed_X = np.vstack([observed_X, world.X[batch]])
        observed_y = np.concatenate([observed_y, y_new])

        # Update forensic "seen classes"
        for cls in np.unique(y_new):
            cls = int(cls)
            if cls != benign_class_id:
                seen_attack_class_ids.add(cls)

        # Update totals
        history["total_vulns"] += count_attacks(y_new)
        history["total_cost"] += float(np.sum(c_new))
        history["total_scans"] += len(batch)

        # Record curves (one point per round)
        history["vulns_curve"].append(history["total_vulns"])
        history["cost_curve"].append(history["total_cost"])
        history["scans_curve"].append(history["total_scans"])
        history["attack_class_coverage_curve"].append(len(seen_attack_class_ids))

        # Record new forensics-success curves (aligned with cost_curve)
        append_forensics_success_curves()

        # --- Final global FN/FP/TN/TP (binary view) ---
        # Binary view: benign=0, attack=1 (y != benign)
        y_true = world.y.astype(int)
        y_true_bin = (y_true != benign_class_id).astype(int)

        # Predict on all samples
        y_pred = model.predict(world.X)
        y_pred_bin = (np.asarray(y_pred).astype(int) != benign_class_id).astype(int)

        tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
        tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
        fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
        fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

        history["final_tp"] = tp
        history["final_tn"] = tn
        history["final_fp"] = fp
        history["final_fn"] = fn





    return history