# single_scan_bams.py
import time
import numpy as np
import config

from main import build_base_world
from models.pca_rf import make_pca_rf_model, uncertainty_entropy
from strategies.baselines import select_random
from optimizers.bams_abc_selector import bams_abc_select_batch

CICIOV_LABEL_NAMES = {
    0: "BENIGN",
    1: "DoS",
    2: "Spoofing-GAS",
    3: "Spoofing-RPM",
    4: "Spoofing-SPEED",
    5: "Spoofing-STEERING_WHEEL",
}
def main():
    rng = np.random.default_rng(getattr(config, "RANDOM_SEED", 20))

    # Build dataset world exactly like the main experiment
    world = build_base_world(rng)

    benign_id = int(getattr(config, "BENIGN_CLASS_ID", 0))
    seed_scans = int(getattr(config, "INITIAL_SEED_SCANS", 20))
    pca_components = int(getattr(config, "PCA_COMPONENTS", 8))

    # --- 1) Seed scans (bootstrap the model) ---
    avail = world.available_indices()
    S = min(seed_scans, len(avail))
    seed_batch = select_random(avail, S, rng)
    y_init, c_init = world.scan(seed_batch)

    observed_X = world.X[seed_batch]
    observed_y = y_init

    # --- 2) Fit model once (PCA + RF) ---
    pca_components = min(pca_components, world.X.shape[1])
    model = make_pca_rf_model(pca_components, seed=getattr(config, "RANDOM_SEED", 20))
    model.fit(observed_X, observed_y)

    # --- 3) Build avail-aligned arrays ---
    avail = np.asarray(world.available_indices(), dtype=int)
    X_avail = world.X[avail]
    cost_avail = world.cost[avail]

    # timing starts here: scoring + selection + scan
    t_total0 = time.perf_counter()

    # predict probabilities on available pool
    t_proba0 = time.perf_counter()
    proba_raw = model.predict_proba(X_avail)
    t_proba = time.perf_counter() - t_proba0

    classes = np.asarray(getattr(model, "classes_", np.arange(proba_raw.shape[1])), dtype=int)

    # align columns to true label ids 0..max_label (same idea as runner.py)
    max_label = int(np.max(world.y.astype(int)))
    K_full = max_label + 1
    proba = np.zeros((proba_raw.shape[0], K_full), dtype=float)
    for j, cls in enumerate(classes):
        c = int(cls)
        if 0 <= c < K_full:
            proba[:, c] = proba_raw[:, j]

    # attack score p
    if proba.shape[1] == 2:
        p = proba[:, 1]  # binary attack prob
    else:
        # multiclass default detection score
        p = 1.0 - proba[:, benign_id]

    # uncertainty score u
    u = uncertainty_entropy(proba)

    # --- 4) BAMS-ABC selects exactly one sample (B=1) ---
    abc_colony = int(getattr(config, "ABC_COLONY_SIZE", 10))
    abc_iters = int(getattr(config, "ABC_ITERS", 8))
    abc_limit = int(getattr(config, "ABC_LIMIT", 6))

    alpha = float(getattr(config, "ALPHA", 1.0))
    beta = float(getattr(config, "BETA", 0.15))
    gamma = float(getattr(config, "GAMMA", 0.0))
    lamb = float(getattr(config, "LAMBDA", 0.0))

    rho_g = float(getattr(config, "BAMS_RHO_G", 0.40))
    rho_f = float(getattr(config, "BAMS_RHO_F", 0.40))

    # For large datasets you typically keep X_norm = None (unless you enable redundancy)
    X_norm = None

    t_sel0 = time.perf_counter()
    batch = bams_abc_select_batch(
        avail=avail,
        X_norm=X_norm,
        p=p,
        u=u,
        cost=cost_avail,
        proba=proba,
        observed_y=observed_y,
        benign_class_id=benign_id,
        B=1,
        rng=rng,
        rho_g=rho_g,
        rho_f=rho_f,
        colony_size=abc_colony,
        iters=abc_iters,
        limit=abc_limit,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lamb=lamb,
    )
    t_sel = time.perf_counter() - t_sel0

    # --- 5) Scan that one selected sample ---
    t_scan0 = time.perf_counter()
    y_new, c_new = world.scan(batch)
    t_scan = time.perf_counter() - t_scan0

    t_total = time.perf_counter() - t_total0

    # --- 6) Report ---
    idx = int(batch[0])
    true_label = int(y_new[0])
    is_attack = (true_label != benign_id)



    # score of selected point (need its position in avail)
    pos = int(np.where(avail == idx)[0][0])
    score = float(p[pos])
    cost_i = float(c_new[0])

    # predicted class for the selected sample (multiclass)
    pred_label = int(np.argmax(proba[pos]))
    pred_prob = float(np.max(proba[pos]))

    dataset_name = getattr(config, "DATASET_NAME", "unknown").lower()
    label_mode = getattr(config, "LABEL_MODE", "binary").lower()

    if dataset_name == "ciciov2024" and label_mode == "multiclass" and proba.shape[1] > 2:
        true_name = CICIOV_LABEL_NAMES.get(true_label, f"label_{true_label}")
        pred_name = CICIOV_LABEL_NAMES.get(pred_label, f"label_{pred_label}")

        # show also top-3 model guesses (useful for analysis)
        topk = np.argsort(proba[pos])[::-1][:3]
        topk_str = ", ".join(
            f"{CICIOV_LABEL_NAMES.get(int(k), f'label_{int(k)}')}={float(proba[pos, int(k)]):.3f}"
            for k in topk
        )

        print(f"True class:      {true_label} ({true_name})")
        print(f"Predicted class: {pred_label} ({pred_name}), prob={pred_prob:.3f}")
        print(f"Top-3 probs:     {topk_str}")
    else:
        print(f"True label: {true_label} -> {'ATTACK' if is_attack else 'BENIGN'}")

    print("\n=== BAMS-ABC single-scan benchmark ===")
    print(f"Dataset: {getattr(config, 'DATASET_NAME', 'unknown')} | LABEL_MODE={getattr(config, 'LABEL_MODE', 'binary')}")
    print(f"Selected index: {idx}")
    print(f"Score p(i): {score:.6f}")
    print(f"True label: {true_label} -> {'ATTACK' if is_attack else 'BENIGN'}")
    print(f"Inspection cost c(i): {cost_i:.6f}")
    print(f"Timing breakdown:")
    print(f"  predict_proba on avail: {t_proba*1000:.2f} ms")
    print(f"  BAMS-ABC selection:     {t_sel*1000:.2f} ms")
    print(f"  scan(batch):            {t_scan*1000:.2f} ms")
    print(f"  TOTAL (score+select+scan): {t_total*1000:.2f} ms")


if __name__ == "__main__":
    main()