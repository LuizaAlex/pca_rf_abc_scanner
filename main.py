# main.py
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils.seed import make_rng

from data.simulator import make_synthetic_world
from data.ciciov2024_loader import load_decimal_csvs
from data.cicmodbus2023_loader import load_modbus2023_flows
from data.cicmalanal2017_loader import load_cicmalanal2017
from data.world_factory import make_world_from_arrays
from data.dohbrw2020_loader import load_dohbrw2020
from evaluation.multi_run import (
    run_many,
    make_common_cost_grid,
    summarize_curves,
    ci95_of_mean,
    describe_runs,
    summarize_metric_curves

)

from evaluation.stats import permutation_test_paired
from evaluation.metrics import area_under_curve

def print_dataset_stats(y: np.ndarray, cost: np.ndarray) -> None:
    """Print basic label distribution and cost stats for sanity checking."""
    unique, counts = np.unique(y, return_counts=True)

    print("\n=== Dataset label distribution ===")
    for u, c in zip(unique, counts):
        pct = (c / len(y)) * 100.0
        print(f"label {u}: {c} ({pct:.2f}%)")

    print("\n=== Cost stats ===")
    print(f"min={cost.min():.3f} max={cost.max():.3f} mean={cost.mean():.3f} std={cost.std():.3f}")


def build_base_world(rng):
    """
    Build the dataset "world" used by the experiment.

    Controlled by config.DATASET_NAME:
      - "ciciov2024": load CICIoV2024 decimal CSVs
      - "modbus2023": load Modbus 2023 preprocessed flows_labeled.csv (capped via MODBUS2023_SAMPLE_PER_CLASS)
      - "cicmalanal2017": load CICMalAnal2017 processed CSV (binary benign vs scareware, produced by preprocessing script)
      -"dohbrw2020" _ laod Dohbrw2020 preprocessed flows_labeled.csv
      - "synthetic": generate synthetic dataset
    """
    dataset = getattr(config, "DATASET_NAME", "ciciov2024").lower()

    if dataset == "ciciov2024":
        file_paths = [Path(p) for p in config.CICIOV_DECIMAL_FILES]

        missing = [str(p) for p in file_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Some CICIoV2024 files were not found:\n"
                + "\n".join(missing)
                + "\n\nFix config.CICIOV_DECIMAL_FILES paths or move the CSVs accordingly."
            )

        X, y, cost = load_decimal_csvs(
            file_paths=file_paths,
            label_mode=config.LABEL_MODE,
            sample_per_class=config.SAMPLE_PER_CLASS,
            random_state=config.RANDOM_SEED
        )

        print(f"Loaded CICIoV2024: X={X.shape}, y={y.shape}, cost={cost.shape}")
        print_dataset_stats(y, cost)
        return make_world_from_arrays(X, y, cost)

    if dataset == "modbus2023":
        csv_path = Path(config.MODBUS2023_FLOWS_CSV).expanduser().resolve()
        print(f"[modbus2023] Using flows CSV: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing Modbus 2023 processed file:\n{csv_path}\n\n"
                f"Run preprocessing first:\n"
                f"  export ZEEK_BIN=/opt/homebrew/bin/zeek\n"
                f"  python -m data.cicmodbus2023_preprocess\n"
            )

        # Sanity check columns for the exact file being loaded
        cols = pd.read_csv(csv_path, nrows=1).columns.tolist()
        print(f"[modbus2023] CSV columns (first row): {cols}")
        if "label" not in cols:
            raise ValueError(f"[modbus2023] 'label' column missing in: {csv_path}")

        X, y, cost = load_modbus2023_flows(
            csv_path=csv_path,
            sample_per_class=getattr(config, "MODBUS2023_SAMPLE_PER_CLASS", 50000),
            random_state=config.RANDOM_SEED
        )

        print(f"Loaded Modbus2023 (capped): X={X.shape}, y={y.shape}, cost={cost.shape}")
        print_dataset_stats(y, cost)
        return make_world_from_arrays(X, y, cost)

    if dataset == "dohbrw2020":
        # This expects already ran preprocessing and produced a single processed CSV with:
        #   - numeric features
        #   - 'label' column (0 = benign, 1 = malicious)
        csv_path = Path(config.DOHBRW2020_CSV).expanduser().resolve()
        print(f"[dohbrw2020] Using processed CSV: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing DoHBrw2020 processed file:\n{csv_path}\n\n"
                f"Run preprocessing first:\n"
                f"  python data/dohbrw2020_preprocess.py\n"
            )

        cols = pd.read_csv(csv_path, nrows=1).columns.tolist()
        print(f"[dohbrw2020] CSV columns (first row): {cols}")
        if "label" not in cols:
            raise ValueError(f"[dohbrw2020] 'label' column missing in: {csv_path}")

        X, y, cost = load_dohbrw2020(processed_csv=str(csv_path))
        print(f"Loaded DoHBrw2020: X={X.shape}, y={y.shape}, cost={cost.shape}")
        print_dataset_stats(y, cost)
        return make_world_from_arrays(X, y, cost)

    if dataset == "cicmalanal2017":
        # This expects an already ran preprocessing and produced a single processed CSV with:
        #   - numeric features
        #   - 'label' column (0 = benign, 1 = scareware)
        csv_path = Path(config.CICMALANAL2017_CSV).expanduser().resolve()
        print(f"[cicmalanal2017] Using processed CSV: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing CICMalAnal2017 processed file:\n{csv_path}\n\n"
                f"Run preprocessing first:\n"
                f"  python data/cicmalanal2017_preprocess.py\n"
            )

        # Sanity check columns for the exact file being loaded
        cols = pd.read_csv(csv_path, nrows=1).columns.tolist()
        print(f"[cicmalanal2017] CSV columns (first row): {cols}")
        if "label" not in cols:
            raise ValueError(f"[cicmalanal2017] 'label' column missing in: {csv_path}")

        X, y, cost = load_cicmalanal2017(processed_csv=str(csv_path))

        print(f"Loaded CICMalAnal2017: X={X.shape}, y={y.shape}, cost={cost.shape}")
        print_dataset_stats(y, cost)
        return make_world_from_arrays(X, y, cost)

    # --- Synthetic fallback ---
    X_world = make_synthetic_world(
        rng=rng,
        n_nodes=config.N_NODES,
        n_features=config.N_FEATURES,
        vuln_rate=config.VULN_RATE
    )
    print("Loaded synthetic world.")
    return X_world


def make_results_dir() -> Path:
    """
    Create a timestamped results folder.
    Also includes dataset name so multiple datasets don't overwrite each other.
    """
    dataset = getattr(config, "DATASET_NAME", "unknown").lower()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base = Path(getattr(config, "RESULTS_DIR", "results"))
    out_dir = base / dataset / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



def per_run_auc_from_curves(per_strategy, strat: str, curve_key: str):
    """
    Return a numpy array of per-run AUCs for a curve stored as:
      per_strategy[strat]["curves_cost"] and per_strategy[strat][curve_key]
    """
    aucs = []
    for cost_curve, y_curve in zip(per_strategy[strat]["curves_cost"], per_strategy[strat].get(curve_key, [])):
        aucs.append(area_under_curve(cost_curve, y_curve))
    return np.array(aucs, dtype=float)

def per_run_auc_norm_by_cost(per_strategy, strat: str, curve_key: str):
    """
    Normalize AUC by max_cost so results are comparable across different scan budgets/cost scales.
    For curves already in [0,1] (e.g., balance, macro-F1).
    """
    aucs = []
    for cost_curve, y_curve in zip(per_strategy[strat]["curves_cost"], per_strategy[strat].get(curve_key, [])):
        max_cost = float(np.max(cost_curve)) if len(cost_curve) else 1.0
        aucs.append(area_under_curve(cost_curve, y_curve) / max_cost)
    return np.array(aucs, dtype=float)


def per_run_auc_norm_by_cost_and_classes(per_strategy, strat: str, curve_key: str, n_attack_classes: int):
    """
    Normalize curves that are counts in [0..n_attack_classes] (coverage, confirmed coverage)
    by n_attack_classes and also normalize AUC by max_cost.
    Produces a dimensionless score ~[0,1].
    """
    den = float(max(1, int(n_attack_classes)))
    aucs = []
    for cost_curve, y_curve in zip(per_strategy[strat]["curves_cost"], per_strategy[strat].get(curve_key, [])):
        max_cost = float(np.max(cost_curve)) if len(cost_curve) else 1.0
        y_norm = np.array(y_curve, dtype=float) / den
        aucs.append(area_under_curve(cost_curve, y_norm) / max_cost)
    return np.array(aucs, dtype=float)


def main():
    rng = make_rng(config.RANDOM_SEED)

    base_world = build_base_world(rng)

    benign_class_id = int(getattr(config, "BENIGN_CLASS_ID", 0))
    all_classes = np.unique(base_world.y.astype(int))
    n_attack_classes = int(np.sum(all_classes != benign_class_id))

    print("X shape:", base_world.X.shape, "dtype:", base_world.X.dtype, "MB:", base_world.X.nbytes / 1e6)
    print("y shape:", base_world.y.shape, "dtype:", base_world.y.dtype)
    print("cost shape:", base_world.cost.shape, "dtype:", base_world.cost.dtype, "MB:", base_world.cost.nbytes / 1e6)

    # Add "fab_abc" for forensics: ABC + batch-level class coverage + forensic anchors.
    strategies = ["random", "greedy", "abc", "bams_abc",  "ga"]

    abc_params = {
        "colony_size": config.ABC_COLONY_SIZE,
        "iters": config.ABC_ITERS,
        "limit": config.ABC_LIMIT
    }
    weights = {
        "alpha": config.ALPHA,
        "beta": config.BETA,
        "gamma": config.GAMMA,
        "lamb": config.LAMBDA
    }

    ga_params = {
        "pop_size": config.GA_POP_SIZE,
        "generations": config.GA_GENERATIONS,
        "elite_frac": config.GA_ELITE_FRAC,
        "tournament_k": config.GA_TOURNAMENT_K,
        "mutation_rate": config.GA_MUTATION_RATE,
        "mutation_swaps": config.GA_MUTATION_SWAPS,
    }

    hybrid_params = {
        "greedy_abc": {
            "shortlist_mult": 10,
            "greedy_beta": 0.15,
        },

        # NEW hybrid: BAMS-ABC
        "bams_abc": {
            "rho_g": 0.30,  # greedy anchors fraction
            "rho_f": 0.30,  # forensic/balance anchors fraction
        }
    }


    N_RUNS = getattr(config, "N_RUNS", 10)
    per_strategy = run_many(
        base_world=base_world,
        strategies=strategies,
        n_runs=N_RUNS,
        rounds=config.ROUNDS,
        batch_size=config.BATCH_SIZE,
        initial_seed_scans=config.INITIAL_SEED_SCANS,
        pca_components=config.PCA_COMPONENTS,
        seed_base=config.RANDOM_SEED,
        abc_params=abc_params,
        ga_params=ga_params,
        hybrid_params=hybrid_params,
        fitness_weights=weights
    )

    print("\n=== Confusion on SCANNED items (binary view) ===")
    print("Strategy | Scanned | TP | TN | FP | FN")
    print("-" * 55)

    for strat in strategies:
        tp = int(per_strategy[strat]["tp_scanned"].sum()) if per_strategy[strat]["tp_scanned"].ndim else int(
            per_strategy[strat]["tp_scanned"])
        tn = int(per_strategy[strat]["tn_scanned"].sum()) if per_strategy[strat]["tn_scanned"].ndim else int(
            per_strategy[strat]["tn_scanned"])
        fp = int(per_strategy[strat]["fp_scanned"].sum()) if per_strategy[strat]["fp_scanned"].ndim else int(
            per_strategy[strat]["fp_scanned"])
        fn = int(per_strategy[strat]["fn_scanned"].sum()) if per_strategy[strat]["fn_scanned"].ndim else int(
            per_strategy[strat]["fn_scanned"])

        scanned = tp + tn + fp + fn
        print(f"{strat:>9} | {scanned:6d} | {tp:3d} | {tn:3d} | {fp:3d} | {fn:3d}")

    out_dir = make_results_dir()
    print(f"\nSaved outputs will be written to: {out_dir.resolve()}")

    # ---- Performance + stability table (terminal + Excel) ----
    print("\n=== AUC (vulns vs cost): performance + stability ===")
    print("Strategy | Mean AUC ±95%CI | Time(mean±CI, min) | Std | CV(std/mean) | Median | Min..Max")
    print("-" * 105)

    table_rows = []
    for strat in strategies:
        aucs = per_strategy[strat]["aucs"]
        times = per_strategy[strat]["times_sec"] / 60.0  # minutes

        mean, ci = ci95_of_mean(aucs)
        t_mean, t_ci = ci95_of_mean(times)

        d = describe_runs(aucs)

        print(
            f"{strat:>7} | "
            f"{mean:10.3f} ± {ci:8.3f} | "
            f"{t_mean:7.2f} ± {t_ci:5.2f} | "
            f"{d['std']:8.3f} | "
            f"{d['cv'] * 100:11.3f}% | "
            f"{d['median']:10.3f} | "
            f"{d['min']:10.3f} .. {d['max']:10.3f}"
        )

        table_rows.append({
            "strategy": strat,
            "mean_auc": mean,
            "ci95": ci,
            "mean_time_min": float(t_mean),  # NEW
            "ci95_time_min": float(t_ci),  # NEW
            "std": d["std"],
            "cv": d["cv"],
            "median": d["median"],
            "min": d["min"],
            "max": d["max"],
            "n_runs": len(aucs),
        })

    df_summary = pd.DataFrame(table_rows)

    print("\n=== Final False Negatives (global, binary view) ===")
    print("Strategy | Mean FN ±95%CI | Std | CV(std/mean) | Median | Min..Max")
    print("-" * 80)

    for strat in strategies:
        fns = per_strategy[strat]["final_fn"]
        mean, ci = ci95_of_mean(fns)
        d = describe_runs(fns)

        print(
            f"{strat:>9} | "
            f"{mean:10.1f} ± {ci:8.1f} | "
            f"{d['std']:8.1f} | "
            f"{d['cv'] * 100:11.3f}% | "
            f"{d['median']:10.1f} | "
            f"{d['min']:10.1f} .. {d['max']:10.1f}"
        )

    # Optional: print raw FN per run (useful for debugging)
    print("\nRaw FN per run:")
    for strat in strategies:
        print(f"{strat:>9}: {per_strategy[strat]['final_fn'].astype(int).tolist()}")

    # Forensic coverage AUC (unique attack classes vs cost)
    print("\n=== Forensics: AUC (unique attack classes vs cost) ===")
    print("Strategy | Mean Coverage-AUC ±95%CI | Std | CV(std/mean) | Median | Min..Max")
    print("-" * 90)

    coverage_table_rows = []
    for strat in strategies:
        # Compute per-run coverage AUC from raw curves
        cover_aucs = []
        for cost_curve, cov_curve in zip(per_strategy[strat]["curves_cost"], per_strategy[strat]["curves_coverage"]):
            # cov_curve is "unique attack classes discovered so far" (0..n_attack_classes)
            max_cost = float(np.max(cost_curve)) if len(cost_curve) else 1.0
            den = float(max(1, n_attack_classes))  # attack classes only (exclude benign)

            cov_norm = np.array(cov_curve, dtype=float) / den
            cover_aucs.append(area_under_curve(cost_curve, cov_norm) / max_cost)


        cover_aucs = np.array(cover_aucs, dtype=float)

        mean, ci = ci95_of_mean(cover_aucs)
        d = describe_runs(cover_aucs)

        print(
            f"{strat:>9} | "
            f"{mean:14.3f} ± {ci:8.3f} | "
            f"{d['std']:8.3f} | "
            f"{d['cv'] * 100:11.3f}% | "
            f"{d['median']:10.3f} | "
            f"{d['min']:10.3f} .. {d['max']:10.3f}"
        )

        coverage_table_rows.append({
            "strategy": strat,
            "mean_cov_auc": mean,
            "ci95": ci,
            "std": d["std"],
            "cv": d["cv"],
            "median": d["median"],
            "min": d["min"],
            "max": d["max"],
            "n_runs": len(cover_aucs),
        })

    df_cov_summary = pd.DataFrame(coverage_table_rows)

    print("\n=== Forensics success: AUCs (confirmed coverage / balance / macro-F1) ===")
    print("Strategy | CCC-AUC(mean±CI) | EBC-AUC(mean±CI) | F1-AUC(mean±CI)")
    print("-" * 90)

    fes_rows = []
    for strat in strategies:
        ccc_aucs = []
        ebc_aucs = []
        f1_aucs = []

        for cost_curve, ccc_curve, ebc_curve, f1_curve in zip(
                per_strategy[strat]["curves_cost"],
                per_strategy[strat]["curves_confirmed_coverage"],
                per_strategy[strat]["curves_balance"],
                per_strategy[strat]["curves_macro_f1"],
        ):
            max_cost = float(np.max(cost_curve)) if len(cost_curve) else 1.0
            den = float(max(1, n_attack_classes))

            ccc_norm = np.array(ccc_curve, dtype=float) / den
            ccc_aucs.append(area_under_curve(cost_curve, ccc_norm) / max_cost)

            ebc_aucs.append(area_under_curve(cost_curve, ebc_curve) / max_cost)
            f1_aucs.append(area_under_curve(cost_curve, f1_curve) / max_cost)

        ccc_aucs = np.array(ccc_aucs, dtype=float)
        ebc_aucs = np.array(ebc_aucs, dtype=float)
        f1_aucs = np.array(f1_aucs, dtype=float)

        ccc_mean, ccc_ci = ci95_of_mean(ccc_aucs)
        ebc_mean, ebc_ci = ci95_of_mean(ebc_aucs)
        f1_mean, f1_ci = ci95_of_mean(f1_aucs)

        times = per_strategy[strat]["times_sec"] / 60.0
        t_mean, t_ci = ci95_of_mean(times)

        print(
            f"{strat:>9} | "
            f"{t_mean:6.2f}±{t_ci:4.2f} min | "
            f"{ccc_mean:10.3f}±{ccc_ci:6.3f} | "
            f"{ebc_mean:10.3f}±{ebc_ci:6.3f} | "
            f"{f1_mean:10.3f}±{f1_ci:6.3f}"
        )

        # Optional: combine into one "FES" per run (normalized by max possible on that run)
        fes_scores = []
        for cost_curve, ccc_curve, ebc_curve, f1_curve, auc_ccc, auc_ebc, auc_f1 in zip(
                per_strategy[strat]["curves_cost"],
                per_strategy[strat]["curves_confirmed_coverage"],
                per_strategy[strat]["curves_balance"],
                per_strategy[strat]["curves_macro_f1"],
                ccc_aucs, ebc_aucs, f1_aucs
        ):
            max_cost = float(np.max(cost_curve)) if len(cost_curve) else 1.0
            max_ccc = float(np.max(ccc_curve)) if len(ccc_curve) else 1.0

            # Normalize AUCs to [0,1] scale-ish
            norm_ccc = float(auc_ccc) / (max_cost * max(1.0, max_ccc))
            norm_ebc = float(auc_ebc) / (max_cost * 1.0)  # balance max is 1
            norm_f1 = float(auc_f1) / (max_cost * 1.0)  # macro-f1 max is 1

            fes = 0.5 * norm_ccc + 0.3 * norm_ebc + 0.2 * norm_f1
            fes_scores.append(fes)

        fes_scores = np.array(fes_scores, dtype=float)
        fes_mean, fes_ci = ci95_of_mean(fes_scores)
        fes_desc = describe_runs(fes_scores)

        fes_rows.append({
            "strategy": strat,
            "mean_ccc_auc": ccc_mean,
            "ci95_ccc": ccc_ci,
            "mean_ebc_auc": ebc_mean,
            "ci95_ebc": ebc_ci,
            "mean_f1_auc": f1_mean,
            "ci95_f1": f1_ci,
            "mean_fes": fes_mean,
            "ci95_fes": fes_ci,
            "std_fes": fes_desc["std"],
            "cv_fes": fes_desc["cv"],
            "median_fes": fes_desc["median"],
            "min_fes": fes_desc["min"],
            "max_fes": fes_desc["max"],
            "n_runs": len(fes_scores),
        })

    df_fes_summary = pd.DataFrame(fes_rows)

    # ---- Permutation test p-values (paired) ----
    print("\n=== Permutation test p-values on AUC (paired) ===")

   # hybrid_name = "greedy_shortlist_abc"

    p_abc_vs_greedy = permutation_test_paired(
        per_strategy["abc"]["aucs"], per_strategy["greedy"]["aucs"], seed=config.RANDOM_SEED
    )
    p_abc_vs_random = permutation_test_paired(
        per_strategy["abc"]["aucs"], per_strategy["random"]["aucs"], seed=config.RANDOM_SEED
    )
    p_ga_vs_greedy = permutation_test_paired(
        per_strategy["ga"]["aucs"], per_strategy["greedy"]["aucs"], seed=config.RANDOM_SEED
    )
    p_ga_vs_random = permutation_test_paired(
        per_strategy["ga"]["aucs"], per_strategy["random"]["aucs"], seed=config.RANDOM_SEED
    )



    print(f"ABC vs Greedy p ≈ {p_abc_vs_greedy:.4f}")
    print(f"ABC vs Random p ≈ {p_abc_vs_random:.4f}")
    print(f" GA vs Greedy p ≈ {p_ga_vs_greedy:.4f}")
    print(f" GA vs Random p ≈ {p_ga_vs_random:.4f}")

    df_pvals = pd.DataFrame([
        {"comparison": "ABC vs Greedy", "p_value": p_abc_vs_greedy},
        {"comparison": "ABC vs Random", "p_value": p_abc_vs_random},
        {"comparison": "GA vs Greedy", "p_value": p_ga_vs_greedy},
        {"comparison": "GA vs Random", "p_value": p_ga_vs_random},

    ])

    print("\n=== Permutation test p-values (paired) for FORENSICS metrics ===")

    # Compute per-run AUC arrays for forensic metrics
    ccc_auc = {s: per_run_auc_norm_by_cost_and_classes(per_strategy, s, "curves_confirmed_coverage", n_attack_classes)
               for s in strategies}
    ebc_auc = {s: per_run_auc_norm_by_cost(per_strategy, s, "curves_balance") for s in strategies}
    f1_auc = {s: per_run_auc_norm_by_cost(per_strategy, s, "curves_macro_f1") for s in strategies}

    # Optional: first-hit class coverage
    cov_auc = {s: per_run_auc_norm_by_cost_and_classes(per_strategy, s, "curves_coverage", n_attack_classes)
               for s in strategies}
    # Optional: per-run combined FES
    fes_auc = {}
    for s in strategies:
        fes_scores = []
        for cost_curve, ccc_curve, ebc_curve, f1_curve in zip(
                per_strategy[s]["curves_cost"],
                per_strategy[s]["curves_confirmed_coverage"],
                per_strategy[s]["curves_balance"],
                per_strategy[s]["curves_macro_f1"],
        ):
            auc_ccc = float(area_under_curve(cost_curve, ccc_curve))
            auc_ebc = float(area_under_curve(cost_curve, ebc_curve))
            auc_f1 = float(area_under_curve(cost_curve, f1_curve))

            max_cost = float(np.max(cost_curve)) if len(cost_curve) else 1.0
            max_ccc = float(np.max(ccc_curve)) if len(ccc_curve) else 1.0

            norm_ccc = auc_ccc / (max_cost * max(1.0, max_ccc))
            norm_ebc = auc_ebc / (max_cost * 1.0)
            norm_f1 = auc_f1 / (max_cost * 1.0)

            fes = 0.5 * norm_ccc + 0.3 * norm_ebc + 0.2 * norm_f1
            fes_scores.append(float(fes))
        fes_auc[s] = np.array(fes_scores, dtype=float)



    comparisons = [
        ("abc", "greedy"),
        ("abc", "random"),
        ("ga", "greedy"),
        ("ga", "random"),
        ("bams_abc", "abc"),
        ("bams_abc", "greedy"),
        ("bams_abc", "random"),
        ("bams_abc", "ga"),
    ]

    # Run tests for each metric
    def pvals_for_metric(metric_name: str, metric_dict: dict):
        rows = []
        for a, b in comparisons:
            if a not in metric_dict or b not in metric_dict:
                continue
            pa = metric_dict[a]
            pb = metric_dict[b]
            # defensive: require same length
            n = min(len(pa), len(pb))
            if n == 0:
                continue
            pval = permutation_test_paired(pa[:n], pb[:n], seed=config.RANDOM_SEED)
            rows.append({"metric": metric_name, "comparison": f"{a} vs {b}", "p_value": float(pval)})
        return rows

    rows = []
    rows += pvals_for_metric("CCC_AUC", ccc_auc)
    rows += pvals_for_metric("EBC_AUC", ebc_auc)
    rows += pvals_for_metric("MACRO_F1_AUC", f1_auc)
    rows += pvals_for_metric("COVERAGE_AUC", cov_auc)
    rows += pvals_for_metric("FES", fes_auc)

    df_pvals_forensics = pd.DataFrame(rows)

    # Print a compact view
    for metric in ["CCC_AUC", "EBC_AUC", "MACRO_F1_AUC", "COVERAGE_AUC", "FES"]:
        sub = df_pvals_forensics[df_pvals_forensics["metric"] == metric]
        if len(sub) == 0:
            continue
        print(f"\n-- {metric} --")
        for _, r in sub.iterrows():
            print(f"{r['comparison']:<28} p ≈ {r['p_value']:.4f}")


    # ---- Plot mean curve + CI band (popup + PNG) ----
    cost_grid = make_common_cost_grid(per_strategy, num_points=250)
    curve_summary = summarize_curves(per_strategy, cost_grid)

    # NEW: summarize forensic coverage curves (unique attack classes vs cost)
    coverage_summary = summarize_metric_curves(per_strategy, cost_grid, "curves_coverage")

    # NEW: forensic success curve summaries
    confirmed_cov_summary = summarize_metric_curves(per_strategy, cost_grid, "curves_confirmed_coverage")
    balance_summary = summarize_metric_curves(per_strategy, cost_grid, "curves_balance")
    macro_f1_summary = summarize_metric_curves(per_strategy, cost_grid, "curves_macro_f1")


    # ---- Plot discovery curve (mean ± 95% CI) ----
    plt.figure()
    for strat in strategies:
        mean_curve = curve_summary[strat]["mean"]
        ci95_curve = curve_summary[strat]["ci95"]

        plt.plot(cost_grid, mean_curve, label=strat)
        plt.fill_between(cost_grid, mean_curve - ci95_curve, mean_curve + ci95_curve, alpha=0.2)

    plt.xlabel("Cumulative inspection cost")
    plt.ylabel("Cumulative attacks found")
    plt.title(f"Discovery Efficiency (mean ± 95% CI), runs={N_RUNS}")
    plt.legend()

    plot_path = out_dir / "discovery_curve.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {plot_path.resolve()}")
    plt.show()

    # ---- Plot forensic coverage curve (mean ± 95% CI) ----
    plt.figure()
    for strat in strategies:
        mean_curve = coverage_summary[strat]["mean"]
        ci95_curve = coverage_summary[strat]["ci95"]

        plt.plot(cost_grid, mean_curve, label=strat)
        plt.fill_between(cost_grid, mean_curve - ci95_curve, mean_curve + ci95_curve, alpha=0.2)

    plt.xlabel("Cumulative inspection cost")
    plt.ylabel("Unique attack classes discovered")
    plt.title(f"Forensic Coverage (mean ± 95% CI), runs={N_RUNS}")
    plt.legend()

    plot_path = out_dir / "coverage_curve.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {plot_path.resolve()}")
    plt.show()

    plt.figure()
    for strat in strategies:
        mean_curve = confirmed_cov_summary[strat]["mean"]
        ci95_curve = confirmed_cov_summary[strat]["ci95"]
        plt.plot(cost_grid, mean_curve, label=strat)
        plt.fill_between(cost_grid, mean_curve - ci95_curve, mean_curve + ci95_curve, alpha=0.2)

    plt.xlabel("Cumulative inspection cost")
    plt.ylabel("Confirmed attack classes (>= m samples)")
    plt.title(f"Confirmed Coverage (mean ± 95% CI), runs={N_RUNS}")
    plt.legend()

    plot_path = out_dir / "confirmed_coverage_curve.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {plot_path.resolve()}")
    plt.show()

    plt.figure()
    for strat in strategies:
        mean_curve = balance_summary[strat]["mean"]
        ci95_curve = balance_summary[strat]["ci95"]
        plt.plot(cost_grid, mean_curve, label=strat)
        plt.fill_between(cost_grid, mean_curve - ci95_curve, mean_curve + ci95_curve, alpha=0.2)

    plt.xlabel("Cumulative inspection cost")
    plt.ylabel("Evidence balance (normalized entropy)")
    plt.title(f"Evidence Balance (mean ± 95% CI), runs={N_RUNS}")
    plt.legend()

    plot_path = out_dir / "evidence_balance_curve.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {plot_path.resolve()}")
    plt.show()

    # ---- Save Excel workbook with summary + raw data ----

    excel_path = out_dir / "results.xlsx"

    # 1) Per-run discovery AUC (already computed)
    auc_rows = []
    for strat in strategies:
        for run_idx, auc in enumerate(per_strategy[strat]["aucs"], start=1):
            auc_rows.append({"strategy": strat, "run": run_idx, "auc_discovery": float(auc)})
    df_aucs = pd.DataFrame(auc_rows)

    # 2) Mean discovery curve points (vulns vs cost)
    curve_rows = []
    for strat in strategies:
        mean_curve = curve_summary[strat]["mean"]
        ci95_curve = curve_summary[strat]["ci95"]
        for x, m, c in zip(cost_grid, mean_curve, ci95_curve):
            curve_rows.append({
                "strategy": strat,
                "cost": float(x),
                "mean_vulns": float(m),
                "ci95": float(c),
                "lower": float(m - c),
                "upper": float(m + c),
            })
    df_curves = pd.DataFrame(curve_rows)

    # 3) Mean first-hit coverage curve points (unique attack classes vs cost)
    coverage_rows = []
    for strat in strategies:
        mean_curve = coverage_summary[strat]["mean"]
        ci95_curve = coverage_summary[strat]["ci95"]
        for x, m, c in zip(cost_grid, mean_curve, ci95_curve):
            coverage_rows.append({
                "strategy": strat,
                "cost": float(x),
                "mean_unique_attack_classes": float(m),
                "ci95": float(c),
                "lower": float(m - c),
                "upper": float(m + c),
            })
    df_coverage_curves = pd.DataFrame(coverage_rows)

    # 4) Mean confirmed coverage curve points (>= m per class)
    ccc_rows = []
    for strat in strategies:
        mean_curve = confirmed_cov_summary[strat]["mean"]
        ci95_curve = confirmed_cov_summary[strat]["ci95"]
        for x, m, c in zip(cost_grid, mean_curve, ci95_curve):
            ccc_rows.append({
                "strategy": strat,
                "cost": float(x),
                "mean_confirmed_attack_classes": float(m),
                "ci95": float(c),
                "lower": float(m - c),
                "upper": float(m + c),
            })
    df_ccc_curves = pd.DataFrame(ccc_rows)

    # 5) Mean evidence balance curve points (normalized entropy)
    bal_rows = []
    for strat in strategies:
        mean_curve = balance_summary[strat]["mean"]
        ci95_curve = balance_summary[strat]["ci95"]
        for x, m, c in zip(cost_grid, mean_curve, ci95_curve):
            bal_rows.append({
                "strategy": strat,
                "cost": float(x),
                "mean_evidence_balance": float(m),
                "ci95": float(c),
                "lower": float(m - c),
                "upper": float(m + c),
            })
    df_balance_curves = pd.DataFrame(bal_rows)

    # 6) Mean macro-F1 curve points
    f1_rows = []
    for strat in strategies:
        mean_curve = macro_f1_summary[strat]["mean"]
        ci95_curve = macro_f1_summary[strat]["ci95"]
        for x, m, c in zip(cost_grid, mean_curve, ci95_curve):
            f1_rows.append({
                "strategy": strat,
                "cost": float(x),
                "mean_macro_f1": float(m),
                "ci95": float(c),
                "lower": float(m - c),
                "upper": float(m + c),
            })
    df_macro_f1_curves = pd.DataFrame(f1_rows)

    # 7) Optional: per-run forensic AUCs + FES per run
    fes_run_rows = []
    for strat in strategies:
        # These curves exist per run (lists), collected in multi_run.py
        for run_idx, (cost_curve, ccc_curve, ebc_curve, f1_curve) in enumerate(
                zip(
                    per_strategy[strat]["curves_cost"],
                    per_strategy[strat]["curves_confirmed_coverage"],
                    per_strategy[strat]["curves_balance"],
                    per_strategy[strat]["curves_macro_f1"],
                ),
                start=1
        ):
            auc_ccc = float(area_under_curve(cost_curve, ccc_curve))
            auc_ebc = float(area_under_curve(cost_curve, ebc_curve))
            auc_f1 = float(area_under_curve(cost_curve, f1_curve))

            max_cost = float(np.max(cost_curve)) if len(cost_curve) else 1.0
            max_ccc = float(np.max(ccc_curve)) if len(ccc_curve) else 1.0

            # normalize to get a unit-ish combined score
            norm_ccc = auc_ccc / (max_cost * max(1.0, max_ccc))
            norm_ebc = auc_ebc / (max_cost * 1.0)
            norm_f1 = auc_f1 / (max_cost * 1.0)

            fes = 0.5 * norm_ccc + 0.3 * norm_ebc + 0.2 * norm_f1

            fes_run_rows.append({
                "strategy": strat,
                "run": run_idx,
                "auc_ccc": auc_ccc,
                "auc_ebc": auc_ebc,
                "auc_macro_f1": auc_f1,
                "fes": float(fes),
            })

    df_fes_per_run = pd.DataFrame(fes_run_rows)

    # --- Write everything to Excel ---
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # summaries
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_cov_summary.to_excel(writer, sheet_name="coverage_summary", index=False)
        df_fes_summary.to_excel(writer, sheet_name="forensics_success", index=False)
        df_pvals.to_excel(writer, sheet_name="p_values", index=False)
        df_pvals_forensics.to_excel(writer, sheet_name="p_values_forensics", index=False)  # NEW
        # per-run values
        df_aucs.to_excel(writer, sheet_name="aucs_per_run", index=False)
        df_fes_per_run.to_excel(writer, sheet_name="forensics_per_run", index=False)

        # curve points (mean ± CI on common grid)
        df_curves.to_excel(writer, sheet_name="curve_points", index=False)
        df_coverage_curves.to_excel(writer, sheet_name="coverage_curve_points", index=False)
        df_ccc_curves.to_excel(writer, sheet_name="confirmed_cov_points", index=False)
        df_balance_curves.to_excel(writer, sheet_name="balance_curve_points", index=False)
        df_macro_f1_curves.to_excel(writer, sheet_name="macro_f1_curve_points", index=False)

    print(f"Saved Excel: {excel_path.resolve()}")

if __name__ == "__main__":
    main()