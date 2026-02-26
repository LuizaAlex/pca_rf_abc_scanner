# evaluation/multi_run.py
import copy
import numpy as np
import gc
from evaluation.runner import run_experiment
from evaluation.metrics import area_under_curve
import time
from data.world_factory import make_world_from_arrays

def interpolate_curve(x, y, x_grid):
    """
    Interpolate y over a common x_grid so we can average curves across runs.
    Assumes x is increasing.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # If curve is empty, return zeros
    if len(x) == 0 or len(y) == 0:
        return np.zeros_like(x_grid, dtype=float)

    # If x/y lengths mismatch, truncate to the shortest (defensive)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    # np.interp uses left/right fill; fill before first point with y[0]
    return np.interp(x_grid, x, y, left=y[0], right=y[-1])
def run_many(
    base_world,
    strategies,
    n_runs,
    rounds,
    batch_size,
    initial_seed_scans,
    pca_components,
    seed_base,
    abc_params,
    ga_params,
    hybrid_params,
    fitness_weights,
    fab_params=None  # kept for backward compatibility; not used
):
    """
    Paired multi-run evaluation:
    - Generate run seeds once.
    - Use the same seed for each strategy in the same run_id.
    This makes comparisons much more statistically reliable.
    """
    rng_master = np.random.default_rng(seed_base)
    run_seeds = [int(rng_master.integers(1, 1_000_000_000)) for _ in range(n_runs)]

    per_strategy = {}
    for strat in strategies:
        per_strategy[strat] = {
            "aucs": [],
            "curves_cost": [],
            "curves_vulns": [],
            "curves_coverage": [],  # forensic: distinct attack classes found over time (first-hit)

            # NEW forensics-success curves:
            "curves_confirmed_coverage": [],
            "curves_balance": [],
            "curves_macro_f1": [],
            "times_sec": [],
        }

    for run_id, seed in enumerate(run_seeds, start=1):
        for strat in strategies:
            print(f"[{strat}] run {run_id}/{n_runs}", flush=True)

            world = make_world_from_arrays(base_world.X, base_world.y, base_world.cost)
            rng = np.random.default_rng(seed)

            t0 = time.perf_counter()

            hist = run_experiment(
                world=world,
                rng=rng,
                strategy=strat,
                rounds=rounds,
                batch_size=batch_size,
                initial_seed_scans=initial_seed_scans,
                pca_components=pca_components,
                seed=seed,
                abc_params=abc_params,
                ga_params=ga_params,
                hybrid_params=hybrid_params,
                fitness_weights=fitness_weights,
            )

            if strat == "random" and run_id == 1:
                print("DEBUG cost_curve first/last:", hist["cost_curve"][:5], "...", hist["cost_curve"][-5:])
                print("DEBUG coverage_curve first/last:", hist.get("attack_class_coverage_curve", [])[:10], "...",
                      hist.get("attack_class_coverage_curve", [])[-10:])
                print("DEBUG unique values coverage:", sorted(set(hist.get("attack_class_coverage_curve", []))))

            elapsed = time.perf_counter() - t0
            per_strategy[strat]["times_sec"].append(float(elapsed))

            # Core discovery AUC (attacks vs cost)
            auc = area_under_curve(hist.get("cost_curve", []), hist.get("vulns_curve", []))
            per_strategy[strat]["aucs"].append(auc)

            # Store raw curves for later interpolation/plotting
            per_strategy[strat]["curves_cost"].append(hist.get("cost_curve", []))
            per_strategy[strat]["curves_vulns"].append(hist.get("vulns_curve", []))

            # Forensics: first-hit class coverage curve
            per_strategy[strat]["curves_coverage"].append(hist.get("attack_class_coverage_curve", []))

            # NEW forensics-success curves (may be missing for older runners; safe defaults)
            per_strategy[strat]["curves_confirmed_coverage"].append(hist.get("confirmed_coverage_curve", []))
            per_strategy[strat]["curves_balance"].append(hist.get("evidence_balance_curve", []))
            per_strategy[strat]["curves_macro_f1"].append(hist.get("macro_f1_curve", []))

            del hist, world, rng
            gc.collect()


    # Convert auc lists to numpy arrays
    for strat in strategies:
        per_strategy[strat]["aucs"] = np.array(per_strategy[strat]["aucs"], dtype=float)
        per_strategy[strat]["times_sec"] = np.array(per_strategy[strat]["times_sec"], dtype=float)  # NEW
    return per_strategy


def make_common_cost_grid(per_strategy, num_points=200):
    """
    Build a shared x-axis (cost grid) common to all strategies.
    Uses the minimum max-cost across strategies/runs to ensure overlap.
    """
    max_costs = []
    for strat_data in per_strategy.values():
        for c in strat_data["curves_cost"]:
            if len(c) > 0:
                max_costs.append(c[-1])

    if not max_costs:
        return np.linspace(0.0, 1.0, num_points)

    common_max = float(min(max_costs))
    return np.linspace(0.0, common_max, num_points)


def summarize_curves(per_strategy, cost_grid):
    """
    Interpolate each run to cost_grid, then compute mean and 95% CI
    for the default discovery curve (vulns vs cost).
    """
    out = {}

    for strat, data in per_strategy.items():
        ys = []
        for x, y in zip(data["curves_cost"], data["curves_vulns"]):
            ys.append(interpolate_curve(x, y, cost_grid))

        # Defensive: if no runs exist, return zeros
        if not ys:
            mean = np.zeros_like(cost_grid)
            ci95 = np.zeros_like(cost_grid)
        else:
            Y = np.vstack(ys)  # (n_runs, grid_points)
            mean = Y.mean(axis=0)
            n = Y.shape[0]
            std = Y.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
            ci95 = 1.96 * std / np.sqrt(max(n, 1))

        out[strat] = {"mean": mean, "ci95": ci95}

    return out


def summarize_metric_curves(per_strategy, cost_grid, curve_key: str):
    """
    Generic curve summarizer (mean ± 95% CI) for any curve stored in per_strategy[strat][curve_key].
    Example: curve_key = "curves_coverage"
    """
    out = {}

    for strat, data in per_strategy.items():
        ys = []
        curves = data.get(curve_key, [])

        for x, y in zip(data["curves_cost"], curves):
            ys.append(interpolate_curve(x, y, cost_grid))

        if not ys:
            mean = np.zeros_like(cost_grid)
            ci95 = np.zeros_like(cost_grid)
        else:
            Y = np.vstack(ys)
            mean = Y.mean(axis=0)
            n = Y.shape[0]
            std = Y.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
            ci95 = 1.96 * std / np.sqrt(max(n, 1))

        out[strat] = {"mean": mean, "ci95": ci95}

    return out


def ci95_of_mean(x):
    x = np.array(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else 0.0, 0.0)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def describe_runs(x):
    """
    Returns summary stats for a list/array of per-run metrics.
    """
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}

    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    cv = float(std / mean) if mean != 0 else 0.0

    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": float(x.min()),
        "median": float(np.median(x)),
        "max": float(x.max()),
    }