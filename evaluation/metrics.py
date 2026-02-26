# evaluation/metrics.py
import numpy as np

def area_under_curve(x, y):
    """Trapezoid area under y(x). x must be increasing."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return 0.0
    return np.trapezoid(y, x)

def summary(history):
    return {
        "total_vulns": history["total_vulns"],
        "total_cost": history["total_cost"],
        "total_scans": history["total_scans"],
        "auc_vulns_vs_cost": area_under_curve(history["cost_curve"], history["vulns_curve"]),
    }
