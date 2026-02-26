# data/cicmodbus2023_loader.py
from pathlib import Path
import numpy as np
import pandas as pd


FEATURE_COLS = [
    "src_port", "dst_port",
    "duration",
    "orig_bytes", "resp_bytes",
    "orig_pkts", "resp_pkts",
    "orig_ip_bytes", "resp_ip_bytes",
    "missed_bytes",
    "ip_proto",
]

LABEL_CANDIDATES = ["label", "y", "attack", "is_attack", "class"]


def load_modbus2023_flows(csv_path: Path,
                          sample_per_class: int | None = None,
                          random_state: int = 0):
    """
    Load preprocessed Modbus 2023 flows dataset.

    label:
      0 = benign, 1 = attack (binary)

    sampling:
      If sample_per_class is set, sample up to that many rows per class.
    """
    df = pd.read_csv(csv_path)

    # Find label column
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            f"Could not find label column in {csv_path}.\n"
            f"Expected one of: {LABEL_CANDIDATES}\n"
            f"Actual columns: {list(df.columns)}"
        )

    # Convert labels to int 0/1 robustly
    y_series = df[label_col]
    if y_series.dtype == object:
        mapped = y_series.astype(str).str.lower().map({
            "0": 0, "1": 1,
            "benign": 0, "normal": 0,
            "attack": 1, "malicious": 1
        })
        if mapped.isna().any():
            mapped = pd.to_numeric(y_series, errors="coerce")
        y = mapped.fillna(0).astype(int).to_numpy()
    else:
        y = pd.to_numeric(y_series, errors="coerce").fillna(0).astype(int).to_numpy()

    # Keep only existing feature columns
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not feat_cols:
        raise ValueError(f"No expected feature columns found in {csv_path}. Columns={list(df.columns)}")

    # Convert feature columns to numeric
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # --- Balanced sampling by indices (stable across pandas versions) ---
    if sample_per_class is not None:
        rng = np.random.default_rng(random_state)

        idx_keep = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) == 0:
                continue
            take = min(sample_per_class, len(cls_idx))
            chosen = rng.choice(cls_idx, size=take, replace=False)
            idx_keep.append(chosen)

        idx_keep = np.concatenate(idx_keep) if idx_keep else np.arange(len(y))
        rng.shuffle(idx_keep)

        df = df.iloc[idx_keep].reset_index(drop=True)
        y = y[idx_keep]

    X = df[feat_cols].to_numpy(dtype=float)

    # Cost proxy: combine bytes + pkts (normalized)
    total_bytes = np.zeros(len(df), dtype=float)
    if "orig_bytes" in df.columns:
        total_bytes += df["orig_bytes"].to_numpy(dtype=float)
    if "resp_bytes" in df.columns:
        total_bytes += df["resp_bytes"].to_numpy(dtype=float)

    total_pkts = np.zeros(len(df), dtype=float)
    if "orig_pkts" in df.columns:
        total_pkts += df["orig_pkts"].to_numpy(dtype=float)
    if "resp_pkts" in df.columns:
        total_pkts += df["resp_pkts"].to_numpy(dtype=float)

    if total_bytes.max() > 0:
        total_bytes = total_bytes / total_bytes.max()
    if total_pkts.max() > 0:
        total_pkts = total_pkts / total_pkts.max()

    cost = 1.0 + 0.7 * total_bytes + 0.3 * total_pkts

    return X, y, cost
