# data/ciciov2024_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


FEATURE_COLS = [ "DATA_0", "DATA_1", "DATA_2", "DATA_3", "DATA_4", "DATA_5", "DATA_6", "DATA_7"]


def build_labels(df: pd.DataFrame, mode: str = "binary") -> np.ndarray:
    """
    mode:
      - "binary": 0=benign, 1=attack
      - "multiclass": 0=BENIGN, then fixed IDs for attack families (DoS and Spoofing variants)

    Expected CICIoV columns:
      - label: BENIGN/ATTACK (or similar)
      - category: e.g., DoS, SPOOFING
      - specific_class: e.g., DoS, GAS, RPM, SPEED, STEERING_WHEEL (may vary by release)
    """
    mode = str(mode).lower()

    # Normalize helper
    def norm(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower()

    if mode == "binary":
        s = norm(df["label"])
        return (s != "benign").astype(int).to_numpy()

    if mode == "multiclass":
        # Fixed mapping (stable across runs/datasets)
        # 0 must remain BENIGN for runner.py forensic logic
        mapping = {
            "benign": 0,
            "dos": 1,
            "spoofing_gas": 2,
            "spoofing_rpm": 3,
            "spoofing_speed": 4,
            "spoofing_steering_wheel": 5,
        }

        label = norm(df["label"])
        category = norm(df["category"])
        spec = norm(df["specific_class"])

        y = np.empty(len(df), dtype=int)

        # BENIGN rows
        benign_mask = (label == "benign")
        y[benign_mask] = mapping["benign"]

        # ATTACK rows: build a normalized family key
        # - DoS can appear as category=dos or specific_class=dos
        # - Spoofing variants can appear as category=spoofing + spec in {gas,rpm,speed,steering_wheel}
        attack_mask = ~benign_mask

        # Default: mark unknown attacks as generic attack family

        unknown = 1  # fallback to DoS-like bucket

        # Handle DoS
        dos_mask = attack_mask & ((category == "dos") | (spec == "dos"))
        y[dos_mask] = mapping["dos"]

        # Handle Spoofing variants
        # Some releases may store spec as "spoofing-gas" already; normalize separators.
        spec_clean = spec.str.replace("-", "_", regex=False)

        gas_mask = attack_mask & (category == "spoofing") & (spec_clean == "gas")
        rpm_mask = attack_mask & (category == "spoofing") & (spec_clean == "rpm")
        speed_mask = attack_mask & (category == "spoofing") & (spec_clean == "speed")
        steer_mask = attack_mask & (category == "spoofing") & (spec_clean == "steering_wheel")

        y[gas_mask] = mapping["spoofing_gas"]
        y[rpm_mask] = mapping["spoofing_rpm"]
        y[speed_mask] = mapping["spoofing_speed"]
        y[steer_mask] = mapping["spoofing_steering_wheel"]


        covered = benign_mask | dos_mask | gas_mask | rpm_mask | speed_mask | steer_mask
        if not covered.all():

            bad = df.loc[~covered, ["label", "category", "specific_class"]].head(10)
            raise ValueError(
                "Unmapped rows found in CICIoV2024 multiclass labeling. "
                "Please check label/category/specific_class formatting.\n"
                f"Examples:\n{bad}"
            )

        return y

    raise ValueError(f"Unknown label mode: {mode}")


def load_decimal_csvs(
    file_paths: list[Path],
    label_mode: str = "binary",
    sample_per_class: int | None = 50_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads multiple class CSVs, concatenates them, returns (X, y, cost).

    sample_per_class:
      - If set, downsample each file to keep the project runnable on a laptop.
      - CICIoV2024 can be very large (millions of rows), so sampling is practical.
    """
    dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp)

        # Some distributions may include extra whitespace or slightly different casing
        df.columns = [c.strip() for c in df.columns]

        # Keep only required cols (features + label fields)
        needed = set(FEATURE_COLS + ["label", "category", "specific_class"])
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{fp.name} is missing columns: {sorted(missing)}")

        # Optional downsample per class file
        if sample_per_class is not None and len(df) > sample_per_class:
            df = df.sample(n=sample_per_class, random_state=random_state)

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Build X
    X = data[FEATURE_COLS].to_numpy(dtype=float)

    # Build y
    y = build_labels(data, mode=label_mode)

    # Define a cost (we don't have scan time here; use something reasonable)
    # Option 1 (simple): constant cost per “inspection”
    # cost = np.ones(len(data), dtype=float)

    # Option 2: cost proxy = message "payload magnitude"
    cost = 1.0 + (np.abs(X[:, 1:]).sum(axis=1) / (255.0 * 8.0))  # 1..2-ish

    return X, y, cost
