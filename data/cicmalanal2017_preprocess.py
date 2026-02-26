# data/cicmalanal2017_preprocess.py
from pathlib import Path
import numpy as np
import pandas as pd

RAW_ROOT = Path("raw/cicmalanal2017")
OUT_PATH = Path("processed/cicmalanal2017/cicmalanal2017_benign_scareware_adware.csv")

CLASS_DIRS = {
    "benign": RAW_ROOT / "benign",
    "scareware": RAW_ROOT / "scareware",
    "adware": RAW_ROOT / "adware",
}

# IMPORTANT: benign must remain 0 for runner.py forensics logic
LABEL_MAP = {
    "benign": 0,
    "scareware": 1,
    "adware": 2,
}

DROP_COLS_EXACT = {
    "Flow ID", "Timestamp",
    "Source IP", "Destination IP",
    "Source Port", "Destination Port",
    "Protocol",
    "Label",
}

def load_all_csv(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder.resolve()}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def make_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_lower = {c.lower() for c in DROP_COLS_EXACT}
    cols_to_drop = [c for c in df.columns if c.strip().lower() in drop_lower]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    for class_name, folder in CLASS_DIRS.items():
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder for class '{class_name}': {folder.resolve()}")

        df = load_all_csv(folder)
        df["label"] = int(LABEL_MAP[class_name])
        parts.append(df)

    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    X = make_numeric_features(X)

    non_numeric = [c for c in X.columns if X[c].dtype == "object"]
    if non_numeric:
        raise ValueError(f"Still non-numeric columns after preprocessing: {non_numeric}")

    out = X.astype(np.float32)
    out["label"] = y.to_numpy(dtype=int)

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} | shape={out.shape}")
    print("Label counts:", out["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()