from pathlib import Path
import numpy as np
import pandas as pd

RAW_BENIGN_DIR = Path("raw/dohbrw2020/benign")
RAW_MAL_DIR = Path("raw/dohbrw2020/malicious")
OUT_PATH = Path("processed/dohbrw2020/dohbrw2020_benign_malicious.csv")

# Columns to drop (identifiers / non-generalizable / string columns)
DROP_COLS = {
    "SourceIP", "DestinationIP",
    "TimeStamp",
    # If present, drop any pre-existing label columns:
    "Label", "DoH",
}

def load_all_csvs(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder.resolve()}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def to_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop known ID/time/label columns (case-insensitive)
    drop_lower = {c.strip().lower() for c in DROP_COLS}
    cols_to_drop = [c for c in df.columns if c.strip().lower() in drop_lower]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Convert any remaining object columns to numeric (coerce -> NaN)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_b = load_all_csvs(RAW_BENIGN_DIR)
    df_m = load_all_csvs(RAW_MAL_DIR)

    #  binary label for the experiment:
    # 0 = benign, 1 = malicious
    df_b["label"] = 0
    df_m["label"] = 1

    df = pd.concat([df_b, df_m], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    X = to_numeric_features(X)

    # Ensure no object columns remain
    non_numeric = [c for c in X.columns if X[c].dtype == "object"]
    if non_numeric:
        raise ValueError(f"Still non-numeric columns after preprocessing: {non_numeric}")

    out = X.astype(np.float32)
    out["label"] = y.to_numpy(dtype=np.int32)

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} | shape={out.shape}")
    print("Label counts:", pd.Series(out["label"]).value_counts().to_dict())

if __name__ == "__main__":
    main()