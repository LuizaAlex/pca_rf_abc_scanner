from pathlib import Path
import numpy as np
import pandas as pd

def load_dohbrw2020(processed_csv: str):
    path = Path(processed_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {path}")

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("Expected column 'label' in processed dataset.")

    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)


    cost = np.ones(len(y), dtype=np.float32)
    return X, y, cost