from pathlib import Path
import numpy as np

def cache_path(dataset: str, seed: int, cache_dir: str = "cache") -> Path:
    p = Path(cache_dir) / dataset.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p / f"seed_{seed}.npz"

def save_seed_cache(path: Path, init_batch: np.ndarray, y_init: np.ndarray, c_init: np.ndarray) -> None:
    np.savez_compressed(path, init_batch=init_batch, y_init=y_init, c_init=c_init)

def load_seed_cache(path: Path):
    data = np.load(path, allow_pickle=False)
    return data["init_batch"], data["y_init"], data["c_init"]