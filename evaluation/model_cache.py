from pathlib import Path
import joblib

def model_cache_path(dataset: str, seed: int, pca_components: int, cache_dir: str = "cache") -> Path:
    p = Path(cache_dir) / dataset.lower() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"pca_rf_seed_{seed}_pca_{pca_components}.joblib"

def save_model(path: Path, model) -> None:
    joblib.dump(model, path)

def load_model(path: Path):
    return joblib.load(path)