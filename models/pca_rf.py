# models/pca_rf.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def make_pca_rf_model(n_components: int, seed: int) -> Pipeline:
    """
    PCA + Random Forest pipeline.
    - Standardize features
    - Reduce dimensionality with PCA
    - Classify vulnerability with RF
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced"
        )),
    ])

def predict_vuln_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return probability of vulnerable class (class=1) for each node."""
    return model.predict_proba(X)[:, 1]

def uncertainty_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy uncertainty measure: high near p=0.5, low near 0 or 1."""
    eps = 1e-9
    return -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))

def uncertainty_entropy(proba: np.ndarray) -> np.ndarray:
    """
    Shannon entropy for binary OR multiclass.
    proba shape:
      - (n, 2) for binary
      - (n, k) for multiclass
    """
    eps = 1e-9
    p = np.clip(proba, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)
