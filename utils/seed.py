import numpy as np

def make_rng(seed: int) -> np.random.Generator:
    """Create a numpy random generator."""
    return np.random.default_rng(seed)
