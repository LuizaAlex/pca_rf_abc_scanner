# data/world_factory.py
import numpy as np
from data.simulator import World

def make_world_from_arrays(X: np.ndarray, y: np.ndarray, cost: np.ndarray) -> World:
    scanned = np.zeros(len(y), dtype=bool)
    return World(X=X, y=y, cost=cost, scanned=scanned)
