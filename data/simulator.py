from dataclasses import dataclass
import numpy as np

@dataclass
class World:
    """
    A simulated network environment.

    - X: per-node feature vectors (what your model sees)
    - y: ground truth vulnerability labels (unknown in real life; known here for evaluation)
    - cost: scan cost per node (time/packets proxy)
    - scanned: tracks which nodes you already scanned
    """
    X: np.ndarray
    y: np.ndarray
    cost: np.ndarray
    scanned: np.ndarray

    def available_indices(self) -> np.ndarray:
        """Return indices of nodes that haven't been scanned yet."""
        return np.where(~self.scanned)[0]

    def scan(self, indices: np.ndarray):
        """
        Simulate scanning:
        - marks nodes as scanned
        - reveals the label (vulnerable or not)
        - returns labels and cost
        """
        self.scanned[indices] = True
        return self.y[indices], self.cost[indices]


def make_synthetic_world(rng: np.random.Generator,
                         n_nodes: int,
                         n_features: int,
                         vuln_rate: float) -> World:
    """
    Create a toy dataset:
    - Some hidden structure correlates with vulnerability.
    - Costs vary per node.
    """
    # Base features
    X = rng.normal(0, 1, size=(n_nodes, n_features))

    # Hidden "risk signal": some linear combination + noise
    w = rng.normal(0, 1, size=(n_features,))
    raw_risk = X @ w + 0.5 * rng.normal(0, 1, size=n_nodes)

    # Convert risk -> probability, then sample vulnerabilities
    # Control prevalence by shifting threshold using percentile.
    threshold = np.quantile(raw_risk, 1.0 - vuln_rate)
    y = (raw_risk >= threshold).astype(int)

    # Cost model: some nodes are more expensive to scan (random + feature-based)
    base_cost = rng.uniform(0.5, 2.0, size=n_nodes)
    feature_cost = 0.15 * np.abs(X[:, 0])  # tie cost slightly to a feature
    cost = base_cost + feature_cost

    scanned = np.zeros(n_nodes, dtype=bool)
    return World(X=X, y=y, cost=cost, scanned=scanned)
