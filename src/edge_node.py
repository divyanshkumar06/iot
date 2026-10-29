"""
src/edge_node.py — Simulated IoT Edge Node
==========================================
Each EdgeNode instance represents one logical IoT device in the simulation.
It holds its local dataset, trains a local model, and produces model updates.

** This is a SOFTWARE SIMULATION. No real hardware or network is involved. **

Author (proposed): Tanu
"""

import time
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from config import FEATURES, TARGET, MAX_ITER_LOCAL
from src.model import (
    build_model, get_weights, set_weights, weights_byte_size
)


class EdgeNode:
    """
    Simulates one IoT edge device participating in federated learning.

    Responsibilities
    ----------------
    1. Store its local (private) sensor dataset.
    2. Accept global model parameters from the coordinator.
    3. Train the model locally on its private data.
    4. Return updated model parameters (NOT raw data) to the coordinator.
    5. Report local metrics and communication statistics.
    """

    def __init__(
        self,
        node_id: str,
        local_data: pd.DataFrame,
        scaler: StandardScaler,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        node_id    : unique identifier, e.g. "node_0"
        local_data : the node's private training DataFrame
        scaler     : pre-fitted StandardScaler (shared, fitted on global data summary)
        seed       : random seed for reproducibility
        """
        self.node_id = node_id
        self.seed = seed

        # ── Private local dataset (never leaves the node in federated mode) ──
        self._local_data = local_data.copy()
        self.n_samples = len(local_data)
        self.fault_rate = float(local_data[TARGET].mean())

        # Pre-process features
        self.X_local = scaler.transform(local_data[FEATURES].values)
        self.y_local = local_data[TARGET].values.astype(int)

        # ── Local model (initialised later with global weights) ──────────────
        self.model: MLPClassifier = build_model(seed=seed)
        # Warm dummy fit so coefs_ and intercepts_ exist
        dummy_X = np.zeros((2, len(FEATURES)))
        dummy_y = np.array([0, 1])
        self.model.max_iter = 1
        self.model.fit(dummy_X, dummy_y)
        self.model.max_iter = MAX_ITER_LOCAL
        self._is_initialised: bool = False

        # ── Stats tracking ───────────────────────────────────────────────────
        self.bytes_sent: int = 0        # total bytes uploaded (model updates)
        self.bytes_received: int = 0    # total bytes downloaded (global model)
        self.last_train_time: float = 0.0
        self.last_local_accuracy: float = 0.0
        self.rounds_participated: int = 0
        self.training_log: list[dict] = []

    # ── Receiving the global model ────────────────────────────────────────────
    def receive_global_weights(self, global_weights: list[np.ndarray]) -> None:
        """
        Download the global model from the coordinator and overwrite local weights.
        Simulates the coordinator sending model parameters to this node.
        """
        set_weights(self.model, global_weights)
        received_bytes = weights_byte_size(global_weights)
        self.bytes_received += received_bytes
        self._is_initialised = True

    # ── Local training ────────────────────────────────────────────────────────
    def train_local(self, local_epochs: int = 5) -> dict:
        """
        Train the local model on this node's private dataset.
        Called once per federated round AFTER receiving global weights.

        Parameters
        ----------
        local_epochs : number of full passes over local data

        Returns
        -------
        dict with n_samples, local_accuracy, train_time
        """
        if not self._is_initialised:
            raise RuntimeError(
                f"{self.node_id}: must call receive_global_weights() before training."
            )

        # Temporarily set max_iter for this round's local epochs
        self.model.max_iter = local_epochs
        self.model.n_iter_no_change = max(local_epochs, 10)

        t0 = time.perf_counter()
        self.model.fit(self.X_local, self.y_local)
        elapsed = time.perf_counter() - t0

        # Local training accuracy (on own training data — optimistic)
        local_acc = float(self.model.score(self.X_local, self.y_local))

        self.last_train_time = elapsed
        self.last_local_accuracy = local_acc
        self.rounds_participated += 1

        entry = {
            "round": self.rounds_participated,
            "node_id": self.node_id,
            "n_samples": self.n_samples,
            "local_accuracy": round(local_acc, 4),
            "train_time_s": round(elapsed, 4),
        }
        self.training_log.append(entry)
        return entry

    # ── Sending model updates ─────────────────────────────────────────────────
    def get_model_update(self) -> tuple[list[np.ndarray], int]:
        """
        Return (model_weights, n_samples) for FedAvg aggregation.
        Only model parameters leave the node — raw data stays private.

        Returns
        -------
        weights   : list of weight arrays
        n_samples : number of training examples (needed for weighted average)
        """
        weights = get_weights(self.model)
        upload_bytes = weights_byte_size(weights)
        self.bytes_sent += upload_bytes
        return weights, self.n_samples

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference using this node's current local model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities where supported."""
        return self.model.predict_proba(X)

    # ── Utility ───────────────────────────────────────────────────────────────
    def get_status(self) -> dict:
        """Return a status dict suitable for dashboard display."""
        return {
            "node_id": self.node_id,
            "n_samples": self.n_samples,
            "fault_rate_pct": round(self.fault_rate * 100, 1),
            "rounds_participated": self.rounds_participated,
            "last_local_accuracy": round(self.last_local_accuracy * 100, 2),
            "last_train_time_s": round(self.last_train_time, 4),
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "is_initialised": self._is_initialised,
        }

    def __repr__(self) -> str:
        return (
            f"EdgeNode(id={self.node_id}, samples={self.n_samples}, "
            f"fault_rate={self.fault_rate:.2%})"
        )

