"""
src/fedavg.py — Federated Averaging (FedAvg) Coordinator
=========================================================
Implements the FedAvg algorithm from:
  McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data," AISTATS 2017.

Mathematical formulation
------------------------
Given K nodes, each with n_k local training samples and local weights w_k:

    w_global = Σ (n_k * w_k) / Σ n_k

This is a sample-weighted average of model parameters.
Nodes with more training data have proportionally greater influence.

Author (proposed): Vishal
"""

import numpy as np
import time
import logging
from typing import Callable, Optional, List, Tuple

from src.edge_node import EdgeNode
from src.model import get_weights, set_weights, weights_byte_size

logger = logging.getLogger(__name__)


def federated_average(
    weight_list: List[List[np.ndarray]],
    sample_counts: List[int],
) -> List[np.ndarray]:
    """
    Compute the sample-weighted average of model weight lists.

    Parameters
    ----------
    weight_list   : list of weight-lists, one per participating node
                    Each element is [W1, b1, W2, b2, …]
    sample_counts : number of training samples for each node

    Returns
    -------
    averaged_weights : list of np.ndarray (same structure as input)

    Raises
    ------
    ValueError if weight_list is empty or shapes mismatch
    """
    if not weight_list:
        raise ValueError("federated_average: weight_list is empty")
    if len(weight_list) != len(sample_counts):
        raise ValueError("weight_list and sample_counts must have the same length")

    total_samples = sum(sample_counts)
    if total_samples == 0:
        raise ValueError("federated_average: total sample count is zero")

    # Initialise accumulators as zeros with same shape as first set of weights
    n_tensors = len(weight_list[0])
    averaged = [np.zeros_like(w) for w in weight_list[0]]

    # Weighted accumulation: w_global += (n_k / total_n) * w_k
    for weights, n_k in zip(weight_list, sample_counts):
        if len(weights) != n_tensors:
            raise ValueError(
                f"Weight tensor count mismatch: expected {n_tensors}, got {len(weights)}"
            )
        fraction = n_k / total_samples
        for i, w in enumerate(weights):
            averaged[i] = averaged[i] + fraction * w

    return averaged


class FedAvgCoordinator:
    """
    Central coordinator that orchestrates federated learning rounds.

    Responsibilities
    ----------------
    1. Distribute the global model to all participating nodes.
    2. Trigger local training on each node.
    3. Collect model updates.
    4. Aggregate updates via FedAvg.
    5. Track communication bytes and round metrics.
    6. Evaluate the global model on a held-out test set.
    """

    def __init__(
        self,
        global_model,           # initialised sklearn MLPClassifier
        nodes: list[EdgeNode],
        scaler,                 # fitted StandardScaler
    ):
        self.global_model = global_model
        self.nodes = nodes
        self.scaler = scaler

        # Communication accounting
        self.total_bytes_downloaded: int = 0   # bytes sent FROM coordinator TO nodes
        self.total_bytes_uploaded: int = 0     # bytes sent FROM nodes TO coordinator
        self.rounds_completed: int = 0

        # Per-round history for dashboard plots
        self.round_history: list[dict] = []

    def _broadcast_global_model(self, participating_nodes: list[EdgeNode]) -> int:
        """
        Send the current global model weights to all participating nodes.
        Returns total bytes transmitted (download side).
        """
        global_weights = get_weights(self.global_model)
        bytes_per_node = weights_byte_size(global_weights)
        total_bytes = 0
        for node in participating_nodes:
            node.receive_global_weights(global_weights)
            total_bytes += bytes_per_node
        return total_bytes

    def _collect_updates(
        self, participating_nodes: list[EdgeNode], local_epochs: int
    ) -> tuple[list[list[np.ndarray]], list[int], list[dict], int]:
        """
        Trigger local training on each node and collect updates.

        Returns
        -------
        weight_list    : list of weight arrays from each node
        sample_counts  : corresponding sample counts
        node_logs      : list of per-node training log entries
        upload_bytes   : total bytes uploaded from nodes to coordinator
        """
        weight_list = []
        sample_counts = []
        node_logs = []
        upload_bytes = 0

        for node in participating_nodes:
            log_entry = node.train_local(local_epochs=local_epochs)
            node_logs.append(log_entry)

            weights, n_samples = node.get_model_update()
            weight_list.append(weights)
            sample_counts.append(n_samples)
            upload_bytes += weights_byte_size(weights)

        return weight_list, sample_counts, node_logs, upload_bytes

    def run_round(
        self,
        local_epochs: int = 5,
        fraction_clients: float = 1.0,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        round_num: int = 1,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Execute one complete federated learning round.

        Parameters
        ----------
        local_epochs      : epochs each node trains locally
        fraction_clients  : fraction of nodes to include (1.0 = all)
        X_test, y_test    : optional held-out set for global eval
        round_num         : round index (for logging)
        progress_callback : optional function(message) for live progress display

        Returns
        -------
        round_metrics dict
        """
        t0 = time.perf_counter()
        log = progress_callback or (lambda msg: logger.info(msg))

        log(f"🔄 Round {round_num}: Selecting participating nodes …")

        # ── Node selection ────────────────────────────────────────────────────
        n_participate = max(1, int(len(self.nodes) * fraction_clients))
        rng = np.random.default_rng(self.rounds_completed)
        indices = rng.choice(len(self.nodes), size=n_participate, replace=False)
        participating = [self.nodes[i] for i in sorted(indices)]
        log(f"   Nodes: {[n.node_id for n in participating]}")

        # ── Step 1: Broadcast global model ────────────────────────────────────
        log(f"📡 Round {round_num}: Broadcasting global model …")
        dl_bytes = self._broadcast_global_model(participating)
        self.total_bytes_downloaded += dl_bytes
        log(f"   Downloaded {dl_bytes:,} bytes across {len(participating)} nodes")

        # ── Step 2: Local training + update collection ─────────────────────
        log(f"🏋️  Round {round_num}: Local training ({local_epochs} epochs each) …")
        weight_list, sample_counts, node_logs, ul_bytes = self._collect_updates(
            participating, local_epochs
        )
        self.total_bytes_uploaded += ul_bytes
        log(f"   Uploaded {ul_bytes:,} bytes of model updates")

        # ── Step 3: FedAvg aggregation ────────────────────────────────────────
        log(f"⚙️  Round {round_num}: Aggregating with FedAvg …")
        averaged_weights = federated_average(weight_list, sample_counts)
        set_weights(self.global_model, averaged_weights)
        log(f"   Aggregated {len(participating)} updates (weighted by sample count)")

        # ── Step 4: Evaluate global model ─────────────────────────────────────
        global_acc = None
        if X_test is not None and y_test is not None:
            global_acc = float(self.global_model.score(X_test, y_test))
            log(f"📊 Round {round_num}: Global test accuracy = {global_acc:.4f}")

        elapsed = time.perf_counter() - t0
        self.rounds_completed += 1

        metrics = {
            "round": round_num,
            "participating_nodes": [n.node_id for n in participating],
            "n_participating": len(participating),
            "download_bytes": dl_bytes,
            "upload_bytes": ul_bytes,
            "total_round_bytes": dl_bytes + ul_bytes,
            "global_accuracy": global_acc,
            "node_logs": node_logs,
            "round_time_s": round(elapsed, 4),
            "sample_counts": sample_counts,
        }
        self.round_history.append(metrics)
        log(f"✅ Round {round_num} complete in {elapsed:.2f}s")
        return metrics

    def run_all_rounds(
        self,
        n_rounds: int = 10,
        local_epochs: int = 5,
        fraction_clients: float = 1.0,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> list[dict]:
        """
        Run n_rounds of federated learning and return the full history.
        """
        log = progress_callback or (lambda msg: logger.info(msg))
        log(f"🚀 Starting FedAvg training: {n_rounds} rounds, "
            f"{len(self.nodes)} nodes, {local_epochs} local epochs each")

        all_metrics = []
        for r in range(1, n_rounds + 1):
            metrics = self.run_round(
                local_epochs=local_epochs,
                fraction_clients=fraction_clients,
                X_test=X_test,
                y_test=y_test,
                round_num=r,
                progress_callback=progress_callback,
            )
            all_metrics.append(metrics)

        total_comm = self.total_bytes_downloaded + self.total_bytes_uploaded
        log(f"\n🏁 Federated training complete!")
        log(f"   Total rounds: {n_rounds}")
        log(f"   Total communication: {total_comm:,} bytes "
            f"(↓{self.total_bytes_downloaded:,} + ↑{self.total_bytes_uploaded:,})")
        return all_metrics

    def get_total_communication_bytes(self) -> int:
        """Return total bytes (upload + download) across all federated rounds."""
        return self.total_bytes_downloaded + self.total_bytes_uploaded
