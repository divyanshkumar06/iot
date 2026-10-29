"""
src/communication.py — Communication Cost Accounting
=====================================================
Documents and calculates the bytes transferred in both centralized
and federated modes.

Methodology (also documented in README)
----------------------------------------
All byte counts are ESTIMATED APPLICATION-LEVEL PAYLOAD bytes.
They do NOT reflect real-world TCP/IP overhead, compression, or
network energy consumption.

Centralized mode
  - Each node uploads its raw training records to the server.
  - Cost = n_records × BYTES_PER_RECORD_CSV

Federated mode
  - Round 0:  Server downloads global model to all nodes.
  - Each round upload:   Each node sends its model weights to server.
  - Each round download: Server sends aggregated weights to all nodes.

Author (proposed): Vishal
"""

import numpy as np
from config import BYTES_PER_RECORD_CSV, BYTES_PER_FLOAT


def bytes_for_raw_records(n_records: int) -> int:
    """
    Estimated bytes to transmit n_records as CSV rows.
    6 features + 1 label ≈ 80 bytes per record (conservative estimate).
    """
    return n_records * BYTES_PER_RECORD_CSV


def bytes_for_weights(weight_list: list[np.ndarray]) -> int:
    """
    Bytes required to transmit a list of weight arrays.
    Assumes float64 (8 bytes per element), no compression.
    """
    return sum(w.nbytes for w in weight_list)


def bytes_for_weight_shapes(shapes: list[tuple]) -> int:
    """
    Estimate bytes from shapes alone (useful before model is fitted).
    shapes: list of (rows, cols) tuples (or (size,) for bias vectors)
    """
    total_params = sum(np.prod(s) for s in shapes)
    return int(total_params) * BYTES_PER_FLOAT


def federated_total_bytes(
    model_weight_bytes: int,
    n_nodes: int,
    n_rounds: int,
) -> dict:
    """
    Estimate total communication bytes for a federated experiment.

    Parameters
    ----------
    model_weight_bytes : bytes for one complete set of model weights
    n_nodes            : number of participating edge nodes
    n_rounds           : number of federated rounds

    Returns
    -------
    dict with breakdown and total
    """
    # Initial broadcast: server → all nodes (round 0)
    initial_broadcast = model_weight_bytes * n_nodes

    # Per-round: each node uploads once, server broadcasts back
    per_round_upload = model_weight_bytes * n_nodes       # nodes → server
    per_round_download = model_weight_bytes * n_nodes     # server → nodes

    total_rounds_upload = per_round_upload * n_rounds
    total_rounds_download = per_round_download * n_rounds

    total = initial_broadcast + total_rounds_upload + total_rounds_download

    return {
        "initial_broadcast_bytes": initial_broadcast,
        "per_round_upload_bytes": per_round_upload,
        "per_round_download_bytes": per_round_download,
        "total_rounds_upload_bytes": total_rounds_upload,
        "total_rounds_download_bytes": total_rounds_download,
        "total_bytes": total,
        "model_weight_bytes": model_weight_bytes,
        "n_nodes": n_nodes,
        "n_rounds": n_rounds,
    }


def communication_savings_pct(fed_bytes: int, centralized_bytes: int) -> float:
    """
    Compute communication reduction percentage.

    Formula:
        savings (%) = 100 × (1 - federated_bytes / centralized_bytes)

    Returns 0.0 if centralized_bytes is zero (avoids division by zero).
    Returns negative if federated costs MORE than centralized.
    """
    if centralized_bytes == 0:
        return 0.0
    return 100.0 * (1.0 - fed_bytes / centralized_bytes)


def format_bytes(n: int) -> str:
    """Format byte count as a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def communication_report(
    fed_bytes: int,
    centralized_bytes: int,
    n_nodes: int,
    n_rounds: int,
    model_weight_bytes: int,
    n_training_records: int,
) -> str:
    """
    Generate a human-readable communication cost report.
    """
    savings = communication_savings_pct(fed_bytes, centralized_bytes)
    lines = [
        "=" * 60,
        "COMMUNICATION COST REPORT (Estimated Application-Level Bytes)",
        "=" * 60,
        f"  Nodes:                  {n_nodes}",
        f"  Federated rounds:       {n_rounds}",
        f"  Training records:       {n_training_records:,}",
        f"  Model size:             {format_bytes(model_weight_bytes)}",
        "",
        "  CENTRALIZED MODE",
        f"    Raw data upload:      {format_bytes(centralized_bytes)}",
        f"    (= {n_training_records} records × {BYTES_PER_RECORD_CSV} bytes/record)",
        "",
        "  FEDERATED MODE",
        f"    Total comm. bytes:    {format_bytes(fed_bytes)}",
        f"    (model updates only — raw data stays on nodes)",
        "",
        f"  COMMUNICATION SAVINGS:  {savings:.1f}%",
        "=" * 60,
        "NOTE: These are estimated payload bytes, not real network traffic.",
        "      TCP/IP headers, TLS overhead, and compression are excluded.",
    ]
    return "\n".join(lines)

