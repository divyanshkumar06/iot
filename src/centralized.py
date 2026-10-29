"""
src/centralized.py — Centralized Training Baseline
===================================================
Implements a traditional centralized learning approach where ALL raw
training data from every node is sent to the central server.

This serves as the comparison baseline for federated learning.

Key difference from federated mode:
  - Centralized: raw sensor records (features + labels) are transmitted
  - Federated:   only model parameter updates are transmitted

Author (proposed): Vishal
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from config import FEATURES, TARGET, GLOBAL_SEED, BYTES_PER_RECORD_CSV, MAX_ITER_LOCAL
from src.model import build_model, get_weights, weights_byte_size


def run_centralized_training(
    node_datasets: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = GLOBAL_SEED,
    max_iter: int = 200,
    progress_callback=None,
) -> dict:
    """
    Train a model on the combined (pooled) dataset from all nodes.

    In a real centralized system, each node would transmit its raw records
    over the network. Here we simulate that by combining DataFrames in memory
    and counting the equivalent bytes.

    Parameters
    ----------
    node_datasets     : dict of per-node DataFrames (training data)
    scaler            : fitted StandardScaler (same as federated mode)
    X_test, y_test    : held-out evaluation set
    seed              : random seed
    max_iter          : training iterations for the centralized model
    progress_callback : optional function(str) for status messages

    Returns
    -------
    dict with model, metrics, and communication bytes
    """
    log = progress_callback or print

    log("📤 Centralized: Simulating raw data upload from nodes …")

    # ── Simulate raw data upload ──────────────────────────────────────────────
    # Count bytes as if every training record were transmitted in CSV format
    all_dfs = list(node_datasets.values())
    total_records = sum(len(df) for df in all_dfs)
    raw_upload_bytes = total_records * BYTES_PER_RECORD_CSV

    log(f"   {total_records:,} records × {BYTES_PER_RECORD_CSV} bytes = "
        f"{raw_upload_bytes:,} bytes uploaded")

    # ── Pool all training data ────────────────────────────────────────────────
    log("🔗 Centralized: Pooling all node data on server …")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Shuffle to avoid ordering artefacts
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    X_train_all = scaler.transform(combined_df[FEATURES].values)
    y_train_all = combined_df[TARGET].values.astype(int)

    log(f"   Training set: {len(combined_df):,} samples "
        f"(fault rate: {y_train_all.mean():.1%})")

    # ── Train model ───────────────────────────────────────────────────────────
    log("🏋️  Centralized: Training model on pooled data …")
    model = build_model(seed=seed)
    model.max_iter = max_iter
    model.warm_start = False   # fresh training (not incremental)

    t0 = time.perf_counter()
    model.fit(X_train_all, y_train_all)
    train_time = time.perf_counter() - t0

    log(f"   Training complete in {train_time:.2f}s "
        f"(converged in {model.n_iter_} iterations)")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log("📊 Centralized: Evaluating on held-out test set …")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Model weight bytes (for reference — no distribution needed in centralized)
    model_bytes = weights_byte_size(get_weights(model))

    log(f"   Test accuracy: {model.score(X_test, y_test):.4f}")

    return {
        "model": model,
        "model_bytes": model_bytes,
        "raw_upload_bytes": raw_upload_bytes,
        "total_comm_bytes": raw_upload_bytes,   # centralized = raw data upload
        "total_records": total_records,
        "train_time_s": round(train_time, 4),
        "n_iterations": model.n_iter_,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "node_sample_counts": {k: len(v) for k, v in node_datasets.items()},
    }

