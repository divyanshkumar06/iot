"""
src/experiment.py — Full Experiment Orchestrator
================================================
Ties together data generation, edge-node simulation, federated training,
centralized training, and evaluation into a single reproducible run.

This module can be imported by the dashboard OR run standalone.

Author (proposed): Divyansh
"""

import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from config import (
    GLOBAL_SEED, TOTAL_SAMPLES, FEATURES, TARGET,
    DEFAULT_NUM_NODES, DEFAULT_ROUNDS, DEFAULT_LOCAL_EPOCHS,
    DEFAULT_FRACTION_CLIENTS, EXPERIMENT_RESULTS_PATH,
    MODEL_PATH, CENTRAL_MODEL_PATH, DATA_DIR,
)
from src.data_generator import generate_node_datasets, generate_holdout_test_set
from src.model import initialise_global_model, get_weights, weights_byte_size
from src.edge_node import EdgeNode
from src.fedavg import FedAvgCoordinator
from src.centralized import run_centralized_training
from src.evaluation import evaluate_model, compare_models, print_comparison_table
from src.communication import (
    communication_savings_pct, communication_report, format_bytes
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_full_experiment(
    num_nodes: int = DEFAULT_NUM_NODES,
    n_rounds: int = DEFAULT_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    total_samples: int = TOTAL_SAMPLES,
    fault_rates: list = None,
    seed: int = GLOBAL_SEED,
    fraction_clients: float = DEFAULT_FRACTION_CLIENTS,
    progress_callback=None,
    save_results: bool = True,
) -> dict:
    """
    Run the complete FedEdge-IoT experiment.

    Steps
    -----
    1. Generate synthetic sensor data (non-IID across nodes)
    2. Create a shared held-out test set
    3. Fit a shared StandardScaler on aggregated feature stats
    4. Initialise a global model
    5. Create EdgeNode instances (simulated IoT devices)
    6. Run centralized training baseline
    7. Run federated training (FedAvg)
    8. Evaluate both models on the same held-out test set
    9. Compare communication costs
    10. Save results

    Parameters
    ----------
    num_nodes         : number of simulated edge nodes
    n_rounds          : federated rounds
    local_epochs      : epochs per node per round
    total_samples     : total training records across all nodes
    fault_rates       : per-node fault rates (list or None for defaults)
    seed              : master random seed
    fraction_clients  : fraction of nodes participating each round
    progress_callback : function(str) for live progress updates
    save_results      : whether to persist results to disk

    Returns
    -------
    Comprehensive results dict
    """
    log = progress_callback or (lambda msg: logger.info(msg))
    overall_t0 = time.perf_counter()

    log("=" * 60)
    log("FedEdge-IoT: Full Experiment Starting")
    log(f"  Nodes={num_nodes}, Rounds={n_rounds}, "
        f"LocalEpochs={local_epochs}, Seed={seed}")
    log("=" * 60)

    # ── Stage 1: Data Generation ──────────────────────────────────────────────
    log("\n📊 Stage 1: Generating synthetic sensor datasets …")
    node_datasets = generate_node_datasets(
        num_nodes=num_nodes,
        total_samples=total_samples,
        fault_rates=fault_rates,
        seed=seed,
    )
    test_df = generate_holdout_test_set(
        n_samples=max(500, total_samples // 5),
        fault_rate=0.30,
        seed=seed + 999,
    )
    log(f"   Generated {sum(len(v) for v in node_datasets.values()):,} "
        f"training records across {num_nodes} nodes")
    log(f"   Held-out test set: {len(test_df):,} records")

    # Save datasets to disk
    for node_id, df in node_datasets.items():
        df.to_csv(os.path.join(DATA_DIR, f"{node_id}_train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_set.csv"), index=False)
    log(f"   Datasets saved to {DATA_DIR}")

    # ── Stage 2: Shared Scaler ────────────────────────────────────────────────
    log("\n⚖️  Stage 2: Fitting shared StandardScaler …")
    # Fit scaler on ALL training data so all nodes use the same normalisation
    all_train = pd.concat(list(node_datasets.values()), ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(all_train[FEATURES].values)

    X_test = scaler.transform(test_df[FEATURES].values)
    y_test = test_df[TARGET].values.astype(int)
    log("   Scaler fitted.")

    # ── Stage 3: Initialise Global Model ─────────────────────────────────────
    log("\n🧠 Stage 3: Initialising global model …")
    # Use a small slice of training data that has both classes
    init_df = all_train.sample(min(200, len(all_train)), random_state=seed)
    # Ensure both classes present
    if init_df[TARGET].nunique() < 2:
        extra = all_train[all_train[TARGET] == 1].head(10)
        init_df = pd.concat([init_df, extra]).drop_duplicates()
    X_init = scaler.transform(init_df[FEATURES].values)
    y_init = init_df[TARGET].values.astype(int)

    global_model, _ = initialise_global_model(X_init, y_init, seed=seed)
    model_weights = get_weights(global_model)
    model_weight_bytes = weights_byte_size(model_weights)
    log(f"   Model initialised. Weight size = {format_bytes(model_weight_bytes)}")

    # ── Stage 4: Create Edge Nodes ────────────────────────────────────────────
    log(f"\n🖥️  Stage 4: Creating {num_nodes} simulated edge nodes …")
    nodes = []
    for i in range(num_nodes):
        node_id = f"node_{i}"
        node = EdgeNode(
            node_id=node_id,
            local_data=node_datasets[node_id],
            scaler=scaler,
            seed=seed + i,
        )
        nodes.append(node)
        log(f"   {node}")

    # ── Stage 5: Centralized Training ────────────────────────────────────────
    log("\n🏢 Stage 5: Centralized Training Baseline …")
    central_result = run_centralized_training(
        node_datasets=node_datasets,
        scaler=scaler,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
        max_iter=200,
        progress_callback=log,
    )
    central_metrics = evaluate_model(
        central_result["model"], X_test, y_test, model_name="Centralized"
    )
    central_metrics["train_time_s"] = central_result["train_time_s"]
    central_metrics["total_comm_bytes"] = central_result["total_comm_bytes"]

    if save_results:
        joblib.dump(central_result["model"], CENTRAL_MODEL_PATH)
        log(f"   Centralized model saved to {CENTRAL_MODEL_PATH}")

    # ── Stage 6: Federated Training (FedAvg) ─────────────────────────────────
    log(f"\n🔗 Stage 6: Federated Training ({n_rounds} rounds) …")
    fed_t0 = time.perf_counter()
    coordinator = FedAvgCoordinator(
        global_model=global_model,
        nodes=nodes,
        scaler=scaler,
    )
    round_history = coordinator.run_all_rounds(
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        fraction_clients=fraction_clients,
        X_test=X_test,
        y_test=y_test,
        progress_callback=log,
    )
    fed_train_time = time.perf_counter() - fed_t0

    fed_comm_bytes = coordinator.get_total_communication_bytes()
    fed_metrics = evaluate_model(
        global_model, X_test, y_test, model_name="Federated (Global)"
    )
    fed_metrics["train_time_s"] = round(fed_train_time, 4)
    fed_metrics["total_comm_bytes"] = fed_comm_bytes

    if save_results:
        joblib.dump(global_model, MODEL_PATH)
        log(f"   Federated model saved to {MODEL_PATH}")

    # ── Stage 7: Communication Comparison ────────────────────────────────────
    log("\n📡 Stage 7: Communication Cost Comparison …")
    savings = communication_savings_pct(fed_comm_bytes, central_result["total_comm_bytes"])
    report = communication_report(
        fed_bytes=fed_comm_bytes,
        centralized_bytes=central_result["total_comm_bytes"],
        n_nodes=num_nodes,
        n_rounds=n_rounds,
        model_weight_bytes=model_weight_bytes,
        n_training_records=sum(len(v) for v in node_datasets.values()),
    )
    log(report)

    # ── Stage 8: Print Comparison ─────────────────────────────────────────────
    print_comparison_table(fed_metrics, central_metrics)

    overall_elapsed = time.perf_counter() - overall_t0
    log(f"\n✅ Experiment complete in {overall_elapsed:.1f}s")

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "config": {
            "num_nodes": num_nodes,
            "n_rounds": n_rounds,
            "local_epochs": local_epochs,
            "total_samples": total_samples,
            "seed": seed,
            "fraction_clients": fraction_clients,
        },
        "node_summaries": [n.get_status() for n in nodes],
        "node_datasets_info": {k: len(v) for k, v in node_datasets.items()},
        "test_set_size": len(test_df),
        "federated_metrics": fed_metrics,
        "central_metrics": central_metrics,
        "round_history": round_history,
        "communication": {
            "fed_bytes": fed_comm_bytes,
            "centralized_bytes": central_result["total_comm_bytes"],
            "savings_pct": round(savings, 2),
            "model_weight_bytes": model_weight_bytes,
            "raw_upload_bytes": central_result["raw_upload_bytes"],
        },
        "total_experiment_time_s": round(overall_elapsed, 2),
        "scaler": scaler,
        "global_model": global_model,
        "central_model": central_result["model"],
        "nodes": nodes,
    }

    if save_results:
        # Save everything except large objects
        save_dict = {k: v for k, v in results.items()
                     if k not in ("scaler", "global_model", "central_model", "nodes")}
        joblib.dump(save_dict, EXPERIMENT_RESULTS_PATH)
        log(f"   Results saved to {EXPERIMENT_RESULTS_PATH}")

    return results


if __name__ == "__main__":
    results = run_full_experiment(
        num_nodes=5,
        n_rounds=5,
        local_epochs=5,
        total_samples=8000,
        seed=42,
        save_results=True,
    )
    print(f"\nFederated accuracy:   {results['federated_metrics']['accuracy']:.4f}")
    print(f"Centralized accuracy: {results['central_metrics']['accuracy']:.4f}")
    print(f"Communication savings: {results['communication']['savings_pct']:.1f}%")

