"""
tests/test_fedavg.py — Unit Tests for FedAvg Algorithm
=======================================================
Verifies that the sample-weighted averaging is mathematically correct.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from src.fedavg import federated_average, FedAvgCoordinator
from src.data_generator import generate_node_datasets, generate_holdout_test_set
from src.model import initialise_global_model, get_weights, set_weights, weights_byte_size
from src.edge_node import EdgeNode
from sklearn.preprocessing import StandardScaler
from config import FEATURES, TARGET, GLOBAL_SEED


def _make_simple_weights(n_params: int = 10, n_sets: int = 3):
    """Helper: create simple 1D weight lists for arithmetic verification."""
    return [[np.array([float(i + j) for j in range(n_params)])]
            for i in range(n_sets)]


class TestFedAvg:

    def test_equal_weights_equal_samples(self):
        """With equal samples, result is the arithmetic mean."""
        w1 = [np.array([1.0, 2.0, 3.0])]
        w2 = [np.array([3.0, 4.0, 5.0])]
        counts = [100, 100]
        result = federated_average([w1, w2], counts)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_weighted_average_correctness(self):
        """Sample-weighted average is computed correctly."""
        w1 = [np.array([0.0, 0.0])]
        w2 = [np.array([4.0, 4.0])]
        # n1=100 contributes 100/(100+300)=0.25, n2=300 contributes 0.75
        counts = [100, 300]
        result = federated_average([w1, w2], counts)
        expected = np.array([3.0, 3.0])
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_single_node(self):
        """With one node, output equals input."""
        w = [np.array([1.5, 2.5, 3.5])]
        result = federated_average([w], [50])
        np.testing.assert_allclose(result[0], w[0], rtol=1e-10)

    def test_many_layers(self):
        """Works correctly with multiple weight tensors (layers)."""
        w1 = [np.array([1.0, 1.0]), np.array([0.5])]
        w2 = [np.array([3.0, 3.0]), np.array([1.5])]
        counts = [1, 1]
        result = federated_average([w1, w2], counts)
        np.testing.assert_allclose(result[0], np.array([2.0, 2.0]), rtol=1e-6)
        np.testing.assert_allclose(result[1], np.array([1.0]), rtol=1e-6)

    def test_2d_weight_matrices(self):
        """Handles 2D weight matrices correctly."""
        W1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        W2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = federated_average([[W1], [W2]], [1, 1])
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_preserves_shape(self):
        """Output shapes match input shapes."""
        shapes = [(32, 6), (32,), (2, 32), (2,)]
        rng = np.random.default_rng(0)
        weights_a = [rng.random(s) for s in shapes]
        weights_b = [rng.random(s) for s in shapes]
        result = federated_average([weights_a, weights_b], [100, 200])
        for res, a, b in zip(result, weights_a, weights_b):
            assert res.shape == a.shape

    def test_empty_raises(self):
        """Empty weight list raises ValueError."""
        with pytest.raises(ValueError):
            federated_average([], [])

    def test_zero_samples_raises(self):
        """Zero total samples raises ValueError."""
        w = [np.array([1.0, 2.0])]
        with pytest.raises(ValueError):
            federated_average([w], [0])

    def test_mismatched_lengths_raises(self):
        """Mismatched weight_list / sample_counts raises ValueError."""
        w1 = [np.array([1.0])]
        w2 = [np.array([2.0])]
        with pytest.raises(ValueError):
            federated_average([w1, w2], [100])  # 2 weight sets but 1 count

    def test_weighted_sum_equals_one(self):
        """The fractions n_k/total always sum to 1.0."""
        counts = [100, 200, 300, 400]
        total = sum(counts)
        fractions = [c / total for c in counts]
        assert abs(sum(fractions) - 1.0) < 1e-10

    def test_full_fedavg_round(self):
        """End-to-end: FedAvgCoordinator runs one round without errors."""
        # Generate small dataset
        datasets = generate_node_datasets(num_nodes=2, total_samples=400, seed=42)
        all_df = pd.concat(list(datasets.values()), ignore_index=True)

        scaler = StandardScaler()
        scaler.fit(all_df[FEATURES].values)

        # Initialise model
        init_sample = all_df.head(50)
        # Ensure both classes
        init_sample = pd.concat([
            all_df[all_df[TARGET] == 0].head(25),
            all_df[all_df[TARGET] == 1].head(25),
        ])
        X_init = scaler.transform(init_sample[FEATURES].values)
        y_init = init_sample[TARGET].values.astype(int)

        global_model, _ = initialise_global_model(X_init, y_init, seed=42)

        nodes = [
            EdgeNode(f"node_{i}", datasets[f"node_{i}"], scaler, seed=42 + i)
            for i in range(2)
        ]
        coordinator = FedAvgCoordinator(global_model, nodes, scaler)

        # Generate test set
        test_df = generate_holdout_test_set(n_samples=100, seed=9999)
        X_test = scaler.transform(test_df[FEATURES].values)
        y_test = test_df[TARGET].values.astype(int)

        metrics = coordinator.run_round(
            local_epochs=2,
            X_test=X_test,
            y_test=y_test,
            round_num=1,
        )

        assert "global_accuracy" in metrics
        assert metrics["global_accuracy"] is not None
        assert 0.0 <= metrics["global_accuracy"] <= 1.0
        assert metrics["n_participating"] == 2
        assert metrics["upload_bytes"] > 0
        assert metrics["download_bytes"] > 0

    def test_global_model_accuracy_improves(self):
        """Global accuracy after training is better than chance (> 0.5 on balanced data)."""
        datasets = generate_node_datasets(num_nodes=3, total_samples=1200, seed=42)
        all_df = pd.concat(list(datasets.values()), ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(all_df[FEATURES].values)

        init_sample = pd.concat([
            all_df[all_df[TARGET] == 0].head(50),
            all_df[all_df[TARGET] == 1].head(50),
        ])
        X_init = scaler.transform(init_sample[FEATURES].values)
        y_init = init_sample[TARGET].values.astype(int)
        global_model, _ = initialise_global_model(X_init, y_init, seed=42)

        nodes = [
            EdgeNode(f"node_{i}", datasets[f"node_{i}"], scaler, seed=42 + i)
            for i in range(3)
        ]
        coordinator = FedAvgCoordinator(global_model, nodes, scaler)

        test_df = generate_holdout_test_set(n_samples=200, seed=9999)
        X_test = scaler.transform(test_df[FEATURES].values)
        y_test = test_df[TARGET].values.astype(int)

        coordinator.run_all_rounds(
            n_rounds=3, local_epochs=5, X_test=X_test, y_test=y_test
        )
        final_acc = global_model.score(X_test, y_test)
        assert final_acc > 0.55, f"Accuracy {final_acc:.4f} too low — check FedAvg"

