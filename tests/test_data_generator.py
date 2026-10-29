"""
tests/test_data_generator.py — Unit Tests for Data Generator
=============================================================
Run with:   pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from src.data_generator import (
    generate_node_datasets,
    generate_holdout_test_set,
    get_node_summary,
    estimate_raw_upload_bytes,
)
from config import FEATURES, TARGET, BYTES_PER_RECORD_CSV


class TestDataGenerator:

    def test_node_count(self):
        """Correct number of nodes are generated."""
        datasets = generate_node_datasets(num_nodes=3, total_samples=600, seed=1)
        assert len(datasets) == 3

    def test_node_keys(self):
        """Node keys follow expected naming convention."""
        datasets = generate_node_datasets(num_nodes=4, total_samples=800, seed=2)
        assert set(datasets.keys()) == {"node_0", "node_1", "node_2", "node_3"}

    def test_total_sample_count(self):
        """Total samples across all nodes equals requested total."""
        n = 1000
        datasets = generate_node_datasets(num_nodes=5, total_samples=n, seed=3)
        total = sum(len(df) for df in datasets.values())
        assert total == n

    def test_feature_columns_present(self):
        """All required feature columns are present in each node dataset."""
        datasets = generate_node_datasets(num_nodes=2, total_samples=200, seed=4)
        for df in datasets.values():
            for feat in FEATURES:
                assert feat in df.columns, f"Missing feature: {feat}"
            assert TARGET in df.columns

    def test_target_binary(self):
        """Target column contains only 0 and 1."""
        datasets = generate_node_datasets(num_nodes=3, total_samples=300, seed=5)
        for df in datasets.values():
            unique_labels = set(df[TARGET].unique())
            assert unique_labels.issubset({0, 1}), f"Unexpected labels: {unique_labels}"

    def test_no_missing_values(self):
        """No NaN values in any feature or label column."""
        datasets = generate_node_datasets(num_nodes=3, total_samples=300, seed=6)
        for name, df in datasets.items():
            assert df[FEATURES + [TARGET]].isnull().sum().sum() == 0, \
                f"{name} has missing values"

    def test_fault_rates_nonzero(self):
        """Fault rates are non-zero (at least some fault examples present)."""
        datasets = generate_node_datasets(num_nodes=5, total_samples=1000, seed=7)
        for name, df in datasets.items():
            fault_pct = df[TARGET].mean()
            assert fault_pct > 0, f"{name} has 0% fault rate"
            assert fault_pct < 1, f"{name} has 100% fault rate"

    def test_feature_value_ranges(self):
        """Feature values are within physically plausible clipped ranges."""
        datasets = generate_node_datasets(num_nodes=3, total_samples=300, seed=8)
        df = list(datasets.values())[0]
        assert df["rms_vibration"].min() >= 0.01
        assert df["rms_vibration"].max() <= 20.0
        assert df["temperature"].min() >= 20.0
        assert df["temperature"].max() <= 120.0
        assert df["rpm"].min() >= 100.0
        assert df["rpm"].max() <= 5000.0

    def test_non_iid_fault_rates(self):
        """Different nodes should have different fault rates (non-IID)."""
        datasets = generate_node_datasets(num_nodes=5, total_samples=2000,
                                          fault_rates=[0.1, 0.2, 0.3, 0.4, 0.5], seed=9)
        rates = [df[TARGET].mean() for df in datasets.values()]
        # Rates should not all be identical
        assert max(rates) - min(rates) > 0.05, "Fault rates appear identical (non-IID not working)"

    def test_holdout_test_set(self):
        """Holdout test set is independent (different seed, separate function)."""
        test_df = generate_holdout_test_set(n_samples=200, fault_rate=0.30, seed=999)
        assert len(test_df) == 200
        for feat in FEATURES:
            assert feat in test_df.columns

    def test_reproducibility(self):
        """Same seed produces same dataset."""
        d1 = generate_node_datasets(num_nodes=3, total_samples=300, seed=42)
        d2 = generate_node_datasets(num_nodes=3, total_samples=300, seed=42)
        for node in d1:
            pd.testing.assert_frame_equal(
                d1[node][FEATURES].reset_index(drop=True),
                d2[node][FEATURES].reset_index(drop=True),
            )

    def test_different_seeds_differ(self):
        """Different seeds produce different datasets."""
        d1 = generate_node_datasets(num_nodes=2, total_samples=200, seed=1)
        d2 = generate_node_datasets(num_nodes=2, total_samples=200, seed=2)
        # At least one node's data should differ
        rms1 = d1["node_0"]["rms_vibration"].values
        rms2 = d2["node_0"]["rms_vibration"].values
        assert not np.allclose(rms1, rms2), "Different seeds produced identical data"

    def test_node_summary(self):
        """get_node_summary returns a DataFrame with correct columns."""
        datasets = generate_node_datasets(num_nodes=3, total_samples=300, seed=10)
        summary = get_node_summary(datasets)
        assert isinstance(summary, pd.DataFrame)
        assert "Node" in summary.columns
        assert "Samples" in summary.columns

    def test_raw_upload_bytes(self):
        """estimate_raw_upload_bytes calculates correctly."""
        datasets = generate_node_datasets(num_nodes=3, total_samples=300, seed=11)
        total_rows = sum(len(df) for df in datasets.values())
        expected = total_rows * BYTES_PER_RECORD_CSV
        actual = estimate_raw_upload_bytes(datasets)
        assert actual == expected

