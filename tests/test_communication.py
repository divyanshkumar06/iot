"""
tests/test_communication.py — Unit Tests for Communication Module
=================================================================
Verifies byte calculations and savings formulas.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.communication import (
    bytes_for_raw_records,
    bytes_for_weights,
    bytes_for_weight_shapes,
    federated_total_bytes,
    communication_savings_pct,
    format_bytes,
    communication_report,
)
from config import BYTES_PER_RECORD_CSV, BYTES_PER_FLOAT


class TestCommunication:

    def test_raw_records_basic(self):
        assert bytes_for_raw_records(100) == 100 * BYTES_PER_RECORD_CSV

    def test_raw_records_zero(self):
        assert bytes_for_raw_records(0) == 0

    def test_weights_basic(self):
        weights = [np.zeros((32, 6)), np.zeros(32), np.zeros((2, 32)), np.zeros(2)]
        # 32*6 + 32 + 2*32 + 2 = 192 + 32 + 64 + 2 = 290 params * 8 bytes
        expected = sum(w.nbytes for w in weights)
        assert bytes_for_weights(weights) == expected

    def test_weights_shapes_consistency(self):
        """bytes_for_weight_shapes matches bytes_for_weights for float64 arrays."""
        shapes = [(32, 6), (32,), (2, 32), (2,)]
        weights = [np.zeros(s, dtype=np.float64) for s in shapes]
        assert bytes_for_weight_shapes(shapes) == bytes_for_weights(weights)

    def test_savings_zero_fed(self):
        """Zero federated bytes → 100% savings."""
        assert communication_savings_pct(0, 1000) == pytest.approx(100.0)

    def test_savings_equal(self):
        """Equal bytes → 0% savings."""
        assert communication_savings_pct(500, 500) == pytest.approx(0.0)

    def test_savings_formula(self):
        """Savings = 100 * (1 - fed/central)."""
        fed = 200
        central = 1000
        expected = 100.0 * (1 - 200 / 1000)  # 80%
        assert communication_savings_pct(fed, central) == pytest.approx(expected)

    def test_savings_zero_central(self):
        """Zero centralized bytes → 0.0 (no division by zero)."""
        result = communication_savings_pct(100, 0)
        assert result == 0.0

    def test_savings_fed_more_expensive(self):
        """Federated costing more than centralized → negative savings."""
        result = communication_savings_pct(2000, 1000)
        assert result < 0

    def test_format_bytes_units(self):
        assert "B"  in format_bytes(500)
        assert "KB" in format_bytes(2048)
        assert "MB" in format_bytes(2 * 1024 ** 2)
        assert "GB" in format_bytes(3 * 1024 ** 3)

    def test_federated_total_bytes_structure(self):
        """federated_total_bytes returns expected keys."""
        result = federated_total_bytes(
            model_weight_bytes=5000,
            n_nodes=5,
            n_rounds=10,
        )
        required_keys = [
            "initial_broadcast_bytes",
            "total_rounds_upload_bytes",
            "total_rounds_download_bytes",
            "total_bytes",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_federated_total_bytes_arithmetic(self):
        """Manual calculation matches function output."""
        model_bytes = 1000
        n_nodes = 3
        n_rounds = 5

        initial = model_bytes * n_nodes                    # 3000
        per_round_up = model_bytes * n_nodes * n_rounds   # 15000
        per_round_dl = model_bytes * n_nodes * n_rounds   # 15000
        expected_total = initial + per_round_up + per_round_dl  # 33000

        result = federated_total_bytes(model_bytes, n_nodes, n_rounds)
        assert result["total_bytes"] == expected_total

    def test_communication_report_returns_string(self):
        report = communication_report(
            fed_bytes=50000,
            centralized_bytes=500000,
            n_nodes=5,
            n_rounds=10,
            model_weight_bytes=5000,
            n_training_records=1000,
        )
        assert isinstance(report, str)
        assert "COMMUNICATION COST REPORT" in report
        assert "SAVINGS" in report

