"""
config.py — FedEdge-IoT Global Configuration
=============================================
Central place for all project-wide constants and defaults.
Change values here to reconfigure experiments without touching source files.
"""
from __future__ import annotations
from typing import List

# ── Reproducibility ─────────────────────────────────────────────────────────
GLOBAL_SEED: int = 42

# ── Dataset ──────────────────────────────────────────────────────────────────
TOTAL_SAMPLES: int = 8000          # total synthetic sensor records
TEST_SPLIT: float = 0.20           # fraction held out for evaluation
FEATURES: List[str] = [
    "rms_vibration",       # root-mean-square vibration (g)
    "peak_vibration",      # peak amplitude (g)
    "vibration_variance",  # variance of vibration signal
    "dominant_frequency",  # dominant FFT frequency (Hz)
    "temperature",         # bearing temperature (°C)
    "rpm",                 # rotational speed (RPM)
]
TARGET: str = "fault"              # 0 = Normal, 1 = Fault

# ── Node defaults ────────────────────────────────────────────────────────────
DEFAULT_NUM_NODES: int = 5
DEFAULT_FAULT_RATES: List[float] = [0.10, 0.20, 0.35, 0.45, 0.25]  # per-node

# ── Model ────────────────────────────────────────────────────────────────────
HIDDEN_LAYER_SIZE: int = 32        # neurons in single hidden layer
LEARNING_RATE: float = 0.01
MAX_ITER_LOCAL: int = 50           # sklearn MLPClassifier max_iter per round
ACTIVATION: str = "relu"

# ── Federated Learning ───────────────────────────────────────────────────────
DEFAULT_ROUNDS: int = 10
DEFAULT_LOCAL_EPOCHS: int = 5
DEFAULT_FRACTION_CLIENTS: float = 1.0   # fraction of nodes per round

# ── Communication accounting ─────────────────────────────────────────────────
BYTES_PER_FLOAT: int = 8           # float64 = 8 bytes
# CSV serialization assumption: each feature value occupies ~8 bytes on avg
BYTES_PER_RECORD_CSV: int = 80     # 6 features * ~8 chars + separators + label

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "generated")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "federated_model.joblib")
CENTRAL_MODEL_PATH = os.path.join(RESULTS_DIR, "central_model.joblib")
EXPERIMENT_RESULTS_PATH = os.path.join(RESULTS_DIR, "experiment_results.joblib")

# Create dirs if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
