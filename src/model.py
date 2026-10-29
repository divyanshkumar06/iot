"""
src/model.py — Simple Neural Network with Explicit NumPy Weights
================================================================
Uses scikit-learn's MLPClassifier wrapped so that weights can be
extracted and averaged via FedAvg.

Why not arbitrary sklearn models?
----------------------------------
sklearn models like RandomForest or SVM do not expose a simple
weight-vector that can be numerically averaged. MLPClassifier does:
  - coefs_  : list of weight matrices per layer
  - intercepts_ : list of bias vectors per layer

These can be averaged element-by-element (FedAvg).

Author (proposed): Vishal
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import copy
from typing import Optional, List, Tuple

from config import (
    HIDDEN_LAYER_SIZE, LEARNING_RATE, MAX_ITER_LOCAL,
    ACTIVATION, GLOBAL_SEED, FEATURES
)


# ── Model architecture ────────────────────────────────────────────────────────
def build_model(seed: int = GLOBAL_SEED) -> MLPClassifier:
    """
    Construct a fresh MLPClassifier with a fixed architecture.
    ALL nodes and the server must use identical architectures so that
    weight matrices have the same shape and can be averaged.

    Architecture: 6 inputs → 32 hidden (ReLU) → 2 outputs (softmax)
    """
    return MLPClassifier(
        hidden_layer_sizes=(HIDDEN_LAYER_SIZE,),
        activation=ACTIVATION,
        solver="adam",
        learning_rate_init=LEARNING_RATE,
        max_iter=MAX_ITER_LOCAL,
        random_state=seed,
        warm_start=True,    # allows incremental training across rounds
        n_iter_no_change=10,
        early_stopping=False,
    )


# ── Weight extraction and injection ──────────────────────────────────────────
def get_weights(model: MLPClassifier) -> List[np.ndarray]:
    """
    Extract all trainable parameters (weights + biases) from a fitted model.
    Returns a flat list: [W1, b1, W2, b2, …]
    """
    params = []
    for W in model.coefs_:
        params.append(W.copy())
    for b in model.intercepts_:
        params.append(b.copy())
    return params


def set_weights(model: MLPClassifier, weights: List[np.ndarray]) -> None:
    """
    Write a flat weight list back into a model's coefs_ and intercepts_.
    If model has not been fitted, initialize dummy attributes first.
    """
    n_layers = len(weights) // 2
    model.coefs_ = [weights[i].copy() for i in range(n_layers)]
    model.intercepts_ = [weights[n_layers + i].copy() for i in range(n_layers)]
    if not hasattr(model, "classes_"):
        model.classes_ = np.array([0, 1])
        model.n_outputs_ = 1
        model.n_layers_ = n_layers + 1
        model.out_activation_ = "logistic" if model.n_outputs_ == 1 else "softmax"


def weights_byte_size(weights: List[np.ndarray]) -> int:
    """
    Compute the number of bytes required to transmit model weights.
    Assumes float64 (8 bytes per element).
    """
    return sum(w.nbytes for w in weights)


def clone_model_with_weights(
    source_model: MLPClassifier,
    weights: List[np.ndarray],
) -> MLPClassifier:
    """Return a deep copy of source_model with new weights applied."""
    new_model = copy.deepcopy(source_model)
    set_weights(new_model, weights)
    return new_model


# ── Warm initialisation ───────────────────────────────────────────────────────
def initialise_global_model(
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    seed: int = GLOBAL_SEED,
) -> tuple[MLPClassifier, StandardScaler]:
    """
    Perform a minimal fit on a tiny sample to initialise the model's
    weight shapes. This must be done before FedAvg rounds start so that
    coefs_/intercepts_ attributes exist.

    Parameters
    ----------
    X_sample : small array of features (just needs both classes present)
    y_sample : labels for X_sample

    Returns
    -------
    (initialized model, fitted scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    model = build_model(seed=seed)
    # One pass to set weight shapes
    model.max_iter = 1
    model.fit(X_scaled, y_sample)
    model.max_iter = MAX_ITER_LOCAL   # restore for later training
    return model, scaler
