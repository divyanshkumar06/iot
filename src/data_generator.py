"""
src/data_generator.py — Synthetic IoT Sensor Data Generator
============================================================
** ALL DATA IN THIS MODULE IS SYNTHETIC SIMULATION DATA **
No values originate from real industrial sensors.

Author (proposed): Tanu
Purpose: Generate realistic-but-synthetic vibration and temperature sensor
         readings for industrial predictive maintenance simulation.

Design notes
------------
* Reproducible via random seeds.
* Non-IID: each node can have its own fault rate, RPM range, and noise level.
* Fault samples exhibit statistically distinguishable — but overlapping — patterns.
* Returns pandas DataFrames so downstream code is readable.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Optional, Dict

from config import FEATURES, TARGET, GLOBAL_SEED, BYTES_PER_RECORD_CSV


# ── per-node heterogeneity profiles ──────────────────────────────────────────
# Each profile entry tweaks the data generation to simulate different machines.
NODE_PROFILES = {
    # (rpm_mean, rpm_std, noise_scale, fault_rate_override or None)
    "light_machine":   dict(rpm_mean=1200, rpm_std=50,  noise=0.5),
    "medium_machine":  dict(rpm_mean=1800, rpm_std=100, noise=1.0),
    "heavy_machine":   dict(rpm_mean=2400, rpm_std=150, noise=1.5),
    "high_speed":      dict(rpm_mean=3000, rpm_std=200, noise=2.0),
    "variable_speed":  dict(rpm_mean=2000, rpm_std=400, noise=1.2),
    "old_machine":     dict(rpm_mean=1500, rpm_std=80,  noise=2.5),
    "precision_unit":  dict(rpm_mean=900,  rpm_std=20,  noise=0.3),
    "conveyor":        dict(rpm_mean=600,  rpm_std=30,  noise=0.8),
    "pump":            dict(rpm_mean=3600, rpm_std=100, noise=1.0),
    "compressor":      dict(rpm_mean=1450, rpm_std=60,  noise=1.3),
}

PROFILE_NAMES = list(NODE_PROFILES.keys())


def _generate_node_samples(
    n_samples: int,
    fault_rate: float,
    profile: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate synthetic sensor records for ONE edge node.

    Parameters
    ----------
    n_samples  : number of records to generate
    fault_rate : fraction of records that represent faulty conditions (0-1)
    profile    : dict with rpm_mean, rpm_std, noise keys
    rng        : numpy random Generator (seeded externally)

    Returns
    -------
    DataFrame with columns matching FEATURES + TARGET
    """
    n_fault = int(n_samples * fault_rate)
    n_normal = n_samples - n_fault

    rpm_mean  = profile["rpm_mean"]
    rpm_std   = profile["rpm_std"]
    noise     = profile["noise"]

    # ── Normal operating condition ──────────────────────────────────────────
    # Low-to-moderate vibration, stable temperature, consistent RPM
    norm_rms   = rng.normal(loc=1.15, scale=0.32 * noise, size=n_normal)
    norm_peak  = norm_rms + rng.exponential(scale=0.30 * noise, size=n_normal)
    norm_var   = rng.normal(loc=0.09, scale=0.03 * noise, size=n_normal)
    norm_freq  = rng.normal(loc=rpm_mean / 60, scale=rpm_std / 50 * noise, size=n_normal)
    norm_temp  = rng.normal(loc=66.5, scale=5.5 * noise, size=n_normal)
    norm_rpm   = rng.normal(loc=rpm_mean, scale=rpm_std, size=n_normal)
    norm_label = np.zeros(n_normal, dtype=int)

    # ── Faulty condition ────────────────────────────────────────────────────
    # Elevated vibration, higher temperature, erratic RPM, shifted frequency
    # Realistic overlap with normal range to make classification challenging
    fault_rms  = rng.normal(loc=1.85, scale=0.45 * noise, size=n_fault)
    fault_peak = fault_rms + rng.exponential(scale=0.55 * noise, size=n_fault)
    fault_var  = rng.normal(loc=0.22, scale=0.07 * noise, size=n_fault)
    fault_freq = rng.normal(loc=rpm_mean / 60 * 1.3, scale=rpm_std / 40 * noise, size=n_fault)
    fault_temp = rng.normal(loc=78.5, scale=7.0 * noise, size=n_fault)
    fault_rpm  = rng.normal(loc=rpm_mean * 0.90, scale=rpm_std * 1.6, size=n_fault)
    fault_label = np.ones(n_fault, dtype=int)

    # Sensor noise / borderline condition (transient spikes and ambiguous states)
    labels = np.concatenate([norm_label, fault_label])
    # ~4% borderline overlap / sensor noise
    noise_mask = rng.random(size=len(labels)) < (0.04 * min(noise, 1.5))
    labels[noise_mask] = 1 - labels[noise_mask]

    # Stack and shuffle
    data = pd.DataFrame({
        "rms_vibration":      np.concatenate([norm_rms,  fault_rms]),
        "peak_vibration":     np.concatenate([norm_peak, fault_peak]),
        "vibration_variance": np.concatenate([norm_var,  fault_var]),
        "dominant_frequency": np.concatenate([norm_freq, fault_freq]),
        "temperature":        np.concatenate([norm_temp, fault_temp]),
        "rpm":                np.concatenate([norm_rpm,  fault_rpm]),
        TARGET:               labels,
    })

    # Clip to physically plausible ranges
    data["rms_vibration"]      = data["rms_vibration"].clip(0.01, 20.0)
    data["peak_vibration"]     = data["peak_vibration"].clip(0.01, 30.0)
    data["vibration_variance"] = data["vibration_variance"].clip(0.0, 5.0)
    data["dominant_frequency"] = data["dominant_frequency"].clip(1.0, 500.0)
    data["temperature"]        = data["temperature"].clip(20.0, 120.0)
    data["rpm"]                = data["rpm"].clip(100.0, 5000.0)

    # Shuffle rows
    data = data.sample(frac=1, random_state=int(rng.integers(0, 99999))).reset_index(drop=True)
    return data


def generate_node_datasets(
    num_nodes: int = 5,
    total_samples: int = 8000,
    fault_rates: Optional[list] = None,
    seed: int = GLOBAL_SEED,
) -> Dict[str, pd.DataFrame]:
    """
    Generate separate synthetic datasets for each simulated edge node.

    Parameters
    ----------
    num_nodes     : number of edge nodes to simulate
    total_samples : total records across all nodes
    fault_rates   : per-node fault rate; if None, spread between 0.10–0.45
    seed          : master random seed

    Returns
    -------
    dict mapping "node_0", "node_1", … to DataFrames
    """
    rng = np.random.default_rng(seed)

    if fault_rates is None:
        # Spread fault rates across nodes for non-IID heterogeneity
        fault_rates = np.linspace(0.10, 0.45, num_nodes).tolist()
    elif len(fault_rates) < num_nodes:
        # Pad if too short
        extra = np.linspace(0.15, 0.40, num_nodes - len(fault_rates)).tolist()
        fault_rates = list(fault_rates) + extra

    # Distribute samples roughly evenly (with slight variance for realism)
    base = total_samples // num_nodes
    sizes = [base] * num_nodes
    remainder = total_samples - base * num_nodes
    for i in range(remainder):
        sizes[i] += 1

    node_datasets: Dict[str, pd.DataFrame] = {}
    for i in range(num_nodes):
        profile_name = PROFILE_NAMES[i % len(PROFILE_NAMES)]
        profile = NODE_PROFILES[profile_name]
        df = _generate_node_samples(
            n_samples=sizes[i],
            fault_rate=float(fault_rates[i]),
            profile=profile,
            rng=rng,
        )
        df["node_id"] = i          # tag each record with its node
        df["profile"] = profile_name
        node_datasets[f"node_{i}"] = df

    return node_datasets


def generate_holdout_test_set(
    n_samples: int = 1500,
    fault_rate: float = 0.30,
    seed: int = GLOBAL_SEED + 999,
) -> pd.DataFrame:
    """
    Generate a held-out evaluation set that is independent of training data.
    Uses a different seed so test records are never seen during training.

    The test set uses a mixed profile to represent the global distribution.
    """
    rng = np.random.default_rng(seed)
    # Use a 'medium_machine' profile as the neutral test profile
    profile = NODE_PROFILES["medium_machine"]
    df = _generate_node_samples(n_samples, fault_rate, profile, rng)
    df["node_id"] = -1          # -1 indicates shared evaluation set
    df["profile"] = "evaluation"
    return df


def estimate_raw_upload_bytes(node_datasets: dict[str, pd.DataFrame]) -> int:
    """
    Estimate bytes required to upload all raw training records to a central server.
    Assumes CSV-like serialization: BYTES_PER_RECORD_CSV bytes per row.
    Evaluation records are excluded (they are not uploaded in either mode).
    """
    total_rows = sum(len(df) for df in node_datasets.values())
    return total_rows * BYTES_PER_RECORD_CSV


def get_node_summary(node_datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a summary table of each node's dataset characteristics."""
    rows = []
    for name, df in node_datasets.items():
        rows.append({
            "Node": name,
            "Samples": len(df),
            "Fault %": f"{df[TARGET].mean() * 100:.1f}%",
            "Avg RMS Vibration": f"{df['rms_vibration'].mean():.3f} g",
            "Avg Temperature (°C)": f"{df['temperature'].mean():.1f}",
            "Avg RPM": f"{df['rpm'].mean():.0f}",
            "Profile": df["profile"].iloc[0],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick smoke-test when run directly
    print("Generating synthetic IoT sensor datasets …")
    nodes = generate_node_datasets(num_nodes=5, total_samples=8000, seed=42)
    test_df = generate_holdout_test_set()

    for name, df in nodes.items():
        fault_pct = df[TARGET].mean() * 100
        print(f"  {name}: {len(df)} samples, fault rate = {fault_pct:.1f}%")

    print(f"  test set: {len(test_df)} samples")
    print("\nNode summary:")
    print(get_node_summary(nodes).to_string(index=False))
    print("\n[OK] Data generator working correctly.")
