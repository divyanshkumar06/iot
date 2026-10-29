# data/README.md — Dataset Documentation

## Dataset Overview

All data files in this directory are **SYNTHETIC SIMULATION DATA**.
No values originate from real industrial sensors or real IoT hardware.

## File Naming Convention

| File | Description |
|------|-------------|
| `node_0_train.csv` | Training data for simulated edge node 0 |
| `node_1_train.csv` | Training data for simulated edge node 1 |
| `node_N_train.csv` | Training data for simulated edge node N |
| `test_set.csv` | Held-out evaluation dataset (never used for training) |

## Features

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `rms_vibration` | float | g | Root-mean-square vibration amplitude |
| `peak_vibration` | float | g | Peak vibration amplitude |
| `vibration_variance` | float | — | Statistical variance of vibration signal |
| `dominant_frequency` | float | Hz | Dominant FFT frequency component |
| `temperature` | float | °C | Bearing or component temperature |
| `rpm` | float | RPM | Rotational speed |
| `fault` | int | 0/1 | Target label: 0=Normal, 1=Fault |
| `node_id` | int | — | Source node index (-1 for test set) |
| `profile` | str | — | Machine profile name used for generation |

## Train/Test Split Strategy

- **Training data** and **test data** are generated with DIFFERENT random seeds.
  - Training: seed = GLOBAL_SEED (default 42)
  - Test: seed = GLOBAL_SEED + 999 (default 1041)
- This ensures NO test records can appear in training data.
- The test set uses a fixed 30% fault rate and "medium_machine" profile.

## Non-IID Distribution

Each node's data is generated with a different:
- Fault rate (configurable, default: linearly spaced 10%–45%)
- Machine profile (different RPM, noise level)

This simulates realistic heterogeneity in a fleet of industrial machines.

## Regenerating Data

```powershell
cd FedEdge-IoT
python -c "from src.data_generator import generate_node_datasets, generate_holdout_test_set; print('Data generator OK')"
```

Data is also auto-generated when you run the Streamlit dashboard or `src/experiment.py`.

## Disclaimer

These files are synthetic and must not be used for real industrial safety decisions.

