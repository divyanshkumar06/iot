# FedEdge-IoT — STAR Method Project Explanation

## How to Use This Document

The **STAR method** (Situation → Task → Action → Result) is a structured way to explain a project.
Use this document to prepare your viva answers and presentation narrative.

---

## S — Situation

**The problem we are solving:**

Modern industrial facilities deploy dozens or hundreds of IoT-connected machines, each generating
continuous streams of sensor data — vibration levels, temperatures, rotational speeds, and more.
Traditional machine learning for predictive maintenance requires sending all of this raw sensor data
to a centralised server or cloud platform.

This centralised approach creates several challenges:

1. **High bandwidth consumption**: Thousands of sensor records per minute per machine across a large
   facility can saturate network connections.

2. **Privacy and operational security risks**: Raw operational data — which may reveal production
   rates, machine utilisation, or proprietary processes — must leave the facility network.

3. **Single point of failure**: If the central server is unavailable, no inference can be performed
   locally.

4. **Regulatory compliance**: Sectors like healthcare, defence, and utilities face strict rules
   about where operational data may be stored and processed.

*Example to say:* "Imagine a factory with 50 machines, each recording 10 sensor readings per second.
That is 500 sensor values per second, 43 million per day, all needing transmission and centralised
storage."

---

## T — Task

**What this project set out to accomplish:**

Our task was to design and demonstrate a **Federated Learning system for IoT predictive maintenance**
that achieves the following objectives:

1. Simulate multiple IoT edge nodes, each holding its own private sensor dataset.
2. Train a machine condition classifier (Normal vs Fault) collaboratively across all nodes
   **without transmitting raw sensor records** to a central server.
3. Implement **Federated Averaging (FedAvg)** from scratch to aggregate model updates.
4. Compare this federated approach against traditional centralised training using the same data,
   same model architecture, and same held-out evaluation set.
5. Quantify the trade-off between model accuracy and communication bandwidth savings.
6. Present all findings through an interactive dashboard suitable for faculty demonstration.

---

## A — Action

**What we actually built and implemented:**

### Data Simulation (Tanu)
- Implemented `data_generator.py` to generate synthetic vibration, temperature, and RPM sensor
  data for each simulated node.
- Introduced **non-IID heterogeneity**: each node has a different fault rate, noise level, and
  RPM profile, simulating a real fleet of different machines.
- Created a separate held-out test set with a different random seed to prevent data leakage.
- Clearly labelled all data as SYNTHETIC SIMULATION DATA.

### Local Training and Edge Nodes (Tanu, Vishal)
- Implemented the `EdgeNode` class: each node holds its local private data, accepts global model
  weights, trains locally, and returns only model updates.
- Used scikit-learn's `MLPClassifier` (6-input → 32-hidden ReLU → 2-output) because its
  `coefs_` and `intercepts_` can be numerically averaged.

### FedAvg and Centralized Baseline (Vishal)
- Implemented `federated_average()` computing the sample-weighted mean:
  **w_global = Σ(n_k × w_k) / Σ(n_k)**
- Implemented `FedAvgCoordinator` orchestrating broadcast → local training → aggregation.
- Implemented `run_centralized_training()` as a fair baseline using the same model,
  same scaler, and same test set.

### Communication Accounting (Vishal)
- Implemented `communication.py` to count estimated application-level bytes for both modes.
- Federated bytes: model parameter size × nodes × rounds (both upload and download).
- Centralized bytes: total training records × 80 bytes/record (CSV serialization assumption).

### Dashboard and Documentation (Hanmant, Divyansh)
- Built a seven-tab Streamlit dashboard with dark navy/teal theme.
- Implemented live sensor prediction, confusion matrices, round-by-round accuracy plots,
  and a one-click full demonstration workflow.
- Wrote README, architecture documentation, and this STAR guide.

### Integration and Experiment (Divyansh)
- Tied all modules together in `experiment.py`.
- Validated correctness with pytest unit tests covering data generation, FedAvg arithmetic,
  and communication byte calculations.

---

## R — Result

**What the experiment produced:**

*(Replace placeholders below with your actual measured values after running the experiment.)*

| Metric | Federated | Centralized | Notes |
|--------|-----------|------------|-------|
| Accuracy | **[MEASURED]** | **[MEASURED]** | Measured on 1,500-record held-out test set |
| F1-Score (macro) | **[MEASURED]** | **[MEASURED]** | — |
| Training time | **[MEASURED]** s | **[MEASURED]** s | Total wall-clock time |
| Communication bytes | **[MEASURED]** KB | **[MEASURED]** KB | Estimated payload bytes |
| Communication savings | **[MEASURED]** % | baseline | 100 × (1 − fed/central) |
| Inference latency | **[MEASURED]** ms | **[MEASURED]** ms | Single-sample, local |

**Key findings to explain:**

1. **Communication savings**: Federated learning transmits only model parameter updates (~2.3 KB per
   round) rather than thousands of raw sensor records, yielding substantial bandwidth savings as the
   dataset grows.

2. **Accuracy trade-off**: Centralized learning may achieve slightly higher accuracy because it
   trains on all data simultaneously. The federated model may converge more slowly due to
   non-IID data distributions across nodes. This is an honest, documented trade-off.

3. **Data locality**: In federated mode, raw sensor records never left individual node storage.
   The coordinator received only floating-point model weights.

4. **Non-IID challenge demonstrated**: The violin plots in the dashboard show clearly different
   RMS vibration and fault rate distributions per node, illustrating why federated learning
   on real industrial data is non-trivial.

---

## 2-Minute Project Pitch

*(Suggested speaking order: Divyansh introduces, Tanu explains data, Vishal explains FL,
Hanmant demonstrates dashboard)*

**Divyansh (30 sec):**
"Our project, FedEdge-IoT, addresses a real challenge in industrial IoT: how do you build a
machine-learning model for predictive maintenance when the training data from different machines
is sensitive and you cannot centralise it? Our solution is federated learning."

**Tanu (30 sec):**
"We simulated five IoT edge nodes, each representing a machine with a vibration sensor. Each node
generates its own local synthetic dataset with different fault rates — this is called non-IID
data, which is a realistic challenge in federated learning."

**Vishal (30 sec):**
"We implemented FedAvg from scratch. Each node trains a small neural network locally, and only
the model weights — not the raw sensor readings — are sent to a central coordinator, which
computes a weighted average to produce a global model."

**Hanmant (30 sec):**
"The Streamlit dashboard shows the entire workflow interactively — node status, round-by-round
accuracy, communication byte comparison, confusion matrices, and live sensor predictions.
You can configure nodes and rounds and run the full experiment with one click."

---

## 5-Minute Presentation Script

**[Slide 1 — Title] — Divyansh (1 min)**
Introduce the team. State the problem: centralised ML for IoT requires transmitting all raw sensor
data to the cloud, raising bandwidth and privacy concerns. State the solution: federated learning
on simulated edge nodes.

**[Slide 2 — Architecture] — Divyansh (45 sec)**
Walk through the five-layer architecture diagram. Emphasise that in federated mode, raw data
stays on the edge node; only model weights cross the communication layer.

**[Slide 3 — Data and Nodes] — Tanu (45 sec)**
Show the node status cards and the fault rate bar chart. Explain non-IID distribution. Show
the violin plot showing different vibration distributions per node.

**[Slide 4 — FedAvg in Action] — Vishal (1 min)**
Show the round-by-round accuracy plot. Explain the FedAvg formula. Show the node participation
log. Show upload/download bytes per round chart.

**[Slide 5 — Comparison] — Vishal (45 sec)**
Show the side-by-side accuracy, F1, and communication cost charts. Explain the trade-off.
State the actual measured communication savings percentage.

**[Slide 6 — Live Demo] — Hanmant (45 sec)**
Open the Live Sensor Prediction tab. Demonstrate a Normal reading and a Fault reading.
Show the probability gauge. Explain the sensor history chart.

**[Slide 7 — Conclusions] — Divyansh (30 sec)**
Summarise: federated learning achieves competitive accuracy while dramatically reducing
communication costs and keeping raw data local. Discuss limitations and future work
(differential privacy, real hardware deployment).

