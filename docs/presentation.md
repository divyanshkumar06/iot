# FedEdge-IoT — Presentation Guide

## Slide Deck Outline (10 slides)

### Slide 1 — Title
**FedEdge-IoT: Federated Learning on Simulated IoT Edge Nodes Using FedAvg**  
Team: Divyansh Kumar | Tanu | Vishal | Hanmant  
Subject: Internet of Things (IoT) and Edge Computing  

*Note: "SYNTHETIC DATA SIMULATION — No real IoT hardware used"*

---

### Slide 2 — Problem Statement
**The Challenge of Centralised IoT ML**
- 🏭 Industrial machines generate continuous sensor streams
- ☁️ Traditional ML = upload ALL data to cloud
- 📶 High bandwidth consumption
- 🔒 Privacy and security risks
- ⏱️ Latency and reliability concerns
- ❓ *Can we train ML models without centralising raw data?*

---

### Slide 3 — Proposed Solution
**Federated Learning on Edge Nodes**
- Each machine keeps its sensor data LOCAL
- Train a model locally on each machine
- Send only MODEL WEIGHTS (not raw data) to a coordinator
- Coordinator aggregates weights using FedAvg
- Global model is broadcast back
- Repeat for multiple rounds

*Key insight: Model weights encode patterns, not raw records*

---

### Slide 4 — System Architecture (show the 5-layer diagram from architecture.md)
**FedEdge-IoT: 5-Layer IoT Architecture**
- Layer 1: Simulated sensors (vibration, temperature, RPM)
- Layer 2: Edge nodes (local training, private data)
- Layer 3: Communication (model updates only)
- Layer 4: FedAvg coordinator
- Layer 5: Streamlit dashboard

---

### Slide 5 — Dataset and Non-IID Design
**Synthetic Sensor Data (SIMULATION)**
- 8,000 total records, 5 nodes
- Features: RMS vibration, peak vibration, variance, frequency, temperature, RPM
- Non-IID: Different fault rates per node (10% to 45%)
- Different machine profiles: light, heavy, high-speed, etc.
- Separate held-out test set (different seed, never seen during training)

*Show violin plot of RMS vibration by node — demonstrates heterogeneity*

---

### Slide 6 — FedAvg Implementation
**FedAvg: Sample-Weighted Averaging**

Formula: w_global = Σ(n_k × w_k) / Σ(n_k)

Round structure:
1. Server → nodes: broadcast global weights
2. Nodes: train locally (private data stays local)
3. Nodes → server: upload weight updates
4. Server: compute weighted average
5. Evaluate on held-out test set

Architecture: 6 inputs → 32 hidden (ReLU) → 2 outputs (Normal / Fault)

---

### Slide 7 — Results: Accuracy Comparison
**Model Performance (Measured Values)**

| Metric | Federated | Centralized |
|--------|-----------|------------|
| Accuracy | [MEASURED] | [MEASURED] |
| Precision | [MEASURED] | [MEASURED] |
| Recall | [MEASURED] | [MEASURED] |
| F1-Score | [MEASURED] | [MEASURED] |

*Show confusion matrices side by side*

---

### Slide 8 — Results: Communication Savings
**Communication Cost Comparison**

| Mode | Total Bytes | Details |
|------|------------|---------|
| Centralized | [MEASURED] | Raw sensor records uploaded |
| Federated | [MEASURED] | Model weights only |
| **Savings** | **[MEASURED]%** | 100 × (1 − fed/central) |

*Show communication bar chart*

Key insight: As dataset grows → centralized cost grows linearly, federated stays bounded

---

### Slide 9 — Live Demonstration
**Interactive Dashboard Demo**
- ✅ Node status cards (5 simulated IoT devices)
- ✅ Round-by-round accuracy chart
- ✅ Communication byte comparison
- ✅ Confusion matrices
- ✅ Live sensor prediction (slide the controls!)
- ✅ Fault probability gauge

---

### Slide 10 — Conclusions and Future Work
**Key Takeaways**
- ✅ Federated learning enables collaborative training without sharing raw data
- ✅ Communication savings: [MEASURED]% less data transmitted
- ✅ Competitive accuracy: [MEASURED] vs [MEASURED] centralized
- ⚠️ Non-IID data is a real challenge (heterogeneous fault rates demonstrated)
- ⚠️ Model weights alone do not guarantee complete privacy

**Future Work:**
- Differential privacy (adding noise to model updates)
- Real hardware (Raspberry Pi + ADXL345 sensor)
- MQTT-based real network simulation
- Federated learning with time-series sensor data

---

## Speaking Order Recommendation

| Speaker | Slides | Duration |
|---------|--------|----------|
| Divyansh | 1, 2, 3 | ~1 min |
| Tanu | 4, 5 | ~1 min |
| Vishal | 6, 8 | ~1 min |
| Hanmant | 7, 9, 10 + live demo | ~2 min |

**Total: ~5 minutes + Q&A**

---

## Key Demo Tips

1. **Before the demo**: Run `python src/experiment.py` to pre-generate results
2. **Dashboard startup**: `streamlit run app.py` — use the Full Demo tab for one-click
3. **Live prediction**: Slide RMS Vibration to 3.0 and Temperature to 88°C → show Fault
4. **Normal reading**: Slide RMS to 0.8 and Temperature to 65°C → show Normal
5. **Communication chart**: Emphasise the bar chart showing federated << centralized
6. **Be honest**: If centralized accuracy is higher, explain the trade-off — don't hide it

---

## Common Faculty Questions — Quick Answers

**"Is this real hardware?"**  
→ "No, this is a software simulation. Each EdgeNode Python object simulates one IoT device."

**"Why use sklearn instead of PyTorch?"**  
→ "sklearn's MLPClassifier exposes coefs_ and intercepts_ that can be numerically averaged. PyTorch would require custom weight extraction, adding complexity without educational benefit."

**"How would you deploy this on real hardware?"**  
→ "Use Raspberry Pi for edge nodes, MQTT for communication, real ADXL345 accelerometers for vibration data. The Python code is compatible."

**"Doesn't federated learning have worse accuracy?"**  
→ "Yes, often slightly. The trade-off is intentional: lower accuracy for better privacy and communication efficiency. We report both honestly."

