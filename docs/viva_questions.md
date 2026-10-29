# FedEdge-IoT — Viva Questions and Answers

## 20+ Important Questions with Technically Correct Answers

---

### 1. Why is this an IoT project rather than only a machine learning project?

**Answer:**  
This project is an IoT project because it simulates the entire IoT stack:  
- **Layer 1**: Simulated sensors (vibration, temperature, RPM) generating data  
- **Layer 2**: Edge nodes that process data locally (edge computing)  
- **Layer 3**: Communication protocols (simulated network byte transfer)  
- **Layer 4**: Cloud/fog coordination (FedAvg aggregation)  
- **Layer 5**: Application dashboard  

The ML model is one component within this architecture. Without the IoT context — heterogeneous
edge devices, limited bandwidth, privacy constraints, local data — the federated learning design
would not be motivated. The project demonstrates *why* federated learning makes sense for IoT,
not just how to implement it.

---

### 2. What is an edge node?

**Answer:**  
An edge node is a computing device located at the "edge" of the network — close to where data
is generated — rather than at a central data centre. Examples include:  
- Raspberry Pi connected to a factory machine  
- Industrial gateway aggregating sensor readings  
- Smart camera with onboard inference  

Edge nodes have limited compute and memory but can preprocess data, run inference, and even
train small ML models locally. In our project, each `EdgeNode` Python object simulates one
such device.

---

### 3. What is edge computing?

**Answer:**  
Edge computing is a distributed computing paradigm where computation and data storage are
performed closer to the data source (the "edge" of the network) rather than in a centralised
cloud server.  

**Benefits over cloud computing:**  
- Lower latency (no round-trip to distant server)  
- Reduced bandwidth consumption (process locally, send summaries)  
- Improved reliability (works even without internet connectivity)  
- Better privacy (sensitive data need not leave the local network)

---

### 4. What is federated learning?

**Answer:**  
Federated learning (FL) is a distributed machine learning approach where:  
1. A global model is distributed to participating devices (edge nodes)  
2. Each device trains the model on its own **local private data**  
3. Each device sends only its **model updates** (weights/gradients) to a coordinator  
4. The coordinator **aggregates** updates into a new global model  
5. Steps 1–4 are repeated for multiple rounds  

**Key principle**: Raw data never leaves the device. Only model parameters are shared.  
FL was introduced by Google (McMahan et al., 2017) for training keyboard language models
on Android phones without uploading private text.

---

### 5. How does FedAvg work mathematically?

**Answer:**  
FedAvg (Federated Averaging) computes a **sample-weighted average** of model parameters:

$$\mathbf{w}_{global} = \frac{\sum_{k=1}^{K} n_k \cdot \mathbf{w}_k}{\sum_{k=1}^{K} n_k}$$

Where:  
- K = number of participating nodes  
- n_k = number of local training samples at node k  
- **w**_k = model weights of node k after local training  

**Why weighted?** Nodes with more training data should contribute proportionally more
to the global model, making the average more representative of the overall data distribution.

In code (`src/fedavg.py`), each weight tensor is accumulated as a fraction-weighted sum:
```python
fraction = n_k / total_samples
averaged[i] += fraction * w_k[i]
```

---

### 6. Why not simply centralise the data?

**Answer:**  
Centralising data creates several problems:  
1. **Bandwidth**: Millions of sensor records per day per facility requires significant network capacity  
2. **Privacy**: Operational data may reveal proprietary production information  
3. **Latency**: Inference requires a round-trip to the cloud  
4. **Regulations**: GDPR, sector-specific rules may prohibit moving data across locations  
5. **Reliability**: Cloud outages halt all inference and learning  

Federated learning addresses all five by keeping raw data local.

---

### 7. What is non-IID data?

**Answer:**  
IID stands for "Independent and Identically Distributed." In standard ML, we assume all
training examples are drawn from the same distribution.

**Non-IID** means different nodes have different data distributions. In our project:  
- Node 0 might have 10% fault rate, Node 4 might have 45%  
- Node 2 uses high RPM machines, Node 5 uses low RPM machines  
- Each node sees only its own subset of the overall distribution  

This is realistic: different factory machines have different operating conditions.
Non-IID data is a core challenge in federated learning — the global model must generalise
across heterogeneous local distributions.

---

### 8. Why might centralized training achieve better accuracy?

**Answer:**  
Centralised training pools all data, so the model:  
- Sees the full distribution (all fault rates, all machine types) in each training step  
- Can identify patterns that span multiple nodes  
- Benefits from a larger effective batch size  

Federated learning may have lower accuracy because:  
- Each node trains on a subset (non-IID) — local models may overfit to local distributions  
- Averaging heterogeneous models can introduce noise  
- Limited number of rounds may mean insufficient convergence  

**Important note**: Our project does not manipulate results to guarantee federated wins.
We report honest measured values.

---

### 9. How are communication bytes calculated?

**Answer:**  
**Centralized mode:**  
`bytes = n_training_records × 80 bytes/record`  
(6 features × ~10 chars + label + CSV delimiters ≈ 80 bytes per row)

**Federated mode:**  
`bytes = model_weight_bytes × n_nodes × (1 + 2 × n_rounds)`  
- `1` = initial broadcast  
- `2 × n_rounds` = one upload + one download per round  

`model_weight_bytes = n_parameters × 8` (float64, no compression)

Our model has 290 parameters, so model_weight_bytes ≈ 2,320 bytes ≈ 2.3 KB.

**Disclaimer**: These are estimated application-level payload bytes, not real network traffic.

---

### 10. Does federated learning guarantee privacy?

**Answer:**  
**No.** Federated learning reduces the transmission of raw data, but it does NOT provide
cryptographic privacy guarantees. Risks include:  
- **Model inversion attacks**: An adversary may recover approximate training samples from weights  
- **Membership inference**: Determining if a specific record was in the training set  
- **Gradient leakage**: Weight updates can inadvertently encode raw data features  

Stronger protections (NOT implemented in this project) include:  
- **Differential Privacy**: Adding calibrated noise to model updates  
- **Secure Aggregation**: Cryptographic protocol so the coordinator only sees the aggregate  
- **Homomorphic Encryption**: Computing on encrypted weights  

Our project demonstrates FL's communication efficiency, not its privacy guarantees.

---

### 11. What is the difference between cloud computing and edge computing?

**Answer:**  

| Aspect | Cloud Computing | Edge Computing |
|--------|----------------|---------------|
| Location | Centralised data centre | Near the data source |
| Latency | High (100s of ms) | Low (<10 ms) |
| Bandwidth | High consumption | Low consumption |
| Reliability | Depends on connectivity | Works offline |
| Compute power | Very high | Limited |
| Privacy | Data leaves the network | Data stays local |

---

### 12. What happens when a node has more samples than other nodes?

**Answer:**  
In FedAvg, that node receives a higher weight in the aggregation:
`contribution = n_k / total_samples`

A node with 3,000 samples in a 5-node 10,000-sample system contributes 30% of the global model.
This is intentional — more data should have more influence on the global model.

**Challenge**: If a single large node has very different data (e.g., all fault examples),
it can dominate the model and harm performance on other nodes. Mitigation strategies include
capping maximum per-node contribution or using uniform averaging.

---

### 13. How could this be implemented using Raspberry Pi or ESP32 devices?

**Answer:**  
**Raspberry Pi** (Python-capable):  
- Run the same `EdgeNode.train_local()` code on each Pi  
- Use `socket` or MQTT to transmit weight arrays to a central server  
- Real sensors: ADXL345 (accelerometer), DS18B20 (temperature) via GPIO  

**ESP32** (MicroPython, limited):  
- Too small for sklearn MLPClassifier  
- Could use a tiny custom neural network with fixed-point arithmetic  
- Or send preprocessed features to a gateway Raspberry Pi for training  
- Use Wi-Fi or BLE for model update transmission  

**Real implementation steps**:  
1. Connect sensors to GPIO pins  
2. Collect sensor readings in a local CSV or ring buffer  
3. Train model locally (Pi) or send to gateway (ESP32)  
4. Serialize model weights as JSON/pickle and POST to coordinator API  
5. Download new global weights from coordinator API  

---

### 14. How does the project simulate sensor readings?

**Answer:**  
`data_generator.py` uses NumPy's `default_rng` (random number generator) to produce synthetic
readings:

- **Normal conditions**: Gaussian distributions centred at typical operating values
- **Fault conditions**: Shifted distributions with higher mean and wider variance
- **Realistic noise**: Each machine profile has a `noise` multiplier  
- **Physical clipping**: Values clipped to physically plausible ranges  
- **Reproducibility**: Fixed seeds ensure the same data every run  

This is entirely statistical simulation — no physical sensors are involved.

---

### 15. What are the limitations of synthetic data?

**Answer:**  
1. **Distribution assumptions**: Real vibration signals have complex frequency components;
   our simple Gaussian model cannot capture all real-world patterns  
2. **No temporal dynamics**: Real sensor data has time series structure; our records are IID  
3. **No missing data or sensor failure**: Real IoT deployments have dropouts and calibration errors  
4. **No adversarial conditions**: Power surges, environmental interference not modelled  
5. **Artificial separability**: Our fault patterns are more cleanly distinguishable than real data  

**Implication**: Model performance on our synthetic data may not generalise to real industrial
sensors without retraining on real data.

---

### 16. How could network latency, dropped packets, and node failures be simulated?

**Answer:**  
The current implementation does not simulate these. Possible extensions:

- **Latency**: Add `time.sleep(random_delay)` before each node transmits updates  
- **Dropped packets**: Randomly skip some nodes' updates each round  
  (`if random() < drop_rate: skip node k`)  
- **Node failure**: Remove nodes from the participation list mid-experiment  
- **Stragglers**: Sort nodes by simulated transmission time and ignore slow ones  
- **Partial participation**: `fraction_clients < 1.0` already simulates partial participation  

---

### 17. Why is accuracy alone insufficient for predictive maintenance?

**Answer:**  
In predictive maintenance, the cost of errors is asymmetric:  
- **False Negative (missed fault)**: Machine fails unexpectedly → production halt, safety risk,
  costly emergency repair  
- **False Positive (false alarm)**: Machine stopped for maintenance it did not need → lost
  production time, unnecessary cost  

A model with 95% accuracy on a dataset with 5% faults could achieve this by simply predicting
"Normal" for every sample — missing ALL faults.

Therefore we also report:  
- **Recall** (especially for the Fault class): fraction of actual faults correctly detected  
- **Precision**: fraction of fault predictions that are correct  
- **F1-score**: harmonic mean of precision and recall  
- **Confusion matrix**: shows exactly which errors are being made  

---

### 18. What is the difference between training time and inference latency?

**Answer:**  
- **Training time**: Total time to fit the model on training data. Measured in seconds.
  Happens offline or in scheduled maintenance windows. In federated mode, this is the
  sum of all local training across all nodes across all rounds.

- **Inference latency**: Time to classify a single new sensor reading. Measured in
  milliseconds. This is what matters for real-time monitoring. Our measurement: average
  of 95 single-sample prediction calls (after 5 warm-up runs).

A model might take 10 seconds to train but classify each reading in 0.1 ms — both are
important for different reasons.

---

### 19. What would change if the number of nodes increased?

**Answer:**  
| Aspect | Effect of more nodes |
|--------|---------------------|
| **Federated comm. bytes** | Increase proportionally (more broadcasts and uploads) |
| **Centralized comm. bytes** | Stay the same (total data is same, just split differently) |
| **Communication savings** | Federated becomes relatively more expensive vs centralized |
| **Convergence** | May be slower with more non-IID heterogeneity |
| **Fault coverage** | Better coverage of different operating conditions |
| **Privacy** | More nodes = harder to isolate any one node's contribution |
| **Bandwidth** | Can run rounds in parallel if truly distributed |

In general, federated learning scales well in terms of **privacy and data locality**, but
the **communication cost** scales with number of nodes.

---

### 20. What are the next steps toward real hardware deployment?

**Answer:**  
1. **Hardware selection**: Raspberry Pi 4 for edge nodes (can run Python + sklearn),
   a laptop or cloud VM as coordinator  
2. **Real sensors**: ADXL345 accelerometer + DS18B20 temperature sensor + RPM encoder  
3. **Communication protocol**: MQTT broker (Mosquitto) for lightweight IoT messaging,
   or REST API with Flask/FastAPI on coordinator  
4. **Model serialisation**: Use `joblib.dumps()` / `joblib.loads()` for weight transmission  
5. **Scheduling**: Cron job or event-driven trigger for local training rounds  
6. **Privacy**: Integrate differential privacy (add Gaussian noise to weights before upload)  
7. **Security**: TLS/HTTPS for coordinator API, device authentication  
8. **Monitoring**: Extend dashboard to receive live metrics from physical nodes via MQTT  

---

### Bonus Q21: What is the FedAvg paper reference?

**Answer:**  
McMahan, H.B., Moore, E., Ramage, D., Hampson, S., and Agüera y Arcas, B. (2017).
"Communication-Efficient Learning of Deep Networks from Decentralized Data."
*Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS).*  
arXiv: 1602.05629

---

### Bonus Q22: How is this different from distributed training?

**Answer:**  
**Distributed training** (e.g., data parallelism with MPI or PyTorch DDP):  
- All nodes have access to the same shared dataset  
- Parallelises the computation of gradients  
- The coordinator sees all raw gradients computed on centralised data  
- Primary goal: speed up training  

**Federated learning**:  
- Each node has its own **private** dataset that is NOT shared  
- Nodes train on different data (non-IID)  
- Primary goal: **privacy** and **data locality**  
- Secondary benefit: bandwidth savings  

Federated learning is a privacy-motivated paradigm; distributed training is a
performance-optimisation paradigm.

