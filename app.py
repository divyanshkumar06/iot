"""
app.py — FedEdge-IoT Streamlit Dashboard
=========================================
Interactive presentation-ready dashboard for the FedEdge-IoT project.

Launch with:
    streamlit run app.py

Author (proposed): Hanmant
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import copy

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    FEATURES, TARGET, DEFAULT_NUM_NODES, DEFAULT_ROUNDS,
    DEFAULT_LOCAL_EPOCHS, GLOBAL_SEED, EXPERIMENT_RESULTS_PATH,
    MODEL_PATH, CENTRAL_MODEL_PATH, DATA_DIR, RESULTS_DIR,
    TOTAL_SAMPLES,
)
from src.data_generator import generate_node_datasets, generate_holdout_test_set, get_node_summary
from src.model import initialise_global_model, get_weights, weights_byte_size
from src.edge_node import EdgeNode
from src.fedavg import FedAvgCoordinator
from src.centralized import run_centralized_training
from src.evaluation import evaluate_model
from src.communication import (
    communication_savings_pct, format_bytes, bytes_for_raw_records
)
from src.experiment import run_full_experiment

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedEdge-IoT",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark navy + teal theme) ───────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1b2a; color: #e0e6ed; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #112240; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #64ffda; }
    
    /* Headers */
    h1 { color: #64ffda !important; }
    h2 { color: #ccd6f6 !important; }
    h3 { color: #a8b2d8 !important; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #172a45;
        border: 1px solid #233554;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="metric-container"] label { color: #64ffda !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ccd6f6 !important;
        font-size: 1.4rem !important;
    }
    
    /* Cards */
    .fed-card {
        background-color: #172a45;
        border: 1px solid #233554;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .fed-card-title { color: #64ffda; font-weight: bold; font-size: 1rem; }
    .fed-badge-normal { background-color: #1e4d2b; color: #6fcf97;
                        border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }
    .fed-badge-fault  { background-color: #4d1e1e; color: #eb5757;
                        border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }
    
    /* Info boxes */
    .info-box {
        background-color: #0f3460;
        border-left: 4px solid #64ffda;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .warn-box {
        background-color: #3d2600;
        border-left: 4px solid #ffb347;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #112240; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; }
    .stTabs [aria-selected="true"] { color: #64ffda !important;
                                      border-bottom-color: #64ffda !important; }
    
    /* Buttons */
    .stButton > button {
        background-color: #64ffda;
        color: #0d1b2a;
        font-weight: bold;
        border: none;
        border-radius: 6px;
    }
    .stButton > button:hover { background-color: #52d9b8; }
    
    /* Tables */
    .dataframe { background-color: #172a45 !important; color: #e0e6ed !important; }
    
    /* Progress */
    .stProgress > div > div { background-color: #64ffda; }
</style>
""", unsafe_allow_html=True)


# ── Plotly theme helper ───────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#172a45",
    plot_bgcolor="#0d1b2a",
    font=dict(color="#ccd6f6", family="monospace"),
    xaxis=dict(gridcolor="#233554", zerolinecolor="#233554"),
    yaxis=dict(gridcolor="#233554", zerolinecolor="#233554"),
    legend=dict(bgcolor="#172a45", bordercolor="#233554"),
    margin=dict(l=40, r=20, t=40, b=40),
)

TEAL    = "#64ffda"
ORANGE  = "#ff9f43"
RED     = "#eb5757"
BLUE    = "#74b9ff"
PURPLE  = "#a29bfe"


# ══════════════════════════════════════════════════════════════════════════════
# Session state helpers
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "experiment_done": False,
        "results": None,
        "node_datasets": None,
        "test_df": None,
        "scaler": None,
        "global_model": None,
        "central_model": None,
        "nodes": None,
        "log_lines": [],
        "sensor_history": [],
        "demo_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _log(msg: str):
    st.session_state["log_lines"].append(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — global configuration
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        num_nodes = st.slider("Edge Nodes", 2, 10, DEFAULT_NUM_NODES, key="cfg_nodes")
        n_rounds  = st.slider("Federated Rounds", 2, 20, DEFAULT_ROUNDS, key="cfg_rounds")
        local_ep  = st.slider("Local Epochs / Round", 2, 20, DEFAULT_LOCAL_EPOCHS, key="cfg_epochs")
        n_samples = st.slider("Total Samples", 2000, 15000, TOTAL_SAMPLES, step=500, key="cfg_samples")
        seed      = st.number_input("Random Seed", value=GLOBAL_SEED, key="cfg_seed")

        st.markdown("---")
        st.markdown("### 🌐 Non-IID Heterogeneity")
        hetero = st.select_slider(
            "Fault Rate Spread",
            options=["Low (similar)", "Medium", "High (very different)"],
            value="Medium",
            key="cfg_hetero",
        )
        hetero_map = {
            "Low (similar)":       (0.20, 0.30),
            "Medium":              (0.10, 0.45),
            "High (very different)": (0.05, 0.55),
        }
        fr_min, fr_max = hetero_map[hetero]
        fault_rates = np.linspace(fr_min, fr_max, num_nodes).tolist()

        st.markdown("---")
        st.caption("🔒 **Privacy note:** In federated mode, raw sensor records stay on each node.")
        st.caption("⚠️ **Disclaimer:** All data is SYNTHETIC SIMULATION DATA.")

    return {
        "num_nodes": num_nodes,
        "n_rounds": n_rounds,
        "local_epochs": local_ep,
        "total_samples": n_samples,
        "seed": int(seed),
        "fault_rates": fault_rates,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tab A — Project Overview
# ══════════════════════════════════════════════════════════════════════════════
def tab_overview():
    st.markdown("# 🌐 FedEdge-IoT")
    st.markdown("### *Federated Learning on Simulated IoT Edge Nodes Using FedAvg*")

    st.markdown("""
    <div class="info-box">
    <b>Subject:</b> Internet of Things (IoT) &amp; Edge Computing &nbsp;|&nbsp;
    <b>Team:</b> Divyansh • Tanu • Vishal • Hanmant
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Problem Statement")
        st.markdown("""
        Modern IoT deployments generate **massive volumes of sensor data** on machines
        and devices. Traditional cloud-centric ML requires shipping every raw sensor
        record to a central server — creating:

        - 📶 High **communication bandwidth** consumption
        - 🔒 Potential **privacy exposure** of operational data
        - ⏱️ High **latency** for time-sensitive decisions
        - 🌐 Single point of **failure** and bottleneck

        **FedEdge-IoT** demonstrates how multiple IoT edge nodes can collaboratively
        train a shared predictive-maintenance model *without transmitting raw sensor data*
        to the central server.
        """)

    with col2:
        st.markdown("### ✅ Key Objectives")
        objectives = [
            "Simulate 5–10 IoT edge nodes with heterogeneous data distributions",
            "Generate synthetic industrial vibration sensor data locally",
            "Classify machine conditions: **Normal** vs **Fault**",
            "Implement FedAvg from scratch with sample-weighted aggregation",
            "Compare federated vs centralized training fairly",
            "Measure accuracy, F1, communication bytes, and latency",
            "Visualise the full IoT → Edge → Server → Dashboard workflow",
            "Explain privacy trade-offs honestly",
        ]
        for obj in objectives:
            st.markdown(f"- {obj}")

    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")

    # Mermaid-like ASCII diagram rendered as code for clarity
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                    FedEdge-IoT ARCHITECTURE                      │
    ├──────────────────────────────────────────────────────────────────┤
    │  LAYER 1: IoT Perception Layer                                   │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
    │  │ Sensor  │  │ Sensor  │  │ Sensor  │  │ Sensor  │  ...       │
    │  │ (Vibr.) │  │ (Temp.) │  │  (RPM)  │  │  (Freq) │           │
    │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
    ├───────┼────────────┼────────────┼────────────┼───────────────── │
    │  LAYER 2: Edge Layer            │                               │
    │  ┌────▼────────────▼────────────▼────────────▼──────────────┐  │
    │  │            Edge Node (local private data)                 │  │
    │  │   ┌─────────────┐   ┌──────────────────────────────┐    │  │
    │  │   │ Local Data  │   │   Local Model Training       │    │  │
    │  │   │ (PRIVATE)   │   │   (MLP · FedAvg weights)     │    │  │
    │  │   └─────────────┘   └──────────────┬───────────────┘    │  │
    │  └────────────────────────────────────┼───────────────────────┘ │
    ├───────────────────────────────────────┼───────────────────────── │
    │  LAYER 3: Communication Layer         │                          │
    │           ← Model updates only →      │ (NO raw data upload)     │
    ├───────────────────────────────────────┼───────────────────────── │
    │  LAYER 4: Fog/Cloud Coordination Layer│                          │
    │  ┌─────────────────────────────────────▼──────────────────────┐  │
    │  │   FedAvg Coordinator                                       │  │
    │  │   w_global = Σ(n_k × w_k) / Σ(n_k)                       │  │
    │  └───────────────────────────────────────────────────────────┘  │
    ├───────────────────────────────────────────────────────────────── │
    │  LAYER 5: Application Layer                                      │
    │  ┌─────────────────────────────────────────────────────────┐     │
    │  │   Streamlit Dashboard · Plotly Charts · Predictions     │     │
    │  └─────────────────────────────────────────────────────────┘     │
    └──────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🌍 What is Edge Computing?")
        st.markdown("""
        **Edge computing** moves computation closer to where data is generated —
        on the device or a nearby gateway — instead of sending everything to a
        distant cloud server.

        **Benefits:** Lower latency · Reduced bandwidth · Improved reliability
        """)

    with col4:
        st.markdown("### 🤝 What is Federated Learning?")
        st.markdown("""
        **Federated Learning (FL)** is a distributed ML approach where each device
        trains a model on its own local data and sends only model *parameter updates*
        to a coordinator, which aggregates them into a global model.

        **Key principle:** *Data stays local. Only model updates are shared.*
        """)

    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Important:</b> All sensor data in this project is <b>SYNTHETIC SIMULATION DATA</b>.
    This is a software simulation — no real IoT hardware, real sensors, or real network
    connections are used. Node names like "node_0" represent logically separate simulated
    processes running within this single Python application.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab B — Edge Node Simulator
# ══════════════════════════════════════════════════════════════════════════════
def tab_nodes(cfg: dict):
    st.markdown("## 🖥️ Edge Node Simulator")
    st.markdown("""
    <div class="info-box">
    Each card represents one simulated IoT edge node. Nodes have <b>different data distributions</b>
    (non-IID) to model realistic industrial heterogeneity.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Generate Node Datasets", key="btn_gen_data"):
        with st.spinner("Generating synthetic sensor data …"):
            node_datasets = generate_node_datasets(
                num_nodes=cfg["num_nodes"],
                total_samples=cfg["total_samples"],
                fault_rates=cfg["fault_rates"],
                seed=cfg["seed"],
            )
            test_df = generate_holdout_test_set(
                n_samples=max(500, cfg["total_samples"] // 5),
                fault_rate=0.30,
                seed=cfg["seed"] + 999,
            )
            st.session_state["node_datasets"] = node_datasets
            st.session_state["test_df"] = test_df
        st.success(f"✅ Generated datasets for {cfg['num_nodes']} nodes!")

    if st.session_state["node_datasets"] is None:
        st.info("👆 Click 'Generate Node Datasets' to create simulated sensor data.")
        return

    node_datasets = st.session_state["node_datasets"]

    # Node status cards
    st.markdown("### Node Status Cards")
    cols = st.columns(min(cfg["num_nodes"], 5))
    for i, (node_id, df) in enumerate(node_datasets.items()):
        col = cols[i % len(cols)]
        fault_pct = df[TARGET].mean() * 100
        with col:
            st.markdown(f"""
            <div class="fed-card">
                <div class="fed-card-title">🖥️ {node_id}</div>
                <hr style="border-color:#233554; margin:6px 0">
                <b>Samples:</b> {len(df):,}<br>
                <b>Fault Rate:</b> {fault_pct:.1f}%<br>
                <b>Avg RMS:</b> {df['rms_vibration'].mean():.3f} g<br>
                <b>Avg RPM:</b> {df['rpm'].mean():.0f}<br>
                <b>Profile:</b> {df['profile'].iloc[0]}<br>
                <span class="fed-badge-normal">🟢 ONLINE</span>
            </div>
            """, unsafe_allow_html=True)

    # Summary table
    st.markdown("### Node Characteristics Table")
    summary_df = get_node_summary(node_datasets)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Distribution charts
    st.markdown("### Sample Distribution")
    col1, col2 = st.columns(2)

    with col1:
        # Samples per node
        names = list(node_datasets.keys())
        counts = [len(df) for df in node_datasets.values()]
        fig = go.Figure(go.Bar(x=names, y=counts, marker_color=TEAL,
                               text=counts, textposition="outside"))
        fig.update_layout(title="Samples per Node", xaxis_title="Node",
                          yaxis_title="Samples", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fault rate per node (demonstrates non-IID)
        fault_rates_actual = [node_datasets[n][TARGET].mean() * 100 for n in names]
        fig2 = go.Figure(go.Bar(x=names, y=fault_rates_actual, marker_color=ORANGE,
                                text=[f"{r:.1f}%" for r in fault_rates_actual],
                                textposition="outside"))
        fig2.update_layout(title="Fault Rate per Node (Non-IID Demo)",
                           xaxis_title="Node", yaxis_title="Fault Rate (%)",
                           **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # Feature distribution — RMS vibration for all nodes
    st.markdown("### RMS Vibration Distribution by Node")
    all_data = pd.concat(
        [df.assign(node=node_id) for node_id, df in node_datasets.items()]
    )
    fig3 = px.violin(
        all_data, x="node", y="rms_vibration", color="node",
        box=True, points=False,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig3.update_layout(showlegend=False,
                       xaxis_title="Node", yaxis_title="RMS Vibration (g)",
                       **PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Sample Sensor Records (first 10 rows, node_0)")
    st.dataframe(
        node_datasets["node_0"].head(10)[FEATURES + [TARGET]].round(4),
        use_container_width=True,
    )
    st.caption("⚠️ SYNTHETIC SIMULATION DATA — not real sensor readings")


# ══════════════════════════════════════════════════════════════════════════════
# Tab C — Federated Training
# ══════════════════════════════════════════════════════════════════════════════
def tab_federated(cfg: dict):
    st.markdown("## 🔗 Federated Training (FedAvg)")
    st.markdown("""
    <div class="info-box">
    <b>FedAvg formula:</b> w_global = Σ(n_k × w_k) / Σ(n_k)
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Raw sensor data <b>never</b> leaves individual edge nodes.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["node_datasets"] is None:
        st.warning("⚠️ Please generate node datasets in the **Edge Node Simulator** tab first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Configured Nodes", cfg["num_nodes"])
    with col2:
        st.metric("Federated Rounds", cfg["n_rounds"])
    with col3:
        st.metric("Local Epochs / Round", cfg["local_epochs"])

    run_btn = st.button("🚀 Start Federated Training", key="btn_fed_train", type="primary")

    log_container = st.container()
    progress_bar = st.progress(0)

    if run_btn:
        st.session_state["log_lines"] = []
        log_area = log_container.empty()

        def progress_cb(msg):
            st.session_state["log_lines"].append(msg)
            log_area.code("\n".join(st.session_state["log_lines"]), language="")

        with st.spinner("Running federated experiment …"):
            results = run_full_experiment(
                num_nodes=cfg["num_nodes"],
                n_rounds=cfg["n_rounds"],
                local_epochs=cfg["local_epochs"],
                total_samples=cfg["total_samples"],
                fault_rates=cfg["fault_rates"],
                seed=cfg["seed"],
                progress_callback=progress_cb,
                save_results=True,
            )

        st.session_state["results"] = results
        st.session_state["experiment_done"] = True
        st.session_state["global_model"] = results["global_model"]
        st.session_state["central_model"] = results["central_model"]
        st.session_state["scaler"] = results["scaler"]
        st.session_state["nodes"] = results["nodes"]
        progress_bar.progress(100)
        st.success("✅ Federated training complete! View results in the Comparison tab.")

    if st.session_state["experiment_done"] and st.session_state["results"]:
        results = st.session_state["results"]
        round_history = results["round_history"]

        st.markdown("---")
        st.markdown("### 📈 Round-by-Round Global Accuracy")
        rounds = [r["round"] for r in round_history]
        acc_values = [r["global_accuracy"] for r in round_history
                      if r["global_accuracy"] is not None]
        if acc_values:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=acc_values,
                mode="lines+markers",
                line=dict(color=TEAL, width=2),
                marker=dict(size=8, color=TEAL),
                name="Global Accuracy",
            ))
            fig.update_layout(
                title="Global Model Accuracy per Federated Round",
                xaxis_title="Round", yaxis_title="Accuracy",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📡 Communication Bytes per Round")
        up_bytes   = [r["upload_bytes"]   for r in round_history]
        dl_bytes   = [r["download_bytes"] for r in round_history]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Upload (nodes→server)", x=rounds, y=up_bytes,
                              marker_color=ORANGE))
        fig2.add_trace(go.Bar(name="Download (server→nodes)", x=rounds, y=dl_bytes,
                              marker_color=BLUE))
        fig2.update_layout(barmode="stack", title="Communication per Round",
                           xaxis_title="Round", yaxis_title="Bytes",
                           **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🖥️ Node Participation Log")
        rows = []
        for r in round_history:
            for n_log in r["node_logs"]:
                rows.append({
                    "Round": r["round"],
                    "Node": n_log["node_id"],
                    "Samples": n_log["n_samples"],
                    "Local Accuracy": f"{n_log['local_accuracy']:.4f}",
                    "Train Time (s)": f"{n_log['train_time_s']:.4f}",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab D — Comparison
# ══════════════════════════════════════════════════════════════════════════════
def tab_comparison():
    st.markdown("## ⚖️ Centralized vs Federated Comparison")

    if not st.session_state["experiment_done"]:
        st.info("▶️ Run the experiment in the **Federated Training** tab first.")
        return

    results = st.session_state["results"]
    fm = results["federated_metrics"]
    cm = results["central_metrics"]
    comm = results["communication"]

    # ── Top metrics ───────────────────────────────────────────────────────────
    st.markdown("### 🏆 Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    def delta_str(a, b): return f"{(a-b)*100:+.2f}pp"

    c1.metric("Fed Accuracy",   f"{fm['accuracy']:.4f}",   delta_str(fm['accuracy'],   cm['accuracy']))
    c2.metric("Fed Precision",  f"{fm['precision']:.4f}",  delta_str(fm['precision'],  cm['precision']))
    c3.metric("Fed Recall",     f"{fm['recall']:.4f}",     delta_str(fm['recall'],     cm['recall']))
    c4.metric("Fed F1-Score",   f"{fm['f1_score']:.4f}",   delta_str(fm['f1_score'],   cm['f1_score']))
    c5.metric("Fed Latency",    f"{fm['inference_latency_ms']:.2f}ms",
              f"{fm['inference_latency_ms']-cm['inference_latency_ms']:+.2f}ms")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Accuracy & F1 Comparison")
        metrics_names = ["accuracy", "precision", "recall", "f1_score"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Federated", x=metrics_names,
            y=[fm[m] for m in metrics_names], marker_color=TEAL,
        ))
        fig.add_trace(go.Bar(
            name="Centralized", x=metrics_names,
            y=[cm[m] for m in metrics_names], marker_color=ORANGE,
        ))
        fig.update_layout(barmode="group", title="Model Performance Comparison",
                          yaxis_title="Score", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📡 Communication Cost")
        fig2 = go.Figure(go.Bar(
            x=["Centralized\n(raw data upload)", "Federated\n(model updates only)"],
            y=[comm["centralized_bytes"], comm["fed_bytes"]],
            marker_color=[RED, TEAL],
            text=[format_bytes(comm["centralized_bytes"]),
                  format_bytes(comm["fed_bytes"])],
            textposition="outside",
        ))
        fig2.update_layout(title="Total Communication Bytes",
                           yaxis_title="Bytes", **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # Communication savings metric
    savings = comm["savings_pct"]
    savings_color = "normal" if savings > 0 else "inverse"
    st.metric(
        "📶 Communication Savings (Federated vs Centralized)",
        f"{savings:.1f}%",
        delta=f"{savings:.1f}% less data transmitted in federated mode",
        delta_color=savings_color,
    )

    st.markdown("---")

    # Confusion matrices
    st.markdown("### 🔲 Confusion Matrices")
    col3, col4 = st.columns(2)

    def plot_cm(cm_data, title):
        z = cm_data
        labels = ["Normal", "Fault"]
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels,
            colorscale=[[0, "#0d1b2a"], [1, TEAL]],
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}",
            showscale=False,
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Predicted", yaxis_title="Actual",
            **PLOTLY_LAYOUT,
        )
        return fig

    with col3:
        st.plotly_chart(plot_cm(fm["confusion_matrix"], "Federated Model"), use_container_width=True)
    with col4:
        st.plotly_chart(plot_cm(cm["confusion_matrix"], "Centralized Model"), use_container_width=True)

    # Training time comparison
    st.markdown("### ⏱️ Training Time")
    fig3 = go.Figure(go.Bar(
        x=["Centralized", "Federated"],
        y=[cm["train_time_s"], fm["train_time_s"]],
        marker_color=[RED, TEAL],
        text=[f"{cm['train_time_s']:.2f}s", f"{fm['train_time_s']:.2f}s"],
        textposition="outside",
    ))
    fig3.update_layout(title="Training Time (seconds)",
                       yaxis_title="Seconds", **PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    # Interpretation
    st.markdown("### 💡 Result Interpretation")
    acc_diff = (fm["accuracy"] - cm["accuracy"]) * 100
    interpretation = "outperforms" if acc_diff > 0 else "slightly underperforms"
    st.markdown(f"""
    <div class="info-box">
    <b>Performance:</b> The federated model {interpretation} centralized by
    <b>{abs(acc_diff):.2f} percentage points</b>. This is expected:
    centralized learning typically has an accuracy advantage because it trains
    on pooled data, but federated learning preserves data locality.<br><br>
    <b>Communication:</b> Federated mode saves <b>{savings:.1f}%</b> of communication bandwidth
    compared to uploading raw sensor records. Savings grow as dataset size increases
    and model size stays constant.<br><br>
    <b>Trade-off:</b> Federated learning trades a possible small accuracy reduction
    for significant privacy, bandwidth, and latency benefits.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Full Results Table")
    comparison_df = pd.DataFrame([
        {
            "Metric": k,
            "Federated": fm.get(k, "N/A"),
            "Centralized": cm.get(k, "N/A"),
        }
        for k in ["accuracy", "precision", "recall", "f1_score",
                   "inference_latency_ms", "train_time_s", "total_comm_bytes"]
    ])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab E — Live Sensor Prediction
# ══════════════════════════════════════════════════════════════════════════════
def tab_prediction(cfg: dict):
    st.markdown("## 🔍 Live Sensor Prediction")
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Disclaimer:</b> These are <b>ML model predictions on synthetic data only</b>.
    This is NOT a safety-certified industrial diagnostic tool. Do not use for real equipment decisions.
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["experiment_done"]:
        st.info("▶️ Run the experiment in the **Federated Training** tab first.")
        return

    scaler = st.session_state["scaler"]
    global_model = st.session_state["global_model"]
    central_model = st.session_state["central_model"]

    model_choice = st.radio(
        "Select Model for Prediction",
        ["Federated (Global)", "Centralized"],
        horizontal=True,
    )
    active_model = global_model if model_choice == "Federated (Global)" else central_model

    st.markdown("---")
    st.markdown("### 🎛️ Sensor Input Controls")

    col1, col2, col3 = st.columns(3)
    with col1:
        rms_vib = st.slider("RMS Vibration (g)",       0.1, 10.0, 1.0, 0.01)
        peak_vib = st.slider("Peak Vibration (g)",      0.1, 15.0, 1.5, 0.01)
    with col2:
        vib_var = st.slider("Vibration Variance",       0.0, 2.0,  0.1, 0.01)
        dom_freq = st.slider("Dominant Frequency (Hz)", 5.0, 200.0, 30.0, 0.5)
    with col3:
        temp = st.slider("Temperature (°C)",            20.0, 110.0, 65.0, 0.5)
        rpm  = st.slider("RPM",                         100.0, 4000.0, 1800.0, 10.0)

    # Auto-generate random reading
    if st.button("🎲 Generate Random Normal Reading"):
        rng = np.random.default_rng()
        rms_vib  = float(rng.normal(0.8, 0.15))
        peak_vib = float(rms_vib + rng.exponential(0.2))
        vib_var  = float(max(0, rng.normal(0.05, 0.01)))
        dom_freq = float(rng.normal(30.0, 3.0))
        temp     = float(rng.normal(65.0, 3.0))
        rpm      = float(rng.normal(1800, 100))
        st.rerun()

    if st.button("🚨 Generate Random Fault Reading"):
        rng = np.random.default_rng()
        rms_vib  = float(rng.normal(2.5, 0.5))
        peak_vib = float(rms_vib + rng.exponential(0.8))
        vib_var  = float(max(0, rng.normal(0.35, 0.08)))
        dom_freq = float(rng.normal(50.0, 5.0))
        temp     = float(rng.normal(85.0, 5.0))
        rpm      = float(rng.normal(1530, 200))
        st.rerun()

    # Predict
    X_input = np.array([[rms_vib, peak_vib, vib_var, dom_freq, temp, rpm]])
    X_scaled = scaler.transform(X_input)
    prediction = active_model.predict(X_scaled)[0]
    proba = active_model.predict_proba(X_scaled)[0]
    confidence = proba[prediction]

    result_label = "🟢 NORMAL" if prediction == 0 else "🔴 FAULT"
    result_color = "#6fcf97" if prediction == 0 else "#eb5757"

    st.markdown("---")
    st.markdown(f"""
    <div style="background-color:#172a45; border:2px solid {result_color};
                border-radius:12px; padding:24px; text-align:center; margin:16px 0;">
        <h2 style="color:{result_color}; margin:0;">{result_label}</h2>
        <p style="color:#ccd6f6; margin:8px 0;">
            Confidence: <b>{confidence:.1%}</b> &nbsp;|&nbsp;
            P(Normal) = {proba[0]:.4f} &nbsp;|&nbsp; P(Fault) = {proba[1]:.4f}
        </p>
        <p style="color:#8892b0; font-size:0.85rem;">
            Model: {model_choice}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba[1] * 100,
        title={"text": "Fault Probability (%)", "font": {"color": "#ccd6f6"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
            "bar": {"color": result_color},
            "bgcolor": "#172a45",
            "bordercolor": "#233554",
            "steps": [
                {"range": [0, 40], "color": "#1e4d2b"},
                {"range": [40, 70], "color": "#4d3600"},
                {"range": [70, 100], "color": "#4d1e1e"},
            ],
            "threshold": {"line": {"color": result_color, "width": 3}, "value": 50},
        },
        number={"font": {"color": result_color}},
    ))
    fig.update_layout(paper_bgcolor="#172a45", font=dict(color="#ccd6f6"),
                      height=300, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Sensor history chart
    reading = {
        "rms_vibration": rms_vib, "temperature": temp, "rpm": rpm,
        "prediction": prediction, "confidence": confidence,
    }
    st.session_state["sensor_history"].append(reading)
    if len(st.session_state["sensor_history"]) > 30:
        st.session_state["sensor_history"] = st.session_state["sensor_history"][-30:]

    hist = pd.DataFrame(st.session_state["sensor_history"])
    hist["t"] = range(len(hist))
    hist["color"] = hist["prediction"].map({0: TEAL, 1: RED})

    st.markdown("### 📈 Sensor History (last 30 readings)")
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=["RMS Vibration (g)", "Temperature (°C)"])
    fig2.add_trace(
        go.Scatter(x=hist["t"], y=hist["rms_vibration"],
                   mode="lines+markers", name="RMS Vibration",
                   line=dict(color=TEAL), marker=dict(color=hist["color"].tolist())),
        row=1, col=1,
    )
    fig2.add_trace(
        go.Scatter(x=hist["t"], y=hist["temperature"],
                   mode="lines+markers", name="Temperature",
                   line=dict(color=ORANGE), marker=dict(color=hist["color"].tolist())),
        row=2, col=1,
    )
    fig2.update_layout(paper_bgcolor="#172a45", plot_bgcolor="#0d1b2a",
                       font=dict(color="#ccd6f6"), legend=dict(bgcolor="#172a45"),
                       margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🟢 Green markers = Normal predictions | 🔴 Red markers = Fault predictions")


# ══════════════════════════════════════════════════════════════════════════════
# Tab F — Privacy & Communication
# ══════════════════════════════════════════════════════════════════════════════
def tab_privacy(cfg: dict):
    st.markdown("## 🔒 Privacy & Communication Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌐 Federated Mode — Data Flow")
        st.markdown("""
        <div class="fed-card">
        <div class="fed-card-title">📍 Raw Data Location</div>
        <hr style="border-color:#233554;">
        Each node's sensor records <b>remain on that node's local storage</b>.<br><br>
        Only <b>model weight updates</b> (floating-point numbers) travel over the network.<br><br>
        The central coordinator <b>never sees raw sensor readings</b> in federated mode.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### ☁️ Centralized Mode — Data Flow")
        st.markdown("""
        <div class="fed-card">
        <div class="fed-card-title">📤 Raw Data Upload</div>
        <hr style="border-color:#233554;">
        Every training record from every node must be <b>transmitted to the central server</b>.<br><br>
        The server holds a <b>complete copy</b> of all operational sensor data.<br><br>
        This creates privacy, security, and bandwidth challenges at scale.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Communication breakdown
    if st.session_state["experiment_done"]:
        results = st.session_state["results"]
        comm = results["communication"]

        st.markdown("### 📊 Communication Cost Breakdown")
        col3, col4, col5 = st.columns(3)
        col3.metric("Centralized Data Upload", format_bytes(comm["centralized_bytes"]))
        col4.metric("Federated Total (model updates)", format_bytes(comm["fed_bytes"]))
        col5.metric("Communication Savings", f"{comm['savings_pct']:.1f}%")

        # Waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Communication Bytes",
            orientation="v",
            x=["Centralized Baseline", "Saved by Federated", "Federated Cost"],
            y=[comm["centralized_bytes"], -(comm["centralized_bytes"] - comm["fed_bytes"]),
               comm["fed_bytes"]],
            connector={"line": {"color": "#233554"}},
            decreasing={"marker": {"color": TEAL}},
            increasing={"marker": {"color": RED}},
            totals={"marker": {"color": BLUE}},
        ))
        fig.update_layout(title="Communication Savings Waterfall",
                          yaxis_title="Bytes", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚠️ Privacy Limitations")
    st.markdown("""
    <div class="warn-box">
    <b>Important Disclaimer:</b><br>
    Federated learning reduces the transmission of raw data, but it does <b>NOT</b> provide
    cryptographic privacy guarantees on its own. Model updates (gradients or weights) can
    potentially leak information about training data through techniques like
    gradient inversion attacks.<br><br>
    <b>Stronger privacy protections</b> not implemented here include:<br>
    • <b>Differential Privacy</b> — adding calibrated noise to model updates<br>
    • <b>Secure Aggregation</b> — cryptographic protocols so the server only sees the aggregate<br>
    • <b>Homomorphic Encryption</b> — computation on encrypted weights<br><br>
    This project demonstrates the <i>communication efficiency</i> and <i>data locality</i>
    benefits of federated learning at the educational level.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📐 Communication Methodology")
    st.markdown(f"""
    All byte counts are **estimated application-level payload bytes**.
    TCP/IP overhead, TLS, and compression are excluded.

    | Mode | What is transferred | Bytes per unit |
    |------|--------------------|-----------------------|
    | Centralized | Raw CSV sensor record | {80} bytes/record |
    | Federated | Float64 model weight | 8 bytes/parameter |

    **Centralized bytes** = total_training_records × 80

    **Federated bytes** = (n_nodes × model_bytes × n_rounds × 2) + initial_broadcast

    These are conservative estimates. Actual network costs depend on serialization format,
    compression, and protocol overhead.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Tab G — Full Demo (one-click)
# ══════════════════════════════════════════════════════════════════════════════
def tab_demo(cfg: dict):
    st.markdown("## 🎬 Full Demonstration Workflow")
    st.markdown("""
    Click the button below to run the **complete end-to-end experiment** in one step.
    This generates data, trains both models, compares results, and produces a summary.
    """)

    st.markdown("""
    <div class="info-box">
    <b>Demonstration steps:</b><br>
    1️⃣ Generate synthetic sensor data &nbsp;
    2️⃣ Display edge node distributions &nbsp;
    3️⃣ Run centralized training &nbsp;
    4️⃣ Run federated training (FedAvg) &nbsp;
    5️⃣ Compare models &nbsp;
    6️⃣ Show communication savings &nbsp;
    7️⃣ Run sample predictions &nbsp;
    8️⃣ Generate summary
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    **Configuration:**  
    Nodes = `{cfg['num_nodes']}` | Rounds = `{cfg['n_rounds']}` |  
    Local Epochs = `{cfg['local_epochs']}` | Samples = `{cfg['total_samples']}` | Seed = `{cfg['seed']}`
    """)

    if st.button("▶️ Run Full Demonstration", type="primary", key="btn_demo"):
        log_container = st.empty()
        progress = st.progress(0)
        log_lines = []

        def demo_log(msg):
            log_lines.append(msg)
            log_container.code("\n".join(log_lines[-30:]), language="")

        demo_log("🎬 FedEdge-IoT Full Demonstration Starting …")
        progress.progress(10)

        results = run_full_experiment(
            num_nodes=cfg["num_nodes"],
            n_rounds=cfg["n_rounds"],
            local_epochs=cfg["local_epochs"],
            total_samples=cfg["total_samples"],
            fault_rates=cfg["fault_rates"],
            seed=cfg["seed"],
            progress_callback=demo_log,
            save_results=True,
        )

        st.session_state.update({
            "results": results,
            "experiment_done": True,
            "global_model": results["global_model"],
            "central_model": results["central_model"],
            "scaler": results["scaler"],
            "nodes": results["nodes"],
            "node_datasets": {
                f"node_{i}": None for i in range(cfg["num_nodes"])
            },
        })

        # Regenerate datasets for display
        node_datasets = generate_node_datasets(
            num_nodes=cfg["num_nodes"],
            total_samples=cfg["total_samples"],
            fault_rates=cfg["fault_rates"],
            seed=cfg["seed"],
        )
        st.session_state["node_datasets"] = node_datasets
        progress.progress(100)
        demo_log("✅ Demonstration complete!")

    if st.session_state["experiment_done"] and st.session_state["results"]:
        results = st.session_state["results"]
        fm = results["federated_metrics"]
        cm_res = results["central_metrics"]
        comm = results["communication"]

        st.markdown("---")
        st.markdown("## 📋 Experiment Summary")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Federated Accuracy", f"{fm['accuracy']:.4f}")
        c2.metric("Centralized Accuracy", f"{cm_res['accuracy']:.4f}")
        c3.metric("Federated F1", f"{fm['f1_score']:.4f}")
        c4.metric("Communication Savings", f"{comm['savings_pct']:.1f}%")

        # Sample predictions
        st.markdown("### 🔮 Sample Predictions")
        scaler = st.session_state["scaler"]
        model = st.session_state["global_model"]

        samples = {
            "Normal Reading": [0.75, 1.1, 0.04, 29.5, 63.0, 1810],
            "Fault Reading":  [2.8,  4.5, 0.38, 48.0, 87.5, 1520],
        }

        for label, values in samples.items():
            X = np.array([values])
            X_sc = scaler.transform(X)
            pred = model.predict(X_sc)[0]
            proba = model.predict_proba(X_sc)[0]
            pred_label = "🟢 Normal" if pred == 0 else "🔴 Fault"
            st.markdown(f"**{label}** → Predicted: {pred_label} (confidence: {proba[pred]:.1%})")

        # Communication summary
        st.markdown("### 📡 Communication Summary")
        st.markdown(f"""
        | Mode | Total Bytes | Details |
        |------|------------|---------|
        | **Centralized** | {format_bytes(comm['centralized_bytes'])} | Raw sensor records uploaded |
        | **Federated** | {format_bytes(comm['fed_bytes'])} | Model updates only |
        | **Savings** | **{comm['savings_pct']:.1f}%** | Reduction in communication |
        """)

        st.markdown("### 🔑 Key Takeaways")
        st.markdown("""
        1. **Data stays local**: Raw sensor records never left individual nodes in federated mode
        2. **Communication efficiency**: Federated transmits only model parameters, not sensor records
        3. **Performance trade-off**: Small accuracy differences are expected between the two approaches
        4. **Scalability**: As dataset size grows, federated communication savings increase
        5. **Non-IID challenge**: Heterogeneous data across nodes is a real federated learning challenge
        """)


# ══════════════════════════════════════════════════════════════════════════════
# Main — Tab layout
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cfg = render_sidebar()

    tabs = st.tabs([
        "📖 Project Overview",
        "🖥️ Edge Node Simulator",
        "🔗 Federated Training",
        "⚖️ Comparison",
        "🔍 Live Prediction",
        "🔒 Privacy & Comm.",
        "🎬 Full Demo",
    ])

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_nodes(cfg)
    with tabs[2]:
        tab_federated(cfg)
    with tabs[3]:
        tab_comparison()
    with tabs[4]:
        tab_prediction(cfg)
    with tabs[5]:
        tab_privacy(cfg)
    with tabs[6]:
        tab_demo(cfg)


if __name__ == "__main__":
    main()

