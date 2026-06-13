"""
streamlit_app.py
----------------
Professional Streamlit dashboard for the Fraud Detection System.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1rem; text-align: center;
        border: 1px solid #dee2e6;
    }
    .fraud-alert {
        background: #ffe3e3; border-left: 5px solid #e74c3c;
        padding: 1rem; border-radius: 8px;
        font-size: 1.3rem; font-weight: 700; color: #c0392b;
    }
    .safe-alert {
        background: #d4edda; border-left: 5px solid #27ae60;
        padding: 1rem; border-radius: 8px;
        font-size: 1.3rem; font-weight: 700; color: #1e8449;
    }
    .risk-label { font-size: 0.85rem; color: #6c757d; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ── Load utilities (lazy — avoids error if models not yet trained) ─────────────
@st.cache_resource
def load_predict():
    from predict import predict, get_feature_names
    return predict, get_feature_names()

@st.cache_resource
def load_explainer():
    from shap_explainer import explain_transaction, plot_shap_bar
    return explain_transaction, plot_shap_bar


def risk_gauge(score: float) -> plt.Figure:
    """Draw a simple semicircular risk gauge using matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})

    # Background arc (grey)
    theta = np.linspace(np.pi, 0, 200)
    ax.fill_between(theta, 0.6, 1.0, color="#ecf0f1", zorder=1)

    # Coloured arc (red portion based on score)
    theta_score = np.linspace(np.pi, np.pi - score * np.pi, 200)
    color = "#e74c3c" if score > 0.5 else ("#f39c12" if score > 0.25 else "#27ae60")
    ax.fill_between(theta_score, 0.6, 1.0, color=color, zorder=2, alpha=0.9)

    ax.set_ylim(0, 1.2)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.axis("off")
    ax.text(0, -0.05, f"{score:.0%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color,
            transform=ax.transData)
    ax.set_title("Risk Score", fontsize=11, pad=0)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/fraud.png", width=70)
    st.title("⚙️ Controls")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home", "🔍 Predict", "📊 Dashboard", "ℹ️ About"])
    st.markdown("---")
    st.caption("Fraud Detection System v1.0\nPowered by XGBoost + SHAP")


# ── HOME PAGE ─────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="main-title">🛡️ Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown("#### End-to-end ML pipeline for real-time credit card fraud detection")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size",  "284,807", "transactions")
    with col2:
        st.metric("Fraud Cases",   "492",     "0.17% of total")
    with col3:
        st.metric("Model",         "XGBoost", "best F1-score")
    with col4:
        st.metric("Explainability","SHAP",    "feature-level")

    st.markdown("---")
    st.markdown("### 🚀 How It Works")
    cols = st.columns(3)
    steps = [
        ("1️⃣ Input", "Enter transaction features in the **Predict** tab or call the REST API."),
        ("2️⃣ Predict", "The model returns a **fraud probability score** (0–1)."),
        ("3️⃣ Explain", "SHAP shows **which features** drove the decision and by how much."),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.info(f"**{title}**\n\n{desc}")


# ── PREDICT PAGE ──────────────────────────────────────────────────────────────
elif page == "🔍 Predict":
    st.header("🔍 Transaction Fraud Predictor")
    st.markdown("Fill in the transaction details below and click **Predict**.")

    try:
        predict_fn, feature_names = load_predict()
    except Exception as e:
        st.error(f"❌ Model not found. Please run `python src/train_model.py` first.\n\n{e}")
        st.stop()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("Transaction Details")
        col1, col2 = st.columns(2)
        with col1:
            time_val   = st.number_input("Time (seconds since first tx)", value=3600.0, step=1.0)
            amount_val = st.number_input("Amount (€)", value=50.0, min_value=0.0, step=0.01)

        st.markdown("**V-Features** (PCA anonymised — set to 0 if unknown)")
        v_cols = st.columns(4)
        v_vals = {}
        for i in range(1, 29):
            col = v_cols[(i - 1) % 4]
            v_vals[f"V{i}"] = col.number_input(f"V{i}", value=0.0, step=0.01,
                                                format="%.4f", key=f"v{i}")

        submitted = st.form_submit_button("🔍 Predict Fraud", use_container_width=True)

    if submitted:
        transaction = {"Time": time_val, "Amount": amount_val, **v_vals}

        with st.spinner("Analysing transaction ..."):
            result = predict_fn(transaction)

        score      = result["risk_score"]
        prediction = result["prediction"]
        is_fraud   = result["label"] == 1

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        col_a, col_b, col_c = st.columns([2, 1, 2])
        with col_a:
            if is_fraud:
                st.markdown(f'<div class="fraud-alert">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-alert">✅ NORMAL TRANSACTION</div>', unsafe_allow_html=True)
        with col_b:
            st.pyplot(risk_gauge(score), use_container_width=True)
        with col_c:
            st.metric("Fraud Probability", f"{score:.2%}")
            risk_level = "🔴 HIGH" if score > 0.7 else ("🟡 MEDIUM" if score > 0.3 else "🟢 LOW")
            st.metric("Risk Level", risk_level)

        # SHAP explanation
        st.markdown("---")
        st.subheader("🧠 Why this prediction? (SHAP Explanation)")
        try:
            explain_fn, plot_fn = load_explainer()
            with st.spinner("Computing SHAP values ..."):
                explanation = explain_fn(transaction)

            fig = plot_fn(explanation)
            st.pyplot(fig, use_container_width=True)

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("##### 🔴 Top Fraud Drivers")
                for feat, val in explanation["top_positive"]:
                    st.write(f"**{feat}** → `{val:+.4f}`")
            with col_neg:
                st.markdown("##### 🔵 Top Normal Drivers")
                for feat, val in explanation["top_negative"]:
                    st.write(f"**{feat}** → `{val:+.4f}`")

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")


# ── DASHBOARD PAGE ────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.header("📊 Model Performance Dashboard")

    metrics_path = os.path.join("models", "model_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        st.subheader("Model Comparison")
        st.dataframe(df.style.highlight_max(
            subset=["F1", "ROC-AUC", "Recall"], color="#d4edda"
        ).highlight_min(subset=["Precision"], color="#fff3cd"), use_container_width=True)

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(df))
        width = 0.2
        for i, metric in enumerate(["Precision", "Recall", "F1", "ROC-AUC"]):
            ax.bar(x + i * width, df[metric], width, label=metric)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df["Model"])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title("Model Metrics Comparison")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Run `python src/train_model.py` to generate metrics.")

    # Evaluation plots
    st.subheader("Evaluation Plots")
    plot_col1, plot_col2 = st.columns(2)
    for col, fname, title in [
        (plot_col1, "confusion_matrix.png", "Confusion Matrix"),
        (plot_col2, "roc_curve.png",        "ROC Curve"),
    ]:
        path = os.path.join("screenshots", fname)
        with col:
            if os.path.exists(path):
                st.image(path, caption=title)
            else:
                st.info(f"{title} not found. Run evaluate_model.py.")


# ── ABOUT PAGE ────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    st.markdown("""
    ## Fraud Detection System

    An end-to-end machine learning pipeline that:
    - Detects fraudulent credit card transactions in real time
    - Provides a fraud risk probability score (0–1)
    - Explains predictions with SHAP feature contributions
    - Exposes a REST API via FastAPI

    ### Technologies
    `Python` • `Scikit-learn` • `XGBoost` • `SHAP` • `Streamlit` • `FastAPI` • `SMOTE`

    ### Dataset
    Kaggle Credit Card Fraud Detection (284,807 transactions, 0.17% fraud)

    ### Author
    Built as a portfolio ML project demonstrating end-to-end model development,
    class imbalance handling, and explainability.
    """)
