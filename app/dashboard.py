"""
dashboard.py
------------
Standalone chart-generation helpers used by the Streamlit dashboard.
Can also be run directly to regenerate all EDA charts.

Run:
    python app/dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PLOTS_DIR       = os.path.join("screenshots")
PROCESSED_PATH  = os.path.join("data", "processed", "creditcard_processed.csv")


def _save(fig: plt.Figure, name: str) -> str:
    """Save a figure to the screenshots folder and return the path."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {path}")
    return path


def plot_class_distribution(df: pd.DataFrame) -> str:
    """Bar chart showing Fraud vs Normal transaction counts."""
    counts = df["Class"].value_counts().rename({0: "Normal", 1: "Fraud"})

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3498db", "#e74c3c"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    ax.set_title("Transaction Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,}", ha="center", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    return _save(fig, "class_distribution.png")


def plot_amount_distribution(df: pd.DataFrame) -> str:
    """Overlapping KDE plots of Amount for Fraud vs Normal."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(0, "#3498db", "Normal"), (1, "#e74c3c", "Fraud")]:
        subset = df[df["Class"] == label]["Amount"]
        sns.kdeplot(subset, ax=ax, fill=True, alpha=0.4, color=color, label=name)
    ax.set_xlabel("Transaction Amount (€)")
    ax.set_title("Amount Distribution: Normal vs Fraud", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return _save(fig, "amount_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Heatmap of feature correlations with the target (Class)."""
    corr = df.corr()["Class"].drop("Class").sort_values()

    fig, ax = plt.subplots(figsize=(4, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr.values.reshape(-1, 1),
        annot=True, fmt=".2f",
        yticklabels=corr.index,
        xticklabels=["Corr w/ Class"],
        cmap=cmap, center=0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation with Target", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "correlation_heatmap.png")


def plot_fraud_percentage(df: pd.DataFrame) -> str:
    """Pie chart showing fraud vs normal percentage."""
    counts = df["Class"].value_counts()
    labels = ["Normal (99.83%)", "Fraud (0.17%)"]
    colors = ["#3498db", "#e74c3c"]
    explode = (0, 0.1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts.values, labels=labels, colors=colors, explode=explode,
           autopct="%1.2f%%", startangle=90, textprops={"fontsize": 11})
    ax.set_title("Fraud vs Normal Transaction Ratio", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "fraud_percentage.png")


def generate_all_charts(df: pd.DataFrame | None = None) -> list[str]:
    """Generate and save all EDA charts. Returns list of saved paths."""
    if df is None:
        if not os.path.exists(PROCESSED_PATH):
            raise FileNotFoundError(
                "Processed data not found. Run `python src/train_model.py` first."
            )
        df = pd.read_csv(PROCESSED_PATH)

    paths = [
        plot_class_distribution(df),
        plot_amount_distribution(df),
        plot_correlation_heatmap(df),
        plot_fraud_percentage(df),
    ]
    return paths


if __name__ == "__main__":
    print("[INFO] Generating all EDA dashboard charts ...")
    paths = generate_all_charts()
    print(f"[INFO] Generated {len(paths)} charts in {PLOTS_DIR}/")
