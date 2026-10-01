"""
CommuteCast - Stage 2: Exploratory Visualizations
Module: src/visualizations.py

Produces 5 presentation-ready figures suitable for academic/college presentation:
1. Traffic volume histogram (with mean, median, and bimodal distribution markers)
2. Average traffic volume by hour (revealing morning and evening peak commute hours)
3. Congestion category distribution (quantile-based Low, Medium, High breakdown)
4. Traffic volume during rain vs no rain (comparative distribution & statistics)
5. Traffic volume by hour with observed hourly averages (scatter/spread with mean curve)

Also creates a combined 5-panel Executive Dashboard figure.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is first in sys.path and remove 'src' from sys.path
# to avoid shadowing Python's standard library modules (such as 'statistics')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use non-interactive Agg backend to avoid GUI hang in background processes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_traffic_data, PROCESSED_DATA_PATH

# Output directory for presentation figures
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# College presentation aesthetic styling
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


def plot_traffic_volume_histogram(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Figure 1: Traffic volume distribution histogram with KDE, Mean, and Median.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tv = df["traffic_volume"]
    mean_val = tv.mean()
    median_val = tv.median()
    
    sns.histplot(
        tv,
        bins=50,
        kde=True,
        color="#2b5c8f",
        edgecolor="#1a365d",
        alpha=0.65,
        ax=ax,
        stat="density"
    )
    
    # Statistical reference lines
    ax.axvline(mean_val, color="#e53e3e", linestyle="--", linewidth=2.2,
               label=f"Sample Mean (E[X]): {mean_val:.1f}")
    ax.axvline(median_val, color="#38a169", linestyle="-.", linewidth=2.2,
               label=f"Median (Q50): {median_val:.1f}")
    
    ax.set_title("1. Interstate Traffic Volume Distribution (Bimodal Behavior)", pad=15)
    ax.set_xlabel("Traffic Volume (Vehicles / Hour)")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e0", loc="upper right")
    
    # Text annotation explaining bimodality
    ax.text(
        0.03, 0.78,
        "Key Observation:\n"
        "Bimodal distribution reflecting:\n"
        "• Overnight lull (< 1,500 veh/hr)\n"
        "• Daytime active commute (3,000–6,000 veh/hr)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7fafc", edgecolor="#cbd5e0", alpha=0.95)
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_avg_traffic_by_hour(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Figure 2: Average traffic volume by hour of the day (0-23).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
    hourly_avg = df.groupby("hour")["traffic_volume"].mean().reset_index()
    
    # Colors: highlight morning peak (7-8) and evening peak (16-17)
    colors = []
    for h in hourly_avg["hour"]:
        if h in [7, 8]:
            colors.append("#dd6b20")  # Morning peak orange
        elif h in [16, 17]:
            colors.append("#c53030")  # Evening peak red
        elif 9 <= h <= 15:
            colors.append("#3182ce")  # Daytime blue
        else:
            colors.append("#718096")  # Night gray
            
    bars = ax.bar(hourly_avg["hour"], hourly_avg["traffic_volume"], color=colors, edgecolor="#2d3748", width=0.72)
    
    # Peak labels
    morning_peak = hourly_avg.loc[hourly_avg["hour"] == 7, "traffic_volume"].values[0]
    evening_peak = hourly_avg.loc[hourly_avg["hour"] == 16, "traffic_volume"].values[0]
    
    ax.annotate(f"Morning Peak: {morning_peak:.0f}",
                xy=(7, morning_peak), xytext=(3, morning_peak + 550),
                arrowprops=dict(facecolor="#dd6b20", shrink=0.08, width=1.5, headwidth=6),
                fontsize=10, fontweight="bold", color="#9c4221")
    
    ax.annotate(f"Evening Peak: {evening_peak:.0f}",
                xy=(16, evening_peak), xytext=(17, evening_peak + 550),
                arrowprops=dict(facecolor="#c53030", shrink=0.08, width=1.5, headwidth=6),
                fontsize=10, fontweight="bold", color="#9b2c2c")
    
    ax.set_title("2. Average Hourly Traffic Volume (Diurnal Commute Profile)", pad=15)
    ax.set_xlabel("Hour of the Day (0:00 to 23:00)")
    ax.set_ylabel("Empirical Average Traffic Volume (Vehicles / Hour)")
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.set_ylim(0, 6000)
    ax.grid(True, axis="y")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_congestion_distribution(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Figure 3: Congestion category distribution (Quantile-based Low, Medium, High).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    counts = df["congestion"].value_counts().reindex(["Low", "Medium", "High"])
    pcts = (df["congestion"].value_counts(normalize=True) * 100).reindex(["Low", "Medium", "High"])
    
    palette = ["#38a169", "#d69e2e", "#e53e3e"]
    bars = ax.bar(["Low", "Medium", "High"], counts.values, color=palette, edgecolor="#2d3748", width=0.55)
    
    # Annotate bar counts and percentages
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 400,
                f"{count:,}\n({pct:.2f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    q25 = df["traffic_volume"].quantile(0.25)
    q75 = df["traffic_volume"].quantile(0.75)
    
    ax.set_title("3. Congestion Category Distribution (Quantile-Based)", pad=15)
    ax.set_xlabel("Congestion Category")
    ax.set_ylabel("Number of Hourly Observations")
    ax.set_ylim(0, 28000)
    ax.grid(True, axis="y")
    
    # Explanatory caption box
    ax.text(
        0.5, 0.88,
        f"Quantile Thresholds:\n"
        f"• Low: Traffic <= {q25:.1f} (Bottom 25%)\n"
        f"• Medium: {q25:.1f} < Traffic <= {q75:.1f} (Middle 50%)\n"
        f"• High: Traffic > {q75:.1f} (Top 25%)",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7fafc", edgecolor="#cbd5e0", alpha=0.95)
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_traffic_rain_vs_norain(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Figure 4: Comparative distribution of traffic volume during Rain vs No Rain.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    no_rain_vol = df[df["rain"] == 0]["traffic_volume"]
    rain_vol = df[df["rain"] == 1]["traffic_volume"]
    
    box_data = [no_rain_vol, rain_vol]
    labels = [
        f"No Rain (rain = 0)\nN = {len(no_rain_vol):,}\nMean = {no_rain_vol.mean():.1f}",
        f"Rain Present (rain = 1)\nN = {len(rain_vol):,}\nMean = {rain_vol.mean():.1f}"
    ]
    
    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        tick_labels=labels,
        widths=0.45,
        medianprops=dict(color="#1a202c", linewidth=2.5),
        meanprops=dict(marker="o", markeredgecolor="#e53e3e", markerfacecolor="#e53e3e", markersize=8),
        showmeans=True
    )
    
    colors = ["#bee3f8", "#63b3ed"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#2b6cb0")
        
    ax.set_title("4. Traffic Volume During Rain vs. No Rain Conditions", pad=15)
    ax.set_ylabel("Traffic Volume (Vehicles / Hour)")
    ax.grid(True, axis="y")
    
    # Explanatory statistical note
    ax.text(
        0.5, 0.12,
        "Statistical Note: Red circles represent sample means (E[X]).\n"
        "Rain observations have slightly higher sample mean (3,373 vs 3,254 veh/hr)\n"
        "because convective rain storms disproportionately occur during daytime peak commute.",
        transform=ax.transAxes,
        fontsize=9.5,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7fafc", edgecolor="#cbd5e0", alpha=0.95)
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_hourly_traffic_with_observed_averages(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Figure 5: Traffic volume by hour with observed hourly averages (empirical expectation line).
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    # Calculate hourly quantiles and means
    hourly_stats = df.groupby("hour")["traffic_volume"].agg(
        mean="mean",
        q25=lambda x: x.quantile(0.25),
        median="median",
        q75=lambda x: x.quantile(0.75),
        q05=lambda x: x.quantile(0.05),
        q95=lambda x: x.quantile(0.95)
    ).reset_index()
    
    # Draw spread band (5th to 95th percentile)
    ax.fill_between(
        hourly_stats["hour"],
        hourly_stats["q05"],
        hourly_stats["q95"],
        color="#cbd5e0",
        alpha=0.4,
        label="5th–95th Percentile Spread"
    )
    
    # Draw IQR band (25th to 75th percentile)
    ax.fill_between(
        hourly_stats["hour"],
        hourly_stats["q25"],
        hourly_stats["q75"],
        color="#90cdf4",
        alpha=0.55,
        label="Interquartile Range (Q25–Q75)"
    )
    
    # Plot empirical expectation line (sample mean)
    ax.plot(
        hourly_stats["hour"],
        hourly_stats["mean"],
        color="#c53030",
        linewidth=3.0,
        marker="o",
        markersize=6,
        label="Observed Hourly Average Mean (E[Traffic|Hour])"
    )
    
    # Plot median line
    ax.plot(
        hourly_stats["hour"],
        hourly_stats["median"],
        color="#2b6cb0",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=5,
        label="Observed Hourly Median (Q50|Hour)"
    )
    
    ax.set_title("5. Hourly Traffic Volume Spread with Observed Empirical Averages", pad=15)
    ax.set_xlabel("Hour of the Day (0:00 to 23:00)")
    ax.set_ylabel("Traffic Volume (Vehicles / Hour)")
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.set_ylim(0, 7500)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e0", loc="upper left")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def generate_executive_dashboard(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Combined Executive 5-Panel Dashboard synthesizing all Stage 2 visual analyses.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25)
    
    # Subplot 1: Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    tv = df["traffic_volume"]
    sns.histplot(tv, bins=40, kde=True, color="#2b5c8f", edgecolor="#1a365d", alpha=0.6, ax=ax1, stat="density")
    ax1.axvline(tv.mean(), color="#e53e3e", linestyle="--", linewidth=1.8, label=f"Mean: {tv.mean():.0f}")
    ax1.axvline(tv.median(), color="#38a169", linestyle="-.", linewidth=1.8, label=f"Median: {tv.median():.0f}")
    ax1.set_title("1. Volume Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Vehicles / Hour")
    ax1.set_ylabel("Density")
    ax1.grid(True)
    ax1.legend(fontsize=8, loc="upper right")
    
    # Subplot 2: Hourly Average
    ax2 = fig.add_subplot(gs[0, 1])
    hourly_avg = df.groupby("hour")["traffic_volume"].mean().reset_index()
    colors = ["#dd6b20" if h in [7, 8] else "#c53030" if h in [16, 17] else "#3182ce" for h in hourly_avg["hour"]]
    ax2.bar(hourly_avg["hour"], hourly_avg["traffic_volume"], color=colors, edgecolor="#2d3748", width=0.7)
    ax2.set_title("2. Average Volume by Hour", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Hour (0-23)")
    ax2.set_ylabel("Avg Vehicles / Hour")
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, axis="y")
    
    # Subplot 3: Congestion Breakdown
    ax3 = fig.add_subplot(gs[0, 2])
    counts = df["congestion"].value_counts().reindex(["Low", "Medium", "High"])
    pcts = (df["congestion"].value_counts(normalize=True) * 100).reindex(["Low", "Medium", "High"])
    bars = ax3.bar(["Low", "Medium", "High"], counts.values, color=["#38a169", "#d69e2e", "#e53e3e"], width=0.55, edgecolor="#2d3748")
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 500, f"{count:,}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax3.set_title("3. Congestion Classification", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Observations")
    ax3.set_ylim(0, 28000)
    ax3.grid(True, axis="y")
    
    # Subplot 4: Rain vs No Rain
    ax4 = fig.add_subplot(gs[1, 0])
    no_rain = df[df["rain"] == 0]["traffic_volume"]
    rain = df[df["rain"] == 1]["traffic_volume"]
    bp = ax4.boxplot([no_rain, rain], patch_artist=True, tick_labels=["No Rain", "Rain"], widths=0.45,
                     medianprops=dict(color="#1a202c", linewidth=2),
                     meanprops=dict(marker="o", markeredgecolor="#e53e3e", markerfacecolor="#e53e3e"),
                     showmeans=True)
    bp["boxes"][0].set_facecolor("#bee3f8")
    bp["boxes"][1].set_facecolor("#63b3ed")
    ax4.set_title("4. Traffic: Rain vs. No Rain", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Vehicles / Hour")
    ax4.grid(True, axis="y")
    
    # Subplot 5: Hourly Spread & Expectation Line (Spanning 2 columns)
    ax5 = fig.add_subplot(gs[1, 1:])
    h_stats = df.groupby("hour")["traffic_volume"].agg(
        mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), median="median"
    ).reset_index()
    ax5.fill_between(h_stats["hour"], h_stats["q25"], h_stats["q75"], color="#90cdf4", alpha=0.5, label="Interquartile Range (Q25-Q75)")
    ax5.plot(h_stats["hour"], h_stats["mean"], color="#c53030", linewidth=2.5, marker="o", label="Observed Mean E[Traffic|Hour]")
    ax5.plot(h_stats["hour"], h_stats["median"], color="#2b6cb0", linewidth=1.8, linestyle="--", label="Observed Median")
    ax5.set_title("5. Hourly Spread with Empirical Expectation Curve", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Hour of the Day (0-23)")
    ax5.set_ylabel("Vehicles / Hour")
    ax5.set_xticks(range(0, 24))
    ax5.grid(True)
    ax5.legend(fontsize=9, loc="upper left")
    
    fig.suptitle("CommuteCast - Stage 2: Exploratory Statistical Analysis Dashboard", fontsize=16, y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_stage3_prior_probabilities(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Stage 3 Figure 1: Marginal Prior Probabilities P(Low), P(Medium), P(High).
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    counts = df["congestion"].value_counts().reindex(["Low", "Medium", "High"])
    probs = (counts / len(df)).values
    
    palette = ["#38a169", "#d69e2e", "#e53e3e"]
    bars = ax.bar(["Low", "Medium", "High"], probs, color=palette, edgecolor="#2d3748", width=0.52)
    
    for bar, prob, count in zip(bars, probs, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            prob + 0.015,
            f"P = {prob:.4f}\n({prob*100:.2f}%)\n[n={count:,}]",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold"
        )
        
    ax.set_title("1. Prior Probabilities of Congestion Regimes: P(A) = count(A) / N", pad=15)
    ax.set_xlabel("Congestion State")
    ax.set_ylabel("Probability P(State)")
    ax.set_ylim(0, 0.65)
    ax.grid(True, axis="y")
    
    ax.text(
        0.5, 0.88,
        f"Verification: P(Low) + P(Medium) + P(High) = {probs.sum():.6f} ≈ 1.0",
        transform=ax.transAxes, ha="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7fafc", edgecolor="#cbd5e0")
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def plot_stage3_rain_conditional_high(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Stage 3 Figure 2: P(High | Rain) vs P(High | No Rain) with baseline P(High).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    rain_mask = (df["rain"] == 1)
    norain_mask = (df["rain"] == 0)
    high_mask = (df["congestion"] == "High")
    
    p_high_rain = (high_mask & rain_mask).sum() / rain_mask.sum()
    p_high_norain = (high_mask & norain_mask).sum() / norain_mask.sum()
    p_high_baseline = high_mask.mean()
    
    labels = ["P(High | Rain)\n[Rain = 1]", "P(High | No Rain)\n[Rain = 0]"]
    values = [p_high_rain, p_high_norain]
    colors = ["#3182ce", "#a0aec0"]
    
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d3748", width=0.48)
    
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            val + 0.008,
            f"{val:.4f}\n({val*100:.2f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
        
    ax.axhline(p_high_baseline, color="#e53e3e", linestyle="--", linewidth=2.0,
               label=f"Baseline P(High) = {p_high_baseline:.4f} ({p_high_baseline*100:.2f}%)")
    
    ax.set_title("2. Conditional Probability of High Congestion Given Rain", pad=15)
    ax.set_ylabel("Probability P(High Congestion | Condition)")
    ax.set_ylim(0, 0.38)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    
    ax.text(
        0.5, 0.72,
        f"Independence Check:\n"
        f"P(High | Rain) = {p_high_rain:.4f} vs P(High) = {p_high_baseline:.4f} (diff = +{p_high_rain - p_high_baseline:.4f})\n"
        f"Rain slightly elevates the probability of High Congestion (+2.02%).",
        transform=ax.transAxes, ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7fafc", edgecolor="#cbd5e0")
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def plot_stage3_rain_expected_traffic(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Stage 3 Figure 3: Empirical Expected Traffic Volume E(Traffic | Rain) vs E(Traffic | No Rain).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    e_rain = df.loc[df["rain"] == 1, "traffic_volume"].mean()
    e_norain = df.loc[df["rain"] == 0, "traffic_volume"].mean()
    e_overall = df["traffic_volume"].mean()
    
    labels = ["E(Traffic | Rain)\n[Rain = 1]", "E(Traffic | No Rain)\n[Rain = 0]"]
    values = [e_rain, e_norain]
    colors = ["#2b6cb0", "#718096"]
    
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d3748", width=0.48)
    
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            val + 70,
            f"{val:.1f} veh/hr",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
        
    ax.axhline(e_overall, color="#e53e3e", linestyle="--", linewidth=2.0,
               label=f"Overall Expected Volume E[Traffic] = {e_overall:.1f} veh/hr")
    
    ax.set_title("3. Empirical Expected Traffic Volume: Rain vs. No Rain", pad=15)
    ax.set_ylabel("Expected Hourly Traffic (Vehicles / Hour)")
    ax.set_ylim(0, 4200)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    
    ax.text(
        0.5, 0.72,
        f"Expectation Analysis:\n"
        f"E(Traffic | Rain) = {e_rain:.1f} veh/hr vs E(Traffic | No Rain) = {e_norain:.1f} veh/hr\n"
        f"Difference: +{e_rain - e_norain:.1f} veh/hr (+1.07%).\n"
        f"Note: Confounded by daytime storm prevalence; does NOT prove causation.",
        transform=ax.transAxes, ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7fafc", edgecolor="#cbd5e0")
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def plot_stage3_peak_conditional_high(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Stage 3 Figure 4: P(High | Peak) vs P(High | Non-Peak) with baseline P(High).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    peak_mask = df["hour"].isin([7, 8, 16, 17])
    high_mask = (df["congestion"] == "High")
    
    p_high_peak = (high_mask & peak_mask).sum() / peak_mask.sum()
    p_high_nonpeak = (high_mask & ~peak_mask).sum() / (~peak_mask).sum()
    p_high_baseline = high_mask.mean()
    
    labels = ["P(High | Peak Hour)\n[7-8 AM, 4-5 PM]", "P(High | Non-Peak)\n[Remaining 20 Hours]"]
    values = [p_high_peak, p_high_nonpeak]
    colors = ["#dd6b20", "#4a5568"]
    
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d3748", width=0.48)
    
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            val + 0.015,
            f"{val:.4f}\n({val*100:.2f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
        
    ax.axhline(p_high_baseline, color="#e53e3e", linestyle="--", linewidth=2.0,
               label=f"Baseline P(High) = {p_high_baseline:.4f} ({p_high_baseline*100:.2f}%)")
    
    ax.set_title("4. Conditional Probability of High Congestion: Peak vs. Non-Peak", pad=15)
    ax.set_ylabel("Probability P(High Congestion | Condition)")
    ax.set_ylim(0, 0.85)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    
    ax.text(
        0.5, 0.68,
        f"Defensible Peak Hours: 7-8 AM & 16-17 PM (Core rush hours)\n"
        f"Peak Hour elevates P(High) by nearly 4x over Non-Peak (65.50% vs 16.82%).\n"
        f"Conclusion: High Congestion and Peak Hours are strongly dependent.",
        transform=ax.transAxes, ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7fafc", edgecolor="#cbd5e0")
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def plot_stage3_weekday_conditional_high(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Stage 3 Figure 5: P(High | Weekday) vs P(High | Weekend).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    weekday_mask = (df["is_weekend"] == 0)
    weekend_mask = (df["is_weekend"] == 1)
    high_mask = (df["congestion"] == "High")
    
    p_high_weekday = (high_mask & weekday_mask).sum() / weekday_mask.sum()
    p_high_weekend = (high_mask & weekend_mask).sum() / weekend_mask.sum()
    p_high_baseline = high_mask.mean()
    
    labels = [
        f"P(High | Weekday)\n[Mon–Fri, n={weekday_mask.sum():,}]",
        f"P(High | Weekend)\n[Sat–Sun, n={weekend_mask.sum():,}]"
    ]
    values = [p_high_weekday, p_high_weekend]
    colors = ["#2c5282", "#b7791f"]
    
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d3748", width=0.48)
    
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            val + 0.008,
            f"{val:.4f}\n({val*100:.2f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
        
    ax.axhline(p_high_baseline, color="#e53e3e", linestyle="--", linewidth=2.0,
               label=f"Baseline P(High) = {p_high_baseline:.4f} ({p_high_baseline*100:.2f}%)")
    
    ax.set_title("5. Conditional Probability of High Congestion: Weekday vs. Weekend", pad=15)
    ax.set_ylabel("Probability P(High Congestion | Condition)")
    ax.set_ylim(0, 0.48)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    
    ax.text(
        0.5, 0.72,
        f"Weekday High Probability (33.67%) is ~11x higher than Weekend (3.09%).\n"
        f"Weekend congestion is almost exclusively Low/Medium.",
        transform=ax.transAxes, ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7fafc", edgecolor="#cbd5e0")
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def plot_stage3_rain_peak_conditional_high(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Stage 3 Figure 6: Joint Condition: P(High | Rain, Peak) vs P(High | No Rain, Peak).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    peak_mask = df["hour"].isin([7, 8, 16, 17])
    rain_mask = (df["rain"] == 1)
    norain_mask = (df["rain"] == 0)
    high_mask = (df["congestion"] == "High")
    
    rain_peak = (rain_mask & peak_mask)
    norain_peak = (norain_mask & peak_mask)
    
    p_high_rain_peak = (high_mask & rain_peak).sum() / rain_peak.sum()
    p_high_norain_peak = (high_mask & norain_peak).sum() / norain_peak.sum()
    p_high_peak = (high_mask & peak_mask).sum() / peak_mask.sum()
    
    labels = [
        f"P(High | Rain, Peak)\n[n={rain_peak.sum():,}]",
        f"P(High | No Rain, Peak)\n[n={norain_peak.sum():,}]"
    ]
    values = [p_high_rain_peak, p_high_norain_peak]
    colors = ["#2b6cb0", "#d69e2e"]
    
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d3748", width=0.48)
    
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            val + 0.015,
            f"{val:.4f}\n({val*100:.2f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
        
    ax.axhline(p_high_peak, color="#e53e3e", linestyle="--", linewidth=2.0,
               label=f"Peak Hour Baseline P(High|Peak) = {p_high_peak:.4f} ({p_high_peak*100:.2f}%)")
    
    ax.set_title("6. Joint Conditional Probability: High Congestion Under Rain + Peak", pad=15)
    ax.set_ylabel("Probability P(High Congestion | Joint Condition)")
    ax.set_ylim(0, 0.88)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    
    ax.text(
        0.5, 0.68,
        f"Joint Analysis Findings:\n"
        f"During Peak Hours, rain raises High Congestion probability from 65.31% to 67.75% (+2.44%).\n"
        f"E(Traffic | Rain, Peak) = 5,053.5 veh/hr vs E(Traffic | No Rain, Peak) = 5,065.4 veh/hr.",
        transform=ax.transAxes, ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7fafc", edgecolor="#cbd5e0")
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def generate_stage3_dashboard(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Combined Stage 3 Probability Dashboard synthesizing all 6 probability visualizations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("CommuteCast - Stage 3: Probability, Expectation & Bayes Analysis Dashboard", fontsize=16, y=0.98, fontweight="bold")
    
    # 1. Priors
    ax1 = axes[0, 0]
    counts = df["congestion"].value_counts().reindex(["Low", "Medium", "High"])
    probs = (counts / len(df)).values
    b1 = ax1.bar(["Low", "Med", "High"], probs, color=["#38a169", "#d69e2e", "#e53e3e"], width=0.5, edgecolor="#2d3748")
    for b, p in zip(b1, probs):
        ax1.text(b.get_x() + b.get_width()/2., p + 0.015, f"{p:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_title("1. Congestion Priors P(A)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 0.62)
    ax1.grid(True, axis="y")
    
    # 2. Rain Conditional
    ax2 = axes[0, 1]
    r_mask = (df["rain"] == 1)
    h_mask = (df["congestion"] == "High")
    p_hr = (h_mask & r_mask).sum() / r_mask.sum()
    p_hnr = (h_mask & ~r_mask).sum() / (~r_mask).sum()
    b2 = ax2.bar(["Rain", "No Rain"], [p_hr, p_hnr], color=["#3182ce", "#a0aec0"], width=0.45, edgecolor="#2d3748")
    for b, p in zip(b2, [p_hr, p_hnr]):
        ax2.text(b.get_x() + b.get_width()/2., p + 0.008, f"{p:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(h_mask.mean(), color="#e53e3e", linestyle="--", linewidth=1.5, label="P(High) Base")
    ax2.set_title("2. P(High | Rain vs No Rain)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.38)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, axis="y")
    
    # 3. Rain Expectation
    ax3 = axes[0, 2]
    e_r = df.loc[r_mask, "traffic_volume"].mean()
    e_nr = df.loc[~r_mask, "traffic_volume"].mean()
    b3 = ax3.bar(["Rain", "No Rain"], [e_r, e_nr], color=["#2b6cb0", "#718096"], width=0.45, edgecolor="#2d3748")
    for b, val in zip(b3, [e_r, e_nr]):
        ax3.text(b.get_x() + b.get_width()/2., val + 80, f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax3.axhline(df["traffic_volume"].mean(), color="#e53e3e", linestyle="--", linewidth=1.5, label="Overall Mean")
    ax3.set_title("3. E(Traffic | Rain vs No Rain)", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 4200)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, axis="y")
    
    # 4. Peak Conditional
    ax4 = axes[1, 0]
    p_mask = df["hour"].isin([7, 8, 16, 17])
    p_hp = (h_mask & p_mask).sum() / p_mask.sum()
    p_hnp = (h_mask & ~p_mask).sum() / (~p_mask).sum()
    b4 = ax4.bar(["Peak", "Non-Peak"], [p_hp, p_hnp], color=["#dd6b20", "#4a5568"], width=0.45, edgecolor="#2d3748")
    for b, p in zip(b4, [p_hp, p_hnp]):
        ax4.text(b.get_x() + b.get_width()/2., p + 0.02, f"{p:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax4.axhline(h_mask.mean(), color="#e53e3e", linestyle="--", linewidth=1.5)
    ax4.set_title("4. P(High | Peak vs Non-Peak)", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 0.85)
    ax4.grid(True, axis="y")
    
    # 5. Weekday Conditional
    ax5 = axes[1, 1]
    w_mask = (df["is_weekend"] == 0)
    p_hw = (h_mask & w_mask).sum() / w_mask.sum()
    p_hwe = (h_mask & ~w_mask).sum() / (~w_mask).sum()
    b5 = ax5.bar(["Weekday", "Weekend"], [p_hw, p_hwe], color=["#2c5282", "#b7791f"], width=0.45, edgecolor="#2d3748")
    for b, p in zip(b5, [p_hw, p_hwe]):
        ax5.text(b.get_x() + b.get_width()/2., p + 0.01, f"{p:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax5.axhline(h_mask.mean(), color="#e53e3e", linestyle="--", linewidth=1.5)
    ax5.set_title("5. P(High | Weekday vs Weekend)", fontsize=11, fontweight="bold")
    ax5.set_ylim(0, 0.48)
    ax5.grid(True, axis="y")
    
    # 6. Joint Rain + Peak
    ax6 = axes[1, 2]
    rp_mask = (r_mask & p_mask)
    nrp_mask = (~r_mask & p_mask)
    p_hrp = (h_mask & rp_mask).sum() / rp_mask.sum()
    p_hnrp = (h_mask & nrp_mask).sum() / nrp_mask.sum()
    b6 = ax6.bar(["Rain & Peak", "No Rain & Peak"], [p_hrp, p_hnrp], color=["#2b6cb0", "#d69e2e"], width=0.45, edgecolor="#2d3748")
    for b, p in zip(b6, [p_hrp, p_hnrp]):
        ax6.text(b.get_x() + b.get_width()/2., p + 0.02, f"{p:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax6.axhline(p_hp, color="#e53e3e", linestyle="--", linewidth=1.5, label="Peak Baseline")
    ax6.set_title("6. P(High | Rain & Peak)", fontsize=11, fontweight="bold")
    ax6.set_ylim(0, 0.88)
    ax6.legend(fontsize=8, loc="upper right")
    ax6.grid(True, axis="y")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def plot_stage4_polynomial_fits(hourly_df: pd.DataFrame, models: Dict[int, Dict[str, Any]], save_path: str = None) -> plt.Figure:
    """
    Stage 4 Figure 12: Comparison of Degree 1, Degree 2, and Degree 3 Polynomial Curve Fits
    against Observed Hourly Mean Traffic Volume.
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    X = hourly_df["hour"].values
    y = hourly_df["mean_traffic_volume"].values
    
    # Dense X domain for smooth continuous curve rendering
    x_dense = np.linspace(0, 23, 250)
    
    # Actual mean traffic scatter
    ax.scatter(X, y, color="#1a202c", s=70, zorder=5, label="Observed Mean Traffic (Actual)", edgecolor="white", linewidth=1.2)
    
    # Model curves
    styles = {
        1: dict(color="#e53e3e", linestyle=":", linewidth=2.2, label=f"Degree 1 (Linear): R² = {models[1]['r2']:.4f}, MAE = {models[1]['mae']:.1f}"),
        2: dict(color="#dd6b20", linestyle="--", linewidth=2.4, label=f"Degree 2 (Quadratic): R² = {models[2]['r2']:.4f}, MAE = {models[2]['mae']:.1f}"),
        3: dict(color="#2b6cb0", linestyle="-", linewidth=3.0, label=f"Degree 3 (Cubic - Selected): R² = {models[3]['r2']:.4f}, MAE = {models[3]['mae']:.1f}")
    }
    
    for deg in [1, 2, 3]:
        p = models[deg]["poly1d"]
        y_dense = p(x_dense)
        ax.plot(x_dense, y_dense, **styles[deg])
        
    # Annotate morning and evening peaks
    ax.annotate("Morning Peak\n(7-8 AM)", xy=(7, 4740), xytext=(4, 5400),
                arrowprops=dict(facecolor="#2d3748", shrink=0.08, width=1.2, headwidth=5),
                fontsize=9.5, fontweight="bold", ha="center")
    ax.annotate("Evening Peak\n(4 PM, 5,664 veh/hr)", xy=(16, 5664), xytext=(17.5, 6200),
                arrowprops=dict(facecolor="#2d3748", shrink=0.08, width=1.2, headwidth=5),
                fontsize=9.5, fontweight="bold", ha="center")
                
    ax.set_title("Interstate Traffic Volume vs. Hour of Day: Polynomial Curve Fitting (Degrees 1, 2, 3)", pad=15)
    ax.set_xlabel("Hour of the Day (0:00 to 23:00)")
    ax.set_ylabel("Mean Traffic Volume (Vehicles / Hour)")
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.set_ylim(0, 6800)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e0", fontsize=9.5)
    
    # Model comparison text box
    ax.text(
        0.02, 0.45,
        "Model Comparison:\n"
        "• Deg 1: Severely underfits (R²=16.1%)\n"
        "• Deg 2: Symmetric parabola (R²=83.5%)\n"
        "• Deg 3: Asymmetric inflection (R²=85.8%, Best fit)",
        transform=ax.transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7fafc", edgecolor="#cbd5e0", alpha=0.95)
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def plot_stage4_residuals(hourly_df: pd.DataFrame, models: Dict[int, Dict[str, Any]], save_path: str = None) -> plt.Figure:
    """
    Stage 4 Figure 13: Residual Analysis across Hours of Day (Actual - Predicted).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Residual Analysis by Hour of Day: Residual = Actual - Predicted", fontsize=15, y=0.98, fontweight="bold")
    
    X = hourly_df["hour"].values
    palette = {1: "#e53e3e", 2: "#dd6b20", 3: "#2b6cb0"}
    
    for idx, deg in enumerate([1, 2, 3]):
        ax = axes[idx]
        res = models[deg]["residuals"]
        mae = models[deg]["mae"]
        rmse = models[deg]["rmse"]
        
        ax.axhline(0, color="#1a202c", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.bar(X, res, color=palette[deg], edgecolor="#2d3748", width=0.6, alpha=0.75)
        ax.plot(X, res, color=palette[deg], marker="o", linewidth=1.5, markersize=5)
        
        ax.set_title(f"Degree {deg} Polynomial Residuals | MAE = {mae:.1f} veh/hr | RMSE = {rmse:.1f} veh/hr | Mean Res = {models[deg]['mean_residual']:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Residual (veh/hr)")
        ax.grid(True, axis="y")
        ax.set_ylim(-2000, 2000)
        
    axes[2].set_xlabel("Hour of the Day (0:00 to 23:00)")
    axes[2].set_xticks(range(0, 24))
    axes[2].set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}", flush=True)
    return fig


def generate_all_visualizations(csv_path: str = PROCESSED_DATA_PATH):
    """
    Run complete visualization pipeline for Stages 2, 3, & 4.
    """
    print(f"Loading data from '{csv_path}' for visualization...", flush=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = preprocess_traffic_data()
        
    print(f"Generating presentation-ready figures in: '{FIGURES_DIR}'...", flush=True)
    
    # Stage 2 figures
    print("Generating Figure 1: Traffic volume histogram...", flush=True)
    f1 = plot_traffic_volume_histogram(df, os.path.join(FIGURES_DIR, "01_traffic_volume_histogram.png"))
    plt.close(f1)
    
    print("Generating Figure 2: Average traffic volume by hour...", flush=True)
    f2 = plot_avg_traffic_by_hour(df, os.path.join(FIGURES_DIR, "02_avg_traffic_by_hour.png"))
    plt.close(f2)
    
    print("Generating Figure 3: Congestion category distribution...", flush=True)
    f3 = plot_congestion_distribution(df, os.path.join(FIGURES_DIR, "03_congestion_category_distribution.png"))
    plt.close(f3)
    
    print("Generating Figure 4: Traffic volume during rain vs no rain...", flush=True)
    f4 = plot_traffic_rain_vs_norain(df, os.path.join(FIGURES_DIR, "04_traffic_rain_vs_norain.png"))
    plt.close(f4)
    
    print("Generating Figure 5: Traffic volume by hour with observed averages...", flush=True)
    f5 = plot_hourly_traffic_with_observed_averages(df, os.path.join(FIGURES_DIR, "05_hourly_traffic_with_averages.png"))
    plt.close(f5)
    
    print("Generating Figure 0: Executive 5-panel dashboard...", flush=True)
    f_dash = generate_executive_dashboard(df, os.path.join(FIGURES_DIR, "00_stage2_executive_dashboard.png"))
    plt.close(f_dash)
    
    # Stage 3 figures
    print("Generating Stage 3 Figure 6: Prior probabilities...", flush=True)
    f6 = plot_stage3_prior_probabilities(df, os.path.join(FIGURES_DIR, "06_stage3_congestion_priors.png"))
    plt.close(f6)
    
    print("Generating Stage 3 Figure 7: P(High | Rain) vs P(High | No Rain)...", flush=True)
    f7 = plot_stage3_rain_conditional_high(df, os.path.join(FIGURES_DIR, "07_stage3_rain_conditional_high.png"))
    plt.close(f7)
    
    print("Generating Stage 3 Figure 8: E(Traffic | Rain) vs E(Traffic | No Rain)...", flush=True)
    f8 = plot_stage3_rain_expected_traffic(df, os.path.join(FIGURES_DIR, "08_stage3_rain_expected_traffic.png"))
    plt.close(f8)
    
    print("Generating Stage 3 Figure 9: P(High | Peak) vs P(High | Non-Peak)...", flush=True)
    f9 = plot_stage3_peak_conditional_high(df, os.path.join(FIGURES_DIR, "09_stage3_peak_conditional_high.png"))
    plt.close(f9)
    
    print("Generating Stage 3 Figure 10: P(High | Weekday) vs P(High | Weekend)...", flush=True)
    f10 = plot_stage3_weekday_conditional_high(df, os.path.join(FIGURES_DIR, "10_stage3_weekday_conditional_high.png"))
    plt.close(f10)
    
    print("Generating Stage 3 Figure 11: P(High | Rain, Peak) vs P(High | No Rain, Peak)...", flush=True)
    f11 = plot_stage3_rain_peak_conditional_high(df, os.path.join(FIGURES_DIR, "11_stage3_rain_peak_conditional_high.png"))
    plt.close(f11)
    
    print("Generating Stage 3 Figure 0: Probability Dashboard...", flush=True)
    f_p_dash = generate_stage3_dashboard(df, os.path.join(FIGURES_DIR, "00_stage3_probability_dashboard.png"))
    plt.close(f_p_dash)
    
    # Stage 4 figures
    try:
        from src.polynomial import prepare_hourly_data, fit_polynomial_models, evaluate_polynomial_models
    except ImportError:
        from polynomial import prepare_hourly_data, fit_polynomial_models, evaluate_polynomial_models
        
    print("Generating Stage 4 Figure 12: Polynomial curve fits comparison...", flush=True)
    hourly_df = prepare_hourly_data(df)
    models = fit_polynomial_models(hourly_df["hour"].values, hourly_df["mean_traffic_volume"].values)
    _ = evaluate_polynomial_models(models, hourly_df["hour"].values, hourly_df["mean_traffic_volume"].values)
    
    f12 = plot_stage4_polynomial_fits(hourly_df, models, os.path.join(FIGURES_DIR, "12_stage4_polynomial_fit_comparison.png"))
    plt.close(f12)
    
    print("Generating Stage 4 Figure 13: Polynomial residuals analysis...", flush=True)
    f13 = plot_stage4_residuals(hourly_df, models, os.path.join(FIGURES_DIR, "13_stage4_polynomial_residuals.png"))
    plt.close(f13)
    
    print("All Stage 2, Stage 3, and Stage 4 visualizations generated and saved successfully!", flush=True)


if __name__ == "__main__":
    generate_all_visualizations()


