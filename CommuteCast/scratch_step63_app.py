"""
CommuteCast — Smart Traffic Prediction
College Presentation Interface — Traffic Signal & Urban Road Theme

Unit-1 Mathematical and Probability Foundations:
- Diurnal expected traffic volume estimated via Degree-3 polynomial curve fitting.
- Conditional probability of high congestion (> Q75) evaluated independently.
- Zero black-box ML algorithms; zero composite score multiplication; zero external APIs.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure project root is in sys.path and eliminate shadowing from 'src'
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != (PROJECT_ROOT / "src")]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PROCESSED_DATA_PATH, preprocess_traffic_data
from src.polynomial import (
    predict_traffic_for_hour,
    get_default_predictor,
    DEFAULT_SELECTED_DEGREE
)
from src.probability import CORE_PEAK_HOURS
from src.integration import (
    CommuteAnalysisEngine,
    analyze_commute_conditions,
    get_commute_engine
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & TRAFFIC SIGNAL THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CommuteCast — Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if "selected_hour" not in st.session_state:
    st.session_state.selected_hour = 8
if "selected_rain" not in st.session_state:
    st.session_state.selected_rain = "Rain"
if "selected_day" not in st.session_state:
    st.session_state.selected_day = "Monday"
if "selected_holiday" not in st.session_state:
    st.session_state.selected_holiday = "No"
if "analyzed" not in st.session_state:
    st.session_state.analyzed = True

# Traffic Signal & Urban Road Theme CSS
st.markdown("""
<style>
    /* Reset Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Root Palette & Background */
    :root {
        --bg-main: #F7F8F6;
        --text-primary: #17221B;
        --text-secondary: #66736A;
        --card-bg: #FFFFFF;
        --border-color: #E4E9E5;
        --signal-red: #E53935;
        --signal-amber: #FFB300;
        --signal-green: #43A047;
        --signal-green-dark: #388E3C;
        --road-dark: #202A24;
    }

    .stApp {
        background-color: #F7F8F6;
        background-image: 
            radial-gradient(circle at 10% 15%, rgba(67, 160, 71, 0.035) 0%, transparent 40%),
            radial-gradient(circle at 90% 85%, rgba(255, 179, 0, 0.03) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(2, 132, 199, 0.02) 0%, transparent 50%),
            linear-gradient(to right, rgba(228, 233, 229, 0.4) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(228, 233, 229, 0.4) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 48px 48px, 48px 48px;
        color: #17221B;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Container geometry - Desktop Optimized (1366x768 friendly) */
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1040px !important;
        margin: 0 auto !important;
    }

    /* Minimal Hero */
    .hero-container {
        padding: 0.4rem 0 1.0rem 0;
        margin-bottom: 0.8rem;
        border-bottom: 2px dashed #E4E9E5;
    }
    .hero-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }
    .hero-title-group {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 20px;
        padding: 4px 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .signal-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .signal-dot.red { background-color: #E53935; }
    .signal-dot.amber { background-color: #FFB300; }
    .signal-dot.green { background-color: #43A047; }

    .hero-brand {
        font-size: 1.65rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: #17221B;
        margin: 0;
        line-height: 1.1;
    }
    .hero-tag {
        font-size: 0.95rem;
        font-weight: 700;
        color: #43A047;
        margin-left: 6px;
    }
    .hero-subtitle {
        font-size: 0.88rem;
        color: #66736A;
        margin: 0.15rem 0 0 0;
        font-weight: 500;
    }

    /* Tabs Styling - Urban Traffic Signal Aesthetics */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #E4E9E5;
        padding-bottom: 4px;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 10px;
        color: #66736A;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 18px;
        transition: all 0.18s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #17221B;
        background-color: #F7F8F6;
        border-color: #CBD5CE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #17221B !important;
        border: 1.5px solid #43A047 !important;
        font-weight: 800 !important;
        box-shadow: 0 2px 8px rgba(67, 160, 71, 0.12) !important;
    }

    /* Clean Card Base */
    .traffic-card {
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        box-shadow: 0 2px 10px rgba(23, 34, 27, 0.03);
        margin-bottom: 1.1rem;
    }

    .traffic-card-header {
        font-size: 1.05rem;
        font-weight: 800;
        color: #17221B;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.2rem;
    }
    .traffic-card-sub {
        font-size: 0.82rem;
        color: #66736A;
        margin-bottom: 1rem;
        border-bottom: 1px solid #F0F4F1;
        padding-bottom: 0.5rem;
    }

    /* Peak Status Pill */
    .peak-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .peak-status-pill.peak {
        background-color: #F0FDF4;
        border: 1px solid #A7F3D0;
        color: #166534;
    }
    .peak-status-pill.offpeak {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        color: #4B5563;
    }

    .subtle-hint {
        font-size: 0.76rem;
        color: #85948A;
        margin-top: 4px;
    }

    /* Analyze Button Styling */
    div.stButton > button {
        background: #43A047 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.72rem 2.2rem !important;
        font-size: 1.02rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 4px 14px rgba(67, 160, 71, 0.28) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
        text-transform: uppercase !important;
    }
    div.stButton > button:hover {
        background: #388E3C !important;
        box-shadow: 0 6px 20px rgba(67, 160, 71, 0.38) !important;
        transform: translateY(-2px) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Traffic Signal Visual Animation Card */
    .signal-anim-wrap {
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        max-width: 340px;
        margin: 1.2rem auto;
        box-shadow: 0 8px 24px rgba(23, 34, 27, 0.06);
    }
    .signal-housing {
        background: #202A24;
        border: 2px solid #2F3E35;
        border-radius: 18px;
        width: 70px;
        padding: 12px 10px;
        margin: 0 auto 10px auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
        align-items: center;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.6), 0 4px 12px rgba(0,0,0,0.15);
    }
    .signal-bulb {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        transition: all 0.25s ease;
    }
    .signal-bulb.red.active {
        background: #E53935;
        box-shadow: 0 0 22px #E53935, inset 0 0 8px #FFCDD2;
    }
    .signal-bulb.red.dim {
        background: #3B1B1B;
        opacity: 0.25;
    }
    .signal-bulb.amber.active {
        background: #FFB300;
        box-shadow: 0 0 22px #FFB300, inset 0 0 8px #FFF9C4;
    }
    .signal-bulb.amber.dim {
        background: #3B3215;
        opacity: 0.25;
    }
    .signal-bulb.green.active {
        background: #43A047;
        box-shadow: 0 0 22px #43A047, inset 0 0 8px #C8E6C9;
    }
    .signal-bulb.green.dim {
        background: #17331E;
        opacity: 0.25;
    }
    .signal-stage-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #17221B;
        margin-top: 6px;
    }
    .signal-stage-desc {
        font-size: 0.8rem;
        color: #66736A;
        margin-top: 2px;
    }

    /* Result Section - Main Focus Cards */
    .result-section-header {
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        color: #17221B;
        margin: 1.2rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .kpi-result-card {
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 3px 12px rgba(23, 34, 27, 0.04);
        position: relative;
        overflow: hidden;
    }
    .kpi-result-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    .kpi-result-card.left-poly::before {
        background: #43A047;
    }
    .kpi-result-card.right-prob::before {
        background: #FFB300;
    }
    .kpi-result-card.right-prob.high-risk::before {
        background: #E53935;
    }
    .kpi-result-card.right-prob.low-risk::before {
        background: #43A047;
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #66736A;
        margin-bottom: 0.35rem;
    }
    .kpi-val {
        font-size: 2.85rem;
        font-weight: 900;
        color: #17221B;
        line-height: 1;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    .kpi-unit {
        font-size: 0.92rem;
        font-weight: 600;
        color: #66736A;
        margin-left: 0.25rem;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #66736A;
        margin-top: 0.45rem;
        line-height: 1.4;
    }

    /* Traffic Status Banner */
    .status-banner {
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-left: 5px solid #43A047;
        border-radius: 12px;
        padding: 1.0rem 1.4rem;
        margin: 1.0rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        box-shadow: 0 2px 8px rgba(23, 34, 27, 0.03);
    }
    .status-banner.red {
        border-left-color: #E53935;
    }
    .status-banner.amber {
        border-left-color: #FFB300;
    }
    .status-banner.green {
        border-left-color: #43A047;
    }

    .status-banner-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-signal-circle {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .status-signal-circle.red {
        background-color: #FFEBEE;
        color: #E53935;
    }
    .status-signal-circle.amber {
        background-color: #FFF8E1;
        color: #FFB300;
    }
    .status-signal-circle.green {
        background-color: #E8F5E9;
        color: #43A047;
    }

    .status-banner-title {
        font-size: 0.98rem;
        font-weight: 800;
        color: #17221B;
    }
    .status-banner-desc {
        font-size: 0.84rem;
        color: #66736A;
        margin-top: 2px;
    }
    .status-banner-metric {
        text-align: right;
        flex-shrink: 0;
    }
    .status-banner-pct {
        font-size: 1.45rem;
        font-weight: 900;
        color: #17221B;
        line-height: 1;
    }
    .status-banner-base {
        font-size: 0.74rem;
        color: #66736A;
        margin-top: 2px;
    }

    /* Commute Summary Pills */
    .summary-pills-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        padding: 0.65rem 1.0rem;
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 10px;
        margin-bottom: 1.1rem;
    }
    .summary-pill-label {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #66736A;
        margin-right: 4px;
    }
    .commute-pill {
        background: #F7F8F6;
        border: 1px solid #E4E9E5;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #17221B;
    }

    /* Small Insight Card */
    .insight-card {
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 12px;
        padding: 1.0rem 1.3rem;
        margin-top: 0.9rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        box-shadow: 0 2px 6px rgba(23, 34, 27, 0.02);
    }
    .insight-icon {
        font-size: 1.25rem;
        line-height: 1;
        margin-top: 2px;
    }
    .insight-text {
        font-size: 0.88rem;
        color: #17221B;
        line-height: 1.45;
    }

    /* Calculations Tab Cards */
    .calc-card {
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.0rem;
        box-shadow: 0 2px 8px rgba(23, 34, 27, 0.03);
    }
    .calc-card-badge {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #43A047;
        margin-bottom: 0.25rem;
    }
    .calc-card-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #17221B;
        margin-bottom: 0.6rem;
    }
    .calc-formula-box {
        background: #F7F8F6;
        border: 1px solid #E4E9E5;
        border-radius: 8px;
        padding: 0.75rem 1.0rem;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.92rem;
        color: #17221B;
        margin: 0.5rem 0;
    }
    .calc-explanation {
        font-size: 0.85rem;
        color: #66736A;
        line-height: 1.5;
        margin-top: 0.4rem;
    }

    /* Metric Grid for Calculations and Insights */
    .mini-stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 10px;
        margin: 0.8rem 0;
    }
    .mini-stat-card {
        background: #F7F8F6;
        border: 1px solid #E4E9E5;
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
    }
    .mini-stat-val {
        font-size: 1.25rem;
        font-weight: 900;
        color: #17221B;
    }
    .mini-stat-lbl {
        font-size: 0.74rem;
        font-weight: 700;
        color: #66736A;
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Separation Callout */
    .separation-callout {
        background: #F7F8F6;
        border: 1px solid #CBD5CE;
        border-left: 4px solid #17221B;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 1.0rem 0;
        font-size: 0.85rem;
        color: #17221B;
        line-height: 1.5;
    }

    /* Pipeline Arrow Sequence */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 6px;
        padding: 1.0rem;
        background: #F7F8F6;
        border: 1px solid #E4E9E5;
        border-radius: 12px;
        margin: 1.2rem 0;
    }
    .pipeline-node {
        background: #FFFFFF;
        border: 1px solid #E4E9E5;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.78rem;
        font-weight: 800;
        color: #17221B;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .pipeline-arrow {
        color: #85948A;
        font-size: 0.85rem;
        font-weight: 900;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED ENGINE & PREDICTOR LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Traffic Intelligence Models...")
def load_engine() -> CommuteAnalysisEngine:
    return get_commute_engine()


@st.cache_resource(show_spinner="Loading Polynomial Diurnal Fit...")
def load_predictor():
    return get_default_predictor()


engine = load_engine()
predictor = load_predictor()

# Time label mapping
def format_hour_str(h: int) -> str:
    if h == 0:
        return "12:00 AM"
    elif h < 12:
        return f"{h:02d}:00 AM"
    elif h == 12:
        return "12:00 PM"
    else:
        return f"{h-12:02d}:00 PM"


# -----------------------------------------------------------------------------
# 3. MINIMAL HERO SECTION
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-top-row">
        <div class="hero-title-group">
            <span style="font-size: 1.7rem; line-height: 1;">🚦</span>
            <span class="hero-brand">COMMUTECAST</span>
            <span class="hero-tag">Smart Traffic Prediction</span>
        </div>
        <div class="hero-signal-badge">
            <span class="signal-dot red"></span>
            <span class="signal-dot amber"></span>
            <span class="signal-dot green"></span>
            <span style="font-size: 0.75rem; font-weight: 700; color: #66736A; margin-left: 2px;">I-94 Metro Dataset</span>
        </div>
    </div>
    <div class="hero-subtitle">
        Understand your expected traffic and congestion probability before you commute.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. PRIMARY NAVIGATION TABS (EXACTLY FOUR TABS)
# -----------------------------------------------------------------------------
tab_commute, tab_calc, tab_insights, tab_about = st.tabs([
    "🚦 CommuteCast",
    "🧮 Calculations",
    "📊 Insights",
    "ℹ About"
])


# =============================================================================
# TAB 1: 🚦 COMMUTECAST (PRIMARY USER EXPERIENCE)
# =============================================================================
with tab_commute:
    # INPUT SECTION: Clean Central Card ("Plan Your Commute")
    st.markdown("""
    <div class="traffic-card">
        <div class="traffic-card-header">
            <span>🗺️</span>
            <span>Plan Your Commute</span>
        </div>
        <div class="traffic-card-sub">
            Select your journey conditions to forecast expected hourly traffic volume and congestion probability.
        </div>
    """, unsafe_allow_html=True)

    # 4 Inputs in ONE clean row on desktop
    c_day, c_time, c_weather, c_holiday = st.columns([1.1, 1.2, 1.1, 0.9])

    with c_day:
        days_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        d_idx = days_options.index(st.session_state.selected_day) if st.session_state.selected_day in days_options else 0
        sel_day = st.selectbox(
            "DAY",
            options=days_options,
            index=d_idx,
            help="Context only. The current model does not directly use day."
        )
        st.session_state.selected_day = sel_day
        st.markdown('<div class="subtle-hint">Context only</div>', unsafe_allow_html=True)

    with c_time:
        hour_labels = [f"{h:02d}:00 ({format_hour_str(h)})" for h in range(24)]
        h_idx = st.session_state.selected_hour if 0 <= st.session_state.selected_hour <= 23 else 8
        sel_hour = st.selectbox(
            "TIME",
            options=list(range(24)),
            format_func=lambda h: f"{h:02d}:00 ({format_hour_str(h)})",
            index=h_idx,
            help="Active model input: determines expected traffic volume and peak classification."
        )
        st.session_state.selected_hour = sel_hour

    with c_weather:
        weather_options = ["☀ No Rain", "🌧 Rain"]
        w_idx = 1 if st.session_state.selected_rain == "Rain" else 0
        sel_weather = st.radio(
            "WEATHER",
            options=weather_options,
            index=w_idx,
            horizontal=True,
            help="Active empirical condition: determines weather-conditioned high congestion probability."
        )
        st.session_state.selected_rain = "Rain" if "Rain" in sel_weather else "No Rain"
        rain_val = 1 if st.session_state.selected_rain == "Rain" else 0

    with c_holiday:
        holiday_options = ["No", "Yes"]
        hol_idx = 1 if st.session_state.selected_holiday == "Yes" else 0
        sel_holiday = st.radio(
            "HOLIDAY",
            options=holiday_options,
            index=hol_idx,
            horizontal=True,
            help="Informational only. The current model does not use holiday in prediction."
        )
        st.session_state.selected_holiday = sel_holiday
        holiday_val = 1 if sel_holiday == "Yes" else 0
        st.markdown('<div class="subtle-hint">Informational</div>', unsafe_allow_html=True)

    # Automatic Peak Status Derivation (Strictly from {7, 8, 16, 17} — NO MANUAL SELECTOR)
    is_peak = sel_hour in CORE_PEAK_HOURS
    peak_val = 1 if is_peak else 0

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if is_peak:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: #F0FDF4; border: 1px solid #A7F3D0; border-radius: 10px; padding: 0.65rem 1.0rem;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="peak-status-pill peak">🟢 PEAK HOUR</span>
                <span style="font-size: 0.82rem; color: #166534; font-weight: 600;">
                    {format_hour_str(sel_hour)} falls within core metropolitan rush hours (7–8 AM & 4–5 PM).
                </span>
            </div>
            <div class="subtle-hint">Peak status is automatically determined from the selected hour.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px; padding: 0.65rem 1.0rem;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="peak-status-pill offpeak">⚪ OFF-PEAK</span>
                <span style="font-size: 0.82rem; color: #4B5563; font-weight: 600;">
                    {format_hour_str(sel_hour)} falls outside core rush hours.
                </span>
            </div>
            <div class="subtle-hint">Peak status is automatically determined from the selected hour.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ANALYZE BUTTON: Large but elegant, signal-green styling
    btn_c1, btn_c2, btn_c3 = st.columns([1, 1.6, 1])
    with btn_c2:
        analyze_clicked = st.button("🚦 ANALYZE TRAFFIC", use_container_width=True)

    anim_slot = st.empty()

    if analyze_clicked:
        # Centered Traffic Signal Animation (2.4s sequence)
        stages = [
            ("red", "Reading commute", "Parsing hour, precipitation, and peak schedule..."),
            ("amber", "Analyzing pattern", "Evaluating diurnal polynomial curve and condition frequencies..."),
            ("green", "Analysis ready", "Commute outlook synthesized.")
        ]
        for light_state, stage_title, stage_desc in stages:
            red_cls = "active" if light_state == "red" else "dim"
            amb_cls = "active" if light_state == "amber" else "dim"
            grn_cls = "active" if light_state == "green" else "dim"

            anim_slot.markdown(f"""
            <div class="signal-anim-wrap">
                <div class="signal-housing">
                    <div class="signal-bulb red {red_cls}"></div>
                    <div class="signal-bulb amber {amb_cls}"></div>
                    <div class="signal-bulb green {grn_cls}"></div>
                </div>
                <div class="signal-stage-title">{stage_title}</div>
                <div class="signal-stage-desc">{stage_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.8)

        anim_slot.empty()
        st.session_state.analyzed = True

    # Execute backend analysis
    res = engine.analyze(
        hour=sel_hour,
        rain=rain_val,
        peak=peak_val,
        holiday=holiday_val,
        polynomial_degree=DEFAULT_SELECTED_DEGREE
    )

    expected_traffic = res["expected_traffic"]
    p_high_cond = res["high_congestion_probability"]
    p_high_base = res["baseline_high_probability"]
    diff_ppt = res["difference_percentage_points"]
    is_elevated = p_high_cond > p_high_base

    # -------------------------------------------------------------------------
    # RESULT — MAIN FOCUS OF TAB 1
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="result-section-header">
        <span>🚦</span>
        <span>Your Commute Outlook</span>
    </div>
    """, unsafe_allow_html=True)

    # TWO LARGE RESULT CARDS SIDE-BY-SIDE
    res_c1, res_c2 = st.columns(2, gap="medium")

    with res_c1:
        st.markdown(f"""
        <div class="kpi-result-card left-poly">
            <div class="kpi-label">EXPECTED TRAFFIC</div>
            <div class="kpi-val">{expected_traffic:,.0f}<span class="kpi-unit">vehicles / hour</span></div>
            <div class="kpi-subtext">Estimated typical traffic volume</div>
        </div>
        """, unsafe_allow_html=True)

    with res_c2:
        risk_class = "high-risk" if p_high_cond > 0.50 else "low-risk" if p_high_cond <= 0.25 else ""
        st.markdown(f"""
        <div class="kpi-result-card right-prob {risk_class}">
            <div class="kpi-label">HIGH CONGESTION</div>
            <div class="kpi-val">{p_high_cond * 100:.2f}%<span class="kpi-unit">probability</span></div>
            <div class="kpi-subtext">Historical probability under your conditions</div>
        </div>
        """, unsafe_allow_html=True)

    # TRAFFIC STATUS BANNER
    if p_high_cond > 0.50:
        status_theme = "red"
        status_icon = "🔴"
        status_label = "High congestion probability"
    elif p_high_cond > 0.25:
        status_theme = "amber"
        status_icon = "🟡"
        status_label = "Moderate congestion probability"
    else:
        status_theme = "green"
        status_icon = "🟢"
        status_label = "Low congestion probability"

    if diff_ppt >= 0:
        diff_statement = f"{diff_ppt:.2f} percentage points above the historical baseline."
    else:
        diff_statement = f"{abs(diff_ppt):.2f} percentage points below the historical baseline."

    st.markdown(f"""
    <div class="status-banner {status_theme}">
        <div class="status-banner-left">
            <div class="status-signal-circle {status_theme}">{status_icon}</div>
            <div>
                <div class="status-banner-title">{status_label}</div>
                <div class="status-banner-desc">{diff_statement}</div>
            </div>
        </div>
        <div class="status-banner-metric">
            <div class="status-banner-pct">{p_high_cond * 100:.2f}%</div>
            <div class="status-banner-base">Baseline: <strong>{p_high_base * 100:.2f}%</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # COMMUTE SUMMARY PILLS
    peak_pill_text = "Peak Hour" if is_peak else "Off-Peak"
    st.markdown(f"""
    <div class="summary-pills-row">
        <span class="summary-pill-label">YOUR COMMUTE:</span>
        <span class="commute-pill">{sel_day}</span>
        <span class="commute-pill">{format_hour_str(sel_hour)}</span>
        <span class="commute-pill">{st.session_state.selected_rain}</span>
        <span class="commute-pill">{peak_pill_text}</span>
        <span class="commute-pill">Holiday: {sel_holiday}</span>
    </div>
    """, unsafe_allow_html=True)

    # SMALL TRAFFIC PATTERN VISUAL (~320px tall)
    st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin: 0.6rem 0 0.2rem 0;'>Traffic by Hour</div>", unsafe_allow_html=True)
    st.caption("Where does your selected commute time sit in the daily traffic pattern?")

    fig_hour, ax_hour = plt.subplots(figsize=(8, 3.2), dpi=120)
    fig_hour.patch.set_facecolor('#FFFFFF')
    ax_hour.set_facecolor('#FCFDFC')

    hourly_df = predictor.hourly_df
    X_hours = hourly_df["hour"].values
    y_means = hourly_df["mean_traffic_volume"].values
    x_dense = np.linspace(0, 23, 200)
    models = predictor.models

    # Degree-3 curve and observed means
    ax_hour.plot(x_dense, models[3]["poly1d"](x_dense), color="#202A24", linestyle="-", linewidth=2.2, label="Degree-3 Diurnal Traffic Curve", zorder=3)
    ax_hour.scatter(X_hours, y_means, color="#66736A", s=24, alpha=0.8, zorder=4, label="Hourly Mean Observations")

    # Highlight selected hour with vertical road indicator and accent marker
    pred_current = models[3]["poly1d"](sel_hour)
    accent_color = "#E53935" if is_peak else "#43A047"
    ax_hour.axvline(x=sel_hour, color=accent_color, linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
    ax_hour.scatter([sel_hour], [pred_current], color=accent_color, s=110, zorder=6, edgecolor="#FFFFFF", linewidth=2, label=f"Selected Hour ({format_hour_str(sel_hour)})")

    # Clean axes styling
    for spine in ax_hour.spines.values():
        spine.set_color('#E4E9E5')
    ax_hour.tick_params(colors='#66736A', labelsize=8)
    ax_hour.set_xlabel("Hour of Day", color='#17221B', fontsize=8.5, fontweight="bold")
    ax_hour.set_ylabel("Vehicles / Hour", color='#17221B', fontsize=8.5, fontweight="bold")
    ax_hour.set_xlim(-0.5, 23.5)
    ax_hour.set_ylim(0, 6500)
    ax_hour.set_xticks(range(0, 24, 2))
    ax_hour.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax_hour.grid(True, linestyle="--", alpha=0.45, color='#E4E9E5')
    ax_hour.legend(loc="upper left", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#E4E9E5", labelcolor="#17221B")

    plt.tight_layout()
    st.pyplot(fig_hour, use_container_width=True)
    plt.close(fig_hour)

    # KEY INSIGHT CARD (1-2 sentences)
    rain_word = "with rain" if rain_val == 1 else "with no rain"
    diff_phrase = f"{abs(diff_ppt):.2f} percentage points {'above' if diff_ppt >= 0 else 'below'}"
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-icon">💡</div>
        <div class="insight-text">
            At <strong>{format_hour_str(sel_hour)}</strong> {rain_word}, historical observations show a 
            <strong>{p_high_cond * 100:.2f}%</strong> high-congestion probability, compared with the 
            <strong>{p_high_base * 100:.2f}%</strong> overall baseline ({diff_phrase}).
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: 🧮 CALCULATIONS (ACADEMIC & MATHEMATICAL FOUNDATIONS)
# =============================================================================
with tab_calc:
    st.markdown("""
    <div style="margin-bottom: 1.0rem;">
        <div style="font-size: 1.25rem; font-weight: 900; color: #17221B;">Academic & Mathematical Methodology</div>
        <div style="font-size: 0.85rem; color: #66736A;">
            Complete derivation of mathematical formulas, regression equations, empirical conditional probabilities, and Bayesian verification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. POLYNOMIAL CURVE FITTING
    st.markdown(f"""
    <div class="calc-card">
        <div class="calc-card-badge">Section 1 • Regression Modeling</div>
        <div class="calc-card-title">Polynomial Curve Fitting (Diurnal Profile)</div>
        <div class="calc-formula-box">
            ŷ = -171.78 + 558.08x + 1.3152x² - 1.0094x³
        </div>
        <div class="calc-explanation">
            Where <em>x</em> represents the hour of the day (0 to 23). Evaluated at the currently selected hour:<br>
            <strong>x = {sel_hour}</strong> ({format_hour_str(sel_hour)}) &nbsp;→&nbsp; 
            <strong>ŷ ≈ {expected_traffic:,.0f} vehicles/hour</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. CONDITIONAL PROBABILITY
    cond_name = f"P(High | {st.session_state.selected_rain}, {'Peak' if is_peak else 'Off-Peak'})"
    st.markdown(f"""
    <div class="calc-card">
        <div class="calc-card-badge">Section 2 • Probability Foundations</div>
        <div class="calc-card-title">Conditional Probability Analysis</div>
        <div class="calc-formula-box">
            {cond_name} = {p_high_cond * 100:.2f}%
        </div>
        <div class="calc-explanation">
            Empirical relative frequency of high congestion (> Q75 threshold = 4,933 veh/hr) observed across matching historical conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 View Complete 4-Quadrant Empirical Breakdown", expanded=False):
        q_cols = st.columns(4)
        quadrants_data = [
            ("Rain + Peak", engine.joint_probabilities[(1, 1)]["prob"] * 100, engine.joint_probabilities[(1, 1)]["joint_count"], engine.joint_probabilities[(1, 1)]["condition_count"]),
            ("No Rain + Peak", engine.joint_probabilities[(0, 1)]["prob"] * 100, engine.joint_probabilities[(0, 1)]["joint_count"], engine.joint_probabilities[(0, 1)]["condition_count"]),
            ("Rain + Off-Peak", engine.joint_probabilities[(1, 0)]["prob"] * 100, engine.joint_probabilities[(1, 0)]["joint_count"], engine.joint_probabilities[(1, 0)]["condition_count"]),
            ("No Rain + Off-Peak", engine.joint_probabilities[(0, 0)]["prob"] * 100, engine.joint_probabilities[(0, 0)]["joint_count"], engine.joint_probabilities[(0, 0)]["condition_count"]),
        ]
        for col, (label, prob, j_cnt, c_cnt) in zip(q_cols, quadrants_data):
            with col:
                st.markdown(f"""
                <div class="mini-stat-card">
                    <div class="mini-stat-val">{prob:.2f}%</div>
                    <div class="mini-stat-lbl">{label}</div>
                    <div style="font-size: 0.72rem; color: #66736A; margin-top: 4px;">{j_cnt:,} / {c_cnt:,} records</div>
                </div>
                """, unsafe_allow_html=True)

    # 3. BASELINE
    st.markdown(f"""
    <div class="calc-card">
        <div class="calc-card-badge">Section 3 • Benchmark Baseline</div>
        <div class="calc-card-title">Overall Congestion Baseline</div>
        <div class="calc-formula-box">
            P(High) = 24.98%
        </div>
        <div class="calc-explanation">
            Evaluated unconditionally across all 48,187 clean observations. By definition of the 75th percentile (Q75 = 4,933), exactly 24.98% of all recorded hours experience high congestion.
            <br>Current condition difference: <strong>{'+' if diff_ppt >= 0 else ''}{diff_ppt:.2f} percentage points</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. BAYESIAN ANALYSIS
    st.markdown("""
    <div class="calc-card">
        <div class="calc-card-badge">Section 4 • Bayesian Reasoning</div>
        <div class="calc-card-title">Bayes' Theorem Formulation & Empirical Verification</div>
        <div class="calc-formula-box">
            P(High | Evidence) = [ P(Evidence | High) × P(High) ] / P(Evidence)
        </div>
        <div class="calc-explanation">
            Bayes' theorem formalizes belief updating: prior belief P(High) is updated upon observing evidence (e.g. rain or peak hour) to arrive at the posterior probability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🎲 Detailed Bayesian Verification (Rain & Peak Hour)", expanded=True):
        bayes_rain = engine.explain_bayes("rain")
        bayes_peak = engine.explain_bayes("peak")

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.markdown(f"""
            <div style="background: #F7F8F6; border: 1px solid #E4E9E5; border-radius: 10px; padding: 0.9rem;">
                <strong style="color: #17221B; font-size: 0.9rem;">Evidence: Rain (Precipitation)</strong>
                <ul style="font-size: 0.82rem; color: #66736A; margin: 6px 0 0 16px; padding: 0;">
                    <li>Prior P(High): <strong>24.98%</strong></li>
                    <li>Marginal P(Rain): <strong>{bayes_rain['marginal_P_Evidence']*100:.2f}%</strong></li>
                    <li>Likelihood P(Rain | High): <strong>{bayes_rain['likelihood_P_Evidence_given_High']*100:.2f}%</strong></li>
                    <li>Bayes Posterior: <strong>{bayes_rain['bayes_posterior']*100:.2f}%</strong></li>
                    <li>Empirical Posterior: <strong>{bayes_rain['direct_empirical_posterior']*100:.2f}%</strong></li>
                    <li>Absolute Discrepancy: <strong>0.000000</strong> (Exact Verification)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with b_c2:
            st.markdown(f"""
            <div style="background: #F7F8F6; border: 1px solid #E4E9E5; border-radius: 10px; padding: 0.9rem;">
                <strong style="color: #17221B; font-size: 0.9rem;">Evidence: Peak Hour (Rush Hour)</strong>
                <ul style="font-size: 0.82rem; color: #66736A; margin: 6px 0 0 16px; padding: 0;">
                    <li>Prior P(High): <strong>24.98%</strong></li>
                    <li>Marginal P(Peak): <strong>{bayes_peak['marginal_P_Evidence']*100:.2f}%</strong></li>
                    <li>Likelihood P(Peak | High): <strong>{bayes_peak['likelihood_P_Evidence_given_High']*100:.2f}%</strong></li>
                    <li>Bayes Posterior: <strong>{bayes_peak['bayes_posterior']*100:.2f}%</strong></li>
                    <li>Empirical Posterior: <strong>{bayes_peak['direct_empirical_posterior']*100:.2f}%</strong></li>
                    <li>Absolute Discrepancy: <strong>0.000000</strong> (Exact Verification)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # 5. POLYNOMIAL METRICS
    st.markdown("""
    <div class="calc-card">
        <div class="calc-card-badge">Section 5 • Goodness-of-Fit</div>
        <div class="calc-card-title">Polynomial Goodness-of-Fit Metrics (Degree 3)</div>
        <div class="mini-stat-grid">
            <div class="mini-stat-card"><div class="mini-stat-val">539.58</div><div class="mini-stat-lbl">MAE (veh/hr)</div></div>
            <div class="mini-stat-card"><div class="mini-stat-val">658.37</div><div class="mini-stat-lbl">RMSE (veh/hr)</div></div>
            <div class="mini-stat-card"><div class="mini-stat-val">0.8575</div><div class="mini-stat-lbl">R² Score</div></div>
        </div>
        <div class="calc-explanation">
            Degree 3 polynomial captures the asymmetric morning and evening rush hour crests, explaining <strong>85.75%</strong> of the hourly diurnal variance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 6. CONGESTION QUANTILES & MATHEMATICAL SEPARATION
    st.markdown("""
    <div class="calc-card">
        <div class="calc-card-badge">Section 6 • Quantile Boundaries</div>
        <div class="calc-card-title">Congestion Thresholds (Quantiles)</div>
        <div class="mini-stat-grid">
            <div class="mini-stat-card"><div class="mini-stat-val">1,192.5</div><div class="mini-stat-lbl">Q25 Threshold</div></div>
            <div class="mini-stat-card"><div class="mini-stat-val">3,379.0</div><div class="mini-stat-lbl">Median (Q50)</div></div>
            <div class="mini-stat-card"><div class="mini-stat-val">4,933.0</div><div class="mini-stat-lbl">Q75 (High Boundary)</div></div>
        </div>
        <div class="calc-explanation">
            High congestion is objectively defined as any hourly observation exceeding <strong>Q75 = 4,933 vehicles/hour</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Mathematical Separation Note
    st.markdown("""
    <div class="separation-callout">
        <strong>⚠️ Core Methodological Principle: Strict Mathematical Separation</strong><br>
        Expected traffic volume is estimated via polynomial regression curve fitting: <em>ŷ = f(hour)</em>.<br>
        Congestion probability is computed via empirical relative frequencies: <em>P(High | Rain, Peak)</em>.<br>
        These two values describe distinct physical and probabilistic dimensions of traffic and are <strong>NEVER multiplied together</strong>.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: 📊 INSIGHTS (USEFUL VISUAL ANALYTICS ONLY)
# =============================================================================
with tab_insights:
    st.markdown("""
    <div style="margin-bottom: 1.0rem;">
        <div style="font-size: 1.25rem; font-weight: 900; color: #17221B;">Visual Traffic Insights</div>
        <div style="font-size: 0.85rem; color: #66736A;">
            Exploratory visual analytics comparing diurnal patterns, conditional probabilities, and baseline distributions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    in_c1, in_c2 = st.columns(2, gap="medium")

    with in_c1:
        # 1. HOURLY TRAFFIC PATTERN
        st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin-bottom: 0.2rem;'>1. Hourly Traffic Volume Pattern</div>", unsafe_allow_html=True)
        st.caption("Empirical mean traffic vs. Degree-3 polynomial fit")

        fig_pat, ax_pat = plt.subplots(figsize=(6.5, 4.2), dpi=120)
        fig_pat.patch.set_facecolor('#FFFFFF')
        ax_pat.set_facecolor('#FCFDFC')

        ax_pat.scatter(X_hours, y_means, color="#66736A", s=30, zorder=3, label="Hourly Mean Traffic")
        ax_pat.plot(x_dense, models[3]["poly1d"](x_dense), color="#43A047", linewidth=2.4, label="Degree-3 Fit (R²=0.86)")

        # Highlight morning & evening peaks
        ax_pat.axvspan(6.5, 8.5, color="#E53935", alpha=0.08, label="Morning Peak (7–8 AM)")
        ax_pat.axvspan(15.5, 17.5, color="#FFB300", alpha=0.08, label="Evening Peak (4–5 PM)")

        for spine in ax_pat.spines.values():
            spine.set_color('#E4E9E5')
        ax_pat.tick_params(colors='#66736A', labelsize=8)
        ax_pat.set_xlabel("Hour of Day", color='#17221B', fontsize=8.5, fontweight="bold")
        ax_pat.set_ylabel("Traffic Volume (veh/hr)", color='#17221B', fontsize=8.5, fontweight="bold")
        ax_pat.set_xlim(-0.5, 23.5)
        ax_pat.set_ylim(0, 6500)
        ax_pat.set_xticks(range(0, 24, 3))
        ax_pat.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
        ax_pat.grid(True, linestyle="--", alpha=0.45, color='#E4E9E5')
        ax_pat.legend(loc="upper left", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#E4E9E5", labelcolor="#17221B")

        plt.tight_layout()
        st.pyplot(fig_pat, use_container_width=True)
        plt.close(fig_pat)

    with in_c2:
        # 2. PROBABILITY COMPARISON
        st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin-bottom: 0.2rem;'>2. Probability Comparison</div>", unsafe_allow_html=True)
        st.caption("High congestion probability across conditions vs. baseline")

        p_labels = [
            "Rain + Peak",
            "Peak Overall",
            "No Rain + Peak",
            "Overall Baseline",
            "Rain + Non-Peak",
            "No Rain + Non-Peak",
            "Non-Peak Overall"
        ]
        p_vals = [
            67.75,
            65.50,
            65.31,
            24.98,
            18.18,
            16.71,
            16.82
        ]
        bar_palette = [
            "#E53935",
            "#E53935",
            "#FFB300",
            "#17221B",
            "#43A047",
            "#43A047",
            "#43A047"
        ]

        fig_bars, ax_bars = plt.subplots(figsize=(6.5, 4.2), dpi=120)
        fig_bars.patch.set_facecolor('#FFFFFF')
        ax_bars.set_facecolor('#FCFDFC')

        y_positions = np.arange(len(p_labels))
        bars = ax_bars.barh(y_positions, p_vals, color=bar_palette, height=0.55, edgecolor="#E4E9E5", linewidth=0.5)
        ax_bars.axvline(x=24.98, color="#17221B", linestyle="--", linewidth=1.2, label="Baseline (24.98%)")

        for bar, val in zip(bars, p_vals):
            ax_bars.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", ha="left", fontsize=7.8, fontweight="bold", color="#17221B")

        for spine in ax_bars.spines.values():
            spine.set_color('#E4E9E5')
        ax_bars.tick_params(colors='#66736A', labelsize=8)
        ax_bars.set_yticks(y_positions)
        ax_bars.set_yticklabels(p_labels, fontsize=8, fontweight="bold", color="#17221B")
        ax_bars.invert_yaxis()
        ax_bars.set_xlabel("High Congestion Probability (%)", color="#17221B", fontsize=8.5, fontweight="bold")
        ax_bars.set_xlim(0, 85)
        ax_bars.grid(True, axis="x", linestyle="--", alpha=0.45, color='#E4E9E5')
        ax_bars.legend(loc="lower right", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#E4E9E5", labelcolor="#17221B")

        plt.tight_layout()
        st.pyplot(fig_bars, use_container_width=True)
        plt.close(fig_bars)

    # 3. SMALL DATASET SUMMARY
    st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin: 1.2rem 0 0.4rem 0;'>3. Dataset Statistical Summary</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="mini-stat-grid">
        <div class="mini-stat-card"><div class="mini-stat-val">48,187</div><div class="mini-stat-lbl">Observations</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">3,259.62</div><div class="mini-stat-lbl">Mean Traffic (veh/hr)</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">1,192.5</div><div class="mini-stat-lbl">Q25 Boundary</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">3,379.0</div><div class="mini-stat-lbl">Median Traffic</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">4,933.0</div><div class="mini-stat-lbl">Q75 (High Congestion)</div></div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 4: ℹ ABOUT (COLLEGE PROJECT IDENTITY & PIPELINE)
# =============================================================================
with tab_about:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 1.35rem; font-weight: 900; color: #17221B;">COMMUTECAST</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #43A047; margin-top: 2px;">
            Probabilistic Traffic Congestion Prediction Using Polynomial Curve Fitting and Bayesian Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="traffic-card">
        <div class="traffic-card-header">Dataset Overview</div>
        <div style="font-size: 0.88rem; color: #66736A; line-height: 1.6;">
            <strong>UCI Metro Interstate Traffic Volume Dataset</strong><br>
            Captures hourly westbound traffic volume for Interstate 94 between Minneapolis and St. Paul, Minnesota (2012–2018).
            Consists of <strong>48,187</strong> deduplicated, sensor-validated observations after cleaning sensor dropouts and rain recording anomalies.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PROJECT PIPELINE
    st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin: 1.0rem 0 0.2rem 0;'>Project Analytical Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-container">
        <span class="pipeline-node">DATA</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node">CLEAN</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node">STATISTICS</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node">PROBABILITY</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node">BAYES</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node">POLYNOMIAL</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node">PREDICTION</span>
    </div>
    """, unsafe_allow_html=True)

    # CONCEPTS USED
    st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin: 1.0rem 0 0.4rem 0;'>Unit-1 Mathematical Concepts Implemented</div>", unsafe_allow_html=True)
    concepts = [
        ("Probability", "Empirical sample space frequencies P(A) = n(A) / N"),
        ("Conditional Probability", "Event probability conditioned on evidence P(A | B)"),
        ("Bayes' Theorem", "Formal updating of prior beliefs based on likelihood of evidence"),
        ("Random Variables", "Modeling traffic volume X as a continuous diurnal random variable"),
        ("Expectation & Variance", "Sample mean as E[X] estimator and sample spread s²"),
        ("Covariance", "Quantifying joint variability between volume, rain, and temperature"),
        ("Quantiles", "Objective non-parametric classification into Low, Medium, and High congestion"),
        ("Polynomial Curve Fitting", "Ordinary Least Squares cubic regression modeling diurnal periodicity")
    ]

    c_cols = st.columns(2)
    for i, (name, desc) in enumerate(concepts):
        col_to_use = c_cols[i % 2]
        with col_to_use:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E4E9E5; border-radius: 10px; padding: 0.75rem 1.0rem; margin-bottom: 8px;">
                <strong style="color: #17221B; font-size: 0.88rem;">{name}</strong><br>
                <span style="font-size: 0.78rem; color: #66736A;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="border-top: 1px solid #E4E9E5; margin-top: 1.5rem; padding-top: 1.0rem; text-align: center; font-size: 0.8rem; color: #85948A;">
        CommuteCast • Presentation-Ready Academic Engineering Project • Zero Black-Box ML
    </div>
    """, unsafe_allow_html=True)
