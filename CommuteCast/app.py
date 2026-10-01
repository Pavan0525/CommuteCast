"""
CommuteCast — Smart Traffic Prediction
College Presentation Interface — Animated Traffic Signal & Urban Road Theme

Unit-1 Mathematical and Probability Foundations:
- Expected vehicle count estimated via Degree-3 polynomial curve fitting over Date (day_num).
- Conditional probability of high congestion (> Q75 of Vehicle_Count) evaluated independently.
- Location (City + Area) probability P(High | City, Area) computed from Traffic dataset.csv.
- Zero black-box ML algorithms; zero composite score multiplication; zero external APIs.
- Active dataset: Traffic dataset.csv (1200 rows x 9 columns).
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

# New dataset imports (active prediction pipeline)
from src.new_preprocessing import (
    preprocess_new_traffic_data, get_city_area_mapping, get_cities, get_vehicle_types,
    validate_dataset_columns
)
from src.new_polynomial import DEFAULT_SELECTED_DEGREE
from src.new_integration import get_new_commute_engine

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LAYERED THEME ARCHITECTURE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CommuteCast — Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State — City + Area + Vehicle_Type (no preselection)
if "selected_city" not in st.session_state:
    st.session_state.selected_city = None
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None
if "selected_vehicle_type" not in st.session_state:
    st.session_state.selected_vehicle_type = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
# Store the last analyzed snapshot to detect input changes
if "last_analyzed_inputs" not in st.session_state:
    st.session_state.last_analyzed_inputs = None

# -----------------------------------------------------------------------------
# 2. BULLETPROOF CSS & LAYER STACKING SPECIFICATION
# Layer 0 (z-index: 0, fixed): Animated SVG Road Canvas + Embedded Veil
# Layer 1 (z-index: 1, relative): Streamlit Content Containers (.stApp, .block-container)
# Layer 2 (z-index: 2+): Foreground UI Cards & Interactive Controls
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Reset Streamlit default UI chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Root Palette & Dimensions */
    :root {
        --bg-main: #F5F7F4;
        --road-asphalt: #D9DEDA;
        --road-curb: #CBD2CC;
        --road-lane: #FFFFFF;
        --text-primary: #17221B;
        --text-secondary: #66736A;
        --card-surface: rgba(255, 255, 255, 0.95);
        --border-subtle: #E4E9E5;
        --signal-red: #E53935;
        --signal-amber: #FFB300;
        --signal-green: #43A047;
        --accent-blue: #3182CE;
        --accent-teal: #159A9C;
    }

    /* Streamlit Root & Body Layout */
    .stApp {
        background-color: #F5F7F4 !important;
        color: #17221B;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Transparent intermediate containers so Layer 0 SVG is visible */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background: transparent !important;
        position: relative !important;
        z-index: 1 !important;
    }

    /* Fixed Background Layer 0 (Behind all Streamlit content) */
    .traffic-ambient-bg {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        pointer-events: none !important;
        z-index: 0 !important;
        overflow: hidden !important;
    }

    .traffic-city-canvas {
        width: 100%;
        height: 100%;
        display: block;
        pointer-events: none !important;
    }

    /* Foreground Content Container Layer 1 (Always above Layer 0) */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1040px !important;
        margin: 0 auto !important;
        position: relative !important;
        z-index: 1 !important;
    }

    /* Continuous Flowing Lane Markings Keyframes */
    @keyframes flowWestToEast {
        from { stroke-dashoffset: 48; }
        to { stroke-dashoffset: 0; }
    }
    @keyframes flowEastToWest {
        from { stroke-dashoffset: 0; }
        to { stroke-dashoffset: 48; }
    }
    @keyframes flowNorthToSouth {
        from { stroke-dashoffset: 48; }
        to { stroke-dashoffset: 0; }
    }
    @keyframes flowSouthToNorth {
        from { stroke-dashoffset: 0; }
        to { stroke-dashoffset: 48; }
    }
    @keyframes telemetryPulse {
        from { stroke-dashoffset: 64; }
        to { stroke-dashoffset: 0; }
    }

    /* Vehicle Movement Keyframes */
    @keyframes carMoveEast1 {
        0% { transform: translate(-60px, 142px); }
        100% { transform: translate(1660px, 142px); }
    }
    @keyframes carMoveEast2 {
        0% { transform: translate(-60px, 142px); }
        100% { transform: translate(1660px, 142px); }
    }
    @keyframes carMoveWest1 {
        0% { transform: translate(1660px, 178px) rotate(180deg); }
        100% { transform: translate(-60px, 178px) rotate(180deg); }
    }
    @keyframes busMoveSouth {
        0% { transform: translate(116px, -60px) rotate(90deg); }
        100% { transform: translate(116px, 1060px) rotate(90deg); }
    }
    @keyframes carMoveNorth {
        0% { transform: translate(1480px, 1060px) rotate(-90deg); }
        100% { transform: translate(1480px, -60px) rotate(-90deg); }
    }
    @keyframes carMoveDiagonal {
        0% { transform: translate(-60px, 730px) rotate(-14deg); }
        50% { transform: translate(750px, 570px) rotate(-10deg); }
        100% { transform: translate(1660px, 410px) rotate(-8deg); }
    }

    /* Traffic Light State Transitions */
    @keyframes tlCycleRed {
        0%, 38% { fill: #E53935; filter: drop-shadow(0 0 5px #E53935); opacity: 1; }
        42%, 96% { fill: #3B1B1B; filter: none; opacity: 0.22; }
        100% { fill: #E53935; filter: drop-shadow(0 0 5px #E53935); opacity: 1; }
    }
    @keyframes tlCycleAmber {
        0%, 37% { fill: #3B3215; filter: none; opacity: 0.22; }
        41%, 52% { fill: #FFB300; filter: drop-shadow(0 0 5px #FFB300); opacity: 1; }
        56%, 100% { fill: #3B3215; filter: none; opacity: 0.22; }
    }
    @keyframes tlCycleGreen {
        0%, 53% { fill: #17331E; filter: none; opacity: 0.22; }
        57%, 95% { fill: #43A047; filter: drop-shadow(0 0 5px #43A047); opacity: 1; }
        99%, 100% { fill: #17331E; filter: none; opacity: 0.22; }
    }

    .tl-bulb-red { animation: tlCycleRed 11s infinite ease-in-out; }
    .tl-bulb-amber { animation: tlCycleAmber 11s infinite ease-in-out; }
    .tl-bulb-green { animation: tlCycleGreen 11s infinite ease-in-out; }

    /* Road Lane Stroke Classes */
    .road-lane-east {
        stroke: #FFFFFF;
        stroke-width: 2;
        stroke-dasharray: 14 18;
        animation: flowWestToEast 2.2s linear infinite;
    }
    .road-lane-west {
        stroke: #FFFFFF;
        stroke-width: 2;
        stroke-dasharray: 14 18;
        animation: flowEastToWest 2.2s linear infinite;
    }
    .road-lane-south {
        stroke: #FFFFFF;
        stroke-width: 2;
        stroke-dasharray: 14 18;
        animation: flowNorthToSouth 2.2s linear infinite;
    }
    .road-lane-north {
        stroke: #FFFFFF;
        stroke-width: 2;
        stroke-dasharray: 14 18;
        animation: flowSouthToNorth 2.2s linear infinite;
    }
    .road-lane-diagonal {
        stroke: #FFFFFF;
        stroke-width: 2;
        stroke-dasharray: 14 18;
        animation: flowWestToEast 2.4s linear infinite;
    }
    .telemetry-flow-line {
        stroke: #159A9C;
        stroke-width: 2.5;
        stroke-dasharray: 4 24;
        opacity: 0.45;
        animation: telemetryPulse 2.8s linear infinite;
    }

    /* Moving Vehicle Classes */
    .v-blue-sedan { animation: carMoveEast1 28s linear infinite; }
    .v-white-van { animation: carMoveEast2 34s linear -14s infinite; }
    .v-amber-taxi { animation: carMoveWest1 24s linear infinite; }
    .v-teal-bus { animation: busMoveSouth 30s linear infinite; }
    .v-green-car { animation: carMoveNorth 26s linear infinite; }
    .v-red-car { animation: carMoveDiagonal 32s linear -8s infinite; }

    /* Accessibility: Respect Reduced Motion Preference */
    @media (prefers-reduced-motion: reduce) {
        .road-lane-east, .road-lane-west, .road-lane-south, .road-lane-north,
        .road-lane-diagonal, .telemetry-flow-line,
        .v-blue-sedan, .v-white-van, .v-amber-taxi, .v-teal-bus, .v-green-car, .v-red-car,
        .tl-bulb-red, .tl-bulb-amber, .tl-bulb-green {
            animation: none !important;
        }
    }

    /* -------------------------------------------------------------------------
       HERO & BRANDING STYLING WITH PHYSICAL TRAFFIC SIGNAL HEAD
       ------------------------------------------------------------------------- */
    .hero-container {
        padding: 0.3rem 0 0.8rem 0;
        margin-bottom: 0.8rem;
        border-bottom: 2px dashed #E4E9E5;
        position: relative;
        z-index: 2;
    }
    .hero-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.2rem;
    }
    .hero-title-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Physical Hero Traffic Signal Head */
    .hero-signal-head {
        background: #17221B;
        border: 1.5px solid #2F3E35;
        border-radius: 12px;
        padding: 5px 6px;
        display: inline-flex;
        gap: 6px;
        align-items: center;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.12);
    }
    .hero-lens {
        width: 14px;
        height: 14px;
        border-radius: 50%;
    }
    .hero-lens.red {
        background: radial-gradient(circle at 35% 35%, #FFCDD2 0%, #E53935 70%);
        box-shadow: 0 0 6px rgba(229, 57, 53, 0.4);
    }
    .hero-lens.amber {
        background: radial-gradient(circle at 35% 35%, #FFF9C4 0%, #FFB300 70%);
        box-shadow: 0 0 6px rgba(255, 179, 0, 0.4);
    }
    .hero-lens.green {
        background: radial-gradient(circle at 35% 35%, #C8E6C9 0%, #43A047 70%);
        box-shadow: 0 0 10px rgba(67, 160, 71, 0.7);
    }

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
        margin-bottom: 1.1rem;
        position: relative;
        z-index: 2;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.94);
        border: 1px solid #E4E9E5;
        border-radius: 10px;
        color: #66736A;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 16px;
        transition: all 0.18s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #17221B;
        background-color: #FFFFFF;
        border-color: #CBD5CE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #17221B !important;
        border: 1.5px solid #43A047 !important;
        font-weight: 800 !important;
        box-shadow: 0 2px 8px rgba(67, 160, 71, 0.12) !important;
    }

    /* Clean UI Card Surfaces (Layer 2) */
    .traffic-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(228, 233, 229, 0.95);
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        box-shadow: 0 4px 18px rgba(23, 34, 27, 0.04);
        margin-bottom: 1.1rem;
        position: relative;
        z-index: 2;
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
        position: relative;
        z-index: 2;
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
        box-shadow: 0 8px 24px rgba(23, 34, 27, 0.08);
        position: relative;
        z-index: 3;
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
        position: relative;
        z-index: 2;
    }

    .kpi-result-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(228, 233, 229, 0.95);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 18px rgba(23, 34, 27, 0.04);
        position: relative;
        overflow: hidden;
        z-index: 2;
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
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(228, 233, 229, 0.95);
        border-left: 5px solid #43A047;
        border-radius: 12px;
        padding: 1.0rem 1.4rem;
        margin: 1.0rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        box-shadow: 0 4px 14px rgba(23, 34, 27, 0.04);
        position: relative;
        z-index: 2;
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
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(228, 233, 229, 0.95);
        border-radius: 10px;
        margin-bottom: 1.1rem;
        position: relative;
        z-index: 2;
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
        background: #F5F7F4;
        border: 1px solid #E4E9E5;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #17221B;
    }

    /* Small Insight Card */
    .insight-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(228, 233, 229, 0.95);
        border-radius: 12px;
        padding: 1.0rem 1.3rem;
        margin-top: 0.9rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        box-shadow: 0 4px 14px rgba(23, 34, 27, 0.03);
        position: relative;
        z-index: 2;
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
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(228, 233, 229, 0.95);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.0rem;
        box-shadow: 0 4px 14px rgba(23, 34, 27, 0.03);
        position: relative;
        z-index: 2;
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
        background: #F5F7F4;
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
        background: #F5F7F4;
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
        background: #F5F7F4;
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
        background: #F5F7F4;
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

<!-- LAYER 0: Fully Contained Fixed Background (Locked at z-index: 0, cannot cover content) -->
<div class="traffic-ambient-bg" aria-hidden="true">
    <svg class="traffic-city-canvas" viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <!-- Embedded Atmospheric Veil (Inside Layer 0 SVG, never floats as external overlay) -->
            <radialGradient id="ambient-veil-grad" cx="50%" cy="32%" r="65%">
                <stop offset="0%" stop-color="#F5F7F4" stop-opacity="0.75"/>
                <stop offset="65%" stop-color="#F5F7F4" stop-opacity="0.86"/>
                <stop offset="100%" stop-color="#F5F7F4" stop-opacity="0.94"/>
            </radialGradient>

            <!-- Vehicle Prototype: Blue Sedan -->
            <g id="car-blue">
                <rect x="-13" y="-7" width="26" height="14" rx="4" fill="rgba(0,0,0,0.18)" transform="translate(1, 2)"/>
                <rect x="-13" y="-7" width="26" height="14" rx="4" fill="#3182CE"/>
                <rect x="-6" y="-5" width="13" height="10" rx="2" fill="#1E293B" opacity="0.8"/>
                <rect x="-4" y="-4" width="9" height="8" rx="1.5" fill="#3182CE" opacity="0.95"/>
                <circle cx="12" cy="-4" r="1.5" fill="#FFFDE7"/>
                <circle cx="12" cy="4" r="1.5" fill="#FFFDE7"/>
                <circle cx="-13" cy="-4" r="1.2" fill="#E53935"/>
                <circle cx="-13" cy="4" r="1.2" fill="#E53935"/>
            </g>

            <!-- Vehicle Prototype: City Cab / Taxi -->
            <g id="car-taxi">
                <rect x="-13" y="-7" width="26" height="14" rx="4" fill="rgba(0,0,0,0.18)" transform="translate(1, 2)"/>
                <rect x="-13" y="-7" width="26" height="14" rx="4" fill="#FFB300"/>
                <rect x="-6" y="-5" width="13" height="10" rx="2" fill="#1E293B" opacity="0.8"/>
                <rect x="-4" y="-4" width="9" height="8" rx="1.5" fill="#FFB300" opacity="0.95"/>
                <rect x="-3" y="-2" width="6" height="4" rx="1" fill="#17221B"/>
                <circle cx="12" cy="-4" r="1.5" fill="#FFFDE7"/>
                <circle cx="12" cy="4" r="1.5" fill="#FFFDE7"/>
                <circle cx="-13" cy="-4" r="1.2" fill="#E53935"/>
                <circle cx="-13" cy="4" r="1.2" fill="#E53935"/>
            </g>

            <!-- Vehicle Prototype: Green Eco-Car -->
            <g id="car-green">
                <rect x="-13" y="-7" width="26" height="14" rx="4" fill="rgba(0,0,0,0.18)" transform="translate(1, 2)"/>
                <rect x="-13" y="-7" width="26" height="14" rx="4" fill="#43A047"/>
                <rect x="-6" y="-5" width="13" height="10" rx="2" fill="#1E293B" opacity="0.8"/>
                <rect x="-4" y="-4" width="9" height="8" rx="1.5" fill="#43A047" opacity="0.95"/>
                <circle cx="12" cy="-4" r="1.5" fill="#FFFDE7"/>
                <circle cx="12" cy="4" r="1.5" fill="#FFFDE7"/>
                <circle cx="-13" cy="-4" r="1.2" fill="#E53935"/>
                <circle cx="-13" cy="4" r="1.2" fill="#E53935"/>
            </g>

            <!-- Vehicle Prototype: Red Commuter -->
            <g id="car-red">
                <rect x="-13" y="-7" width="26" height="14" rx="4" fill="rgba(0,0,0,0.18)" transform="translate(1, 2)"/>
                <rect x="-13" y="-7" width="26" height="14" rx="4" fill="#E53935"/>
                <rect x="-6" y="-5" width="13" height="10" rx="2" fill="#1E293B" opacity="0.8"/>
                <rect x="-4" y="-4" width="9" height="8" rx="1.5" fill="#E53935" opacity="0.95"/>
                <circle cx="12" cy="-4" r="1.5" fill="#FFFDE7"/>
                <circle cx="12" cy="4" r="1.5" fill="#FFFDE7"/>
                <circle cx="-13" cy="-4" r="1.2" fill="#E53935"/>
                <circle cx="-13" cy="4" r="1.2" fill="#E53935"/>
            </g>

            <!-- Vehicle Prototype: Metro Transit Bus -->
            <g id="bus-teal">
                <rect x="-18" y="-8" width="38" height="16" rx="4" fill="rgba(0,0,0,0.2)" transform="translate(1, 2)"/>
                <rect x="-18" y="-8" width="38" height="16" rx="4" fill="#159A9C"/>
                <rect x="-14" y="-6" width="30" height="12" rx="2" fill="#1E293B" opacity="0.75"/>
                <rect x="-12" y="-5" width="26" height="10" rx="1.5" fill="#159A9C" opacity="0.9"/>
                <circle cx="19" cy="-5" r="1.6" fill="#FFFDE7"/>
                <circle cx="19" cy="5" r="1.6" fill="#FFFDE7"/>
                <circle cx="-18" cy="-5" r="1.3" fill="#E53935"/>
                <circle cx="-18" cy="5" r="1.3" fill="#E53935"/>
            </g>

            <!-- Vehicle Prototype: White Fleet Van -->
            <g id="van-white">
                <rect x="-15" y="-7.5" width="30" height="15" rx="3" fill="rgba(0,0,0,0.18)" transform="translate(1, 2)"/>
                <rect x="-15" y="-7.5" width="30" height="15" rx="3" fill="#FFFFFF" stroke="#CBD5CE" stroke-width="0.8"/>
                <rect x="-4" y="-5.5" width="14" height="11" rx="1.5" fill="#1E293B" opacity="0.75"/>
                <circle cx="14" cy="-4.5" r="1.5" fill="#FFFDE7"/>
                <circle cx="14" cy="4.5" r="1.5" fill="#FFFDE7"/>
                <circle cx="-15" cy="-4.5" r="1.2" fill="#E53935"/>
                <circle cx="-15" cy="4.5" r="1.2" fill="#E53935"/>
            </g>
        </defs>

        <!-- Base Background Fill -->
        <rect width="100%" height="100%" fill="#F5F7F4"/>

        <!-- 1. DISTANT GEOMETRIC SKYLINE (Subtle 0.045 opacity) -->
        <g class="city-skyline" opacity="0.045" fill="#17221B">
            <rect x="40" y="45" width="60" height="85" rx="2"/>
            <rect x="110" y="25" width="45" height="105" rx="2"/>
            <polygon points="132,10 130,25 135,25"/>
            <rect x="165" y="55" width="70" height="75" rx="2"/>
            <rect x="245" y="35" width="50" height="95" rx="2"/>
            <rect x="310" y="65" width="80" height="65" rx="2"/>
            <rect x="520" y="30" width="55" height="100" rx="2"/>
            <rect x="585" y="48" width="75" height="82" rx="2"/>
            <rect x="670" y="20" width="50" height="110" rx="2"/>
            <polygon points="695,5 693,20 697,20"/>
            <rect x="730" y="50" width="65" height="80" rx="2"/>
            <rect x="980" y="40" width="50" height="90" rx="2"/>
            <rect x="1040" y="25" width="65" height="105" rx="2"/>
            <rect x="1115" y="55" width="75" height="75" rx="2"/>
            <rect x="1340" y="35" width="55" height="95" rx="2"/>
            <rect x="1405" y="20" width="50" height="110" rx="2"/>
            <rect x="1465" y="45" width="70" height="85" rx="2"/>
        </g>

        <!-- 2. ROAD NETWORK GEOMETRY (Light Asphalt #D9DEDA with Lane Edges) -->
        <!-- Top Horizontal Expressway (y = 124 to 196, width 72px) -->
        <rect x="0" y="124" width="1600" height="72" fill="#D9DEDA"/>
        <line x1="0" y1="124" x2="1600" y2="124" stroke="#CBD2CC" stroke-width="2"/>
        <line x1="0" y1="196" x2="1600" y2="196" stroke="#CBD2CC" stroke-width="2"/>
        <line x1="0" y1="160" x2="1600" y2="160" stroke="#BFC7C0" stroke-width="2.5"/>

        <!-- Left Vertical Arterial Boulevard (x = 84 to 148, width 64px) -->
        <rect x="84" y="0" width="64" height="1000" fill="#D9DEDA"/>
        <line x1="84" y1="0" x2="84" y2="1000" stroke="#CBD2CC" stroke-width="2"/>
        <line x1="148" y1="0" x2="148" y2="1000" stroke="#CBD2CC" stroke-width="2"/>
        <line x1="116" y1="0" x2="116" y2="1000" stroke="#BFC7C0" stroke-width="2"/>

        <!-- Right Vertical Arterial Boulevard (x = 1448 to 1512, width 64px) -->
        <rect x="1448" y="0" width="64" height="1000" fill="#D9DEDA"/>
        <line x1="1448" y1="0" x2="1448" y2="1000" stroke="#CBD2CC" stroke-width="2"/>
        <line x1="1512" y1="0" x2="1512" y2="1000" stroke="#CBD2CC" stroke-width="2"/>
        <line x1="1480" y1="0" x2="1480" y2="1000" stroke="#BFC7C0" stroke-width="2"/>

        <!-- Diagonal Sweeping Highway (Lower quadrant) -->
        <path d="M -20,740 Q 400,680 800,580 T 1620,400" fill="none" stroke="#D9DEDA" stroke-width="60"/>
        <path d="M -20,710 Q 400,650 800,550 T 1620,370" fill="none" stroke="#CBD2CC" stroke-width="2"/>
        <path d="M -20,770 Q 400,710 800,610 T 1620,430" fill="none" stroke="#CBD2CC" stroke-width="2"/>

        <!-- Intersection Markings & Crosswalk Stripes -->
        <!-- Left Intersection (x: 84 to 148, y: 124 to 196) -->
        <g stroke="#FFFFFF" stroke-width="2" opacity="0.85">
            <line x1="72" y1="130" x2="72" y2="190" stroke-dasharray="4 6"/>
            <line x1="160" y1="130" x2="160" y2="190" stroke-dasharray="4 6"/>
            <line x1="90" y1="112" x2="142" y2="112" stroke-dasharray="4 6"/>
            <line x1="90" y1="208" x2="142" y2="208" stroke-dasharray="4 6"/>
            <line x1="78" y1="126" x2="78" y2="158" stroke-width="3"/>
            <line x1="154" y1="162" x2="154" y2="194" stroke-width="3"/>
        </g>

        <!-- Right Intersection (x: 1448 to 1512, y: 124 to 196) -->
        <g stroke="#FFFFFF" stroke-width="2" opacity="0.85">
            <line x1="1436" y1="130" x2="1436" y2="190" stroke-dasharray="4 6"/>
            <line x1="1524" y1="130" x2="1524" y2="190" stroke-dasharray="4 6"/>
            <line x1="1454" y1="112" x2="1506" y2="112" stroke-dasharray="4 6"/>
            <line x1="1454" y1="208" x2="1506" y2="208" stroke-dasharray="4 6"/>
            <line x1="1442" y1="126" x2="1442" y2="158" stroke-width="3"/>
            <line x1="1518" y1="162" x2="1518" y2="194" stroke-width="3"/>
        </g>

        <!-- 3. ANIMATED MOVING LANE MARKINGS (Continuous GPU flow) -->
        <line x1="0" y1="142" x2="1600" y2="142" class="road-lane-east"/>
        <line x1="0" y1="178" x2="1600" y2="178" class="road-lane-west"/>
        <line x1="100" y1="0" x2="100" y2="1000" class="road-lane-south"/>
        <line x1="132" y1="0" x2="132" y2="1000" class="road-lane-north"/>
        <line x1="1464" y1="0" x2="1464" y2="1000" class="road-lane-south"/>
        <line x1="1496" y1="0" x2="1496" y2="1000" class="road-lane-north"/>
        <path d="M -20,740 Q 400,680 800,580 T 1620,400" fill="none" class="road-lane-diagonal"/>

        <!-- 4. PREDICTIVE TELEMETRY STREAM (Subtle data pulses) -->
        <path d="M 0,142 Q 400,142 800,142 T 1600,142" fill="none" class="telemetry-flow-line"/>
        <path d="M -20,740 Q 400,680 800,580 T 1620,400" fill="none" class="telemetry-flow-line" opacity="0.35"/>

        <!-- 5. INTERSECTION TRAFFIC SIGNALS (Coordinated 3-lens masts) -->
        <g class="traffic-light-rig" transform="translate(168, 98)">
            <line x1="5" y1="28" x2="5" y2="40" stroke="#2F3E35" stroke-width="2.5"/>
            <rect x="0" y="0" width="10" height="28" rx="3" fill="#17221B" stroke="#2F3E35" stroke-width="0.8"/>
            <circle cx="5" cy="5.5" r="3" class="tl-bulb-red"/>
            <circle cx="5" cy="14" r="3" class="tl-bulb-amber"/>
            <circle cx="5" cy="22.5" r="3" class="tl-bulb-green"/>
        </g>

        <g class="traffic-light-rig" transform="translate(1432, 205)">
            <line x1="5" y1="0" x2="5" y2="12" stroke="#2F3E35" stroke-width="2.5"/>
            <rect x="0" y="12" width="10" height="28" rx="3" fill="#17221B" stroke="#2F3E35" stroke-width="0.8"/>
            <circle cx="5" cy="17.5" r="3" class="tl-bulb-green" style="animation-delay: -5.5s;"/>
            <circle cx="5" cy="26" r="3" class="tl-bulb-amber" style="animation-delay: -5.5s;"/>
            <circle cx="5" cy="34.5" r="3" class="tl-bulb-red" style="animation-delay: -5.5s;"/>
        </g>

        <!-- 6. MOVING VEHICLES (Pure CSS animated SVG vehicles) -->
        <g class="v-blue-sedan"><use href="#car-blue"/></g>
        <g class="v-white-van"><use href="#van-white"/></g>
        <g class="v-amber-taxi"><use href="#car-taxi"/></g>
        <g class="v-teal-bus"><use href="#bus-teal"/></g>
        <g class="v-green-car"><use href="#car-green"/></g>
        <g class="v-red-car"><use href="#car-red"/></g>

        <!-- 7. EMBEDDED ATMOSPHERIC DIFFUSION VEIL (Locked inside SVG Layer 0) -->
        <rect width="100%" height="100%" fill="url(#ambient-veil-grad)" pointer-events="none"/>
    </svg>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. CACHED ENGINE & DATASET LOADERS (active pipeline: Traffic dataset.csv)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Traffic Intelligence Models...")
def load_new_engine():
    return get_new_commute_engine()


@st.cache_resource(show_spinner="Loading Location & Dataset Info...")
def load_dataset_meta():
    try:
        df = preprocess_new_traffic_data()
        validate_dataset_columns(df)
    except Exception as e:
        st.error(f"❌ Dataset Validation Failed: {e}")
        st.stop()
    city_area_map = get_city_area_mapping(df)
    cities = get_cities(df)
    vehicle_types = get_vehicle_types(df)
    return city_area_map, cities, vehicle_types


new_engine = load_new_engine()
city_area_map, all_cities, all_vehicle_types = load_dataset_meta()

# Convenience alias so the predictor object is accessible for Insights charts
new_predictor = new_engine.predictor


# -----------------------------------------------------------------------------
# 4. HERO SECTION WITH PHYSICAL TRAFFIC SIGNAL IDENTITY
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-top-row">
        <div class="hero-title-group">
            <!-- Physical 3-Lamp Traffic Signal Head -->
            <div class="hero-signal-head" title="Live Traffic Signal System">
                <div class="hero-lens red"></div>
                <div class="hero-lens amber"></div>
                <div class="hero-lens green"></div>
            </div>
            <div class="hero-brand">COMMUTECAST<span class="hero-tag">Smart Traffic Prediction</span></div>
        </div>
        <div style="background: rgba(255, 255, 255, 0.9); border: 1px solid #E4E9E5; border-radius: 20px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; color: #66736A; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
            📍 City Traffic Dataset — Maharashtra
        </div>
    </div>
    <div class="hero-subtitle">
        Understand your expected traffic and congestion probability before you commute.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. PRIMARY NAVIGATION TABS (EXACTLY FOUR TABS)
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
    # INPUT SECTION: Clean Central Card ("Historical Traffic Analysis")
    st.markdown("""
    <div class="traffic-card">
        <div class="traffic-card-header">
            <span>🗺️</span>
            <span>Historical Traffic Analysis</span>
        </div>
        <div class="traffic-card-sub">
            Select City, Area and Vehicle Type to evaluate expected vehicle count and congestion probability based on historical observations.
        </div>
    """, unsafe_allow_html=True)

    # --- LOCATION & VEHICLE SELECTOR: City + Area + Vehicle Type ---
    st.markdown("<div style='font-size: 0.85rem; font-weight: 800; color: #17221B; margin-bottom: 0.35rem;'>📍 Select City, Area and Vehicle Type</div>", unsafe_allow_html=True)
    loc_c1, loc_c2, loc_c3 = st.columns([1.2, 1.2, 1.0])

    with loc_c1:
        city_display_options = ["Select city"] + all_cities
        if st.session_state.selected_city in all_cities:
            city_idx = all_cities.index(st.session_state.selected_city) + 1
        else:
            city_idx = 0
        sel_city_raw = st.selectbox(
            "CITY",
            options=city_display_options,
            index=city_idx,
            help="Select the city from the dataset locations.",
            key="city_selector"
        )
        if sel_city_raw == "Select city":
            sel_city = None
            st.session_state.selected_city = None
            st.session_state.selected_area = None
            if "area_selector" in st.session_state:
                st.session_state["area_selector"] = "Select area"
        else:
            if sel_city_raw != st.session_state.selected_city:
                # City changed — reset area
                st.session_state.selected_area = None
                if "area_selector" in st.session_state:
                    st.session_state["area_selector"] = "Select area"
            sel_city = sel_city_raw
            st.session_state.selected_city = sel_city

    with loc_c2:
        # Area options depend on selected city (strict cascading)
        if sel_city and sel_city in city_area_map:
            available_areas = city_area_map[sel_city]
        else:
            available_areas = []
        area_display_options = ["Select area"] + available_areas
        if st.session_state.selected_area in available_areas:
            area_idx = available_areas.index(st.session_state.selected_area) + 1
        else:
            area_idx = 0
            st.session_state.selected_area = None
            if "area_selector" in st.session_state and st.session_state["area_selector"] not in available_areas:
                st.session_state["area_selector"] = "Select area"
        sel_area_raw = st.selectbox(
            "AREA",
            options=area_display_options,
            index=area_idx,
            help="Area options update based on the selected city.",
            key="area_selector"
        )
        if sel_area_raw == "Select area" or sel_area_raw not in available_areas:
            sel_area = None
            st.session_state.selected_area = None
        else:
            sel_area = sel_area_raw
            st.session_state.selected_area = sel_area

    with loc_c3:
        vt_display_options = ["Select type"] + all_vehicle_types
        if st.session_state.selected_vehicle_type in all_vehicle_types:
            vt_idx = all_vehicle_types.index(st.session_state.selected_vehicle_type) + 1
        else:
            vt_idx = 0
        sel_vt_raw = st.selectbox(
            "VEHICLE TYPE",
            options=vt_display_options,
            index=vt_idx,
            help="Vehicle type from the dataset.",
            key="vehicle_type_selector"
        )
        if sel_vt_raw == "Select type":
            sel_vehicle_type = None
            st.session_state.selected_vehicle_type = None
        else:
            sel_vehicle_type = sel_vt_raw
            st.session_state.selected_vehicle_type = sel_vehicle_type

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Detect if inputs have changed since last analysis ----
    current_inputs = (st.session_state.selected_city, st.session_state.selected_area,
                      st.session_state.selected_vehicle_type)
    if st.session_state.last_analyzed_inputs is not None and current_inputs != st.session_state.last_analyzed_inputs:
        # Input changed after analysis — invalidate previous result
        st.session_state.analyzed = False

    # ANALYZE BUTTON
    btn_c1, btn_c2, btn_c3 = st.columns([1, 1.6, 1])
    with btn_c2:
        analyze_clicked = st.button("🚦 ANALYZE TRAFFIC", use_container_width=True)

    anim_slot = st.empty()

    if analyze_clicked:
        # VALIDATION: Location must be selected first
        missing = []
        if sel_city is None:
            missing.append("City")
        if sel_area is None:
            missing.append("Area")
        if sel_vehicle_type is None:
            missing.append("Vehicle Type")

        if missing:
            st.warning(f"⚠️ Please select a location before analyzing traffic. **{', '.join(missing)}** required.")
            st.session_state.analyzed = False
        elif sel_city and sel_area and sel_area not in city_area_map.get(sel_city, []):
            st.error(f"⚠️ Validation Error: Area '{sel_area}' does not belong to City '{sel_city}'.")
            st.session_state.analyzed = False
        else:
            # Traffic Signal Animation (RED → AMBER → GREEN)
            stages = [
                ("red", "Reading location data...", f"Loading {sel_city} — {sel_area} traffic history..."),
                ("amber", "Analyzing traffic pattern...", f"Computing empirical expectation and conditional probability for {sel_vehicle_type}..."),
                ("green", "Analysis ready", f"Historical traffic outlook synthesized for {sel_city} — {sel_area} ({sel_vehicle_type}).")
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
            st.session_state.last_analyzed_inputs = current_inputs

    # -------------------------------------------------------------------------
    # RESULTS — Only shown after successful analysis
    # -------------------------------------------------------------------------
    if st.session_state.analyzed and st.session_state.last_analyzed_inputs is not None:
        # Use the saved inputs from the last successful analysis
        _city, _area, _vt = st.session_state.last_analyzed_inputs

        res = new_engine.analyze(
            city=_city,
            area=_area,
            vehicle_type=_vt,
            polynomial_degree=DEFAULT_SELECTED_DEGREE
        )

        expected_traffic = res["expected_vehicle_count"]
        p_high_cond = res["high_congestion_probability"]
        p_high_base = res["baseline_high_probability"]
        diff_ppt = res["difference_percentage_points"]
        condition_used = res["condition_used"]
        fallback = res.get("fallback", False)

        st.markdown("""
        <div class="result-section-header">
            <span>🚦</span>
            <span>Your Commute Outlook</span>
        </div>
        """, unsafe_allow_html=True)

        # LOCATION DISPLAY
        st.markdown(f"""
        <div style="background: #F0FDF4; border: 1px solid #A7F3D0; border-radius: 10px; padding: 0.6rem 1.0rem;
                    margin-bottom: 0.8rem; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.1rem;">📍</span>
            <div>
                <span style="font-size: 0.98rem; font-weight: 800; color: #17221B;">{_city} — {_area}</span>
                <span style="font-size: 0.8rem; color: #66736A; margin-left: 8px;">Vehicle: {_vt}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # TWO LARGE RESULT CARDS SIDE-BY-SIDE
        res_c1, res_c2 = st.columns(2, gap="medium")

        with res_c1:
            st.markdown(f"""
            <div class="kpi-result-card left-poly">
                <div class="kpi-label">EXPECTED TRAFFIC</div>
                <div class="kpi-val">{expected_traffic:,.0f}<span class="kpi-unit">vehicles</span></div>
                <div class="kpi-subtext">Empirical mean: {condition_used} ({res.get('subset_rows', 0)} records){' • ' + res.get('fallback_level', '') if fallback else ''}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            risk_class = "high-risk" if p_high_cond > 0.50 else "low-risk" if p_high_cond <= 0.25 else ""
            st.markdown(f"""
            <div class="kpi-result-card right-prob {risk_class}">
                <div class="kpi-label">HIGH CONGESTION</div>
                <div class="kpi-val">{p_high_cond * 100:.2f}%<span class="kpi-unit">probability</span></div>
                <div class="kpi-subtext">P(High | {condition_used}) ({res.get('subset_rows', 0)} records)</div>
            </div>
            """, unsafe_allow_html=True)

        # DEBUG TELEMETRY AUDIT INFORMATION
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px dashed #94A3B8; border-radius: 8px; padding: 0.55rem 0.9rem;
                    margin-top: 0.75rem; margin-bottom: 0.8rem; font-family: monospace; font-size: 0.78rem; color: #334155;">
            <div style="font-weight: 700; color: #1E293B; margin-bottom: 2px;">🔧 PREDICTION TELEMETRY AUDIT</div>
            <div>City: <strong>{_city}</strong> &nbsp;|&nbsp; Area: <strong>{_area}</strong> &nbsp;|&nbsp; Vehicle Type: <strong>{_vt}</strong></div>
            <div>Matching rows: <strong>{res.get('subset_rows', 0)}</strong> &nbsp;|&nbsp; Fallback: <strong>{res.get('fallback_level', 'Exact')}</strong> &nbsp;|&nbsp; Expected Vehicle Count: <strong>{expected_traffic:,.2f}</strong></div>
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

        # HISTORICAL SELECTION SUMMARY PILLS
        st.markdown(f"""
        <div class="summary-pills-row">
            <span class="summary-pill-label">HISTORICAL FILTER:</span>
            <span class="commute-pill">📍 {_city} — {_area}</span>
            <span class="commute-pill">🚗 Vehicle: {_vt}</span>
        </div>
        """, unsafe_allow_html=True)

        # VEHICLE COUNT TREND CHART (polynomial curve over time)
        st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin: 0.6rem 0 0.2rem 0;'>Vehicle Count Trend</div>", unsafe_allow_html=True)
        st.caption("Daily mean vehicle count across the dataset period with Degree-3 polynomial fit")

        fig_trend, ax_trend = plt.subplots(figsize=(8, 3.2), dpi=120)
        fig_trend.patch.set_facecolor('#FFFFFF')
        ax_trend.set_facecolor('#FCFDFC')

        daily_df = new_predictor.daily_df
        X_days = daily_df["day_num"].values
        y_vcount = daily_df["mean_vehicle_count"].values
        x_dense_d = np.linspace(X_days.min(), X_days.max(), 300)
        models_d = new_predictor.models

        ax_trend.scatter(X_days, y_vcount, color="#66736A", s=8, alpha=0.5, zorder=3, label="Daily Mean Vehicle Count")
        ax_trend.plot(x_dense_d, models_d[3]["poly1d"](x_dense_d), color="#43A047", linewidth=2.2,
                      label="Degree-3 Polynomial Fit", zorder=4)

        for spine in ax_trend.spines.values():
            spine.set_color('#E4E9E5')
        ax_trend.tick_params(colors='#66736A', labelsize=8)
        ax_trend.set_xlabel("Day Number (from Jan 1, 2025)", color='#17221B', fontsize=8.5, fontweight="bold")
        ax_trend.set_ylabel("Vehicle Count", color='#17221B', fontsize=8.5, fontweight="bold")
        ax_trend.grid(True, linestyle="--", alpha=0.45, color='#E4E9E5')
        ax_trend.legend(loc="upper right", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#E4E9E5", labelcolor="#17221B")

        plt.tight_layout()
        st.pyplot(fig_trend, use_container_width=True)
        plt.close(fig_trend)

        # KEY INSIGHT CARD
        diff_phrase = f"{abs(diff_ppt):.2f} percentage points {'above' if diff_ppt >= 0 else 'below'}"
        vt_prob = res.get('p_high_vehicle_type', p_high_base)
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-icon">💡</div>
            <div class="insight-text">
                At <strong>{_city} — {_area}</strong>, historical observations show a
                <strong>{p_high_cond * 100:.2f}%</strong> high-congestion probability
                ({condition_used}), compared with the
                <strong>{p_high_base * 100:.2f}%</strong> dataset baseline ({diff_phrase}).
                Vehicle type ({_vt}) probability: <strong>{vt_prob * 100:.2f}%</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # PRE-ANALYSIS EMPTY STATE
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem 1.5rem; background: rgba(255,255,255,0.85);
                    border: 1px dashed #CBD5CE; border-radius: 14px; margin-top: 1.2rem; position: relative; z-index: 2;">
            <div style="font-size: 2rem; margin-bottom: 0.6rem;">🚦</div>
            <div style="font-size: 1.02rem; font-weight: 800; color: #17221B; margin-bottom: 0.4rem;">Your commute outlook will appear here after analysis.</div>
            <div style="font-size: 0.85rem; color: #66736A;">Complete your commute inputs above and click <strong>Analyze Traffic</strong> to view your prediction.</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: 🧮 CALCULATIONS (ACADEMIC & MATHEMATICAL FOUNDATIONS)
# =============================================================================
with tab_calc:
    st.markdown("""
    <div style="margin-bottom: 1.0rem;">
        <div style="font-size: 1.25rem; font-weight: 900; color: #17221B;">Academic &amp; Mathematical Methodology</div>
        <div style="font-size: 0.85rem; color: #66736A;">
            Complete derivation of mathematical formulas, regression equations, empirical conditional probabilities, and Bayesian verification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analyzed or st.session_state.last_analyzed_inputs is None:
        # Placeholder before analysis
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem 1.5rem; background: rgba(255,255,255,0.85);
                    border: 1px dashed #CBD5CE; border-radius: 14px; margin-top: 0.5rem; position: relative; z-index: 2;">
            <div style="font-size: 2rem; margin-bottom: 0.6rem;">🧮</div>
            <div style="font-size: 1.02rem; font-weight: 800; color: #17221B; margin-bottom: 0.4rem;">Run a commute analysis to view the calculations.</div>
            <div style="font-size: 0.85rem; color: #66736A;">Complete your commute inputs on the <strong>CommuteCast</strong> tab and click <strong>Analyze Traffic</strong> to see the full mathematical derivation for your selected conditions.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Retrieve last analyzed inputs for calculations
        if len(st.session_state.last_analyzed_inputs) == 3:
            _city, _area, _vt = st.session_state.last_analyzed_inputs
        else:
            _city, _area, _vt = st.session_state.last_analyzed_inputs[:3]

        res_c = new_engine.analyze(
            city=_city,
            area=_area,
            vehicle_type=_vt,
            polynomial_degree=DEFAULT_SELECTED_DEGREE
        )
        c_expected_traffic = res_c["expected_vehicle_count"]
        c_p_high_cond = res_c["high_congestion_probability"]
        c_p_high_base = res_c["baseline_high_probability"]
        c_diff_ppt = res_c["difference_percentage_points"]
        c_condition_used = res_c["condition_used"]
        c_fallback = res_c.get("fallback", False)
        c_loc_n = res_c["location_obs"]

        # Get polynomial model metrics from predictor
        _poly_models = new_predictor.models
        _poly_eval = new_predictor.eval_df
        _deg3_row = _poly_eval[_poly_eval['Degree'] == 3].iloc[0]
        _deg3_eq = _poly_models[3]['equation_str']

        # 1. EMPIRICAL EXPECTATION
        _c_high_cnt = res_c.get("high_count", 0)
        _c_var = res_c.get("variance_sample", 0.0)
        _c_std = res_c.get("std_sample", 0.0)
        st.markdown(f"""
        <div class="calc-card">
            <div class="calc-card-badge">Section 1 • Empirical Expectation (Location &amp; Vehicle)</div>
            <div class="calc-card-title">Sample Expectation — E[Vehicle_Count | {c_condition_used}]</div>
            <div class="calc-formula-box">
                E[Vehicle_Count] = (1 / n) × Σ x_i = {c_expected_traffic:,.2f} vehicles
            </div>
            <div class="calc-explanation">
                Calculated directly from the <strong>{c_loc_n:,} matching dataset rows</strong> in <em>Traffic dataset.csv</em>.<br>
                Condition: <strong>{c_condition_used}</strong> • Level: <strong>{res_c.get('fallback_level', 'Exact')}</strong><br>
                Sample variance: s² = {_c_var:,.1f} • Standard deviation: s = {_c_std:,.1f} vehicles.
                {'(Fallback used: ' + res_c.get("fallback_reason", "") + ')' if c_fallback else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. CONDITIONAL PROBABILITY — Location
        st.markdown(f"""
        <div class="calc-card">
            <div class="calc-card-badge">Section 2 • Probability Foundations</div>
            <div class="calc-card-title">Empirical Conditional Probability Analysis</div>
            <div class="calc-formula-box">
                P(High | {c_condition_used}) = count(High) / n = {_c_high_cnt} / {c_loc_n} = {c_p_high_cond * 100:.2f}%
            </div>
            <div class="calc-explanation">
                Empirical relative frequency of high congestion (Congestion_Level == 'High') observed
                for the selected subset ({c_loc_n:,} observations used).<br>
                Baseline dataset P(High) = {c_p_high_base * 100:.2f}% (difference: <strong>{'+' if c_diff_ppt >= 0 else ''}{c_diff_ppt:.2f} percentage points</strong>).
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2b. Conditional by Vehicle Type
        _vt_prob = res_c.get('p_high_vehicle_type', new_engine.p_high_baseline)
        _vt_n = res_c.get('vehicle_type_obs', 0)
        st.markdown(f"""
        <div class="calc-card">
            <div class="calc-card-badge">Section 2b • Conditional Probability — Vehicle Type</div>
            <div class="calc-card-title">P(High | Vehicle_Type)</div>
            <div class="calc-formula-box">
                P(High | Vehicle_Type = {_vt}) = {_vt_prob * 100:.2f}%
            </div>
            <div class="calc-explanation">
                Based on {_vt_n:,} observations of {_vt} across all locations in the dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. BASELINE
        st.markdown(f"""
        <div class="calc-card">
            <div class="calc-card-badge">Section 3 • Benchmark Baseline</div>
            <div class="calc-card-title">Overall Congestion Baseline</div>
            <div class="calc-formula-box">
                P(High) = {c_p_high_base * 100:.2f}%
            </div>
            <div class="calc-explanation">
                Evaluated unconditionally across all 1,200 observations. By definition of the 75th percentile
                (Q75 = 12,237.75 vehicles), exactly 25.00% of all records experience high congestion.
                <br>Current condition difference: <strong>{'+' if c_diff_ppt >= 0 else ''}{c_diff_ppt:.2f} percentage points</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4. BAYESIAN ANALYSIS — Vehicle Type
        bayes_vt = new_engine.explain_bayes(_vt)
        st.markdown(f"""
        <div class="calc-card">
            <div class="calc-card-badge">Section 4 • Bayesian Reasoning</div>
            <div class="calc-card-title">Bayes' Theorem — P(High | Vehicle_Type = {_vt})</div>
            <div class="calc-formula-box">
                P(High | {_vt}) = [ P({_vt} | High) × P(High) ] / P({_vt})
            </div>
            <div class="calc-explanation">
                Bayes' theorem formalizes belief updating: prior belief P(High) is updated upon observing
                evidence (Vehicle_Type = {_vt}) to arrive at the posterior probability.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"🎲 Detailed Bayesian Verification — Vehicle_Type = {_vt}", expanded=True):
            b_c1, b_c2 = st.columns(2)
            with b_c1:
                st.markdown(f"""
                <div style="background: #F5F7F4; border: 1px solid #E4E9E5; border-radius: 10px; padding: 0.9rem;">
                    <strong style="color: #17221B; font-size: 0.9rem;">Bayes Calculation: P(High | {_vt})</strong>
                    <ul style="font-size: 0.82rem; color: #66736A; margin: 6px 0 0 16px; padding: 0;">
                        <li>Prior P(High): <strong>{bayes_vt['prior_P_High']*100:.4f}%</strong></li>
                        <li>Marginal P({_vt}): <strong>{bayes_vt['marginal_P_VehicleType']*100:.4f}%</strong></li>
                        <li>Likelihood P({_vt} | High): <strong>{bayes_vt['likelihood_P_VehicleType_given_High']*100:.4f}%</strong></li>
                        <li>Bayes Posterior: <strong>{bayes_vt['bayes_posterior']*100:.4f}%</strong></li>
                        <li>Empirical Posterior: <strong>{bayes_vt['direct_empirical_posterior']*100:.4f}%</strong></li>
                        <li>Discrepancy: <strong>{bayes_vt['absolute_difference']:.2e}</strong> {'(Exact Match)' if bayes_vt['is_verified'] else '(Check)'}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with b_c2:
                st.markdown(f"""
                <div style="background: #F5F7F4; border: 1px solid #E4E9E5; border-radius: 10px; padding: 0.9rem;">
                    <strong style="color: #17221B; font-size: 0.9rem;">P(High | Vehicle_Type) — All Types</strong>
                    <ul style="font-size: 0.82rem; color: #66736A; margin: 6px 0 0 16px; padding: 0;">
                        {''.join(f'<li>{vt}: <strong>{new_engine.p_high_by_vehicle_type[vt]*100:.2f}%</strong></li>' for vt in new_engine.vehicle_types)}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        # 5. POLYNOMIAL METRICS
        st.markdown(f"""
        <div class="calc-card">
            <div class="calc-card-badge">Section 5 • Goodness-of-Fit</div>
            <div class="calc-card-title">Polynomial Curve Fitting &amp; Goodness-of-Fit (Degree 1, 2, 3)</div>
            <div class="calc-formula-box">
                {_deg3_eq}
            </div>
            <div class="mini-stat-grid">
                <div class="mini-stat-card"><div class="mini-stat-val">{_poly_eval[_poly_eval['Degree']==3]['MAE'].values[0]:,.1f}</div><div class="mini-stat-lbl">MAE Deg-3</div></div>
                <div class="mini-stat-card"><div class="mini-stat-val">{_poly_eval[_poly_eval['Degree']==3]['RMSE'].values[0]:,.1f}</div><div class="mini-stat-lbl">RMSE Deg-3</div></div>
                <div class="mini-stat-card"><div class="mini-stat-val">{_poly_eval[_poly_eval['Degree']==3]['R2'].values[0]:.4f}</div><div class="mini-stat-lbl">R² Deg-3</div></div>
                <div class="mini-stat-card"><div class="mini-stat-val">{_poly_eval[_poly_eval['Degree']==2]['R2'].values[0]:.4f}</div><div class="mini-stat-lbl">R² Deg-2</div></div>
                <div class="mini-stat-card"><div class="mini-stat-val">{_poly_eval[_poly_eval['Degree']==1]['R2'].values[0]:.4f}</div><div class="mini-stat-lbl">R² Deg-1</div></div>
            </div>
            <div class="calc-explanation">
                Degree 3 polynomial fitted on daily aggregated Vehicle_Count vs day_num across 2025.<br>
                Note: The dataset spans a single calendar year (2025) with no strong temporal drift;
                R² values reflect flat distribution rather than modeling failure.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 6. CONGESTION QUANTILES
        _stats = new_engine.get_stats()
        st.markdown(f"""
        <div class="calc-card">
            <div class="calc-card-badge">Section 6 • Quantile Boundaries</div>
            <div class="calc-card-title">Congestion Thresholds (Vehicle_Count Quantiles)</div>
            <div class="mini-stat-grid">
                <div class="mini-stat-card"><div class="mini-stat-val">{new_engine.q25:,.2f}</div><div class="mini-stat-lbl">Q25 Boundary</div></div>
                <div class="mini-stat-card"><div class="mini-stat-val">{new_engine.q50:,.1f}</div><div class="mini-stat-lbl">Median (Q50)</div></div>
                <div class="mini-stat-card"><div class="mini-stat-val">{new_engine.q75:,.2f}</div><div class="mini-stat-lbl">Q75 (High Boundary)</div></div>
                <div class="mini-stat-card"><div class="mini-stat-val">{_stats['mean']:,.1f}</div><div class="mini-stat-lbl">Mean Vehicle Count</div></div>
                <div class="mini-stat-card"><div class="mini-stat-val">{_stats['variance_sample']:,.0f}</div><div class="mini-stat-lbl">Sample Variance</div></div>
            </div>
            <div class="calc-explanation">
                High congestion is statistically defined as Vehicle_Count &gt; Q75 = {new_engine.q75:,.2f}.
                Exactly 25.00% of the 1,200 observations fall in the High category.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 7. COVARIANCE
        _covs = new_engine.get_covariances()
        st.markdown(f"""
        <div class="calc-card">
            <div class="calc-card-badge">Section 7 • Expectation &amp; Covariance</div>
            <div class="calc-card-title">Sample Covariance Analysis</div>
            <div class="calc-formula-box">
                E[Vehicle_Count] = {_stats['mean']:,.2f} vehicles
            </div>
            <div class="calc-explanation">
                Cov(Vehicle_Count, Accidents) = {_covs.get('cov_VehicleCount_Accidents', 0):,.2f}
                (r = {_covs.get('corr_VehicleCount_Accidents', 0):.4f})<br>
                Cov(Vehicle_Count, Traffic_Violations) = {_covs.get('cov_VehicleCount_Traffic_Violations', 0):,.2f}
                (r = {_covs.get('corr_VehicleCount_Traffic_Violations', 0):.4f})<br>
                Cov(Vehicle_Count, Avg_Speed_kmph) = {_covs.get('cov_VehicleCount_Avg_Speed_kmph', 0):,.2f}
                (r = {_covs.get('corr_VehicleCount_Avg_Speed_kmph', 0):.4f})<br>
                <em>Covariance is NOT causation. These represent empirical linear co-variation only.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Core Mathematical Separation Note
        st.markdown("""
        <div class="separation-callout">
            <strong>⚠️ Core Methodological Principle: Strict Mathematical Separation</strong><br>
            Expected traffic volume is calculated directly as the empirical sample mean of historical observations matching the selected City, Area, and Vehicle Type: <em>E[Vehicle_Count | City, Area, Vehicle_Type]</em>.<br>
            Congestion probability is computed via empirical relative frequencies: <em>P(High | City, Area, Vehicle_Type)</em>.<br>
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
            Exploratory visual analytics comparing historical daily patterns, conditional probabilities, and baseline distributions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load chart data from new predictor
    _ins_daily_df = new_predictor.daily_df
    _ins_X_days = _ins_daily_df["day_num"].values
    _ins_y_means = _ins_daily_df["mean_vehicle_count"].values
    _ins_x_dense = np.linspace(_ins_X_days.min(), _ins_X_days.max(), 300)
    _ins_models = new_predictor.models
    _ins_stats = new_engine.get_stats()

    in_c1, in_c2 = st.columns(2, gap="medium")

    with in_c1:
        # 1. DAILY VEHICLE COUNT TREND
        st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin-bottom: 0.2rem;'>1. Daily Vehicle Count Trend</div>", unsafe_allow_html=True)
        st.caption("Daily mean vehicle count vs. Degree-3 polynomial fit (2025)")

        fig_pat, ax_pat = plt.subplots(figsize=(6.5, 4.2), dpi=120)
        fig_pat.patch.set_facecolor('#FFFFFF')
        ax_pat.set_facecolor('#FCFDFC')

        ax_pat.scatter(_ins_X_days, _ins_y_means, color="#66736A", s=8, alpha=0.5, zorder=3, label="Daily Mean")
        ax_pat.plot(_ins_x_dense, _ins_models[3]["poly1d"](_ins_x_dense), color="#43A047", linewidth=2.4,
                    label=f"Degree-3 Fit")

        for spine in ax_pat.spines.values():
            spine.set_color('#E4E9E5')
        ax_pat.tick_params(colors='#66736A', labelsize=8)
        ax_pat.set_xlabel("Day Number", color='#17221B', fontsize=8.5, fontweight="bold")
        ax_pat.set_ylabel("Vehicle Count", color='#17221B', fontsize=8.5, fontweight="bold")
        ax_pat.grid(True, linestyle="--", alpha=0.45, color='#E4E9E5')
        ax_pat.legend(loc="upper right", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#E4E9E5", labelcolor="#17221B")

        plt.tight_layout()
        st.pyplot(fig_pat, use_container_width=True)
        plt.close(fig_pat)

    with in_c2:
        # 2. P(High) BY CITY
        st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin-bottom: 0.2rem;'>2. P(High) by City</div>", unsafe_allow_html=True)
        st.caption("High congestion probability across cities vs. dataset baseline (25.00%)")

        city_names = list(new_engine.p_high_by_city.keys())
        city_probs = [new_engine.p_high_by_city[c] * 100 for c in city_names]
        bar_palette_city = ["#E53935" if p > 27 else "#FFB300" if p > 25 else "#43A047" for p in city_probs]

        fig_city, ax_city = plt.subplots(figsize=(6.5, 4.2), dpi=120)
        fig_city.patch.set_facecolor('#FFFFFF')
        ax_city.set_facecolor('#FCFDFC')

        y_pos = np.arange(len(city_names))
        bars_city = ax_city.barh(y_pos, city_probs, color=bar_palette_city, height=0.55,
                                  edgecolor="#E4E9E5", linewidth=0.5)
        ax_city.axvline(x=25.0, color="#17221B", linestyle="--", linewidth=1.2, label="Baseline (25.00%)")

        for bar, val in zip(bars_city, city_probs):
            ax_city.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{val:.2f}%", va="center", ha="left", fontsize=7.8, fontweight="bold", color="#17221B")

        for spine in ax_city.spines.values():
            spine.set_color('#E4E9E5')
        ax_city.tick_params(colors='#66736A', labelsize=8)
        ax_city.set_yticks(y_pos)
        ax_city.set_yticklabels(city_names, fontsize=8, fontweight="bold", color="#17221B")
        ax_city.invert_yaxis()
        ax_city.set_xlabel("P(High Congestion) %", color="#17221B", fontsize=8.5, fontweight="bold")
        ax_city.set_xlim(0, 40)
        ax_city.grid(True, axis="x", linestyle="--", alpha=0.45, color='#E4E9E5')
        ax_city.legend(loc="lower right", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#E4E9E5", labelcolor="#17221B")

        plt.tight_layout()
        st.pyplot(fig_city, use_container_width=True)
        plt.close(fig_city)

    # 3. P(High) BY VEHICLE TYPE
    in_c3, in_c4 = st.columns(2, gap="medium")

    with in_c3:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin: 1.2rem 0 0.2rem 0;'>3. P(High) by Vehicle Type</div>", unsafe_allow_html=True)
        st.caption("High congestion probability by vehicle type")

        vt_names = list(new_engine.p_high_by_vehicle_type.keys())
        vt_probs = [new_engine.p_high_by_vehicle_type[vt] * 100 for vt in vt_names]
        bar_palette_vt = ["#E53935" if p > 27 else "#FFB300" if p > 25 else "#43A047" for p in vt_probs]

        fig_vt, ax_vt = plt.subplots(figsize=(6.5, 3.5), dpi=120)
        fig_vt.patch.set_facecolor('#FFFFFF')
        ax_vt.set_facecolor('#FCFDFC')

        y_vt = np.arange(len(vt_names))
        bars_vt = ax_vt.barh(y_vt, vt_probs, color=bar_palette_vt, height=0.55,
                               edgecolor="#E4E9E5", linewidth=0.5)
        ax_vt.axvline(x=25.0, color="#17221B", linestyle="--", linewidth=1.2, label="Baseline")

        for bar, val in zip(bars_vt, vt_probs):
            ax_vt.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                       f"{val:.2f}%", va="center", ha="left", fontsize=7.8, fontweight="bold", color="#17221B")

        for spine in ax_vt.spines.values():
            spine.set_color('#E4E9E5')
        ax_vt.tick_params(colors='#66736A', labelsize=8)
        ax_vt.set_yticks(y_vt)
        ax_vt.set_yticklabels(vt_names, fontsize=8, fontweight="bold", color="#17221B")
        ax_vt.invert_yaxis()
        ax_vt.set_xlabel("P(High Congestion) %", color="#17221B", fontsize=8.5, fontweight="bold")
        ax_vt.set_xlim(0, 40)
        ax_vt.grid(True, axis="x", linestyle="--", alpha=0.45, color='#E4E9E5')
        ax_vt.legend(loc="lower right", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#E4E9E5", labelcolor="#17221B")

        plt.tight_layout()
        st.pyplot(fig_vt, use_container_width=True)
        plt.close(fig_vt)

    with in_c4:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin: 1.2rem 0 0.2rem 0;'>4. Congestion Distribution</div>", unsafe_allow_html=True)
        st.caption("Statistical congestion category distribution across all 1,200 records")

        cong_dist = new_engine.get_congestion_distribution()
        cong_labels = cong_dist['Category'].tolist()
        cong_counts = cong_dist['Count'].tolist()
        cong_colors = ["#43A047", "#FFB300", "#E53935"]

        fig_cong, ax_cong = plt.subplots(figsize=(6.5, 3.5), dpi=120)
        fig_cong.patch.set_facecolor('#FFFFFF')
        ax_cong.set_facecolor('#FCFDFC')

        wedges, texts, autotexts = ax_cong.pie(
            cong_counts, labels=cong_labels, colors=cong_colors,
            autopct='%1.1f%%', startangle=140,
            textprops={'fontsize': 9, 'color': '#17221B', 'fontweight': 'bold'},
            wedgeprops={'edgecolor': '#FFFFFF', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_color('#FFFFFF')
            at.set_fontweight('bold')

        ax_cong.set_title('Vehicle_Count Congestion (Q25/Q75 thresholds)', fontsize=8.5, color='#17221B', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_cong, use_container_width=True)
        plt.close(fig_cong)

    # 5. DATASET SUMMARY
    st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #17221B; margin: 1.2rem 0 0.4rem 0;'>5. Dataset Statistical Summary</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mini-stat-grid">
        <div class="mini-stat-card"><div class="mini-stat-val">1,200</div><div class="mini-stat-lbl">Observations</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">{_ins_stats['mean']:,.1f}</div><div class="mini-stat-lbl">Mean Vehicle Count</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">{new_engine.q25:,.1f}</div><div class="mini-stat-lbl">Q25 Boundary</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">{new_engine.q50:,.1f}</div><div class="mini-stat-lbl">Median (Q50)</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">{new_engine.q75:,.1f}</div><div class="mini-stat-lbl">Q75 (High Threshold)</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">{len(new_engine.cities)}</div><div class="mini-stat-lbl">Cities</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">{sum(len(v) for v in new_engine.city_area_mapping.values())}</div><div class="mini-stat-lbl">Areas</div></div>
        <div class="mini-stat-card"><div class="mini-stat-val">{len(new_engine.vehicle_types)}</div><div class="mini-stat-lbl">Vehicle Types</div></div>
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
            <strong>Traffic dataset.csv — Active Dataset</strong><br>
            1,200 records covering multiple cities and areas within Maharashtra, India (2025).<br>
            Columns: <strong>Date, City, Area, Vehicle_Type, Vehicle_Count, Accidents, Traffic_Violations, Avg_Speed_kmph, Congestion_Level</strong>.<br>
            Zero missing values across all columns. No deduplication required.<br><br>
            <strong>Location</strong> in CommuteCast represents a <strong>City + Area</strong> combination present in this dataset —
            not a GPS position, not live traffic, and not any external API.<br>
            Cities: Mumbai, Nashik, Nagpur, Pune, Thane (5 cities × 5 areas = 25 locations).<br>
            Vehicle Types: Auto, Bike, Bus, Car, Truck.<br><br>
            The previous Metro Interstate Traffic Volume (Minnesota/UCI) dataset remains in the project
            but is NOT used in the active prediction pipeline.
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
        ("Probability", "Empirical sample space frequencies P(High) = count(High) / N"),
        ("Conditional Probability", "P(High | City, Area) and P(High | Vehicle_Type) from empirical frequencies"),
        ("Bayes' Theorem", "P(High | VehicleType) = [P(VehicleType | High) × P(High)] / P(VehicleType)"),
        ("Random Variables", "Vehicle_Count modeled as a random variable; congestion derived from Q75 threshold"),
        ("Expectation & Variance", "E[Vehicle_Count] = sample mean; Var computed with ddof=1"),
        ("Covariance", "Cov(Vehicle_Count, Accidents), Cov(Vehicle_Count, Avg_Speed_kmph)"),
        ("Quantiles", "Q25, Q50, Q75 of Vehicle_Count for objective Low/Medium/High classification"),
        ("Polynomial Curve Fitting", "OLS Degree 1/2/3 on day_num → mean Vehicle_Count (daily aggregation)")
    ]

    c_cols = st.columns(2)
    for i, (name, desc) in enumerate(concepts):
        col_to_use = c_cols[i % 2]
        with col_to_use:
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.95); border: 1px solid #E4E9E5; border-radius: 10px; padding: 0.75rem 1.0rem; margin-bottom: 8px;">
                <strong style="color: #17221B; font-size: 0.88rem;">{name}</strong><br>
                <span style="font-size: 0.78rem; color: #66736A;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="border-top: 1px solid #E4E9E5; margin-top: 1.5rem; padding-top: 1.0rem; text-align: center; font-size: 0.8rem; color: #85948A;">
        CommuteCast • Presentation-Ready Academic Engineering Project • Zero Black-Box ML
    </div>
    """, unsafe_allow_html=True)
