"""
CommuteCast - New Dataset Preprocessing
Module: src/new_preprocessing.py
"""

import os
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NEW_RAW_DATA_PATH = os.path.join("data", "Traffic dataset.csv")

REQUIRED_DATASET_COLUMNS = [
    "Date", "City", "Area", "Vehicle_Type", "Vehicle_Count",
    "Accidents", "Traffic_Violations", "Avg_Speed_kmph", "Congestion_Level"
]


def validate_dataset_columns(df: pd.DataFrame) -> None:
    """Validate that all required 9 columns exist in the dataset."""
    missing = [c for c in REQUIRED_DATASET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset validation error: Missing required column(s): {', '.join(missing)}. "
            f"Required columns are: {', '.join(REQUIRED_DATASET_COLUMNS)}"
        )


def load_new_dataset(filepath=None):
    if filepath is None:
        filepath = NEW_RAW_DATA_PATH
    abs_path = filepath if os.path.isabs(filepath) else str(
        Path(__file__).resolve().parent.parent / filepath
    )
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"New dataset not found at: {abs_path}")
    logger.info(f"Loading new dataset from {abs_path}...")
    df = pd.read_csv(abs_path)
    validate_dataset_columns(df)
    logger.info(f"New dataset loaded: {len(df):,} rows.")
    return df


def preprocess_new_traffic_data(filepath=None):
    if filepath is None:
        filepath = NEW_RAW_DATA_PATH
    df = load_new_dataset(filepath)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    date_min = df["Date"].min()
    df["day_num"] = (df["Date"] - date_min).dt.days
    q25 = float(df["Vehicle_Count"].quantile(0.25))
    q50 = float(df["Vehicle_Count"].median())
    q75 = float(df["Vehicle_Count"].quantile(0.75))
    logger.info(f"Vehicle_Count Q25={q25:.2f}, Q50={q50:.2f}, Q75={q75:.2f}")
    congestion_series = pd.cut(
        df["Vehicle_Count"],
        bins=[-np.inf, q25, q75, np.inf],
        labels=["Low", "Medium", "High"]
    )
    df["congestion"] = congestion_series.astype(str)
    df.attrs["q25"] = q25
    df.attrs["q50"] = q50
    df.attrs["q75"] = q75
    return df


def get_city_area_mapping(df):
    mapping = {}
    for city, group in df.groupby("City"):
        mapping[city] = sorted(group["Area"].unique().tolist())
    return mapping


def get_cities(df):
    return sorted(df["City"].unique().tolist())


def get_vehicle_types(df):
    return sorted(df["Vehicle_Type"].unique().tolist())
