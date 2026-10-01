"""
CommuteCast - Stage 2: Data Preprocessing
Module: src/preprocessing.py

Responsibilities:
1. Load raw dataset without modifying original file.
2. Deduplicate exact duplicate records (reporting count).
3. Handle sensor quality issues:
   - Identify temp == 0 K (sensor dropouts) and set to NaN for temperature analysis.
   - Investigate extreme rainfall (9831.3 mm) without arbitrary deletion or inventing values.
4. Correctly parse "None" string for holidays (avoiding false NaN imputation).
5. Extract temporal features: hour, day_of_week, month, year, is_weekend.
6. Generate binary rain indicator: rain = 1 if rain_1h > 0 else 0.
7. Compute temp_celsius from Kelvin.
8. Classify traffic volume into quantile-based congestion classes: Low, Medium, High.
"""

import os
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join("data", "Metro_Interstate_Traffic_Volume.csv")
PROCESSED_DATA_PATH = os.path.join("data", "preprocessed_traffic_volume.csv")


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw CSV dataset.
    Note: keep_default_na=False is critical because the dataset uses the literal
    string 'None' to denote non-holiday days. Standard pandas parsing would convert
    'None' to NaN, destroying the holiday column semantics.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    
    logger.info(f"Loading raw dataset from '{filepath}'...")
    df = pd.read_csv(filepath, keep_default_na=False)
    logger.info(f"Raw dataset loaded successfully with {len(df):,} rows and {len(df.columns)} columns.")
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Identify and remove exact duplicate rows.
    Preserves first occurrence of each unique observation.
    """
    duplicate_mask = df.duplicated()
    num_duplicates = int(duplicate_mask.sum())
    logger.info(f"Identified {num_duplicates} exact duplicate rows.")
    
    df_cleaned = df.drop_duplicates(keep="first").copy()
    logger.info(f"Dataset deduplicated: {len(df_cleaned):,} rows retained.")
    return df_cleaned, num_duplicates


def handle_temperature_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle sensor anomalies in temperature.
    Investigation revealed 10 records with temp == 0.0 K (-273.15 °C, absolute zero).
    These represent sensor transmission dropouts, not physical atmospheric temperatures.
    
    Decision:
    - Retain all traffic volume and temporal records (do not delete the observations).
    - Set invalid temperature readings to np.nan for temperature-specific statistical analysis.
    - Convert valid Kelvin temperatures to Celsius: temp_celsius = temp - 273.15.
    """
    df = df.copy()
    temp_zero_mask = (df["temp"] == 0.0)
    num_temp_zeros = int(temp_zero_mask.sum())
    logger.info(f"Found {num_temp_zeros} records with temp == 0.0 K (sensor dropouts).")
    
    # Store clean Kelvin values with NaN for invalid readings
    df["temp_kelvin_clean"] = df["temp"].replace(0.0, np.nan)
    
    # Convert to Celsius for human-readable meteorological analysis
    df["temp_celsius"] = df["temp_kelvin_clean"] - 273.15
    return df


def handle_rainfall_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle rainfall distribution and extreme sensor recording anomalies.
    
    Investigation:
    - 2016-07-11 17:00:00 recorded rain_1h = 9831.3 mm (~9.8 meters of rain in 1 hour).
    - The next highest rainfall record in the dataset is 55.63 mm.
    - Meteorological records confirm thunderstorm conditions during this evening rush hour
      (traffic_volume = 5,535).
    
    Data-Quality Decisions:
    1. Do NOT arbitrarily delete the observation row:
       The traffic volume measurement (5,535 vehicles/hour) is valid rush-hour commute data.
    2. Do NOT invent an arbitrary replacement value:
       We do not fabricate a replacement estimate (e.g. 9.8 mm or 55 mm).
    3. Binary rain indicator:
       rain = 1 if rain_1h > 0 else 0.
       Since 9831.3 mm > 0 and severe rain occurred, rain = 1 accurately reflects rain presence.
    4. Valid numerical rainfall column (rain_1h_valid):
       For numerical rainfall analysis and covariance, sensor anomalies (> 500 mm) are set
       to np.nan without inventing artificial values.
    """
    df = df.copy()
    extreme_rain_mask = (df["rain_1h"] > 500.0)
    num_extreme_rain = int(extreme_rain_mask.sum())
    logger.info(f"Found {num_extreme_rain} extreme rainfall outlier(s) (> 500 mm/h).")
    
    # Binary rain indicator
    df["rain"] = (df["rain_1h"] > 0).astype(int)
    
    # rain_1h_valid treats physically impossible rainfall as NaN for numerical calculations
    df["rain_1h_valid"] = df["rain_1h"].copy()
    df.loc[extreme_rain_mask, "rain_1h_valid"] = np.nan
    return df


def handle_holiday_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary holiday indicator.
    The raw dataset uses the literal string 'None' for non-holidays.
    is_holiday = 1 if holiday != 'None' else 0.
    """
    df = df.copy()
    df["is_holiday"] = (df["holiday"] != "None").astype(int)
    holiday_count = int(df["is_holiday"].sum())
    logger.info(f"Created 'is_holiday' indicator: {holiday_count:,} holiday records, {len(df)-holiday_count:,} non-holiday records.")
    return df


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date_time as datetime and extract temporal features:
    - hour (0 to 23)
    - day_of_week (0 = Monday, ..., 6 = Sunday)
    - month (1 to 12)
    - year
    - is_weekend (1 if day_of_week in [5, 6] else 0)
    """
    df = df.copy()
    logger.info("Parsing 'date_time' column into datetime objects...")
    df["date_time"] = pd.to_datetime(df["date_time"])
    
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    logger.info("Extracted temporal features: 'hour', 'day_of_week', 'month', 'year', 'is_weekend'.")
    return df


def assign_congestion_categories(
    df: pd.DataFrame,
    q25: Optional[float] = None,
    q75: Optional[float] = None
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Classify traffic volume into quantile-based congestion categories:
    - Low: traffic_volume <= Q25
    - Medium: Q25 < traffic_volume <= Q75
    - High: traffic_volume > Q75
    
    Uses empirical 25th and 75th percentiles to ensure balanced, objective thresholds.
    """
    df = df.copy()
    if q25 is None:
        q25 = float(df["traffic_volume"].quantile(0.25))
    if q75 is None:
        q75 = float(df["traffic_volume"].quantile(0.75))
    median_val = float(df["traffic_volume"].median())
    
    congestion_series = pd.cut(
        df["traffic_volume"],
        bins=[-np.inf, q25, q75, np.inf],
        labels=["Low", "Medium", "High"]
    )
    df["congestion"] = congestion_series.astype(str)
    
    logger.info(f"Congestion classified: Q25={q25:.1f}, Median={median_val:.1f}, Q75={q75:.1f}")
    return df, q25, median_val, q75


def preprocess_traffic_data(
    raw_path: str = RAW_DATA_PATH,
    output_path: Optional[str] = PROCESSED_DATA_PATH,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Complete end-to-end preprocessing pipeline:
    1. Load raw CSV without altering it.
    2. Deduplicate exact duplicate rows.
    3. Treat temp == 0 K as invalid sensor observations (NaN).
    4. Handle rainfall extreme values without deleting rows or inventing values.
    5. Handle holiday indicator ('None' vs named holidays).
    6. Extract datetime features (hour, day_of_week, month, year, is_weekend).
    7. Assign quantile-based congestion categories (Low, Medium, High).
    8. Optionally save preprocessed dataset to output_path.
    """
    logger.info("=== Starting Preprocessing Pipeline ===")
    
    # 1. Load raw
    raw_df = load_raw_data(raw_path)
    
    # 2. Deduplicate
    df, num_dups = remove_duplicate_rows(raw_df)
    
    # 3. Temperature
    df = handle_temperature_anomalies(df)
    
    # 4. Rainfall
    df = handle_rainfall_anomalies(df)
    
    # 5. Holidays
    df = handle_holiday_indicator(df)
    
    # 6. Datetime features
    df = extract_datetime_features(df)
    
    # 7. Congestion categories
    df, q25, q50, q75 = assign_congestion_categories(df)
    
    # 8. Save if requested
    if save_output and output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed dataset saved to: '{output_path}' ({len(df):,} rows).")
    
    logger.info("=== Preprocessing Pipeline Completed Successfully ===")
    return df


if __name__ == "__main__":
    df_preprocessed = preprocess_traffic_data()
    print("\n--- Preprocessed Data Summary ---")
    print(f"Total Rows: {len(df_preprocessed):,}")
    print(f"Columns: {list(df_preprocessed.columns)}")
    print("\nSample Rows:")
    print(df_preprocessed[["date_time", "hour", "is_weekend", "temp_celsius", "rain", "is_holiday", "traffic_volume", "congestion"]].head(5))
