"""
CommuteCast - New Dataset Probability Module
Module: src/new_probability.py

Calculates all Unit-1 probability and empirical statistics using Traffic dataset.csv:
- Empirical expected vehicle count matching user's selected location (City + Area + Vehicle_Type)
- Hierarchical transparent fallback (City+Area+Vehicle_Type -> City+Area -> City -> Global)
- Empirical conditional probability of high congestion: P(High | City, Area, Vehicle_Type)
- P(High), P(High | City), P(High | Area), P(High | Vehicle_Type), P(High | City, Area)
- Bayes theorem: P(High | Vehicle_Type) verified against direct empirical
- Independence analysis
- Expectation, Variance, Covariance (Vehicle_Count, Accidents, Traffic_Violations, Avg_Speed_kmph)
- Zero Poisson, zero Gaussian, zero arbitrary multipliers, zero ML black-boxes.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.new_preprocessing import preprocess_new_traffic_data, NEW_RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_OBS_THRESHOLD = 5


def get_high_congestion_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean series for High congestion records from dataset."""
    if 'Congestion_Level' in df.columns:
        return df['Congestion_Level'] == 'High'
    elif 'congestion' in df.columns:
        return df['congestion'] == 'High'
    else:
        q75 = df['Vehicle_Count'].quantile(0.75)
        return df['Vehicle_Count'] > q75


def calculate_empirical_expected_volume(
    df: pd.DataFrame,
    city: Optional[str] = None,
    area: Optional[str] = None,
    vehicle_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute transparent empirical expected vehicle count from matching dataset rows.
    
    Hierarchical fallback:
      LEVEL 1 — Exact combination: City + Area + Vehicle_Type (if all provided and matching rows > 0)
      LEVEL 2 — Area fallback: City + Area (if available and matching rows > 0)
      LEVEL 3 — City fallback: City (if available and matching rows > 0)
      LEVEL 4 — Global fallback: Entire dataset mean
    
    Returns exact mean, row count, condition description, and fallback metadata.
    Zero arbitrary multipliers or synthetic weights. Never uses city mean when matching area rows exist.
    """
    city_clean = city.strip() if isinstance(city, str) and city.strip() else None
    area_clean = area.strip() if isinstance(area, str) and area.strip() else None
    vt_clean = vehicle_type.strip() if isinstance(vehicle_type, str) and vehicle_type.strip() else None

    # LEVEL 1 — Exact match: City + Area + Vehicle_Type
    if city_clean and area_clean and vt_clean:
        mask = (df['City'] == city_clean) & (df['Area'] == area_clean) & (df['Vehicle_Type'] == vt_clean)
        subset = df[mask]
        if len(subset) > 0:
            return {
                'expected_vehicle_count': float(subset['Vehicle_Count'].mean()),
                'subset_rows': len(subset),
                'condition_used': f"{city_clean} — {area_clean} ({vt_clean})",
                'fallback_level': "Exact (City + Area + Vehicle_Type)",
                'fallback': False,
                'fallback_reason': "",
                'variance_sample': float(subset['Vehicle_Count'].var(ddof=1)) if len(subset) > 1 else 0.0,
                'std_sample': float(subset['Vehicle_Count'].std(ddof=1)) if len(subset) > 1 else 0.0
            }

    # LEVEL 2 — Area fallback: City + Area
    if city_clean and area_clean:
        mask = (df['City'] == city_clean) & (df['Area'] == area_clean)
        subset = df[mask]
        if len(subset) > 0:
            is_fallback = bool(vt_clean is not None)
            reason = f"No rows for {city_clean} + {area_clean} + {vt_clean}; fell back to City + Area mean" if is_fallback else ""
            return {
                'expected_vehicle_count': float(subset['Vehicle_Count'].mean()),
                'subset_rows': len(subset),
                'condition_used': f"{city_clean} — {area_clean}",
                'fallback_level': "City + Area",
                'fallback': is_fallback,
                'fallback_reason': reason,
                'variance_sample': float(subset['Vehicle_Count'].var(ddof=1)) if len(subset) > 1 else 0.0,
                'std_sample': float(subset['Vehicle_Count'].std(ddof=1)) if len(subset) > 1 else 0.0
            }

    # LEVEL 3 — City fallback: City
    if city_clean:
        mask = df['City'] == city_clean
        subset = df[mask]
        if len(subset) > 0:
            is_fallback = bool(area_clean is not None or vt_clean is not None)
            reason = f"Fell back to {city_clean} citywide mean" if is_fallback else ""
            return {
                'expected_vehicle_count': float(subset['Vehicle_Count'].mean()),
                'subset_rows': len(subset),
                'condition_used': f"{city_clean} (Citywide)",
                'fallback_level': "City",
                'fallback': is_fallback,
                'fallback_reason': reason,
                'variance_sample': float(subset['Vehicle_Count'].var(ddof=1)) if len(subset) > 1 else 0.0,
                'std_sample': float(subset['Vehicle_Count'].std(ddof=1)) if len(subset) > 1 else 0.0
            }

    # LEVEL 4 — Global fallback: Entire dataset mean
    global_mean = float(df['Vehicle_Count'].mean())
    return {
        'expected_vehicle_count': global_mean,
        'subset_rows': len(df),
        'condition_used': "All Locations (Global Mean)",
        'fallback_level': "Global Dataset Mean",
        'fallback': True,
        'fallback_reason': "Fell back to global dataset mean",
        'variance_sample': float(df['Vehicle_Count'].var(ddof=1)),
        'std_sample': float(df['Vehicle_Count'].std(ddof=1))
    }


def calculate_empirical_congestion_probability(
    df: pd.DataFrame,
    city: Optional[str] = None,
    area: Optional[str] = None,
    vehicle_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate empirical probability P(Congestion_Level == 'High' | subset)
    = count(Congestion_Level == 'High' in subset) / count(subset).
    
    Uses matching subset with transparent hierarchical fallback.
    """
    high_mask = get_high_congestion_mask(df)
    baseline_p_high = float(high_mask.mean())

    # 1. Exact: City + Area + Vehicle_Type
    if city and area and vehicle_type:
        mask = (df['City'] == city) & (df['Area'] == area) & (df['Vehicle_Type'] == vehicle_type)
        n = int(mask.sum())
        if n > 0:
            high_count = int((high_mask & mask).sum())
            return {
                'probability': float(high_count / n),
                'high_count': high_count,
                'subset_rows': n,
                'condition_used': f"{city} — {area} ({vehicle_type})",
                'fallback_level': "Exact (City + Area + Vehicle_Type)",
                'fallback': False,
                'fallback_reason': ""
            }

    # 2. City + Area
    if city and area:
        mask = (df['City'] == city) & (df['Area'] == area)
        n = int(mask.sum())
        if n > 0:
            high_count = int((high_mask & mask).sum())
            return {
                'probability': float(high_count / n),
                'high_count': high_count,
                'subset_rows': n,
                'condition_used': f"{city} — {area}",
                'fallback_level': "City + Area",
                'fallback': bool(vehicle_type is not None),
                'fallback_reason': f"City+Area fallback for P(High)"
            }

    # 3. City
    if city:
        mask = df['City'] == city
        n = int(mask.sum())
        if n > 0:
            high_count = int((high_mask & mask).sum())
            return {
                'probability': float(high_count / n),
                'high_count': high_count,
                'subset_rows': n,
                'condition_used': f"{city}",
                'fallback_level': "City",
                'fallback': True,
                'fallback_reason': f"City fallback for P(High)"
            }

    # 4. Global
    high_count = int(high_mask.sum())
    return {
        'probability': baseline_p_high,
        'high_count': high_count,
        'subset_rows': len(df),
        'condition_used': "Baseline",
        'fallback_level': "Global Baseline",
        'fallback': True,
        'fallback_reason': "Global baseline P(High)"
    }


def calculate_p_high_location(df: pd.DataFrame, city: str, area: str) -> Dict[str, Any]:
    """Calculate P(High | City, Area) using empirical relative frequency."""
    high_mask = get_high_congestion_mask(df)
    city_area_mask = (df['City'] == city) & (df['Area'] == area)
    n_loc = int(city_area_mask.sum())

    if n_loc >= MIN_OBS_THRESHOLD:
        p = float((high_mask & city_area_mask).sum() / n_loc)
        return {'probability': p, 'n_obs': n_loc, 'condition_used': f'City={city}, Area={area}', 'fallback': False}

    # Fallback 1: Area
    area_mask = df['Area'] == area
    n_area = int(area_mask.sum())
    if n_area >= MIN_OBS_THRESHOLD:
        p = float((high_mask & area_mask).sum() / n_area)
        return {'probability': p, 'n_obs': n_area, 'condition_used': f'Area={area}', 'fallback': True, 'fallback_reason': f'City+Area has only {n_loc} obs'}

    # Fallback 2: City
    c_mask = df['City'] == city
    n_city = int(c_mask.sum())
    p = float((high_mask & c_mask).sum() / n_city) if n_city > 0 else float(high_mask.mean())
    return {'probability': p, 'n_obs': n_city, 'condition_used': f'City={city}', 'fallback': True, 'fallback_reason': f'Area has only {n_area} obs'}


def calculate_p_high_city(df: pd.DataFrame, city: str) -> Dict[str, Any]:
    """Calculate P(High | City) via empirical frequency."""
    high_mask = get_high_congestion_mask(df)
    c_mask = df['City'] == city
    n = int(c_mask.sum())
    p = float((high_mask & c_mask).sum() / n) if n > 0 else float(high_mask.mean())
    return {'probability': p, 'n_obs': n, 'condition_used': f'City={city}'}


def calculate_p_high_area(df: pd.DataFrame, area: str) -> Dict[str, Any]:
    """Calculate P(High | Area) via empirical frequency."""
    high_mask = get_high_congestion_mask(df)
    a_mask = df['Area'] == area
    n = int(a_mask.sum())
    p = float((high_mask & a_mask).sum() / n) if n > 0 else float(high_mask.mean())
    return {'probability': p, 'n_obs': n, 'condition_used': f'Area={area}'}


def calculate_p_high_vehicle_type(df: pd.DataFrame, vehicle_type: str) -> Dict[str, Any]:
    """P(High | Vehicle_Type) via direct empirical frequency."""
    high_mask = get_high_congestion_mask(df)
    vt_mask = df['Vehicle_Type'] == vehicle_type
    n = int(vt_mask.sum())
    if n == 0:
        return {'probability': float(high_mask.mean()), 'n_obs': 0, 'condition_used': 'baseline'}
    p = float((high_mask & vt_mask).sum() / n)
    return {'probability': p, 'n_obs': n, 'condition_used': f'Vehicle_Type={vehicle_type}', 'fallback': False}


def run_bayes_vehicle_type(df: pd.DataFrame, vehicle_type: str) -> Dict[str, Any]:
    """
    Bayes theorem: P(High | Vehicle_Type)
    = P(Vehicle_Type | High) * P(High) / P(Vehicle_Type)
    Verified against direct empirical calculation.
    """
    high_mask = get_high_congestion_mask(df)
    vt_mask = df['Vehicle_Type'] == vehicle_type
    n = len(df)
    count_high = int(high_mask.sum())
    count_vt = int(vt_mask.sum())
    count_joint = int((high_mask & vt_mask).sum())

    p_high = count_high / n
    p_vt = count_vt / n
    if count_high == 0 or p_vt == 0:
        return {'bayes_posterior': p_high, 'direct_empirical_posterior': p_high, 'is_verified': True}

    p_vt_given_high = count_joint / count_high
    p_direct = count_joint / count_vt
    p_bayes = (p_vt_given_high * p_high) / p_vt

    return {
        'vehicle_type': vehicle_type,
        'prior_P_High': p_high,
        'marginal_P_VehicleType': p_vt,
        'likelihood_P_VehicleType_given_High': p_vt_given_high,
        'bayes_posterior': float(p_bayes),
        'direct_empirical_posterior': float(p_direct),
        'absolute_difference': float(abs(p_bayes - p_direct)),
        'is_verified': bool(abs(p_bayes - p_direct) <= 1e-6),
        'n_joint': count_joint,
        'n_high': count_high,
        'n_vehicle_type': count_vt
    }


def compute_vehicle_count_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute descriptive statistics for Vehicle_Count."""
    vc = df['Vehicle_Count']
    return {
        'count': int(vc.count()),
        'mean': float(vc.mean()),
        'variance_sample': float(vc.var(ddof=1)),
        'variance_population': float(vc.var(ddof=0)),
        'std_sample': float(vc.std(ddof=1)),
        'min': float(vc.min()),
        'q25': float(vc.quantile(0.25)),
        'median': float(vc.median()),
        'q75': float(vc.quantile(0.75)),
        'max': float(vc.max()),
        'iqr': float(vc.quantile(0.75) - vc.quantile(0.25))
    }


def compute_covariances(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute covariance between Vehicle_Count and other numeric variables."""
    vc = df['Vehicle_Count']
    result = {}
    for col in ['Accidents', 'Traffic_Violations', 'Avg_Speed_kmph']:
        if col in df.columns:
            result[f'cov_VehicleCount_{col}'] = float(vc.cov(df[col]))
            result[f'corr_VehicleCount_{col}'] = float(vc.corr(df[col]))
    return result


def compute_congestion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency distribution of statistical congestion categories."""
    target_col = 'Congestion_Level' if 'Congestion_Level' in df.columns else 'congestion'
    counts = df[target_col].value_counts().reindex(['Low', 'Medium', 'High'])
    pcts = (df[target_col].value_counts(normalize=True) * 100).reindex(['Low', 'Medium', 'High'])
    return pd.DataFrame({'Category': ['Low', 'Medium', 'High'], 'Count': counts.values, 'Percentage (%)': pcts.values})


def evaluate_independence(p_target: float, p_target_given_cond: float, name: str, threshold: float = 0.03) -> str:
    """Assess empirical independence by comparing marginal vs conditional probability."""
    diff = abs(p_target_given_cond - p_target)
    if diff <= threshold:
        return (f'P(High | {name}) = {p_target_given_cond:.4f} vs P(High) = {p_target:.4f} '
                f'(diff={diff:.4f} <= {threshold:.2f}): consistent with approximate independence.')
    return (f'P(High | {name}) = {p_target_given_cond:.4f} vs P(High) = {p_target:.4f} '
            f'(diff={diff:.4f} > {threshold:.2f}): observed data does NOT support independence.')
