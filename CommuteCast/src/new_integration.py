"""
CommuteCast - New Dataset Integration Engine
Module: src/new_integration.py

Integrates:
- Transparent empirical expected vehicle count matching selected location (City + Area + Vehicle_Type)
- Hierarchical fallback (City+Area+Vehicle_Type -> City+Area -> City -> Global mean)
- Empirical conditional probability: P(High | City, Area, Vehicle_Type)
- Unit-1 Polynomial curve fitting over date/day_num preserved for historical trend
- Bayes theorem verification
- Expectation, Variance, Covariance

Maintains strict mathematical separation:
- Expected traffic volume (empirical subset mean) and
- Probability of high congestion (empirical conditional frequency)
are NEVER multiplied together or combined into an arbitrary composite score.
Zero Poisson, zero Gaussian, zero synthetic multipliers.
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

from src.new_preprocessing import (
    preprocess_new_traffic_data, get_city_area_mapping, get_cities, get_vehicle_types
)
from src.new_polynomial import get_new_predictor, DEFAULT_SELECTED_DEGREE
from src.new_probability import (
    calculate_empirical_expected_volume, calculate_empirical_congestion_probability,
    calculate_p_high_location, calculate_p_high_vehicle_type, calculate_p_high_city,
    calculate_p_high_area, run_bayes_vehicle_type, compute_vehicle_count_statistics,
    compute_covariances, compute_congestion_distribution, evaluate_independence,
    get_high_congestion_mask
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewCommuteAnalysisEngine:
    """
    Unified analysis engine for Traffic dataset.csv.
    Calculates empirical expected vehicle count and empirical probabilities
    directly from dataset rows matching selected City, Area, and Vehicle_Type.
    """
    def __init__(self):
        self.df = preprocess_new_traffic_data()
        self.predictor = get_new_predictor()
        self._init_empirical_probabilities()

    def _init_empirical_probabilities(self):
        df = self.df
        high_mask = get_high_congestion_mask(df)

        self.p_high_baseline = float(high_mask.mean())
        self.count_total = len(df)
        self.count_high = int(high_mask.sum())

        self.q25 = df.attrs.get('q25', float(df['Vehicle_Count'].quantile(0.25)))
        self.q50 = df.attrs.get('q50', float(df['Vehicle_Count'].median()))
        self.q75 = df.attrs.get('q75', float(df['Vehicle_Count'].quantile(0.75)))

        self.city_area_mapping = get_city_area_mapping(df)
        self.cities = get_cities(df)
        self.vehicle_types = get_vehicle_types(df)

        # Precompute P(High | City) and P(High | Vehicle_Type)
        self.p_high_by_city = {}
        for city in self.cities:
            mask = df['City'] == city
            n = int(mask.sum())
            self.p_high_by_city[city] = float((high_mask & mask).sum() / n) if n > 0 else self.p_high_baseline

        self.p_high_by_vehicle_type = {}
        for vt in self.vehicle_types:
            mask = df['Vehicle_Type'] == vt
            n = int(mask.sum())
            self.p_high_by_vehicle_type[vt] = float((high_mask & mask).sum() / n) if n > 0 else self.p_high_baseline

        # Vehicle_Type joint probabilities for Bayes
        self.bayes_by_vehicle_type = {}
        for vt in self.vehicle_types:
            self.bayes_by_vehicle_type[vt] = run_bayes_vehicle_type(df, vt)

    def analyze(
        self,
        city: str,
        area: str,
        vehicle_type: str,
        polynomial_degree: int = DEFAULT_SELECTED_DEGREE
    ) -> Dict[str, Any]:
        """
        Execute empirical commute analysis based on matching dataset rows.
        
        Expected vehicle count is the transparent empirical mean of matching rows:
            df[(df["City"] == city) & (df["Area"] == area) & (df["Vehicle_Type"] == vehicle_type)]
        with hierarchical fallback if empty.
        
        High congestion probability is the empirical relative frequency of High congestion:
            count(Congestion_Level == "High" in subset) / count(subset)
        
        Returns separate expected volume and probability (NEVER combined/multiplied).
        """
        # Area validation against dataset mapping
        if city in self.city_area_mapping and area and area not in self.city_area_mapping[city]:
            logger.warning(
                f"Area validation warning: Area '{area}' does not belong to City '{city}'. "
                f"Valid areas for {city}: {self.city_area_mapping[city]}"
            )

        # 1. Empirical Expected Vehicle Count from matching subset with fallback
        vol_info = calculate_empirical_expected_volume(
            self.df, city=city, area=area, vehicle_type=vehicle_type
        )
        expected_vehicle_count = vol_info['expected_vehicle_count']
        subset_rows = vol_info['subset_rows']
        vol_condition = vol_info['condition_used']
        fallback = vol_info['fallback']
        fallback_level = vol_info['fallback_level']
        fallback_reason = vol_info['fallback_reason']

        # Log debug audit telemetry during calculation
        logger.info(
            f"CommuteCast Audit -> City: {city} | Area: {area} | Vehicle Type: {vehicle_type} | "
            f"Matching rows: {subset_rows} | Fallback: {fallback_level} | "
            f"Expected Vehicle Count: {expected_vehicle_count:.2f}"
        )

        # 2. Empirical Probability of High Congestion from matching subset
        prob_info = calculate_empirical_congestion_probability(
            self.df, city=city, area=area, vehicle_type=vehicle_type
        )
        p_high_cond = prob_info['probability']
        high_count = prob_info['high_count']

        # Vehicle type specific probability for context
        vt_info = calculate_p_high_vehicle_type(self.df, vehicle_type)
        p_high_vt = vt_info['probability']

        # Location (City + Area) probability
        loc_info = calculate_p_high_location(self.df, city, area)
        p_high_loc = loc_info['probability']

        diff = p_high_cond - self.p_high_baseline
        diff_ppt = diff * 100.0

        interpretation = (
            'Location has above-average high congestion probability relative to dataset baseline.'
            if p_high_cond > self.p_high_baseline
            else 'Location has at or below average high congestion probability relative to dataset baseline.'
        )

        return {
            'city': city,
            'area': area,
            'vehicle_type': vehicle_type,
            'expected_vehicle_count': float(expected_vehicle_count),
            'high_congestion_probability': float(p_high_cond),
            'baseline_high_probability': float(self.p_high_baseline),
            'difference_from_baseline': float(diff),
            'difference_percentage_points': float(diff_ppt),
            'subset_rows': int(subset_rows),
            'location_obs': int(subset_rows),
            'condition_used': vol_condition,
            'fallback_level': fallback_level,
            'fallback': fallback,
            'fallback_reason': fallback_reason,
            'high_count': int(high_count),
            'p_high_vehicle_type': float(p_high_vt),
            'p_high_location': float(p_high_loc),
            'vehicle_type_obs': vt_info['n_obs'],
            'polynomial_degree': int(polynomial_degree),
            'variance_sample': vol_info.get('variance_sample', 0.0),
            'std_sample': vol_info.get('std_sample', 0.0),
            'interpretation': interpretation,
            'debug_info': {
                'selected_city': city,
                'selected_area': area,
                'selected_vehicle_type': vehicle_type,
                'matching_rows': int(subset_rows),
                'fallback_level': fallback_level,
                'expected_vehicle_count': float(expected_vehicle_count)
            }
        }

    def explain_bayes(self, vehicle_type: str) -> Dict[str, Any]:
        """Return Bayes theorem verification for P(High | Vehicle_Type)."""
        if vehicle_type not in self.bayes_by_vehicle_type:
            raise ValueError(f'Vehicle type {vehicle_type} not in dataset.')
        return self.bayes_by_vehicle_type[vehicle_type]

    def get_stats(self) -> Dict[str, Any]:
        """Return descriptive statistics for Vehicle_Count."""
        return compute_vehicle_count_statistics(self.df)

    def get_covariances(self) -> Dict[str, Any]:
        """Return covariances between Vehicle_Count and other numeric variables."""
        return compute_covariances(self.df)

    def get_congestion_distribution(self) -> pd.DataFrame:
        """Return frequency distribution of statistical congestion categories."""
        return compute_congestion_distribution(self.df)


_ENGINE_INSTANCE: Optional[NewCommuteAnalysisEngine] = None


def get_new_commute_engine() -> NewCommuteAnalysisEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = NewCommuteAnalysisEngine()
    return _ENGINE_INSTANCE
