"""
CommuteCast - Stage 5: Integrated Commute Analysis Engine
Module: src/integration.py

Responsibilities:
1. Cohesive Integration:
   - Integrates the Stage 4 Polynomial Component (Expected Traffic Volume = f(Hour))
     with the Stage 3 Probability Component (P(High | observed conditions)).
   - Maintains strict mathematical separation: does NOT multiply or combine them into
     an ad-hoc composite score.
2. Input Validation:
   - hour: integer in [0, 23]
   - rain: binary integer in {0, 1}
   - peak: binary integer in {0, 1} (or automatically inferred from hour)
   - holiday: optional binary integer in {0, 1}
3. Transparent Outputs:
   - Expected Traffic Volume (veh/hr) from selected Degree 3 polynomial.
   - Empirical High Congestion Probability P(High | conditions).
   - Baseline High Congestion Probability P(High).
   - Difference from baseline (in probability and percentage points).
   - Clear rule-based interpretation relative to the dataset baseline.
4. Bayes Educational Integration:
   - Exposes Bayes theorem reconstruction alongside direct empirical conditional probabilities.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

# Ensure project root is first in sys.path and remove 'src' from sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PROCESSED_DATA_PATH, preprocess_traffic_data
from src.polynomial import predict_traffic_for_hour, DEFAULT_SELECTED_DEGREE
from src.probability import CORE_PEAK_HOURS
from src.bayes import verify_bayes_with_empirical_probability


class CommuteAnalysisEngine:
    """
    Unified, explainable CommuteCast integration engine.
    """
    def __init__(self, csv_path: str = PROCESSED_DATA_PATH):
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = preprocess_traffic_data()
            
        self._init_empirical_probabilities()
        
    def _init_empirical_probabilities(self):
        """Precompute and cache exact empirical probabilities from the cleaned dataset."""
        df = self.df
        total_n = len(df)
        high_mask = (df["congestion"] == "High")
        rain_mask = (df["rain"] == 1)
        peak_mask = df["hour"].isin(CORE_PEAK_HOURS)
        holiday_mask = (df["is_holiday"] == 1)
        
        # Baseline P(High)
        self.p_high_baseline = float(high_mask.mean())
        self.count_total = total_n
        self.count_high = int(high_mask.sum())
        
        # Marginal probabilities
        self.p_rain = float(rain_mask.mean())
        self.p_peak = float(peak_mask.mean())
        
        # Single-condition conditional probabilities
        self.p_high_given_rain = float((high_mask & rain_mask).sum() / rain_mask.sum())
        self.p_high_given_norain = float((high_mask & ~rain_mask).sum() / (~rain_mask).sum())
        self.p_high_given_peak = float((high_mask & peak_mask).sum() / peak_mask.sum())
        self.p_high_given_nonpeak = float((high_mask & ~peak_mask).sum() / (~peak_mask).sum())
        
        # 4 Quadrants of (Rain, Peak)
        self.joint_probabilities = {
            (1, 1): {
                "label": "Rain = 1, Peak = 1",
                "condition_desc": "Rain Present during Peak Commute Hour",
                "n_obs": int((rain_mask & peak_mask).sum()),
                "n_high": int((high_mask & rain_mask & peak_mask).sum()),
                "prob": float((high_mask & rain_mask & peak_mask).sum() / (rain_mask & peak_mask).sum()),
                "expected_traffic_subset": float(df.loc[rain_mask & peak_mask, "traffic_volume"].mean())
            },
            (0, 1): {
                "label": "Rain = 0, Peak = 1",
                "condition_desc": "Dry Weather during Peak Commute Hour",
                "n_obs": int((~rain_mask & peak_mask).sum()),
                "n_high": int((high_mask & ~rain_mask & peak_mask).sum()),
                "prob": float((high_mask & ~rain_mask & peak_mask).sum() / (~rain_mask & peak_mask).sum()),
                "expected_traffic_subset": float(df.loc[~rain_mask & peak_mask, "traffic_volume"].mean())
            },
            (1, 0): {
                "label": "Rain = 1, Peak = 0",
                "condition_desc": "Rain Present during Off-Peak Hour",
                "n_obs": int((rain_mask & ~peak_mask).sum()),
                "n_high": int((high_mask & rain_mask & ~peak_mask).sum()),
                "prob": float((high_mask & rain_mask & ~peak_mask).sum() / (rain_mask & ~peak_mask).sum()),
                "expected_traffic_subset": float(df.loc[rain_mask & ~peak_mask, "traffic_volume"].mean())
            },
            (0, 0): {
                "label": "Rain = 0, Peak = 0",
                "condition_desc": "Dry Weather during Off-Peak Hour",
                "n_obs": int((~rain_mask & ~peak_mask).sum()),
                "n_high": int((high_mask & ~rain_mask & ~peak_mask).sum()),
                "prob": float((high_mask & ~rain_mask & ~peak_mask).sum() / (~rain_mask & ~peak_mask).sum()),
                "expected_traffic_subset": float(df.loc[~rain_mask & ~peak_mask, "traffic_volume"].mean())
            }
        }
        
        # Holiday condition
        if holiday_mask.sum() > 0:
            self.p_high_given_holiday = float((high_mask & holiday_mask).sum() / holiday_mask.sum())
        else:
            self.p_high_given_holiday = 0.0

    def analyze(
        self,
        hour: int,
        rain: int,
        peak: Optional[int] = None,
        holiday: Optional[int] = None,
        polynomial_degree: int = DEFAULT_SELECTED_DEGREE,
        city: Optional[str] = None,
        area: Optional[str] = None,
        vehicle_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute integrated commute analysis for given conditions.
        
        Parameters:
        - hour: integer 0 to 23
        - rain: integer 0 or 1
        - peak: optional integer 0 or 1 (if None, inferred from hour in CORE_PEAK_HOURS)
        - holiday: optional integer 0 or 1
        - polynomial_degree: integer 1, 2, or 3 (defaults to 3)
        - city: optional city name
        - area: optional area name
        - vehicle_type: optional vehicle type
        """
        # 1. Validate Hour
        if not isinstance(hour, (int, np.integer)):
            raise TypeError(f"Hour must be an integer, got {type(hour).__name__}.")
        if not (0 <= hour <= 23):
            raise ValueError(f"Hour {hour} is invalid; must be an integer between 0 and 23.")
            
        # 2. Validate Rain
        if not isinstance(rain, (int, np.integer)):
            raise TypeError(f"Rain must be an integer (0 or 1), got {type(rain).__name__}.")
        if rain not in (0, 1):
            raise ValueError(f"Rain {rain} is invalid; must be 0 (No Rain) or 1 (Rain).")
            
        # 3. Validate / Infer Peak
        if peak is None:
            # Objective inference from Stage 2/3 core peak hours
            peak = 1 if hour in CORE_PEAK_HOURS else 0
        else:
            if not isinstance(peak, (int, np.integer)):
                raise TypeError(f"Peak must be an integer (0 or 1), got {type(peak).__name__}.")
            if peak not in (0, 1):
                raise ValueError(f"Peak {peak} is invalid; must be 0 (Non-Peak) or 1 (Peak).")
                
        # 4. Validate Holiday if provided
        if holiday is not None:
            if not isinstance(holiday, (int, np.integer)):
                raise TypeError(f"Holiday must be an integer (0 or 1), got {type(holiday).__name__}.")
            if holiday not in (0, 1):
                raise ValueError(f"Holiday {holiday} is invalid; must be 0 (Non-Holiday) or 1 (Holiday).")

        # 5. Expected Traffic Volume
        if city is not None or area is not None or vehicle_type is not None:
            from src.new_preprocessing import load_new_dataset
            from src.new_probability import calculate_empirical_expected_volume
            new_df = load_new_dataset()
            vol_res = calculate_empirical_expected_volume(new_df, city=city, area=area, vehicle_type=vehicle_type)
            expected_traffic = vol_res['expected_vehicle_count']
        else:
            # Polynomial Component: Expected Traffic Volume = f(Hour)
            expected_traffic = predict_traffic_for_hour(hour, degree=polynomial_degree)
        
        # 6. Probability Component: P(High | conditions)
        quadrant_info = self.joint_probabilities[(rain, peak)]
        p_high_cond = quadrant_info["prob"]
        n_supporting_obs = quadrant_info["n_obs"]
        condition_label = quadrant_info["condition_desc"]
        
        # 7. Difference from Dataset Baseline
        diff = p_high_cond - self.p_high_baseline
        diff_ppt = diff * 100.0  # percentage points
        
        # 8. Objective Unit-1 Rule-Based Interpretation
        if p_high_cond > self.p_high_baseline:
            rule_interpretation = "Observed high-congestion probability is above the overall dataset baseline."
        else:
            rule_interpretation = "Observed high-congestion probability is at or below the overall dataset baseline."
            
        return {
            "hour": int(hour),
            "rain": int(rain),
            "peak": int(peak),
            "holiday": int(holiday) if holiday is not None else None,
            "expected_traffic": float(expected_traffic),
            "high_congestion_probability": float(p_high_cond),
            "baseline_high_probability": float(self.p_high_baseline),
            "difference_from_baseline": float(diff),
            "difference_percentage_points": float(diff_ppt),
            "supporting_observations": int(n_supporting_obs),
            "condition_label": condition_label,
            "polynomial_degree": int(polynomial_degree),
            "interpretation": rule_interpretation,
            "single_condition_probabilities": {
                "P_High_given_Rain": self.p_high_given_rain,
                "P_High_given_NoRain": self.p_high_given_norain,
                "P_High_given_Peak": self.p_high_given_peak,
                "P_High_given_NonPeak": self.p_high_given_nonpeak
            }
        }

    def explain_bayes(self, evidence: str = "rain") -> Dict[str, Any]:
        """
        Expose Bayes' Theorem calculation for educational transparency:
        P(High | Rain) = [P(Rain | High) * P(High)] / P(Rain)
        """
        high_mask = (self.df["congestion"] == "High")
        if evidence.lower() == "rain":
            ev_mask = (self.df["rain"] == 1)
            ev_name = "Rain"
        elif evidence.lower() == "peak":
            ev_mask = self.df["hour"].isin(CORE_PEAK_HOURS)
            ev_name = "Peak Hour"
        else:
            raise ValueError(f"Unsupported evidence '{evidence}'. Supported: 'rain', 'peak'.")
            
        verification = verify_bayes_with_empirical_probability(
            df=self.df,
            target_condition=high_mask,
            evidence_condition=ev_mask,
            target_name="High Congestion",
            evidence_name=ev_name
        )
        # Add educational alias keys for notebooks and UI
        verification["prior_P_High"] = verification["prior_P_A"]
        verification["marginal_P_Evidence"] = verification["marginal_P_B"]
        verification["likelihood_P_Evidence_given_High"] = verification["likelihood_P_B_given_A"]
        verification["bayes_P_High_given_Evidence"] = verification["bayes_posterior"]
        verification["direct_empirical_P_High_given_Evidence"] = verification["direct_empirical_posterior"]
        verification["exact_match_discrepancy"] = verification["absolute_difference"]
        return verification


# Module-level singleton engine
_ENGINE_INSTANCE: Optional[CommuteAnalysisEngine] = None

def get_commute_engine() -> CommuteAnalysisEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = CommuteAnalysisEngine()
    return _ENGINE_INSTANCE


def analyze_commute_conditions(
    hour: int,
    rain: int,
    peak: Optional[int] = None,
    holiday: Optional[int] = None,
    polynomial_degree: int = DEFAULT_SELECTED_DEGREE,
    city: Optional[str] = None,
    area: Optional[str] = None,
    vehicle_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Primary functional interface for Stage 5 integrated commute analysis.
    
    Parameters:
    - hour: integer from 0 to 23
    - rain: integer 0 or 1
    - peak: optional integer 0 or 1 (inferred from hour if omitted)
    - holiday: optional integer 0 or 1
    - polynomial_degree: polynomial degree (1, 2, or 3; default 3)
    - city: optional city name
    - area: optional area name
    - vehicle_type: optional vehicle type
    
    Returns:
    Dictionary containing expected traffic, high congestion probability, baseline comparison,
    and rule-based interpretation.
    """
    engine = get_commute_engine()
    return engine.analyze(
        hour=hour,
        rain=rain,
        peak=peak,
        holiday=holiday,
        polynomial_degree=polynomial_degree,
        city=city,
        area=area,
        vehicle_type=vehicle_type
    )


def format_readable_analysis(result: Dict[str, Any]) -> str:
    """
    Format analysis result into a clean, human-readable college presentation layout.
    """
    h = result["hour"]
    time_str = f"{h % 12 if h % 12 != 0 else 12} {'AM' if h < 12 else 'PM'}"
    rain_str = "Yes (rain = 1)" if result["rain"] == 1 else "No (rain = 0)"
    peak_str = "Yes (Peak Hour)" if result["peak"] == 1 else "No (Off-Peak Hour)"
    
    diff_sign = "+" if result["difference_from_baseline"] >= 0 else ""
    
    lines = [
        "CommuteCast Analysis",
        "--------------------",
        f"Hour                      : {h:02d}:00 ({time_str})",
        f"Rain Condition            : {rain_str}",
        f"Peak Commute Window       : {peak_str}",
        "",
        "A. Expected Traffic Volume (Polynomial Component):",
        f"   {result['expected_traffic']:,.1f} veh/hr  (Fitted via Degree {result['polynomial_degree']} Polynomial)",
        "",
        "B. High Congestion Probability (Empirical Component):",
        f"   P(High | Conditions)   : {result['high_congestion_probability'] * 100:.2f}%  ({result['condition_label']})",
        f"   Overall Dataset Base   : {result['baseline_high_probability'] * 100:.2f}%",
        f"   Difference             : {diff_sign}{result['difference_percentage_points']:.2f} percentage points",
        f"   Supporting Observations: {result['supporting_observations']:,} historical hours",
        "",
        "C. Interpretation & Statistical Boundaries:",
        f"   {result['interpretation']}",
        "   Explanation: The polynomial component estimates typical traffic volume for the selected",
        "   hour, while the probability component estimates the empirical likelihood of high",
        "   congestion under the selected conditions. They remain mathematically separate."
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Test example: 8 AM + Rain (Peak=1)
    res = analyze_commute_conditions(hour=8, rain=1)
    print(format_readable_analysis(res))
