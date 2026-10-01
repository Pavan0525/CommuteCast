"""
CommuteCast - Stage 3: Probability, Conditional Probability & Expectation
Module: src/probability.py

Responsibilities:
1. Basic Probability:
   - P(Low), P(Medium), P(High)
   - P(Rain), P(No Rain)
2. Conditional Probability:
   - P(Congestion | Rain) vs P(Congestion | No Rain)
   - P(High | Peak) vs P(High | Non-Peak)
   - P(High | Weekday) vs P(High | Weekend)
   - P(High | Holiday) vs P(High | Non-Holiday)
   - P(High | Rain, Peak) vs P(High | No Rain, Peak)
3. Empirical Expectation:
   - E(Traffic), E(Traffic | Rain), E(Traffic | No Rain)
   - E(Traffic | High Congestion)
   - E(Traffic | Peak), E(Traffic | Non-Peak)
   - E(Traffic | Weekday), E(Traffic | Weekend)
   - E(Traffic | Holiday), E(Traffic | Non-Holiday)
4. Event Independence Evaluation:
   - Compare P(A | B) with P(A) without making unfounded causal claims.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

# Ensure project root is first in sys.path and remove 'src' from sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PROCESSED_DATA_PATH, preprocess_traffic_data

# Defensible Core Peak Hours based on Stage 2 diurnal commute profile:
# Morning rush (7-8 AM) & Evening rush (16-17 PM)
CORE_PEAK_HOURS = [7, 8, 16, 17]


def calculate_probability(df: pd.DataFrame, condition: pd.Series) -> float:
    """
    Calculate empirical probability of an event A:
    P(A) = count(A) / total_observations
    """
    total = len(df)
    if total == 0:
        raise ValueError("Cannot calculate probability on an empty DataFrame.")
    count_a = int(condition.sum())
    prob = count_a / total
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"Calculated probability {prob} is outside [0, 1].")
    return float(prob)


def calculate_conditional_probability(
    df: pd.DataFrame,
    target_condition: pd.Series,
    given_condition: pd.Series
) -> float:
    """
    Calculate empirical conditional probability P(A | B):
    P(A | B) = count(A and B) / count(B)
    """
    count_b = int(given_condition.sum())
    if count_b == 0:
        raise ZeroDivisionError("Condition B has zero occurrences; conditional probability undefined.")
    count_a_and_b = int((target_condition & given_condition).sum())
    cond_prob = count_a_and_b / count_b
    if not (0.0 <= cond_prob <= 1.0):
        raise ValueError(f"Calculated conditional probability {cond_prob} is outside [0, 1].")
    return float(cond_prob)


def calculate_expectation(
    df: pd.DataFrame,
    target_col: str = "traffic_volume",
    condition: Optional[pd.Series] = None
) -> float:
    """
    Calculate empirical expectation E[X | Condition] using the arithmetic sample mean:
    E[X] = (1 / n) * sum(x_i)
    
    Theoretical note: The sample mean is the minimum-variance unbiased estimator (MVUE)
    of the true theoretical expectation under the Law of Large Numbers.
    """
    if condition is not None:
        sub_series = df.loc[condition, target_col].dropna()
    else:
        sub_series = df[target_col].dropna()
        
    if len(sub_series) == 0:
        raise ValueError("Cannot calculate expectation on an empty subset.")
    exp_val = float(sub_series.mean())
    if np.isnan(exp_val) or np.isinf(exp_val):
        raise ValueError(f"Calculated expectation {exp_val} is NaN or Infinite.")
    return exp_val


def calculate_congestion_probabilities(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate marginal probabilities for congestion categories:
    P(Low), P(Medium), P(High)
    """
    n = len(df)
    low_cnt = int((df["congestion"] == "Low").sum())
    med_cnt = int((df["congestion"] == "Medium").sum())
    high_cnt = int((df["congestion"] == "High").sum())
    
    p_low = low_cnt / n
    p_med = med_cnt / n
    p_high = high_cnt / n
    prob_sum = p_low + p_med + p_high
    
    return {
        "count_low": low_cnt,
        "count_medium": med_cnt,
        "count_high": high_cnt,
        "total_observations": n,
        "P_Low": p_low,
        "P_Medium": p_med,
        "P_High": p_high,
        "sum_probabilities": prob_sum
    }


def calculate_rain_probabilities(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate marginal probabilities for rain presence:
    P(Rain) and P(No Rain) where rain = 1 if rain_1h > 0 else 0
    """
    n = len(df)
    rain_cnt = int((df["rain"] == 1).sum())
    norain_cnt = int((df["rain"] == 0).sum())
    
    p_rain = rain_cnt / n
    p_norain = norain_cnt / n
    prob_sum = p_rain + p_norain
    
    return {
        "count_rain": rain_cnt,
        "count_norain": norain_cnt,
        "total_observations": n,
        "P_Rain": p_rain,
        "P_NoRain": p_norain,
        "sum_probabilities": prob_sum
    }


def calculate_rain_conditional_probabilities(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate conditional probabilities of congestion given rain status:
    P(High | Rain), P(High | No Rain)
    P(Medium | Rain), P(Medium | No Rain)
    P(Low | Rain), P(Low | No Rain)
    """
    rain_mask = (df["rain"] == 1)
    norain_mask = (df["rain"] == 0)
    
    high_mask = (df["congestion"] == "High")
    med_mask = (df["congestion"] == "Medium")
    low_mask = (df["congestion"] == "Low")
    
    return {
        "P_High_given_Rain": calculate_conditional_probability(df, high_mask, rain_mask),
        "P_High_given_NoRain": calculate_conditional_probability(df, high_mask, norain_mask),
        "P_Medium_given_Rain": calculate_conditional_probability(df, med_mask, rain_mask),
        "P_Medium_given_NoRain": calculate_conditional_probability(df, med_mask, norain_mask),
        "P_Low_given_Rain": calculate_conditional_probability(df, low_mask, rain_mask),
        "P_Low_given_NoRain": calculate_conditional_probability(df, low_mask, norain_mask),
    }


def calculate_peak_hour_analysis(df: pd.DataFrame, peak_hours: List[int] = CORE_PEAK_HOURS) -> Dict[str, Any]:
    """
    Calculate peak-hour probabilities and expectations.
    
    Peak-Hour Selection Justification:
    Based on Stage 2 exploratory hourly analysis, traffic volume exhibits a distinct bimodal diurnal shape:
    - Morning crest: Hours 7:00 and 8:00 AM (average volume: 4,740 and 4,587 veh/hr; P(High) = 64.7% and 62.0%)
    - Evening crest: Hours 16:00 and 17:00 PM (average volume: 5,664 and 5,310 veh/hr; P(High) = 69.5% and 66.0%)
    Hence, CORE_PEAK_HOURS = [7, 8, 16, 17] captures the 4 core rush-hour commuting windows.
    """
    peak_mask = df["hour"].isin(peak_hours)
    nonpeak_mask = ~peak_mask
    high_mask = (df["congestion"] == "High")
    
    p_peak = calculate_probability(df, peak_mask)
    p_nonpeak = calculate_probability(df, nonpeak_mask)
    
    p_high_given_peak = calculate_conditional_probability(df, high_mask, peak_mask)
    p_high_given_nonpeak = calculate_conditional_probability(df, high_mask, nonpeak_mask)
    
    e_peak = calculate_expectation(df, condition=peak_mask)
    e_nonpeak = calculate_expectation(df, condition=nonpeak_mask)
    
    return {
        "peak_hours": peak_hours,
        "count_peak": int(peak_mask.sum()),
        "count_nonpeak": int(nonpeak_mask.sum()),
        "P_Peak": p_peak,
        "P_NonPeak": p_nonpeak,
        "P_High_given_Peak": p_high_given_peak,
        "P_High_given_NonPeak": p_high_given_nonpeak,
        "E_Traffic_Peak": e_peak,
        "E_Traffic_NonPeak": e_nonpeak
    }


def calculate_rain_and_peak_joint_analysis(df: pd.DataFrame, peak_hours: List[int] = CORE_PEAK_HOURS) -> Dict[str, Any]:
    """
    Examine joint conditional behavior: Rain + Peak Hour
    Calculates:
    - P(High | Rain, Peak Hour)
    - P(High | No Rain, Peak Hour)
    - E(Traffic | Rain, Peak Hour)
    - E(Traffic | No Rain, Peak Hour)
    """
    peak_mask = df["hour"].isin(peak_hours)
    rain_mask = (df["rain"] == 1)
    norain_mask = (df["rain"] == 0)
    high_mask = (df["congestion"] == "High")
    
    rain_peak_mask = (rain_mask & peak_mask)
    norain_peak_mask = (norain_mask & peak_mask)
    
    p_high_given_rain_peak = calculate_conditional_probability(df, high_mask, rain_peak_mask)
    p_high_given_norain_peak = calculate_conditional_probability(df, high_mask, norain_peak_mask)
    
    e_traffic_rain_peak = calculate_expectation(df, condition=rain_peak_mask)
    e_traffic_norain_peak = calculate_expectation(df, condition=norain_peak_mask)
    
    return {
        "count_rain_peak": int(rain_peak_mask.sum()),
        "count_norain_peak": int(norain_peak_mask.sum()),
        "P_High_given_Rain_Peak": p_high_given_rain_peak,
        "P_High_given_NoRain_Peak": p_high_given_norain_peak,
        "E_Traffic_Rain_Peak": e_traffic_rain_peak,
        "E_Traffic_NoRain_Peak": e_traffic_norain_peak
    }


def calculate_temporal_conditional_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate conditional probabilities and expectations for:
    1. Weekday vs Weekend (is_weekend = 0 vs 1)
    2. Holiday vs Non-Holiday (is_holiday = 1 vs 0)
    """
    high_mask = (df["congestion"] == "High")
    
    # Weekday vs Weekend
    weekday_mask = (df["is_weekend"] == 0)
    weekend_mask = (df["is_weekend"] == 1)
    
    p_high_weekday = calculate_conditional_probability(df, high_mask, weekday_mask)
    p_high_weekend = calculate_conditional_probability(df, high_mask, weekend_mask)
    e_weekday = calculate_expectation(df, condition=weekday_mask)
    e_weekend = calculate_expectation(df, condition=weekend_mask)
    
    # Holiday vs Non-Holiday
    holiday_mask = (df["is_holiday"] == 1)
    nonholiday_mask = (df["is_holiday"] == 0)
    
    p_high_holiday = calculate_conditional_probability(df, high_mask, holiday_mask)
    p_high_nonholiday = calculate_conditional_probability(df, high_mask, nonholiday_mask)
    e_holiday = calculate_expectation(df, condition=holiday_mask)
    e_nonholiday = calculate_expectation(df, condition=nonholiday_mask)
    
    return {
        "count_weekday": int(weekday_mask.sum()),
        "count_weekend": int(weekend_mask.sum()),
        "P_High_given_Weekday": p_high_weekday,
        "P_High_given_Weekend": p_high_weekend,
        "E_Traffic_Weekday": e_weekday,
        "E_Traffic_Weekend": e_weekend,
        
        "count_holiday": int(holiday_mask.sum()),
        "count_nonholiday": int(nonholiday_mask.sum()),
        "P_High_given_Holiday": p_high_holiday,
        "P_High_given_NonHoliday": p_high_nonholiday,
        "E_Traffic_Holiday": e_holiday,
        "E_Traffic_NonHoliday": e_nonholiday
    }


def evaluate_statistical_independence(p_target: float, p_target_given_condition: float, event_name: str, threshold: float = 0.03) -> str:
    """
    Compare P(A | B) with P(A) to assess empirical independence:
    If |P(A | B) - P(A)| < threshold, the data is consistent with approximate independence.
    Otherwise, the observed data does NOT support treating these events as independent.
    """
    diff = abs(p_target_given_condition - p_target)
    if diff <= threshold:
        return (
            f"P(High | {event_name}) = {p_target_given_condition:.4f} is close to P(High) = {p_target:.4f} "
            f"(diff = {diff:.4f} <= {threshold:.2f}). The data is consistent with approximate independence."
        )
    else:
        return (
            f"P(High | {event_name}) = {p_target_given_condition:.4f} differs noticeably from P(High) = {p_target:.4f} "
            f"(diff = {diff:.4f} > {threshold:.2f}). The observed data does not support treating these events as independent."
        )


def run_complete_probability_analysis(csv_path: str = PROCESSED_DATA_PATH) -> Dict[str, Any]:
    """
    Execute full Stage 3 probability analysis workflow and return structured dictionary.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = preprocess_traffic_data()
        
    congestion_probs = calculate_congestion_probabilities(df)
    rain_probs = calculate_rain_probabilities(df)
    rain_cond_probs = calculate_rain_conditional_probabilities(df)
    
    # Expectations
    e_overall = calculate_expectation(df)
    e_rain = calculate_expectation(df, condition=(df["rain"] == 1))
    e_norain = calculate_expectation(df, condition=(df["rain"] == 0))
    e_high = calculate_expectation(df, condition=(df["congestion"] == "High"))
    
    peak_analysis = calculate_peak_hour_analysis(df)
    rain_peak_analysis = calculate_rain_and_peak_joint_analysis(df)
    temporal_analysis = calculate_temporal_conditional_analysis(df)
    
    # Independence evaluations
    ind_rain = evaluate_statistical_independence(congestion_probs["P_High"], rain_cond_probs["P_High_given_Rain"], "Rain", threshold=0.03)
    ind_peak = evaluate_statistical_independence(congestion_probs["P_High"], peak_analysis["P_High_given_Peak"], "Peak Hour", threshold=0.03)
    ind_weekday = evaluate_statistical_independence(congestion_probs["P_High"], temporal_analysis["P_High_given_Weekday"], "Weekday", threshold=0.03)
    
    # Covariance & Correlation for rainfall
    cov_rain_raw = float(df["traffic_volume"].cov(df["rain_1h"]))
    corr_rain_raw = float(df["traffic_volume"].corr(df["rain_1h"]))
    cov_rain_valid = float(df["traffic_volume"].cov(df["rain_1h_valid"]))
    corr_rain_valid = float(df["traffic_volume"].corr(df["rain_1h_valid"]))
    
    return {
        "congestion_probs": congestion_probs,
        "rain_probs": rain_probs,
        "rain_cond_probs": rain_cond_probs,
        "expectations": {
            "E_Traffic": e_overall,
            "E_Traffic_Rain": e_rain,
            "E_Traffic_NoRain": e_norain,
            "E_Traffic_High": e_high
        },
        "peak_analysis": peak_analysis,
        "rain_peak_analysis": rain_peak_analysis,
        "temporal_analysis": temporal_analysis,
        "independence": {
            "rain_vs_high": ind_rain,
            "peak_vs_high": ind_peak,
            "weekday_vs_high": ind_weekday
        },
        "rainfall_linear_metrics": {
            "cov_rain_raw": cov_rain_raw,
            "corr_rain_raw": corr_rain_raw,
            "cov_rain_valid": cov_rain_valid,
            "corr_rain_valid": corr_rain_valid
        }
    }


if __name__ == "__main__":
    results = run_complete_probability_analysis()
    print("=== CommuteCast: Stage 3 Probability Analysis Completed Successfully ===")
    print(f"P(Low)    : {results['congestion_probs']['P_Low']:.4f}")
    print(f"P(Medium) : {results['congestion_probs']['P_Medium']:.4f}")
    print(f"P(High)   : {results['congestion_probs']['P_High']:.4f}")
    print(f"P(Rain)   : {results['rain_probs']['P_Rain']:.4f}")
    print(f"P(High | Rain)   : {results['rain_cond_probs']['P_High_given_Rain']:.4f}")
    print(f"P(High | No Rain): {results['rain_cond_probs']['P_High_given_NoRain']:.4f}")
    print(f"E(Traffic)        : {results['expectations']['E_Traffic']:.2f} veh/hr")
    print(f"E(Traffic | Rain) : {results['expectations']['E_Traffic_Rain']:.2f} veh/hr")
    print(f"E(Traffic | No Rain): {results['expectations']['E_Traffic_NoRain']:.2f} veh/hr")
