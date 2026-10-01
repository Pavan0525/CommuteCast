"""
CommuteCast - Stage 2: Exploratory Statistical Analysis
Module: src/statistics.py

Responsibilities:
1. Compute descriptive statistics for traffic_volume:
   - Count, Mean (empirical expectation E[X]), Variance, Standard Deviation,
     Minimum, Q25, Median (Q50), Q75, Maximum, and IQR.
2. Formulate theoretical expectation foundations:
   - Explain why sample mean acts as empirical estimator of expected value E[X].
3. Compute sample covariance between traffic_volume and numerical features:
   - rain_1h (reporting both raw and sensor-validated metrics)
   - temp_celsius (pairwise complete observations)
   - clouds_all
4. Explain Covariance vs. Causation strictly within Unit-1 foundational concepts.
5. Quantile-based Congestion analysis:
   - Report exact thresholds (Q25, Q50, Q75)
   - Report counts and percentage distribution for Low, Medium, High categories.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly from any folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.preprocessing import (
        load_raw_data,
        preprocess_traffic_data,
        PROCESSED_DATA_PATH
    )
except ImportError:
    from preprocessing import (
        load_raw_data,
        preprocess_traffic_data,
        PROCESSED_DATA_PATH
    )


def compute_traffic_volume_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute foundational descriptive statistics for traffic_volume:
    - Count (N)
    - Mean (x_bar): Sample mean as empirical estimator of expected traffic volume E[X]
    - Sample Variance (s^2, ddof=1) & Population Variance (sigma^2, ddof=0)
    - Sample Std Dev (s, ddof=1) & Population Std Dev (sigma, ddof=0)
    - Five-number summary: Min, Q25, Median (Q50), Q75, Max
    - Interquartile Range (IQR = Q75 - Q25)
    """
    tv = df["traffic_volume"]
    n = int(tv.count())
    mean_val = float(tv.mean())
    var_sample = float(tv.var(ddof=1))
    var_pop = float(tv.var(ddof=0))
    std_sample = float(tv.std(ddof=1))
    std_pop = float(tv.std(ddof=0))
    min_val = float(tv.min())
    q25 = float(tv.quantile(0.25))
    median_val = float(tv.median())
    q75 = float(tv.quantile(0.75))
    max_val = float(tv.max())
    iqr = q75 - q25

    return {
        "count": n,
        "mean_expected_value": mean_val,
        "sample_variance": var_sample,
        "population_variance": var_pop,
        "sample_std_dev": std_sample,
        "population_std_dev": std_pop,
        "min": min_val,
        "q25": q25,
        "median": median_val,
        "q75": q75,
        "max": max_val,
        "iqr": iqr
    }


def compute_covariances(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute sample covariance between traffic_volume and numerical predictors:
    Cov(X, Y) = (1 / (N - 1)) * sum((x_i - x_bar) * (y_i - y_bar))
    
    Selected variables:
    1. clouds_all (cloud coverage percentage 0-100%)
    2. temp_celsius (atmospheric temperature in Celsius; 10 sensor dropouts excluded)
    3. rain_1h (rainfall in mm; both raw and sensor-validated series reported)
    """
    tv = df["traffic_volume"]
    
    # Cloud coverage
    cov_clouds = float(tv.cov(df["clouds_all"]))
    corr_clouds = float(tv.corr(df["clouds_all"]))
    
    # Temperature (Celsius) - automatically drops 10 NaN sensor dropout rows
    cov_temp = float(tv.cov(df["temp_celsius"]))
    corr_temp = float(tv.corr(df["temp_celsius"]))
    valid_temp_count = int(df["temp_celsius"].notna().sum())
    
    # Rainfall (raw)
    cov_rain_raw = float(tv.cov(df["rain_1h"]))
    corr_rain_raw = float(tv.corr(df["rain_1h"]))
    
    # Rainfall (validated: excluding physical sensor glitch 9831.3 mm)
    if "rain_1h_valid" in df.columns:
        cov_rain_valid = float(tv.cov(df["rain_1h_valid"]))
        corr_rain_valid = float(tv.corr(df["rain_1h_valid"]))
        valid_rain_count = int(df["rain_1h_valid"].notna().sum())
    else:
        rain_valid = df["rain_1h"].replace(df[df["rain_1h"] > 500]["rain_1h"], np.nan)
        cov_rain_valid = float(tv.cov(rain_valid))
        corr_rain_valid = float(tv.corr(rain_valid))
        valid_rain_count = int(rain_valid.notna().sum())

    return {
        "cov_traffic_clouds": cov_clouds,
        "corr_traffic_clouds": corr_clouds,
        "cov_traffic_temp": cov_temp,
        "corr_traffic_temp": corr_temp,
        "valid_temp_observations": valid_temp_count,
        "cov_traffic_rain_raw": cov_rain_raw,
        "corr_traffic_rain_raw": corr_rain_raw,
        "cov_traffic_rain_valid": cov_rain_valid,
        "corr_traffic_rain_valid": corr_rain_valid,
        "valid_rain_observations": valid_rain_count
    }


def compute_congestion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frequency counts and percentages for congestion classes:
    - Low: traffic_volume <= Q25
    - Medium: Q25 < traffic_volume <= Q75
    - High: traffic_volume > Q75
    """
    counts = df["congestion"].value_counts().reindex(["Low", "Medium", "High"])
    percentages = (df["congestion"].value_counts(normalize=True) * 100).reindex(["Low", "Medium", "High"])
    
    summary_df = pd.DataFrame({
        "Category": ["Low", "Medium", "High"],
        "Count": counts.values,
        "Percentage (%)": percentages.values
    })
    return summary_df


def format_unit1_report(stats: Dict[str, Any], covs: Dict[str, Any], congestion_df: pd.DataFrame) -> str:
    """
    Format descriptive statistics, expectation analysis, covariance evaluation,
    and congestion distribution into an academic presentation-ready report.
    """
    lines = []
    lines.append("=" * 78)
    lines.append("   COMMUTECAST: UNIT-1 EXPLORATORY STATISTICAL ANALYSIS REPORT")
    lines.append("=" * 78)
    lines.append("")
    
    lines.append("1. TRAFFIC VOLUME DESCRIPTIVE STATISTICS & EMPIRICAL EXPECTATION")
    lines.append("-" * 78)
    lines.append(f"  Sample Size (N)             : {stats['count']:,} observations")
    lines.append(f"  Empirical Expectation E[X]  : {stats['mean_expected_value']:.2f} vehicles/hour (Sample Mean x_bar)")
    lines.append(f"  Sample Variance (s^2)       : {stats['sample_variance']:,.2f} (ddof=1)")
    lines.append(f"  Population Variance (sigma^2): {stats['population_variance']:,.2f} (ddof=0)")
    lines.append(f"  Sample Std Deviation (s)    : {stats['sample_std_dev']:.2f} vehicles/hour")
    lines.append(f"  Population Std Dev (sigma)  : {stats['population_std_dev']:.2f} vehicles/hour")
    lines.append(f"  Minimum Volume              : {stats['min']:.0f} vehicles/hour")
    lines.append(f"  First Quartile (Q25)        : {stats['q25']:.1f} vehicles/hour")
    lines.append(f"  Median / Second Quartile (Q50): {stats['median']:.1f} vehicles/hour")
    lines.append(f"  Third Quartile (Q75)        : {stats['q75']:.1f} vehicles/hour")
    lines.append(f"  Maximum Volume              : {stats['max']:.0f} vehicles/hour")
    lines.append(f"  Interquartile Range (IQR)   : {stats['iqr']:.1f} vehicles/hour")
    lines.append("")
    lines.append("  [Theoretical Foundation of Expected Value]:")
    lines.append("  In probability theory, the expected value E[X] represents the probability-")
    lines.append("  weighted average of all possible outcomes. In empirical observational")
    lines.append("  settings where the true population probability density function f(x) is")
    lines.append("  unknown, the arithmetic sample mean x_bar = (1/n) * sum(x_i) is used as")
    lines.append("  the empirical, unbiased point estimator of E[X] per the Law of Large Numbers.")
    lines.append("")
    
    lines.append("2. SAMPLE COVARIANCE ANALYSIS & CAUSATION WARNING")
    lines.append("-" * 78)
    lines.append(f"  Cov(traffic_volume, clouds_all)      : +{covs['cov_traffic_clouds']:,.2f}  (r = {covs['corr_traffic_clouds']:+.4f})")
    lines.append(f"  Cov(traffic_volume, temp_celsius)    : +{covs['cov_traffic_temp']:,.2f}  (r = {covs['corr_traffic_temp']:+.4f}, N = {covs['valid_temp_observations']:,})")
    lines.append(f"  Cov(traffic_volume, rain_1h [raw])   : +{covs['cov_traffic_rain_raw']:,.2f}  (r = {covs['corr_traffic_rain_raw']:+.4f})")
    lines.append(f"  Cov(traffic_volume, rain_1h [valid]) : {covs['cov_traffic_rain_valid']:,.2f}  (r = {covs['corr_traffic_rain_valid']:+.4f}, N = {covs['valid_rain_observations']:,})")
    lines.append("")
    lines.append("  [CRITICAL: Covariance is NOT Causation]:")
    lines.append("  Covariance quantifies the degree to which two random variables co-vary linearly:")
    lines.append("  Cov(X, Y) = E[(X - E[X])(Y - E[Y])]. A positive covariance between temperature")
    lines.append("  and traffic volume does NOT imply that hotter ambient weather causes people")
    lines.append("  to drive. Rather, both variables share latent diurnal and seasonal confounding")
    lines.append("  cycles: daytime hours have both higher solar heating and commuter rush hours;")
    lines.append("  summer months exhibit higher seasonal travel alongside warmer temperatures.")
    lines.append("")
    
    lines.append("3. QUANTILE-BASED CONGESTION CLASSIFICATION")
    lines.append("-" * 78)
    lines.append(f"  Thresholds : Low <= {stats['q25']:.1f} | Medium: ({stats['q25']:.1f}, {stats['q75']:.1f}] | High > {stats['q75']:.1f}")
    lines.append("")
    for _, row in congestion_df.iterrows():
        lines.append(f"  Category {row['Category']:<6}: {int(row['Count']):>6,} observations ({row['Percentage (%)']:>6.2f}%)")
    total_count = int(congestion_df['Count'].sum())
    lines.append(f"  Total Cleaned Observations : {total_count:,} (100.00%)")
    lines.append("=" * 78)
    
    return "\n".join(lines)


def run_statistical_analysis(csv_path: str = PROCESSED_DATA_PATH) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, str]:
    """
    Execute statistical analysis workflow:
    - Loads preprocessed dataset (or processes raw data if needed).
    - Computes descriptive statistics and expectation.
    - Computes covariances.
    - Computes congestion frequency breakdown.
    - Returns structured results and formatted report string.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = preprocess_traffic_data()
        
    stats = compute_traffic_volume_statistics(df)
    covs = compute_covariances(df)
    congestion_df = compute_congestion_distribution(df)
    report_text = format_unit1_report(stats, covs, congestion_df)
    
    return stats, covs, congestion_df, report_text


if __name__ == "__main__":
    stats, covs, congestion_df, report = run_statistical_analysis()
    print(report)
