"""
CommuteCast - Stage 3: Bayes' Theorem Implementation & Empirical Verification
Module: src/bayes.py

Responsibilities:
1. Implement Bayes' Theorem explicitly from foundational probability principles:
   P(A | B) = [P(B | A) * P(A)] / P(B)
2. Verify Bayes' theorem calculations against direct empirical conditional probabilities:
   Direct P(A | B) = count(A and B) / count(B)
3. Provide rigorous comparisons and display absolute differences down to machine precision.
4. Apply to two distinct domain examples:
   - Example 1: P(High Congestion | Rain)
   - Example 2: P(High Congestion | Peak Hour)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Ensure project root is first in sys.path and remove 'src' from sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PROCESSED_DATA_PATH, preprocess_traffic_data
from src.probability import CORE_PEAK_HOURS, calculate_probability, calculate_conditional_probability


def bayes_probability(
    prior_a: float,
    likelihood_b_given_a: float,
    marginal_b: float
) -> float:
    """
    Compute posterior probability P(A | B) using Bayes' Theorem:
    
                     P(B | A) * P(A)
        P(A | B) = -----------------
                          P(B)
                          
    Parameters:
    - prior_a: Prior probability of hypothesis A, P(A)
    - likelihood_b_given_a: Likelihood of evidence B given hypothesis A, P(B | A)
    - marginal_b: Marginal probability of evidence B, P(B)
    
    Validation:
    - All inputs must be valid probabilities in [0, 1].
    - marginal_b must be strictly greater than 0 to prevent division by zero.
    """
    for name, val in [("prior_a", prior_a), ("likelihood_b_given_a", likelihood_b_given_a), ("marginal_b", marginal_b)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Input '{name}' = {val} must be in range [0, 1].")
            
    if marginal_b == 0.0:
        raise ZeroDivisionError("Marginal probability P(B) is zero; Bayes' posterior undefined.")
        
    posterior = (likelihood_b_given_a * prior_a) / marginal_b
    
    # Due to floating point operations, clip to [0, 1] if minor precision noise
    if posterior < 0.0 and posterior > -1e-12:
        posterior = 0.0
    elif posterior > 1.0 and posterior < 1.0 + 1e-12:
        posterior = 1.0
        
    if not (0.0 <= posterior <= 1.0):
        raise ValueError(f"Calculated Bayes posterior {posterior} is outside [0, 1]. Check likelihood and prior consistency.")
        
    return float(posterior)


def verify_bayes_with_empirical_probability(
    df: pd.DataFrame,
    target_condition: pd.Series,
    evidence_condition: pd.Series,
    target_name: str = "A",
    evidence_name: str = "B",
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compare Bayes' Theorem calculation with direct empirical calculation.
    
    1. Direct calculation:
       P(A | B) = count(A and B) / count(B)
       
    2. Bayes' Theorem calculation:
       Prior: P(A) = count(A) / N
       Marginal: P(B) = count(B) / N
       Likelihood: P(B | A) = count(B and A) / count(A)
       Bayes Posterior: P(A | B) = [P(B | A) * P(A)] / P(B)
       
    3. Verification:
       Absolute Difference = |Posterior_Bayes - Posterior_Direct| <= tolerance
    """
    total_n = len(df)
    count_a = int(target_condition.sum())
    count_b = int(evidence_condition.sum())
    count_a_and_b = int((target_condition & evidence_condition).sum())
    
    # Direct empirical conditional probability
    p_direct = count_a_and_b / count_b
    
    # Component probabilities for Bayes
    prior_a = count_a / total_n
    marginal_b = count_b / total_n
    likelihood_b_given_a = count_a_and_b / count_a
    
    # Bayes theorem
    p_bayes = bayes_probability(
        prior_a=prior_a,
        likelihood_b_given_a=likelihood_b_given_a,
        marginal_b=marginal_b
    )
    
    abs_difference = abs(p_bayes - p_direct)
    is_verified = bool(abs_difference <= tolerance)
    
    return {
        "target_name": target_name,
        "evidence_name": evidence_name,
        "total_observations": total_n,
        "count_target": count_a,
        "count_evidence": count_b,
        "count_joint": count_a_and_b,
        "prior_P_A": prior_a,
        "marginal_P_B": marginal_b,
        "likelihood_P_B_given_A": likelihood_b_given_a,
        "bayes_posterior": p_bayes,
        "direct_empirical_posterior": p_direct,
        "absolute_difference": abs_difference,
        "is_verified": is_verified
    }


def run_bayes_analysis(csv_path: str = PROCESSED_DATA_PATH) -> Dict[str, Any]:
    """
    Execute both Bayes examples:
    - Example 1: P(High | Rain)
    - Example 2: P(High | Peak Hour)
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = preprocess_traffic_data()
        
    high_mask = (df["congestion"] == "High")
    rain_mask = (df["rain"] == 1)
    peak_mask = df["hour"].isin(CORE_PEAK_HOURS)
    
    # Example 1: P(High | Rain)
    example1 = verify_bayes_with_empirical_probability(
        df=df,
        target_condition=high_mask,
        evidence_condition=rain_mask,
        target_name="High Congestion",
        evidence_name="Rain"
    )
    
    # Example 2: P(High | Peak)
    example2 = verify_bayes_with_empirical_probability(
        df=df,
        target_condition=high_mask,
        evidence_condition=peak_mask,
        target_name="High Congestion",
        evidence_name="Peak Hour"
    )
    
    return {
        "example1_rain": example1,
        "example2_peak": example2
    }


def format_bayes_report(results: Dict[str, Any]) -> str:
    """Format Bayes verification results into a presentation report."""
    lines = []
    lines.append("=" * 78)
    lines.append("   COMMUTECAST: STAGE 3 BAYES' THEOREM EMPIRICAL VERIFICATION REPORT")
    lines.append("=" * 78)
    lines.append("")
    
    # Example 1
    ex1 = results["example1_rain"]
    lines.append("EXAMPLE 1: BAYES' THEOREM FOR P(High Congestion | Rain)")
    lines.append("-" * 78)
    lines.append("  Formula:")
    lines.append("                P(Rain | High) * P(High)")
    lines.append("  P(High|Rain) = ------------------------")
    lines.append("                        P(Rain)")
    lines.append("")
    lines.append(f"  Prior Probability P(High)            : {ex1['prior_P_A']:.8f}  ({ex1['count_target']:,} / {ex1['total_observations']:,})")
    lines.append(f"  Marginal Evidence P(Rain)            : {ex1['marginal_P_B']:.8f}  ({ex1['count_evidence']:,} / {ex1['total_observations']:,})")
    lines.append(f"  Likelihood P(Rain | High)            : {ex1['likelihood_P_B_given_A']:.8f}  ({ex1['count_joint']:,} / {ex1['count_target']:,})")
    lines.append("")
    lines.append(f"  Bayes Theorem Result                 : {ex1['bayes_posterior']:.8f}")
    lines.append(f"  Direct Empirical Result              : {ex1['direct_empirical_posterior']:.8f}")
    lines.append(f"  Absolute Difference                  : {ex1['absolute_difference']:.2e}")
    lines.append(f"  Mathematically Verified?             : {'YES (EXACT MATCH)' if ex1['is_verified'] else 'NO'}")
    lines.append("")
    
    # Example 2
    ex2 = results["example2_peak"]
    lines.append("EXAMPLE 2: BAYES' THEOREM FOR P(High Congestion | Peak Hour)")
    lines.append("-" * 78)
    lines.append("  Formula:")
    lines.append("                P(Peak | High) * P(High)")
    lines.append("  P(High|Peak) = ------------------------")
    lines.append("                        P(Peak)")
    lines.append("")
    lines.append(f"  Prior Probability P(High)            : {ex2['prior_P_A']:.8f}  ({ex2['count_target']:,} / {ex2['total_observations']:,})")
    lines.append(f"  Marginal Evidence P(Peak)            : {ex2['marginal_P_B']:.8f}  ({ex2['count_evidence']:,} / {ex2['total_observations']:,})")
    lines.append(f"  Likelihood P(Peak | High)            : {ex2['likelihood_P_B_given_A']:.8f}  ({ex2['count_joint']:,} / {ex2['count_target']:,})")
    lines.append("")
    lines.append(f"  Bayes Theorem Result                 : {ex2['bayes_posterior']:.8f}")
    lines.append(f"  Direct Empirical Result              : {ex2['direct_empirical_posterior']:.8f}")
    lines.append(f"  Absolute Difference                  : {ex2['absolute_difference']:.2e}")
    lines.append(f"  Mathematically Verified?             : {'YES (EXACT MATCH)' if ex2['is_verified'] else 'NO'}")
    lines.append("=" * 78)
    
    return "\n".join(lines)


if __name__ == "__main__":
    bayes_res = run_bayes_analysis()
    report = format_bayes_report(bayes_res)
    print(report)
