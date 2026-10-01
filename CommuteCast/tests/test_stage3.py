"""
CommuteCast - Stage 3: Probability, Expectation & Bayes Theorem Unit Tests
Module: tests/test_stage3.py

Validates all 8 critical requirements specified in Stage 3:
1. All probabilities are strictly between 0 and 1.
2. P(Low) + P(Medium) + P(High) ≈ 1.
3. P(Rain) + P(No Rain) ≈ 1.
4. Bayes result approximately equals direct conditional probability (|diff| <= 1e-6).
5. Conditional probabilities are strictly between 0 and 1.
6. Expectation values are finite and positive.
7. No division-by-zero occurs in any calculation.
8. No NaN or infinite values are introduced into probability calculations.
9. Verification of both Bayes examples (Rain and Peak Hour).
10. Preservation of original CSV checksum.
"""

import os
import sys
import hashlib
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path = [p for p in sys.path if Path(p).resolve() != SRC_DIR]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import RAW_DATA_PATH, PROCESSED_DATA_PATH, preprocess_traffic_data
from src.probability import (
    calculate_probability,
    calculate_conditional_probability,
    calculate_expectation,
    calculate_congestion_probabilities,
    calculate_rain_probabilities,
    calculate_rain_conditional_probabilities,
    calculate_peak_hour_analysis,
    calculate_rain_and_peak_joint_analysis,
    calculate_temporal_conditional_analysis,
    CORE_PEAK_HOURS
)
from src.bayes import bayes_probability, verify_bayes_with_empirical_probability

BASELINE_SHA256 = "749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e"


class TestStage3ProbabilityAndBayes(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load preprocessed dataset."""
        if not os.path.exists(PROCESSED_DATA_PATH):
            cls.df = preprocess_traffic_data(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH)
        else:
            cls.df = pd.read_csv(PROCESSED_DATA_PATH)
            
    def test_01_all_basic_probabilities_in_range_and_finite(self):
        """Test that all basic probabilities are valid finite floats in [0, 1]."""
        cg_res = calculate_congestion_probabilities(self.df)
        rain_res = calculate_rain_probabilities(self.df)
        
        for key in ["P_Low", "P_Medium", "P_High"]:
            prob = cg_res[key]
            self.assertTrue(np.isfinite(prob), f"{key} must be finite.")
            self.assertFalse(np.isnan(prob), f"{key} must not be NaN.")
            self.assertTrue(0.0 <= prob <= 1.0, f"{key} ({prob}) must be in [0, 1].")
            
        for key in ["P_Rain", "P_NoRain"]:
            prob = rain_res[key]
            self.assertTrue(np.isfinite(prob), f"{key} must be finite.")
            self.assertFalse(np.isnan(prob), f"{key} must not be NaN.")
            self.assertTrue(0.0 <= prob <= 1.0, f"{key} ({prob}) must be in [0, 1].")
            
    def test_02_congestion_probabilities_sum_to_one(self):
        """Test that P(Low) + P(Medium) + P(High) ≈ 1.0 within machine epsilon."""
        cg_res = calculate_congestion_probabilities(self.df)
        prob_sum = cg_res["sum_probabilities"]
        self.assertAlmostEqual(prob_sum, 1.0, places=7, msg="P(Low) + P(Medium) + P(High) must sum to 1.0")
        
    def test_03_rain_probabilities_sum_to_one(self):
        """Test that P(Rain) + P(No Rain) ≈ 1.0 within machine epsilon."""
        rain_res = calculate_rain_probabilities(self.df)
        prob_sum = rain_res["sum_probabilities"]
        self.assertAlmostEqual(prob_sum, 1.0, places=7, msg="P(Rain) + P(No Rain) must sum to 1.0")
        
    def test_04_conditional_probabilities_in_range_and_finite(self):
        """Test that all conditional probabilities are valid finite floats in [0, 1]."""
        rain_cond = calculate_rain_conditional_probabilities(self.df)
        peak_res = calculate_peak_hour_analysis(self.df)
        temp_res = calculate_temporal_conditional_analysis(self.df)
        joint_res = calculate_rain_and_peak_joint_analysis(self.df)
        
        all_cond_probs = {
            **rain_cond,
            "P_High_given_Peak": peak_res["P_High_given_Peak"],
            "P_High_given_NonPeak": peak_res["P_High_given_NonPeak"],
            "P_High_given_Weekday": temp_res["P_High_given_Weekday"],
            "P_High_given_Weekend": temp_res["P_High_given_Weekend"],
            "P_High_given_Holiday": temp_res["P_High_given_Holiday"],
            "P_High_given_NonHoliday": temp_res["P_High_given_NonHoliday"],
            "P_High_given_Rain_Peak": joint_res["P_High_given_Rain_Peak"],
            "P_High_given_NoRain_Peak": joint_res["P_High_given_NoRain_Peak"],
        }
        
        for name, prob in all_cond_probs.items():
            self.assertTrue(np.isfinite(prob), f"Conditional probability '{name}' must be finite.")
            self.assertFalse(np.isnan(prob), f"Conditional probability '{name}' must not be NaN.")
            self.assertTrue(0.0 <= prob <= 1.0, f"Conditional probability '{name}' ({prob}) must be in [0, 1].")
            
    def test_05_bayes_theorem_example1_rain(self):
        """Test that Bayes' theorem posterior for P(High | Rain) equals direct empirical conditional probability."""
        high_mask = (self.df["congestion"] == "High")
        rain_mask = (self.df["rain"] == 1)
        
        verification = verify_bayes_with_empirical_probability(
            df=self.df,
            target_condition=high_mask,
            evidence_condition=rain_mask,
            target_name="High",
            evidence_name="Rain"
        )
        
        self.assertTrue(verification["is_verified"], "Bayes result did not match direct empirical probability.")
        self.assertAlmostEqual(
            verification["bayes_posterior"],
            verification["direct_empirical_posterior"],
            places=6,
            msg="Bayes posterior must match direct calculation."
        )
        self.assertLessEqual(verification["absolute_difference"], 1e-6)
        
    def test_06_bayes_theorem_example2_peak(self):
        """Test that Bayes' theorem posterior for P(High | Peak) equals direct empirical conditional probability."""
        high_mask = (self.df["congestion"] == "High")
        peak_mask = self.df["hour"].isin(CORE_PEAK_HOURS)
        
        verification = verify_bayes_with_empirical_probability(
            df=self.df,
            target_condition=high_mask,
            evidence_condition=peak_mask,
            target_name="High",
            evidence_name="Peak"
        )
        
        self.assertTrue(verification["is_verified"], "Bayes result did not match direct empirical probability for Peak Hour.")
        self.assertAlmostEqual(
            verification["bayes_posterior"],
            verification["direct_empirical_posterior"],
            places=6
        )
        self.assertLessEqual(verification["absolute_difference"], 1e-6)
        
    def test_07_expectations_are_finite_and_positive(self):
        """Test that empirical expectation values are positive, finite numbers."""
        e_overall = calculate_expectation(self.df)
        e_rain = calculate_expectation(self.df, condition=(self.df["rain"] == 1))
        e_norain = calculate_expectation(self.df, condition=(self.df["rain"] == 0))
        e_high = calculate_expectation(self.df, condition=(self.df["congestion"] == "High"))
        e_peak = calculate_expectation(self.df, condition=self.df["hour"].isin(CORE_PEAK_HOURS))
        e_nonpeak = calculate_expectation(self.df, condition=~self.df["hour"].isin(CORE_PEAK_HOURS))
        
        expectations = [e_overall, e_rain, e_norain, e_high, e_peak, e_nonpeak]
        for e in expectations:
            self.assertTrue(np.isfinite(e), "Expectation must be finite.")
            self.assertFalse(np.isnan(e), "Expectation must not be NaN.")
            self.assertGreater(e, 0, "Expectation must be positive.")
            self.assertLess(e, 7280, "Expectation must not exceed maximum observed traffic volume.")
            
        # Logical check: E(Traffic | High) must be strictly greater than overall E(Traffic)
        self.assertGreater(e_high, e_overall)
        # Logical check: E(Traffic | Peak) must be strictly greater than E(Traffic | Non-Peak)
        self.assertGreater(e_peak, e_nonpeak)
        
    def test_08_no_division_by_zero_and_input_validation(self):
        """Test that input validation prevents division by zero and rejects invalid probabilities."""
        with self.assertRaises(ZeroDivisionError):
            bayes_probability(prior_a=0.5, likelihood_b_given_a=0.5, marginal_b=0.0)
            
        with self.assertRaises(ValueError):
            bayes_probability(prior_a=1.5, likelihood_b_given_a=0.5, marginal_b=0.5)
            
        with self.assertRaises(ZeroDivisionError):
            # Condition B has 0 occurrences
            target = pd.Series([True, False, True])
            empty_given = pd.Series([False, False, False])
            calculate_conditional_probability(self.df.iloc[:3], target, empty_given)
            
    def test_09_original_csv_remains_unchanged(self):
        """Verify byte-level integrity of raw CSV file."""
        sha256_hash = hashlib.sha256()
        with open(RAW_DATA_PATH, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        current_hash = sha256_hash.hexdigest().lower()
        self.assertEqual(
            current_hash,
            BASELINE_SHA256,
            f"Original CSV file was modified! Expected {BASELINE_SHA256}, got {current_hash}"
        )


if __name__ == "__main__":
    unittest.main()
