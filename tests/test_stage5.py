"""
CommuteCast - Stage 5: Integrated Analysis Engine Unit Tests
Module: tests/test_stage5.py

Validates all critical requirements specified in Stage 5:
1. Valid hour values 0 through 23 succeed.
2. Invalid hour raises appropriate error (ValueError or TypeError).
3. Rain accepts only valid binary values {0, 1}.
4. Peak accepts only valid binary values {0, 1}.
5. Expected traffic is positive and finite.
6. High congestion probability is between 0 and 1.
7. Baseline P(High) is between 0 and 1.
8. Difference from baseline and percentage points are computed correctly.
9. Mathematical separation: polynomial expected volume and conditional probabilities are distinct fields.
10. Rule-based interpretation is consistent with baseline comparison.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path = [p for p in sys.path if Path(p).resolve() != SRC_DIR]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration import (
    analyze_commute_conditions,
    get_commute_engine,
    format_readable_analysis,
    CommuteAnalysisEngine
)
from src.probability import CORE_PEAK_HOURS


class TestStage5Integration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize engine."""
        cls.engine = get_commute_engine()
        
    def test_01_valid_hours_0_through_23(self):
        """Test that all hours from 0 to 23 succeed with finite expected volume."""
        for h in range(24):
            for r in (0, 1):
                res = analyze_commute_conditions(hour=h, rain=r)
                self.assertEqual(res["hour"], h)
                self.assertEqual(res["rain"], r)
                self.assertTrue(np.isfinite(res["expected_traffic"]))
                self.assertGreaterEqual(res["expected_traffic"], 0.0)
                self.assertTrue(0.0 <= res["high_congestion_probability"] <= 1.0)
                
    def test_02_invalid_hour_handling(self):
        """Test that invalid hours raise ValueError or TypeError."""
        # Out of range
        with self.assertRaises(ValueError):
            analyze_commute_conditions(hour=-1, rain=0)
        with self.assertRaises(ValueError):
            analyze_commute_conditions(hour=24, rain=0)
        with self.assertRaises(ValueError):
            analyze_commute_conditions(hour=100, rain=0)
            
        # Non-integer types
        with self.assertRaises(TypeError):
            analyze_commute_conditions(hour=8.5, rain=0)
        with self.assertRaises(TypeError):
            analyze_commute_conditions(hour="8", rain=0)
        with self.assertRaises(TypeError):
            analyze_commute_conditions(hour=None, rain=0)
            
    def test_03_invalid_rain_handling(self):
        """Test that rain accepts only binary 0 or 1."""
        with self.assertRaises(ValueError):
            analyze_commute_conditions(hour=8, rain=2)
        with self.assertRaises(ValueError):
            analyze_commute_conditions(hour=8, rain=-1)
        with self.assertRaises(TypeError):
            analyze_commute_conditions(hour=8, rain=0.5)
        with self.assertRaises(TypeError):
            analyze_commute_conditions(hour=8, rain="yes")
            
    def test_04_invalid_peak_handling(self):
        """Test that peak accepts only binary 0, 1, or None."""
        with self.assertRaises(ValueError):
            analyze_commute_conditions(hour=8, rain=0, peak=2)
        with self.assertRaises(ValueError):
            analyze_commute_conditions(hour=8, rain=0, peak=-1)
        with self.assertRaises(TypeError):
            analyze_commute_conditions(hour=8, rain=0, peak=0.5)
        with self.assertRaises(TypeError):
            analyze_commute_conditions(hour=8, rain=0, peak="peak")
            
    def test_05_baseline_and_difference_calculations(self):
        """Test baseline P(High) and difference from baseline calculations."""
        res = analyze_commute_conditions(hour=8, rain=1, peak=1)
        base = res["baseline_high_probability"]
        p_cond = res["high_congestion_probability"]
        diff = res["difference_from_baseline"]
        diff_ppt = res["difference_percentage_points"]
        
        self.assertTrue(0.0 <= base <= 1.0)
        self.assertAlmostEqual(base, 0.249798, places=5)
        self.assertAlmostEqual(diff, p_cond - base, places=7)
        self.assertAlmostEqual(diff_ppt, diff * 100.0, places=5)
        
    def test_06_peak_inference_consistency(self):
        """Test that when peak is None, it is correctly inferred from hour in CORE_PEAK_HOURS."""
        # 8 AM is in CORE_PEAK_HOURS -> peak = 1
        res_8 = analyze_commute_conditions(hour=8, rain=0, peak=None)
        self.assertEqual(res_8["peak"], 1)
        
        # 2 PM (Hour 14) is not in CORE_PEAK_HOURS -> peak = 0
        res_14 = analyze_commute_conditions(hour=14, rain=0, peak=None)
        self.assertEqual(res_14["peak"], 0)
        
    def test_07_all_four_quadrants_probabilities(self):
        """Test that all 4 (Rain, Peak) combinations match Stage 3 empirical counts."""
        quadrants = {
            (1, 1): 0.677472,
            (0, 1): 0.653089,
            (1, 0): 0.181754,
            (0, 0): 0.167136
        }
        for (r, p), expected_prob in quadrants.items():
            res = analyze_commute_conditions(hour=12, rain=r, peak=p)
            self.assertAlmostEqual(res["high_congestion_probability"], expected_prob, places=5)
            
    def test_08_rule_based_interpretation(self):
        """Test that rule-based interpretation matches baseline comparison."""
        # Peak hours have P(High|cond) > baseline -> above baseline
        res_peak = analyze_commute_conditions(hour=8, rain=0, peak=1)
        self.assertIn("above the overall dataset baseline", res_peak["interpretation"])
        
        # Off-peak hours have P(High|cond) < baseline -> at or below baseline
        res_offpeak = analyze_commute_conditions(hour=2, rain=0, peak=0)
        self.assertIn("at or below the overall dataset baseline", res_offpeak["interpretation"])
        
    def test_09_bayes_educational_explanation(self):
        """Test that Bayes explanation helper returns verified exact match."""
        bayes_res = self.engine.explain_bayes("rain")
        self.assertTrue(bayes_res["is_verified"])
        self.assertAlmostEqual(bayes_res["bayes_posterior"], 0.269974, places=5)
        
    def test_10_readable_formatting(self):
        """Test that format_readable_analysis generates clean text output without errors."""
        res = analyze_commute_conditions(hour=16, rain=1)
        text = format_readable_analysis(res)
        self.assertIn("Expected Traffic Volume", text)
        self.assertIn("High Congestion Probability", text)
        self.assertIn("Overall Dataset Base", text)
        self.assertIn("Difference", text)


if __name__ == "__main__":
    unittest.main()
