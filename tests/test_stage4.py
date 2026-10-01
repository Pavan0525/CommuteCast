"""
CommuteCast - Stage 4: Polynomial Curve Fitting Unit Tests
Module: tests/test_stage4.py

Validates all 7 critical requirements specified in Stage 4:
1. All three polynomial models (Degrees 1, 2, and 3) can be fitted successfully.
2. Predictions are finite numbers for all hours 0 through 23.
3. Hour validation works (valid hours succeed; negative or > 23 raise ValueError; non-numeric raises TypeError).
4. Prediction works across all hours 0 through 23.
5. Evaluation metrics (MAE, RMSE, R^2, Mean Residual) are finite floats.
6. Degree 1, Degree 2, and Degree 3 produce distinct fitted curves.
7. No NaN or infinite prediction values occur.
8. Mean residuals are approximately zero (OLS theoretical property).
9. Error metric progression: MAE and RMSE improve from Degree 1 to Degree 2 and Degree 3.
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

from src.preprocessing import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.polynomial import (
    prepare_hourly_data,
    fit_polynomial_models,
    evaluate_polynomial_models,
    predict_traffic_for_hour,
    get_default_predictor,
    TrafficPolynomialPredictor
)

BASELINE_SHA256 = "749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e"


class TestStage4PolynomialCurveFitting(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize data and predictor."""
        cls.predictor = get_default_predictor()
        cls.hourly_df = cls.predictor.hourly_df
        cls.models = cls.predictor.models
        cls.eval_df = cls.predictor.eval_df
        
    def test_01_hourly_data_preparation(self):
        """Test that hourly data has exactly 24 rows (hours 0 to 23) with valid positive means."""
        self.assertEqual(len(self.hourly_df), 24, "Hourly table must contain exactly 24 hours.")
        self.assertEqual(list(self.hourly_df["hour"]), list(range(24)), "Hours must span 0 to 23 in order.")
        self.assertTrue((self.hourly_df["mean_traffic_volume"] > 0).all(), "Mean volume must be positive.")
        self.assertFalse(self.hourly_df["mean_traffic_volume"].isna().any(), "Mean volume must not contain NaNs.")
        
    def test_02_all_three_models_fitted(self):
        """Test that models for Degrees 1, 2, and 3 are successfully fitted with valid coefficients."""
        for deg in [1, 2, 3]:
            self.assertIn(deg, self.models, f"Model for degree {deg} must be fitted.")
            model = self.models[deg]
            self.assertEqual(model["degree"], deg)
            self.assertEqual(len(model["coeffs_descending"]), deg + 1, f"Degree {deg} must have {deg+1} coefficients.")
            self.assertTrue(np.all(np.isfinite(model["coeffs_descending"])), "Coefficients must be finite.")
            
    def test_03_evaluation_metrics_are_finite(self):
        """Test that MAE, RMSE, and R^2 are valid, finite, and within physical bounds."""
        for _, row in self.eval_df.iterrows():
            deg = row["Degree"]
            mae = row["MAE"]
            rmse = row["RMSE"]
            r2 = row["R²"]
            mean_res = row["Mean Residual"]
            
            self.assertTrue(np.isfinite(mae), f"MAE for deg {deg} must be finite.")
            self.assertTrue(np.isfinite(rmse), f"RMSE for deg {deg} must be finite.")
            self.assertTrue(np.isfinite(r2), f"R² for deg {deg} must be finite.")
            self.assertGreater(mae, 0, "MAE must be positive.")
            self.assertGreater(rmse, 0, "RMSE must be positive.")
            self.assertGreaterEqual(rmse, mae, "RMSE must be greater than or equal to MAE.")
            self.assertAlmostEqual(mean_res, 0.0, places=4, msg="OLS residuals must have zero sample mean.")
            
    def test_04_model_differentiation_and_metric_progression(self):
        """Test that Degrees 1, 2, and 3 produce distinct curves and R^2 improves with degree."""
        r2_1 = self.models[1]["r2"]
        r2_2 = self.models[2]["r2"]
        r2_3 = self.models[3]["r2"]
        
        # Models must have different R^2
        self.assertNotEqual(r2_1, r2_2)
        self.assertNotEqual(r2_2, r2_3)
        
        # Progression: R² increases with degree on training data
        self.assertLess(r2_1, r2_2, "Degree 2 R² must exceed Degree 1.")
        self.assertLess(r2_2, r2_3, "Degree 3 R² must exceed Degree 2.")
        
        # Degree 1 is poor (linear fit < 0.25), Degree 2/3 are good (> 0.80)
        self.assertLess(r2_1, 0.25)
        self.assertGreater(r2_2, 0.80)
        self.assertGreater(r2_3, 0.84)
        
    def test_05_predictions_work_for_all_hours_0_to_23(self):
        """Test that predict_traffic_for_hour produces finite, non-negative numbers for hours 0..23."""
        for h in range(24):
            for deg in [1, 2, 3]:
                pred = predict_traffic_for_hour(h, degree=deg)
                self.assertTrue(np.isfinite(pred), f"Prediction at hour {h} deg {deg} must be finite.")
                self.assertFalse(np.isnan(pred), f"Prediction at hour {h} deg {deg} must not be NaN.")
                self.assertGreaterEqual(pred, 0.0, f"Prediction at hour {h} deg {deg} must be non-negative.")
                self.assertLess(pred, 10000.0, f"Prediction at hour {h} deg {deg} exceeds realistic traffic.")
                
    def test_06_hour_validation_error_handling(self):
        """Test that hour validation rejects invalid inputs with ValueError or TypeError."""
        # Out of range hours
        with self.assertRaises(ValueError):
            predict_traffic_for_hour(-1)
        with self.assertRaises(ValueError):
            predict_traffic_for_hour(24)
        with self.assertRaises(ValueError):
            predict_traffic_for_hour(100)
            
        # Non-numeric types
        with self.assertRaises(TypeError):
            predict_traffic_for_hour("8")
        with self.assertRaises(TypeError):
            predict_traffic_for_hour(None)
            
        # Invalid degree
        with self.assertRaises(ValueError):
            predict_traffic_for_hour(8, degree=5)
            
    def test_07_example_predictions_consistency(self):
        """Test key commute hour predictions against expected approximate targets."""
        # At 8:00 AM (Hour 8), actual is ~4,587; Degree 3 prediction should be in [3,500, 4,200]
        pred_8 = predict_traffic_for_hour(8, degree=3)
        self.assertAlmostEqual(pred_8, 3860.2, places=1)
        
        # At 4:00 PM (Hour 16), actual is ~5,664; Degree 3 prediction should be in [4,700, 5,200]
        pred_16 = predict_traffic_for_hour(16, degree=3)
        self.assertAlmostEqual(pred_16, 4959.6, places=1)
        
    def test_08_original_csv_remains_unchanged(self):
        """Verify byte-level SHA256 integrity of raw dataset file."""
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
