"""
CommuteCast - Stage 2: Unit-1 Quality & Statistical Validation Tests
Module: tests/test_stage2.py

Validates all 7 critical requirements specified in Stage 2:
1. No unexpected parsing errors occur.
2. traffic_volume has valid values (non-negative, integer).
3. congestion contains only 'Low', 'Medium', 'High'.
4. Quantile ordering: Q25 < Q50 < Q75.
5. The congestion counts sum to the number of cleaned observations (48,187).
6. No NaN values accidentally enter the traffic target (traffic_volume).
7. The original CSV remains unchanged (verified via SHA256 checksum).
8. Exactly 17 duplicate rows removed.
9. 10 records with temp == 0 K correctly converted to NaN for temperature analysis without dropping traffic records.
10. Extreme rainfall observation (9831.3 mm) preserved without arbitrary deletion or invented values.
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

from src.preprocessing import preprocess_traffic_data, RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.statistics import compute_traffic_volume_statistics, compute_congestion_distribution

# Known baseline SHA256 of data/Metro_Interstate_Traffic_Volume.csv
BASELINE_SHA256 = "749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e"


class TestStage2PreprocessingAndStatistics(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Run preprocessing pipeline and cache preprocessed DataFrame."""
        cls.df = preprocess_traffic_data(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH, save_output=True)
        cls.stats = compute_traffic_volume_statistics(cls.df)
        cls.congestion_df = compute_congestion_distribution(cls.df)
        
    def test_01_original_csv_unchanged(self):
        """CRITICAL: Test that the original CSV file was preserved completely unchanged."""
        sha256_hash = hashlib.sha256()
        with open(RAW_DATA_PATH, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        current_hash = sha256_hash.hexdigest().lower()
        self.assertEqual(
            current_hash,
            BASELINE_SHA256,
            f"Original CSV file has been modified! Expected {BASELINE_SHA256}, got {current_hash}"
        )
        
    def test_02_deduplication_exact_count(self):
        """Verify that exactly 17 duplicate rows were removed (48,204 -> 48,187)."""
        raw_df = pd.read_csv(RAW_DATA_PATH, keep_default_na=False)
        self.assertEqual(len(raw_df), 48204, "Raw CSV row count should be exactly 48,204.")
        self.assertEqual(raw_df.duplicated().sum(), 17, "Raw CSV must contain exactly 17 duplicate rows.")
        self.assertEqual(len(self.df), 48187, "Cleaned dataset must contain exactly 48,187 unique rows.")
        
    def test_03_traffic_volume_no_nans_and_valid_range(self):
        """Test that no NaN values enter the traffic target and all values are non-negative."""
        nan_count = self.df["traffic_volume"].isna().sum()
        self.assertEqual(nan_count, 0, f"Found {nan_count} NaN values in traffic_volume target.")
        self.assertTrue((self.df["traffic_volume"] >= 0).all(), "Found negative traffic volume values.")
        self.assertEqual(self.df["traffic_volume"].min(), 0, "Minimum traffic volume should be 0.")
        self.assertEqual(self.df["traffic_volume"].max(), 7280, "Maximum traffic volume should be 7,280.")
        
    def test_04_congestion_classes_validity(self):
        """Test that congestion contains only 'Low', 'Medium', 'High' with no nulls."""
        unique_categories = set(self.df["congestion"].unique())
        expected_categories = {"Low", "Medium", "High"}
        self.assertEqual(
            unique_categories,
            expected_categories,
            f"Congestion categories must be strictly {expected_categories}, found: {unique_categories}"
        )
        self.assertFalse(self.df["congestion"].isna().any(), "Congestion column must not contain NaNs.")
        
    def test_05_quantiles_ordering(self):
        """Test that Q25 < Q50 < Q75 strictly holds."""
        q25 = self.stats["q25"]
        q50 = self.stats["median"]
        q75 = self.stats["q75"]
        
        self.assertLess(q25, q50, f"Q25 ({q25}) must be strictly less than Q50 ({q50})")
        self.assertLess(q50, q75, f"Q50 ({q50}) must be strictly less than Q75 ({q75})")
        self.assertAlmostEqual(q25, 1192.5, places=1)
        self.assertAlmostEqual(q50, 3379.0, places=1)
        self.assertAlmostEqual(q75, 4933.0, places=1)
        
    def test_06_congestion_counts_sum_to_total(self):
        """Test that Low + Medium + High counts sum exactly to 48,187 (100% of observations)."""
        counts = self.df["congestion"].value_counts()
        total_classified = counts.sum()
        self.assertEqual(
            total_classified,
            len(self.df),
            f"Congestion counts ({total_classified}) must equal cleaned dataset size ({len(self.df)})"
        )
        self.assertEqual(counts["Low"], 12047, "Low category should contain 12,047 records (~25.00%).")
        self.assertEqual(counts["Medium"], 24103, "Medium category should contain 24,103 records (~50.02%).")
        self.assertEqual(counts["High"], 12037, "High category should contain 12,037 records (~24.98%).")
        
    def test_07_temperature_handling(self):
        """Test that temp == 0 K records (10 rows) are converted to NaN for temp_celsius without dropping traffic records."""
        # Cleaned dataset must preserve all 48,187 records
        self.assertEqual(len(self.df), 48187)
        # Exactly 10 rows should have NaN in temp_celsius
        nan_temps = self.df["temp_celsius"].isna().sum()
        self.assertEqual(nan_temps, 10, f"Expected exactly 10 NaN temp values, got {nan_temps}")
        # Traffic volume in those 10 rows must be preserved and non-null
        invalid_temp_rows = self.df[self.df["temp_celsius"].isna()]
        self.assertFalse(invalid_temp_rows["traffic_volume"].isna().any())
        self.assertEqual(len(invalid_temp_rows), 10)
        
    def test_08_rainfall_extreme_handling(self):
        """Test that extreme rainfall record (9831.3 mm) is preserved without arbitrary deletion or invented values."""
        extreme_rows = self.df[self.df["rain_1h"] > 500]
        self.assertEqual(len(extreme_rows), 1, "Expected exactly 1 extreme rainfall record.")
        outlier = extreme_rows.iloc[0]
        self.assertEqual(outlier["rain_1h"], 9831.3)
        self.assertEqual(outlier["rain"], 1, "Binary rain indicator must be 1 for rain_1h = 9831.3 mm.")
        self.assertEqual(outlier["traffic_volume"], 5535, "Traffic volume must remain untouched (5,535).")
        # rain_1h_valid should be NaN for this single sensor glitch
        self.assertTrue(np.isnan(outlier["rain_1h_valid"]))
        
    def test_09_holiday_literal_none_handling(self):
        """Test that literal string 'None' is treated as non-holiday (is_holiday = 0)."""
        self.assertEqual(self.df["is_holiday"].sum(), 61, "Expected exactly 61 holiday records.")
        none_holiday_rows = self.df[self.df["holiday"] == "None"]
        self.assertTrue((none_holiday_rows["is_holiday"] == 0).all(), "All 'None' holiday rows must have is_holiday = 0.")
        named_holiday_rows = self.df[self.df["holiday"] != "None"]
        self.assertTrue((named_holiday_rows["is_holiday"] == 1).all(), "All named holiday rows must have is_holiday = 1.")
        
    def test_10_temporal_features_ranges(self):
        """Test that extracted temporal features are within valid ranges."""
        self.assertTrue(self.df["hour"].between(0, 23).all(), "Hours must be in [0, 23]")
        self.assertTrue(self.df["day_of_week"].between(0, 6).all(), "Day of week must be in [0, 6]")
        self.assertTrue(self.df["month"].between(1, 12).all(), "Month must be in [1, 12]")
        self.assertTrue(self.df["year"].between(2012, 2018).all(), "Year must be in [2012, 2018]")
        self.assertTrue(set(self.df["is_weekend"].unique()).issubset({0, 1}), "is_weekend must be binary {0, 1}")


if __name__ == "__main__":
    unittest.main()
