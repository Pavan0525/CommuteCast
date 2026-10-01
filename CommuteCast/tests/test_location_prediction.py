"""
CommuteCast - Unit Tests for Location-Based Prediction Calculation
Module: tests/test_location_prediction.py

Validates the empirical dataset calculation fix:
1. Different cities produce the correct dataset-derived means.
2. Different areas within the same city produce the correct dataset-derived means.
3. Different vehicle types produce the correct subset means.
4. No hardcoded 9572 returned across locations.
5. Empty subset fallback hierarchy works transparently (City+Area -> City -> Global).
6. Congestion probability is calculated strictly from empirical dataset counts.
7. Zero Poisson, zero Gaussian, zero synthetic multipliers.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path = [p for p in sys.path if Path(p).resolve() != (PROJECT_ROOT / "src")]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.new_integration import get_new_commute_engine
from src.new_probability import (
    calculate_empirical_expected_volume,
    calculate_empirical_congestion_probability,
    get_high_congestion_mask
)


class TestLocationPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = get_new_commute_engine()
        cls.df = cls.engine.df

    def test_01_different_cities_produce_correct_dataset_means(self):
        """Test that different cities produce their exact dataset-derived means."""
        cities = ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane"]
        city_means = {}

        for c in cities:
            expected_mean = float(self.df[self.df["City"] == c]["Vehicle_Count"].mean())
            res = calculate_empirical_expected_volume(self.df, city=c)
            city_means[c] = res["expected_vehicle_count"]
            self.assertAlmostEqual(res["expected_vehicle_count"], expected_mean, places=2)
            self.assertEqual(res["subset_rows"], int((self.df["City"] == c).sum()))

        # Verify cities have distinct empirical means
        self.assertNotEqual(city_means["Mumbai"], city_means["Pune"])
        self.assertNotEqual(city_means["Pune"], city_means["Nagpur"])
        self.assertNotEqual(city_means["Mumbai"], city_means["Thane"])

    def test_02_different_areas_produce_correct_dataset_means(self):
        """Test that different areas within the same city produce distinct, exact subset means."""
        # Mumbai areas: Andheri vs Lower Parel
        andheri_sub = self.df[(self.df["City"] == "Mumbai") & (self.df["Area"] == "Andheri")]
        lp_sub = self.df[(self.df["City"] == "Mumbai") & (self.df["Area"] == "Lower Parel")]

        andheri_res = calculate_empirical_expected_volume(self.df, city="Mumbai", area="Andheri")
        lp_res = calculate_empirical_expected_volume(self.df, city="Mumbai", area="Lower Parel")

        self.assertAlmostEqual(andheri_res["expected_vehicle_count"], float(andheri_sub["Vehicle_Count"].mean()), places=2)
        self.assertAlmostEqual(lp_res["expected_vehicle_count"], float(lp_sub["Vehicle_Count"].mean()), places=2)
        self.assertNotEqual(andheri_res["expected_vehicle_count"], lp_res["expected_vehicle_count"])

        # Pune areas: Kothrud vs Wakad
        kothrud_sub = self.df[(self.df["City"] == "Pune") & (self.df["Area"] == "Kothrud")]
        wakad_sub = self.df[(self.df["City"] == "Pune") & (self.df["Area"] == "Wakad")]

        kothrud_res = calculate_empirical_expected_volume(self.df, city="Pune", area="Kothrud")
        wakad_res = calculate_empirical_expected_volume(self.df, city="Pune", area="Wakad")

        self.assertAlmostEqual(kothrud_res["expected_vehicle_count"], float(kothrud_sub["Vehicle_Count"].mean()), places=2)
        self.assertAlmostEqual(wakad_res["expected_vehicle_count"], float(wakad_sub["Vehicle_Count"].mean()), places=2)
        self.assertNotEqual(kothrud_res["expected_vehicle_count"], wakad_res["expected_vehicle_count"])

    def test_03_different_vehicle_types_produce_correct_subset_means(self):
        """Test that different vehicle types produce exact matching subset means."""
        city = "Mumbai"
        area = "Lower Parel"
        vt_results = {}

        for vt in ["Car", "Auto", "Bus", "Bike", "Truck"]:
            sub = self.df[(self.df["City"] == city) & (self.df["Area"] == area) & (self.df["Vehicle_Type"] == vt)]
            expected_mean = float(sub["Vehicle_Count"].mean())
            res = self.engine.analyze(city=city, area=area, vehicle_type=vt)
            vt_results[vt] = res["expected_vehicle_count"]

            self.assertAlmostEqual(res["expected_vehicle_count"], expected_mean, places=2)
            self.assertEqual(res["subset_rows"], len(sub))

        # Check variation across vehicle types in the same location
        self.assertNotEqual(vt_results["Car"], vt_results["Auto"])
        self.assertNotEqual(vt_results["Car"], vt_results["Bike"])

    def test_04_no_hardcoded_9572(self):
        """Test that diverse locations do NOT return the hardcoded 9572 value."""
        test_combos = [
            ("Mumbai", "Lower Parel", "Car"),
            ("Mumbai", "Andheri", "Auto"),
            ("Pune", "Wakad", "Truck"),
            ("Pune", "Kothrud", "Bike"),
            ("Nagpur", "Sitabuldi", "Bus"),
            ("Nashik", "Gangapur Road", "Car"),
            ("Thane", "Wagle Estate", "Auto")
        ]

        for city, area, vt in test_combos:
            res = self.engine.analyze(city=city, area=area, vehicle_type=vt)
            count = res["expected_vehicle_count"]
            self.assertNotEqual(round(count), 9572,
                                f"Location {city} - {area} ({vt}) returned 9572!")

    def test_05_empty_subset_fallback_works_correctly(self):
        """Test that non-existent combinations fall back transparently through the hierarchy."""
        # 1. Non-existent vehicle type falls back to City + Area
        res_vt = calculate_empirical_expected_volume(self.df, city="Mumbai", area="Lower Parel", vehicle_type="Hovercraft")
        expected_area_mean = float(self.df[(self.df["City"] == "Mumbai") & (self.df["Area"] == "Lower Parel")]["Vehicle_Count"].mean())
        self.assertTrue(res_vt["fallback"])
        self.assertEqual(res_vt["fallback_level"], "City + Area")
        self.assertAlmostEqual(res_vt["expected_vehicle_count"], expected_area_mean, places=2)

        # 2. Non-existent area falls back to City
        res_area = calculate_empirical_expected_volume(self.df, city="Mumbai", area="Atlantis", vehicle_type="Car")
        expected_city_mean = float(self.df[self.df["City"] == "Mumbai"]["Vehicle_Count"].mean())
        self.assertTrue(res_area["fallback"])
        self.assertEqual(res_area["fallback_level"], "City")
        self.assertAlmostEqual(res_area["expected_vehicle_count"], expected_city_mean, places=2)

        # 3. Non-existent city falls back to Global Dataset Mean
        res_city = calculate_empirical_expected_volume(self.df, city="Gotham", area="Downtown", vehicle_type="Batmobile")
        expected_global_mean = float(self.df["Vehicle_Count"].mean())
        self.assertTrue(res_city["fallback"])
        self.assertEqual(res_city["fallback_level"], "Global Dataset Mean")
        self.assertAlmostEqual(res_city["expected_vehicle_count"], expected_global_mean, places=2)

    def test_06_probability_is_calculated_from_empirical_counts(self):
        """Test that congestion probability is calculated from empirical counts, not Poisson/Gaussian."""
        city = "Mumbai"
        area = "Lower Parel"
        vt = "Car"

        mask = (self.df["City"] == city) & (self.df["Area"] == area) & (self.df["Vehicle_Type"] == vt)
        sub = self.df[mask]
        high_mask = get_high_congestion_mask(self.df)
        expected_prob = float((high_mask & mask).sum() / len(sub))

        res = self.engine.analyze(city=city, area=area, vehicle_type=vt)
        self.assertAlmostEqual(res["high_congestion_probability"], expected_prob, places=4)
        self.assertEqual(res["high_count"], int((high_mask & mask).sum()))

        # Global baseline
        baseline = float(high_mask.mean())
        self.assertAlmostEqual(res["baseline_high_probability"], baseline, places=4)


    def test_07_minimum_five_required_combinations(self):
        """
        Verify the exact 5 minimum required combinations match the empirical dataset mean:
        1. Mumbai + Lower Parel + Car
        2. Mumbai + another available Area (Andheri) + Car
        3. Pune + Kothrud + Bike
        4. Pune + another available Area (Wakad) + Bike
        5. Nagpur + Sitabuldi + Bus
        """
        required_combos = [
            ("Mumbai", "Lower Parel", "Car"),
            ("Mumbai", "Andheri", "Car"),
            ("Pune", "Kothrud", "Bike"),
            ("Pune", "Wakad", "Bike"),
            ("Nagpur", "Sitabuldi", "Bus")
        ]

        for city, area, vt in required_combos:
            expected = float(
                self.df[
                    (self.df["City"] == city) &
                    (self.df["Area"] == area) &
                    (self.df["Vehicle_Type"] == vt)
                ]["Vehicle_Count"].mean()
            )
            res = self.engine.analyze(city=city, area=area, vehicle_type=vt)
            actual = res["expected_vehicle_count"]

            self.assertAlmostEqual(
                actual, expected, places=2,
                msg=f"Mismatch for {city} + {area} + {vt}: got {actual}, expected {expected}"
            )
            self.assertFalse(res["fallback"], f"Unexpected fallback for existing combo {city} + {area} + {vt}")
            self.assertIn("Exact", res["fallback_level"])
            self.assertEqual(res["debug_info"]["matching_rows"], res["subset_rows"])

    def test_08_two_different_areas_in_same_city(self):
        """Explicitly test that two different areas in the SAME city produce distinct expected counts."""
        # Mumbai: Lower Parel vs Bandra vs Andheri
        res_lp = self.engine.analyze(city="Mumbai", area="Lower Parel", vehicle_type="Car")
        res_bandra = self.engine.analyze(city="Mumbai", area="Bandra", vehicle_type="Car")
        res_andheri = self.engine.analyze(city="Mumbai", area="Andheri", vehicle_type="Car")

        self.assertNotEqual(res_lp["expected_vehicle_count"], res_bandra["expected_vehicle_count"])
        self.assertNotEqual(res_lp["expected_vehicle_count"], res_andheri["expected_vehicle_count"])
        self.assertNotEqual(res_bandra["expected_vehicle_count"], res_andheri["expected_vehicle_count"])

        # Pune: Kothrud vs Wakad
        res_kothrud = self.engine.analyze(city="Pune", area="Kothrud", vehicle_type="Bike")
        res_wakad = self.engine.analyze(city="Pune", area="Wakad", vehicle_type="Bike")
        self.assertNotEqual(res_kothrud["expected_vehicle_count"], res_wakad["expected_vehicle_count"])

    def test_09_two_different_vehicle_types_in_same_area(self):
        """Explicitly test that two different vehicle types in the SAME area produce distinct expected counts."""
        res_car = self.engine.analyze(city="Mumbai", area="Lower Parel", vehicle_type="Car")
        res_bike = self.engine.analyze(city="Mumbai", area="Lower Parel", vehicle_type="Bike")
        self.assertNotEqual(res_car["expected_vehicle_count"], res_bike["expected_vehicle_count"])

    def test_10_two_different_cities(self):
        """Explicitly test that two different cities produce distinct expected counts."""
        res_mumbai = self.engine.analyze(city="Mumbai", area="Lower Parel", vehicle_type="Car")
        res_pune = self.engine.analyze(city="Pune", area="Kothrud", vehicle_type="Car")
        self.assertNotEqual(res_mumbai["expected_vehicle_count"], res_pune["expected_vehicle_count"])

    def test_11_anti_hardcoding_source_scan(self):
        """
        Anti-hardcoding test: Scans the prediction and integration source files
        and verifies there are NO hardcoded expected vehicle counts (e.g. 9572, 9572.03)
        being assigned as constant return values.
        """
        src_files = [
            PROJECT_ROOT / "src" / "new_integration.py",
            PROJECT_ROOT / "src" / "new_probability.py",
            PROJECT_ROOT / "src" / "integration.py"
        ]

        forbidden_tokens = ["9572", "9572.03", "9572.0", "9572.00"]

        for file_path in src_files:
            if not file_path.exists():
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            for token in forbidden_tokens:
                self.assertNotIn(
                    token, content,
                    f"Found forbidden hardcoded token '{token}' in {file_path.name}!"
                )


if __name__ == "__main__":
    unittest.main()

