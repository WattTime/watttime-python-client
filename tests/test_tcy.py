import unittest
import unittest.mock as mock
from datetime import datetime
import pandas as pd
import pytz
from pathlib import Path
from watttime import TCYCalculator

REGION = "CAISO_NORTH"

class TestTCY(unittest.TestCase):
    def setUp(self):
        self.calculator = TCYCalculator(
            region=REGION,
            timezone="America/Los_Angeles"
        )

    def test_tcy_weekday_weekend_differentiation(self):
        """Test that TCY properly differentiates between weekdays and weekends"""
        dates = pd.date_range(
            start='2023-01-01',
            end='2023-01-31',
            freq='h',
            tz='America/Los_Angeles'
        )
        values = [100 if self.calculator._is_weekday(d) else 200 for d in dates]
        test_data = pd.DataFrame({'value': values}, index=dates)

        ref_table = self.calculator._create_reference_table(test_data)

        weekday_value = ref_table[
            (ref_table['month'] == 1) &
            (ref_table['hour'] == 12) &
            (ref_table['is_weekday'] == True)
        ]['value'].iloc[0]
        weekend_value = ref_table[
            (ref_table['month'] == 1) &
            (ref_table['hour'] == 12) &
            (ref_table['is_weekday'] == False)
        ]['value'].iloc[0]
        
        self.assertEqual(weekday_value, 100)
        self.assertEqual(weekend_value, 200)

    def test_tcy_holiday_handling(self):
        """Test that holidays are correctly identified for different years"""
        # Test current year holiday
        new_years_2024 = pd.Timestamp('2024-01-01', tz='America/Los_Angeles')
        self.assertFalse(self.calculator._is_weekday(new_years_2024))
        
        # Test past year holiday
        new_years_2018 = pd.Timestamp('2018-01-01', tz='America/Los_Angeles')
        self.assertFalse(self.calculator._is_weekday(new_years_2018))
        
        # Test future year holiday
        new_years_2025 = pd.Timestamp('2025-01-01', tz='America/Los_Angeles')
        self.assertFalse(self.calculator._is_weekday(new_years_2025))
        
        # Test regular weekday
        regular_day = pd.Timestamp('2024-01-02', tz='America/Los_Angeles')
        self.assertTrue(self.calculator._is_weekday(regular_day))

    def test_tcy_timezone_handling(self):
        """Test that timezone conversion works correctly with real data"""
        historical_data = self.calculator._get_historical_data()
        
        # Check historical data timezone
        self.assertEqual(historical_data.index.tz.zone, "America/Los_Angeles")
        self.assertTrue('+00:00' not in str(historical_data.index[0]))  # Not UTC
        self.assertTrue('-08:00' in str(historical_data.index[0]) or '-07:00' in str(historical_data.index[0]))  # Pacific time
        
        # Check TCY result timezone
        tcy = self.calculator.calculate_tcy(2024)
        self.assertEqual(tcy.index.tz.zone, "America/Los_Angeles")
        self.assertEqual(str(tcy.index[0])[:4], "2024")  # Starts at beginning of year
        self.assertTrue('-08:00' in str(tcy.index[0]) or '-07:00' in str(tcy.index[0]))  # Pacific time

    def test_tcy_different_years_same_pattern(self):
        """Test that different target years use same recent data patterns"""
        # Calculate TCY for different years
        tcy_2018 = self.calculator.calculate_tcy(2018)
        tcy_2024 = self.calculator.calculate_tcy(2024)

        # Same calendar pattern days (e.g., first Wednesday of January at 2pm)
        # should have similar values despite different years
        jan_2018_wed = tcy_2018['2018-01-03 14:00:00-08:00']  # A Wednesday
        jan_2024_wed = tcy_2024['2024-01-03 14:00:00-08:00']  # A Wednesday
        
        # Use assertAlmostEqual or numpy.testing.assert_allclose instead of assertEqual
        self.assertAlmostEqual(jan_2018_wed, jan_2024_wed, places=2)  # Compare up to 2 decimal places

        # But holidays should follow their respective years
        new_years_2018 = tcy_2018['2018-01-01 12:00:00-08:00']
        new_years_2024 = tcy_2024['2024-01-01 12:00:00-08:00']
        
        # Both should use holiday patterns (different from weekday patterns)
        self.assertNotEqual(new_years_2018, jan_2018_wed)
        self.assertNotEqual(new_years_2024, jan_2024_wed)

if __name__ == "__main__":
    unittest.main()