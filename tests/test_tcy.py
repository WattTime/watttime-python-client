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
            freq='H',
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

    def test_tcy_new_years_holiday(self):
        """Test that New Year's Day is treated as a weekend/holiday"""
        new_years = pd.Timestamp('2024-01-01', tz='America/Los_Angeles')
        self.assertFalse(self.calculator._is_weekday(new_years))
        
        # Test same date but not a holiday
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

if __name__ == "__main__":
    unittest.main()