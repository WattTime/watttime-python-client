import unittest
from datetime import datetime, timedelta
from pytz import timezone, UTC
from watttime_sdk import WattTimeBase, WattTimeHistorical

import pandas as pd
import requests

class TestWattTimeBase(unittest.TestCase):
    def setUp(self):
        self.base = WattTimeBase()

    def test_login_with_real_api(self):
        self.base.login()
        assert self.base.token is not None
        assert self.base.token_valid_until > datetime.now()

    def test_parse_dates_with_string(self):
        start = "2022-01-01"
        end = "2022-01-31"

        parsed_start, parsed_end = self.base.parse_dates(start, end)

        self.assertIsInstance(parsed_start, datetime)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_start, datetime(2022, 1, 1))
        self.assertEqual(parsed_end, datetime(2022, 1, 31))

    def test_parse_dates_with_datetime(self):
        # Case 1: User provides a string with no timezone
        start_str = "2022-01-01"
        end_str = "2022-01-31"
        parsed_start, parsed_end = self.base.parse_dates(start_str, end_str)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 2: User provides a string with non-UTC timezone
        start_str = "2022-01-01 12:00:00+02:00"
        end_str = "2022-01-31 12:00:00+02:00"
        parsed_start, parsed_end = self.base.parse_dates(start_str, end_str)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 3: User provides a string with UTC timezone
        start_str = "2022-01-01 12:00:00Z"
        end_str = "2022-01-31 12:00:00Z"
        parsed_start, parsed_end = self.base.parse_dates(start_str, end_str)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 4: User provides a datetime with no timezone
        start_dt = datetime(2022, 1, 1)
        end_dt = datetime(2022, 1, 31)
        parsed_start, parsed_end = self.base.parse_dates(start_dt, end_dt)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 5: User provides a datetime with non-UTC timezone
        start_dt = datetime(2022, 1, 1, tzinfo=timezone('US/Eastern'))
        end_dt = datetime(2022, 1, 31, tzinfo=timezone('US/Eastern'))
        parsed_start, parsed_end = self.base.parse_dates(start_dt, end_dt)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 6: User provides a datetime with UTC timezone
        start_dt = datetime(2022, 1, 1, tzinfo=UTC)
        end_dt = datetime(2022, 1, 31, tzinfo=UTC)
        parsed_start, parsed_end = self.base.parse_dates(start_dt, end_dt)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)


class TestWattTimeHistorical(unittest.TestCase):
    def setUp(self):
        self.historical = WattTimeHistorical()

    def test_get_historical_jsons_3_months(self):
        start = "2021-01-01 00:00Z"
        end = "2021-12-31 00:00Z"
        region = "CAISO_NORTH"

        jsons = self.historical.get_historical_jsons(start, end, region)

        self.assertIsInstance(jsons, list)
        self.assertGreaterEqual(len(jsons), 1)
        self.assertIsInstance(jsons[0], dict)
        
    def test_get_historical_jsons_1_week(self):
        start = "2021-01-01 00:00Z"
        end = "2021-01-07 00:00Z"
        region = "CAISO_NORTH"

        jsons = self.historical.get_historical_jsons(start, end, region)

        self.assertIsInstance(jsons, list)
        self.assertGreaterEqual(len(jsons), 1)
        self.assertIsInstance(jsons[0], dict)

    def test_get_historical_pandas(self):
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        region = "CAISO_NORTH"

        df = self.historical.get_historical_pandas(start, end, region)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("point_time", df.columns)

if __name__ == '__main__':
    unittest.main()