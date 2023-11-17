import unittest
from datetime import datetime, timedelta
from dateutil.parser import parse
from pytz import timezone, UTC
from watttime_sdk import WattTimeBase, WattTimeHistorical, WattTimeMyAccess, WattTimeForecast

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
        start_dt = datetime(2022, 1, 1, tzinfo=timezone("US/Eastern"))
        end_dt = datetime(2022, 1, 31, tzinfo=timezone("US/Eastern"))
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
        start = "2022-01-01 00:00Z"
        end = "2022-12-31 00:00Z"
        region = "CAISO_NORTH"

        jsons = self.historical.get_historical_jsons(start, end, region)

        self.assertIsInstance(jsons, list)
        self.assertGreaterEqual(len(jsons), 1)
        self.assertIsInstance(jsons[0], dict)

    def test_get_historical_jsons_1_week(self):
        start = "2022-01-01 00:00Z"
        end = "2022-01-07 00:00Z"
        region = "CAISO_NORTH"

        jsons = self.historical.get_historical_jsons(start, end, region)

        self.assertIsInstance(jsons, list)
        self.assertGreaterEqual(len(jsons), 1)
        self.assertIsInstance(jsons[0], dict)

    def test_get_historical_jsons_signal_types(self):
        start = "2022-01-01 00:00Z"
        end = "2022-01-07 00:00Z"
        signal_types = ["co2_moer", "co2_aoer", "health_damage"]
        for signal_type in signal_types:
            if signal_type == "co2_aoer":
                region = "CAISO"
            else:
                region = "CAISO_NORTH"

            jsons = self.historical.get_historical_jsons(
                start, end, region, signal_type=signal_type
            )
            self.assertIsInstance(jsons, list)
            self.assertGreaterEqual(len(jsons), 1)
            self.assertIsInstance(jsons[0], dict)
            self.assertEqual(jsons[0]["meta"]["signal_type"], signal_type)
            self.assertEqual(jsons[0]["meta"]["region"], region)

    def test_get_historical_pandas(self):
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        region = "CAISO_NORTH"

        df = self.historical.get_historical_pandas(start, end, region)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("point_time", df.columns)
        self.assertIn("value", df.columns)

    def test_get_historical_pandas_meta(self):
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        region = "CAISO_NORTH"

        df = self.historical.get_historical_pandas(
            start, end, region, include_meta=True
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("point_time", df.columns)
        self.assertIn("value", df.columns)
        self.assertIn("meta", df.columns)


class TestWattTimeMyAccess(unittest.TestCase):
    def setUp(self):
        self.access = WattTimeMyAccess()

    def test_access_json_structure(self):
        json = self.access.get_access_json()
        self.assertIsInstance(json, dict)
        self.assertIn("signal_types", json)
        self.assertIn("regions", json["signal_types"][0])
        self.assertIn("signal_type", json["signal_types"][0])
        self.assertIn("region", json["signal_types"][0]["regions"][0])
        self.assertIn("region_full_name", json["signal_types"][0]["regions"][0])
        self.assertIn("parent", json["signal_types"][0]["regions"][0])
        self.assertIn(
            "data_point_period_seconds", json["signal_types"][0]["regions"][0]
        )
        self.assertIn("endpoints", json["signal_types"][0]["regions"][0])
        self.assertIn("endpoint", json["signal_types"][0]["regions"][0]["endpoints"][0])
        self.assertIn("models", json["signal_types"][0]["regions"][0]["endpoints"][0])
        self.assertIn(
            "model", json["signal_types"][0]["regions"][0]["endpoints"][0]["models"][0]
        )
        self.assertIn(
            "data_start",
            json["signal_types"][0]["regions"][0]["endpoints"][0]["models"][0],
        )
        self.assertIn(
            "train_start",
            json["signal_types"][0]["regions"][0]["endpoints"][0]["models"][0],
        )
        self.assertIn(
            "train_end",
            json["signal_types"][0]["regions"][0]["endpoints"][0]["models"][0],
        )
        self.assertIn(
            "type", json["signal_types"][0]["regions"][0]["endpoints"][0]["models"][0]
        )

    def test_access_pandas(self):
        df = self.access.get_access_pandas()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("signal_type", df.columns)
        self.assertIn("ba_abbrev", df.columns)
        self.assertIn("region_name", df.columns)
        self.assertIn("endpoint", df.columns)
        self.assertIn("model_date", df.columns)
        self.assertIn("model", df.columns)
        self.assertIn("data_start", df.columns)
        self.assertIn("train_start", df.columns)
        self.assertIn("train_end", df.columns)
        self.assertIn("type", df.columns)
        self.assertGreaterEqual(len(df), 1)


class TestWattTimeForecast(unittest.TestCase):
    def setUp(self):
        self.forecast = WattTimeForecast()

    def test_get_current_jsons(self):
        region = "CAISO_NORTH"

        json = self.forecast.get_forecast_json(region=region)

        self.assertIsInstance(json, dict)
        self.assertIn('meta', json)
        self.assertEqual(len(json['data']), 288)
        self.assertIn('point_time', json['data'][0])
        
    def test_get_current_pandas(self):
        region = "CAISO_NORTH"

        df = self.forecast.get_forecast_pandas(region=region)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("point_time", df.columns)
        self.assertIn("value", df.columns)

if __name__ == "__main__":
    unittest.main()
