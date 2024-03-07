import unittest
import unittest.mock as mock
from datetime import datetime, timedelta
from dateutil.parser import parse
from pytz import timezone, UTC
import os
from watttime import (
    WattTimeBase,
    WattTimeHistorical,
    WattTimeMyAccess,
    WattTimeForecast,
    WattTimeMaps,
)
from pathlib import Path

import pandas as pd
import requests

REGION = "CAISO_NORTH"


def mocked_register(*args, **kwargs):
    url = args[0]

    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

        def raise_for_status(self):
            assert self.status_code == 200

    if (
        (url == "https://api.watttime.org/register")
        & (kwargs["json"]["email"] == os.getenv("WATTTIME_EMAIL"))
        & (kwargs["json"]["username"] == os.getenv("WATTTIME_USER"))
        & (kwargs["json"]["password"] == os.getenv("WATTTIME_PASSWORD"))
    ):
        return MockResponse(
            {"ok": "User created", "user": kwargs["json"]["username"]}, 200
        )
    else:
        raise MockResponse({"error": "Failed to create user"}, 400)


class TestWattTimeBase(unittest.TestCase):
    def setUp(self):
        self.base = WattTimeBase()

    def test_login_with_real_api(self):
        self.base._login()
        assert self.base.token is not None
        assert self.base.token_valid_until > datetime.now()

    def test_parse_dates_with_string(self):
        start = "2022-01-01"
        end = "2022-01-31"

        parsed_start, parsed_end = self.base._parse_dates(start, end)

        self.assertIsInstance(parsed_start, datetime)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_start, datetime(2022, 1, 1, tzinfo=UTC))
        self.assertEqual(parsed_end, datetime(2022, 1, 31, tzinfo=UTC))

    def test_parse_dates_with_datetime(self):
        # Case 1: User provides a string with no timezone
        start_str = "2022-01-01"
        end_str = "2022-01-31"
        parsed_start, parsed_end = self.base._parse_dates(start_str, end_str)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 2: User provides a string with non-UTC timezone
        start_str = "2022-01-01 12:00:00+02:00"
        end_str = "2022-01-31 12:00:00+02:00"
        parsed_start, parsed_end = self.base._parse_dates(start_str, end_str)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 3: User provides a string with UTC timezone
        start_str = "2022-01-01 12:00:00Z"
        end_str = "2022-01-31 12:00:00Z"
        parsed_start, parsed_end = self.base._parse_dates(start_str, end_str)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 4: User provides a datetime with no timezone
        start_dt = datetime(2022, 1, 1)
        end_dt = datetime(2022, 1, 31)
        parsed_start, parsed_end = self.base._parse_dates(start_dt, end_dt)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 5: User provides a datetime with non-UTC timezone
        start_dt = datetime(2022, 1, 1, tzinfo=timezone("US/Eastern"))
        end_dt = datetime(2022, 1, 31, tzinfo=timezone("US/Eastern"))
        parsed_start, parsed_end = self.base._parse_dates(start_dt, end_dt)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

        # Case 6: User provides a datetime with UTC timezone
        start_dt = datetime(2022, 1, 1, tzinfo=UTC)
        end_dt = datetime(2022, 1, 31, tzinfo=UTC)
        parsed_start, parsed_end = self.base._parse_dates(start_dt, end_dt)
        self.assertIsInstance(parsed_start, datetime)
        self.assertEqual(parsed_start.tzinfo, UTC)
        self.assertIsInstance(parsed_end, datetime)
        self.assertEqual(parsed_end.tzinfo, UTC)

    @mock.patch("requests.post", side_effect=mocked_register)
    def test_mock_register(self, mock_post):
        resp = self.base.register(email=os.getenv("WATTTIME_EMAIL"))
        self.assertEqual(len(mock_post.call_args_list), 1)


class TestWattTimeHistorical(unittest.TestCase):
    def setUp(self):
        self.historical = WattTimeHistorical()

    def test_get_historical_jsons_3_months(self):
        start = "2022-01-01 00:00Z"
        end = "2022-12-31 00:00Z"
        jsons = self.historical.get_historical_jsons(start, end, REGION)

        self.assertIsInstance(jsons, list)
        self.assertGreaterEqual(len(jsons), 1)
        self.assertIsInstance(jsons[0], dict)

    def test_get_historical_jsons_1_week(self):
        start = "2022-01-01 00:00Z"
        end = "2022-01-07 00:00Z"
        jsons = self.historical.get_historical_jsons(start, end, REGION)

        self.assertIsInstance(jsons, list)
        self.assertGreaterEqual(len(jsons), 1)
        self.assertIsInstance(jsons[0], dict)

    def test_get_historical_jsons_signal_types(self):
        start = "2023-01-01 00:00Z"
        end = "2023-01-07 00:00Z"
        signal_types = ["co2_moer", "co2_aoer", "health_damage"]
        for signal_type in signal_types:
            if signal_type == "co2_aoer":
                region = "CAISO"
            else:
                region = REGION
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
        df = self.historical.get_historical_pandas(start, end, REGION)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("point_time", df.columns)
        self.assertIn("value", df.columns)

    def test_get_historical_pandas_meta(self):
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        df = self.historical.get_historical_pandas(
            start, end, REGION, include_meta=True
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("point_time", df.columns)
        self.assertIn("value", df.columns)
        self.assertIn("meta", df.columns)

        assert pd.api.types.is_datetime64_any_dtype(df["point_time"].dtype)

    def test_get_historical_csv(self):
        start = parse("2022-01-01 00:00Z")
        end = parse("2022-01-02 00:00Z")
        self.historical.get_historical_csv(start, end, REGION)

        fp = (
            Path.home()
            / "watttime_historical_csvs"
            / f"{REGION}_co2_moer_{start.date()}_{end.date()}.csv"
        )
        assert fp.exists()
        fp.unlink()

    def test_multi_model_range(self):
        """If model is not specified, we should only return the most recent model data"""
        myaccess = WattTimeMyAccess()
        access = myaccess.get_access_pandas()
        access = access.loc[
            (access["signal_type"] == "co2_moer") & (access["region"] == REGION)
        ].sort_values("model", ascending=False)
        assert len(access) > 1

        # start request one month before data_start of most recent model
        start = access["data_start"].values[0] - pd.Timedelta(days=30)
        end = access["data_start"].values[0] + pd.Timedelta(days=30)
        df = self.historical.get_historical_pandas(
            start, end, REGION, include_meta=True
        )

        # should not span into an older model
        self.assertEqual(df.iloc[0]["meta"]["model"]["date"], access.iloc[0]["model"])

        self.assertEqual(df.iloc[-1]["meta"]["model"]["date"], access.iloc[0]["model"])

        # first point_time should be data_start from my-acces
        self.assertAlmostEqual(
            df.iloc[0]["point_time"],
            access.iloc[0]["data_start"].tz_localize("UTC"),
            delta=pd.Timedelta(days=1),
        )


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
        self.assertIn("region", df.columns)
        self.assertIn("region_name", df.columns)
        self.assertIn("endpoint", df.columns)
        self.assertIn("model", df.columns)
        self.assertIn("data_start", df.columns)
        self.assertIn("train_start", df.columns)
        self.assertIn("train_end", df.columns)
        self.assertIn("type", df.columns)
        self.assertGreaterEqual(len(df), 1)

        assert pd.api.types.is_datetime64_any_dtype(df["data_start"])
        assert pd.api.types.is_datetime64_any_dtype(df["train_start"])
        assert pd.api.types.is_datetime64_any_dtype(df["train_end"])


class TestWattTimeForecast(unittest.TestCase):
    def setUp(self):
        self.forecast = WattTimeForecast()

    def test_get_current_json(self):
        json = self.forecast.get_forecast_json(region=REGION)

        self.assertIsInstance(json, dict)
        self.assertIn("meta", json)
        self.assertEqual(len(json["data"]), 288)
        self.assertIn("point_time", json["data"][0])

    def test_get_current_pandas(self):
        df = self.forecast.get_forecast_pandas(region=REGION)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("point_time", df.columns)
        self.assertIn("value", df.columns)

    def test_historical_forecast_jsons(self):
        start = "2023-01-01 00:00Z"
        end = "2023-01-03 00:00Z"
        json_list = self.forecast.get_historical_forecast_json(
            start, end, region=REGION
        )
        first_json = json_list[0]

        self.assertIsInstance(json_list, list)
        self.assertIn("meta", first_json)
        self.assertEqual(len(first_json["data"]), 288)
        self.assertIn("generated_at", first_json["data"][0])

    def test_historical_forecast_pandas(self):
        start = "2023-01-01 00:00Z"
        end = "2023-01-03 00:00Z"
        df = self.forecast.get_historical_forecast_pandas(start, end, region=REGION)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        self.assertIn("point_time", df.columns)
        self.assertIn("value", df.columns)
        self.assertIn("generated_at", df.columns)

    def test_horizon_hours(self):
        json = self.forecast.get_forecast_json(region=REGION, horizon_hours=0)
        self.assertIsInstance(json, dict)
        self.assertIn("meta", json)
        self.assertEqual(len(json["data"]), 1)
        self.assertIn("point_time", json["data"][0])

        json2 = self.forecast.get_forecast_json(region=REGION, horizon_hours=24)
        self.assertIsInstance(json2, dict)
        self.assertIn("meta", json2)
        self.assertEqual(len(json2["data"]), 288)
        self.assertIn("point_time", json2["data"][0])

        json3 = self.forecast.get_forecast_json(region=REGION, horizon_hours=72)
        self.assertIsInstance(json3, dict)
        self.assertIn("meta", json3)
        self.assertEqual(len(json3["data"]), 864)
        self.assertIn("point_time", json3["data"][0])


class TestWattTimeMaps(unittest.TestCase):
    def setUp(self):
        self.maps = WattTimeMaps()

    def test_get_maps_json_moer(self):
        moer = self.maps.get_maps_json(signal_type="co2_moer")
        self.assertEqual(moer["type"], "FeatureCollection")
        self.assertEqual(moer["meta"]["signal_type"], "co2_moer")
        self.assertGreater(
            parse(moer["meta"]["last_updated"]), parse("2023-01-01 00:00Z")
        )
        self.assertGreater(len(moer["features"]), 100)  # 172 as of 2023-12-01

    def test_get_maps_json_aoer(self):
        aoer = self.maps.get_maps_json(signal_type="co2_aoer")
        self.assertEqual(aoer["type"], "FeatureCollection")
        self.assertEqual(aoer["meta"]["signal_type"], "co2_aoer")
        self.assertGreater(
            parse(aoer["meta"]["last_updated"]), parse("2023-01-01 00:00Z")
        )
        self.assertGreater(len(aoer["features"]), 50)  # 87 as of 2023-12-01

    def test_get_maps_json_health(self):
        health = self.maps.get_maps_json(signal_type="health_damage")
        self.assertEqual(health["type"], "FeatureCollection")
        self.assertEqual(health["meta"]["signal_type"], "health_damage")
        self.assertGreater(
            parse(health["meta"]["last_updated"]), parse("2022-01-01 00:00Z")
        )
        self.assertGreater(len(health["features"]), 100)  # 114 as of 2023-12-01

    def test_region_from_loc(self):
        region = self.maps.region_from_loc(
            latitude=39.7522, longitude=-105.0, signal_type="co2_moer"
        )
        self.assertEqual(region["region"], "PSCO")
        self.assertEqual(region["region_full_name"], "Public Service Co of Colorado")
        self.assertEqual(region["signal_type"], "co2_moer")


if __name__ == "__main__":
    unittest.main()
