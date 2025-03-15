import unittest
from dateutil.parser import parse

from watttime import WattTimeAnalysisData as WattTimeData

import pandas as pd


REGION = "CAISO_NORTH"


class TestAnalysisDataHandler(unittest.TestCase):
    def test_get_moers(self):
        dh = WattTimeData(
            region=REGION,
            eval_start=parse("2024-01-01 00:00Z"),
            eval_end=parse("2024-01-01 23:59Z"),
            forecast_max_horizon=0,
        )
        assert isinstance(dh.moers, pd.DataFrame)
        assert all(dh.moers.columns == ["signal_value"])
        assert dh.moers.index.name == "point_time"
        assert len(dh.moers) == 24 * 12

        dh = WattTimeData(
            region=REGION,
            eval_start=parse("2024-01-01 00:00Z"),
            eval_end=parse("2024-01-01 23:59Z"),
            forecast_max_horizon=60 * 24,
        )
        assert len(dh.moers) == (12 * 24) * 2  # also gets through forecast max horizon

    def test_get_forecasts(self):
        dh = WattTimeData(
            region=REGION,
            eval_start=parse("2024-01-01 00:00Z"),
            eval_end=parse("2024-01-01 23:59Z"),
        )
        assert isinstance(dh.forecasts, pd.DataFrame)
        assert all(
            dh.forecasts.columns
            == ["point_time", "predicted_value", "generated_at", "horizon_mins"]
        )
        assert len(dh.forecasts) == (12 * 24) * (12 * 24)

    def test_forecast_v_moer(self):
        dh = WattTimeData(
            region=REGION,
            eval_start=parse("2024-01-01 00:00Z"),
            eval_end=parse("2024-01-01 23:59Z"),
        )
        assert isinstance(dh.forecast_v_moer, pd.DataFrame)
        assert all(
            dh.forecast_v_moer.columns
            == [
                "predicted_value",
                "horizon_mins",
                "signal_value",
            ]
        )
        assert len(dh.forecast_v_moer) == (12 * 24) * (12 * 24)
        assert sum(dh.forecast_v_moer.isna()["signal_value"]) == 0
        assert sum(dh.forecast_v_moer.isna()["predicted_value"]) == 0

    def test_forecast_v_moer_sampled(self):
        # Sample 100%
        dh = WattTimeData(
            region=REGION,
            eval_start=parse("2024-01-01 00:00Z"),
            eval_end=parse("2024-01-02 23:59Z"),
            forecast_sample_perc=1.0,
        )
        assert len(dh.eval_days) == 2
        assert len(dh.sample_days) == 2
        assert len(set(dh.forecasts["generated_at"].dt.date)) == 2

        # Sample 50%
        dh = WattTimeData(
            region=REGION,
            eval_start=parse("2024-01-01 00:00Z"),
            eval_end=parse("2024-01-02 23:59Z"),
            forecast_sample_perc=0.5,
        )
        assert len(dh.eval_days) == 2
        assert len(dh.sample_days) == 1

        assert len(set(dh.forecasts["generated_at"].dt.date)) == 1
