import unittest
from datetime import datetime, date
from watttime.evaluation import report
from typing import List, Union, Dict
import plotly.graph_objects as go
import random
import pandas as pd
import numpy as np
from watttime.evaluation.get_wt_api_forecast_evaluation_data import DataHandlerFactory, AnalysisDataHandler

REGION_LIST: List[str] = ["CAISO_NORTH"]
MODEL_DATE_LIST: List[str] = ["2023-03-01", "2024-10-01"]
SIGNAL_TYPE: str = "co2_moer"
EVAL_START: Union[str, datetime, date] = "2024-10-01T00:00-02:00"
EVAL_END: Union[str, datetime, date] = "2024-11-01T00:00Z"
FORECAST_SAMPLE = 3


class TestPlotCreationMultiModel(unittest.TestCase):
    def setUp(self):

        self.factory = DataHandlerFactory(
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            regions=REGION_LIST,
            model_dates=MODEL_DATE_LIST,
            signal_types=SIGNAL_TYPE,
            forecast_sample_size=FORECAST_SAMPLE,
        )

    def validate_figure(self, fig_dict: Dict[str, go.Figure]):
        self.assertIsInstance(fig_dict, dict)
        self.assertIsInstance(fig_dict[REGION_LIST[0]], go.Figure)
        self.assertGreater(len(fig_dict[REGION_LIST[0]].data), 0)
        _ = fig_dict[REGION_LIST[0]].to_html(
            full_html=False,
            include_plotlyjs=False,
            include_mathjax=False,
            validate=True,
        )

    def test_plot_distribution_moers(self):
        fig_dict = report.plot_distribution_moers(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_heatmaps(self):
        fig_dict = report.plot_heatmaps(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_norm_mae(self):
        fig_dict = report.plot_norm_mae(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_rank_corr(self):
        fig_dict = report.plot_rank_corr(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_impact_forecast_metrics(self):
        fig_dict = report.plot_impact_forecast_metrics(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_bland_altman(self):
        fig_dict = report.plot_bland_altman(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_fuelmix_heatmap(self):
        fig_dict = report.plot_fuelmix_heatmap(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_max_impact_potential(self):
        fig_dict = report.plot_max_impact_potential(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_sample_fuelmix(self):
        fig_dict = report.plot_sample_fuelmix(self.factory)
        self.validate_figure(fig_dict)


class TestPlotCreationSingleModel(unittest.TestCase):
    def setUp(self):

        self.factory = DataHandlerFactory(
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            regions=REGION_LIST,
            model_dates=MODEL_DATE_LIST[0],
            signal_types=SIGNAL_TYPE,
            forecast_sample_size=FORECAST_SAMPLE,
        )

    def validate_figure(self, fig_dict: Dict[str, go.Figure]):
        self.assertIsInstance(fig_dict, dict)
        self.assertIsInstance(fig_dict[REGION_LIST[0]], go.Figure)
        self.assertGreater(len(fig_dict[REGION_LIST[0]].data), 0)
        _ = fig_dict[REGION_LIST[0]].to_html(
            full_html=False,
            include_plotlyjs=False,
            include_mathjax=False,
            validate=True,
        )

    def test_plot_sample_moers_single_job(self):
        fig_dict = report.plot_sample_moers(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_distribution_moers_single_job(self):
        fig_dict = report.plot_distribution_moers(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_heatmaps_single_job(self):
        fig_dict = report.plot_heatmaps(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_norm_mae_single_job(self):
        fig_dict = report.plot_norm_mae(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_rank_corr_single_job(self):
        fig_dict = report.plot_rank_corr(self.factory)
        self.validate_figure(fig_dict)

    def test_plot_impact_forecast_metrics_single_job(self):
        fig_dict = report.plot_impact_forecast_metrics(self.factory)
        self.validate_figure(fig_dict)


class TestSimulateCharge(unittest.TestCase):
    def setUp(self):
        """Creates a sample DataFrame where each generated_at has 4 point_time values."""
        generated_at_times = pd.date_range("2025-03-11 00:00:00", periods=4, freq="5T")
        point_time_offsets = pd.to_timedelta([0, 5, 10, 15], unit="m")

        index = pd.MultiIndex.from_tuples(
            [
                (gen_at + offset, gen_at)
                for gen_at in generated_at_times
                for offset in point_time_offsets
            ],
            names=["point_time", "generated_at"],
        )

        data = {
            "window_start": [pd.Timestamp("2025-03-11 00:00:00")] * len(index),
            "forecast_value": np.random.randint(
                1, 20, len(index)
            ),  # Random forecast values
            "sort_col": np.tile(
                [1, 2, 3, 4], len(generated_at_times)
            ),  # Ranks within each generated_at
        }

        self.df = pd.DataFrame(data, index=index)

    def test_charge_status_output(self):
        """Ensure the function outputs a charge_status column with correct type."""
        charge_status = report.simulate_charge(
            self.df, sort_col="sort_col", charge_mins=30
        )
        self.assertIsInstance(charge_status, pd.Series)
        self.assertTrue(charge_status.dtype == bool)

    def test_correct_length(self):
        """Check that the charge_status output has the correct length."""
        charge_status = report.simulate_charge(
            self.df, sort_col="sort_col", charge_mins=30
        )
        self.assertEqual(len(charge_status), len(self.df))

    def test_all_point_times_gte_generated_at(self):
        """Ensure all point_time values are greater than or equal to generated_at."""
        self.assertTrue(
            (
                self.df.index.get_level_values("point_time")
                >= self.df.index.get_level_values("generated_at")
            ).all()
        )

    def test_charging_fully_schedules(self):
        """Verify that a full charge is scheduled correctly."""
        charge_status = report.simulate_charge(
            self.df, sort_col="sort_col", charge_mins=30
        )
        charge_needed = 20 // 5  # Number of charge periods required
        charged_windows = self.df.loc[charge_status].groupby("window_start").size()
        self.assertTrue((charged_windows >= charge_needed - 1).all())
        self.assertTrue((charged_windows <= charge_needed).all())

    def test_assign_windows_daily_overnight(self):
        """assign_windows should include timestamps in [start, start+window) even when the window crosses midnight."""
        tz = "America/Los_Angeles"
        timestamps = pd.DatetimeIndex(
            [
                "2025-03-11 18:55",  # before daily start
                "2025-03-11 19:00",  # at start
                "2025-03-11 22:00",  # inside
                "2025-03-12 00:10",  # inside (next day)
                "2025-03-12 02:55",  # inside (next day)
                "2025-03-12 03:00",  # at end (excluded)
                "2025-03-12 03:05",  # after end
            ],
            tz=tz,
        )

        anchors = report.assign_windows(timestamps, window_size=8 * 60, window_start_time="19:00")

        expected_anchor = pd.Timestamp("2025-03-11 19:00", tz=tz)

        self.assertTrue(pd.isna(anchors[0]))
        self.assertEqual(anchors[1], expected_anchor)
        self.assertEqual(anchors[2], expected_anchor)
        self.assertEqual(anchors[3], expected_anchor)
        self.assertEqual(anchors[4], expected_anchor)
        self.assertTrue(pd.isna(anchors[5]))
        self.assertTrue(pd.isna(anchors[6]))

    def test_assign_windows_continuous_tiling(self):
        """Without window_start_time, assign continuous non-overlapping windows of given length from first local midnight."""
        tz = "America/Los_Angeles"
        start = pd.Timestamp("2025-03-11 00:00", tz=tz)
        timestamps = pd.date_range(start=start, periods=25, freq="5T")  # 2 hours + 5 minutes

        anchors = report.assign_windows(timestamps, window_size=30)

        # 00:00-00:30 -> anchor 00:00, 00:30-01:00 -> anchor 00:30, 01:00-01:30 -> anchor 01:00, etc.
        self.assertEqual(anchors[0], start)  # 00:00 -> anchor 00:00
        self.assertEqual(anchors[6], start + pd.Timedelta(minutes=30))  # 00:30 -> anchor 00:30
        self.assertEqual(anchors[7], start + pd.Timedelta(minutes=30))  # 00:35 -> anchor 00:30
        self.assertEqual(anchors[12], start + pd.Timedelta(minutes=60))  # 01:00 -> anchor 01:00

    def test_calc_max_potential_equals_rank_compare_potential_when_constrained_inputs_match(self):
        """When both functions operate over the same constrained candidate set (earliest-per-pull rows after
        window filtering), their 'potential' should match.

        Steps:
        - Build a synthetic forecasts_v_moers across a single 2-hour window.
        - Reproduce the window filtering used by calc_rank_compare_metrics, then reduce to earliest-per-pull rows.
        - Feed that constrained candidate set as a moers DataFrame into calc_max_potential.
        - Assert that calc_max_potential['potential'] equals calc_rank_compare_metrics()['potential'].
        """

        tz = "America/Los_Angeles"
        start = pd.Timestamp("2024-01-01 19:00", tz=tz)
        window_mins = 60
        charge_mins = 10  # 2 five-minute slots
        load_kw = 100

        # Build forecasts_v_moers: signal high then low to make ordering visible
        rows = []
        for ga in pd.date_range(start, start + pd.Timedelta(hours=2), freq="5T"):
            for pt_i in pd.date_range(ga, ga + pd.Timedelta(minutes=55), freq="5T"):
                rows.append(
                    {
                        "generated_at": ga,
                        "point_time": pt_i,
                        "predicted_value": random.random(),
                        "signal_value": 1000 - pt_i.minute * 10
                    }
                )
        fvm = (
            pd.DataFrame(rows)
            .set_index(["generated_at", "point_time"])
            .sort_index()
        )

        # Compute rank-compare potential
        rc = report.calc_rank_compare_metrics(
            fvm,
            charge_mins=charge_mins,
            window_mins=window_mins,
            window_start_time="19:00",
            pred_col="predicted_value",
            truth_col="signal_value",
            load_kw=load_kw,
        )
        rc_potential = rc["potential"]

        # Reproduce the same candidate set (earliest-per-pull after window filtering)
        wm = pd.Timedelta(minutes=window_mins)
        window_starts = report.assign_windows(
            fvm.index.get_level_values("generated_at"), wm, "19:00"
        )
        df = fvm.assign(
            window_start=window_starts,
            window_end=[i + wm for i in window_starts],
        )
        # out-of-window filter
        df.loc[
            (df.index.get_level_values("point_time") >= df["window_end"])
            | (df.index.get_level_values("generated_at") >= df["window_end"])
            | (df.index.get_level_values("generated_at") < df["window_start"]),
            "window_start",
        ] = pd.NaT
        df = df.dropna(subset=["window_start", "window_end"]).copy()
        # sufficient generated_at per window
        n_gen = df.groupby("window_start").transform(
            lambda x: x.reset_index()["generated_at"].nunique()
        )["predicted_value"]
        df = df.loc[n_gen >= (charge_mins // 5) - 1]
        
        mp = report.calc_max_potential(
            df,
            charge_mins=charge_mins,
            window_mins=window_mins,
            window_start_time="19:00",
            truth_col="signal_value",
            load_kw=load_kw,
        )["potential"]

        self.assertAlmostEqual(mp, rc_potential, places=6)

class TestAnalysisDataHandler(unittest.TestCase):

    def setUp(self) -> None:
        self.handler = AnalysisDataHandler(
            region=REGION_LIST[0],
            model_date=MODEL_DATE_LIST[0],
            signal_type=SIGNAL_TYPE,
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            forecast_sample_size=2,
            forecast_max_horizon=72*60
        )
                
    def test_data_handler_initialization(self):
        """Test that AnalysisDataHandler initializes correctly with given parameters."""

        self.assertEqual(self.handler.region, REGION_LIST[0])
        self.assertEqual(self.handler.model_date, MODEL_DATE_LIST[0])
        self.assertEqual(self.handler.signal_type, SIGNAL_TYPE)
        self.assertEqual(self.handler.tz, "America/Los_Angeles")
        self.assertEqual(len(self.handler.sample_days), 2*3)

    def test_fetch_moer_data(self):
        """Test that MOER data is fetched and returned as a DataFrame."""
        self.assertIsInstance(self.handler.moers, pd.DataFrame)
        self.assertEqual(["signal_value"], self.handler.moers.columns)
        self.assertIsInstance(self.handler.moers.index, pd.DatetimeIndex)
        self.assertEqual(self.handler.moers.index.tz.zone, self.handler.tz)
        expected_date_range = pd.date_range(
            start=pd.to_datetime(EVAL_START).tz_convert(self.handler.tz),
            end=pd.to_datetime(EVAL_END).tz_convert(self.handler.tz) + pd.Timedelta(minutes=self.handler.forecast_max_horizon),
            freq="5min",
        )
        self.assertTrue(
            self.handler.moers.index.equals(expected_date_range),
            "MOER DataFrame index does not match expected date range.",
        )
    def test_fetch_forecast_data(self):
        """Test that forecast data is fetched and returned as a DataFrame."""
        self.assertIsInstance(self.handler.forecasts, pd.DataFrame)
        self.assertEqual(
            ['point_time', 'generated_at', 'horizon_mins', 'predicted_value'],
            list(self.handler.forecasts.columns),
        )
        self.assertEqual(
            self.handler.forecasts['point_time'].dt.tz.zone,
            self.handler.tz,
        )
        self.assertEqual(
            self.handler.forecasts['generated_at'].dt.tz.zone,
            self.handler.tz,
        )
        expected_gen_ats = len(self.handler.sample_days) * ((60 // 5) * 24)
        actual_gen_ats = self.handler.forecasts['generated_at'].nunique()
        self.assertEqual(
            actual_gen_ats,
            expected_gen_ats,
            "Number of unique generated_at timestamps does not match expected count.",
        )

        expected_length = expected_gen_ats * (self.handler.forecast_max_horizon // 5)
        self.assertEqual(
            len(self.handler.forecasts),
            expected_length,
            "Forecast DataFrame length does not match expected number of samples.",
        )

    def test_forecasts_v_moers(self):
        """Test that forecasted values align correctly with MOER data."""
        self.assertEqual(
            len(self.handler.forecasts_v_moers),
            len(self.handler.forecasts),
            "Forecasts vs MOERs DataFrame length does not match Forecasts DataFrame length.",
        )
        self.assertEqual(
            list(self.handler.forecasts_v_moers.columns),
            ['horizon_mins', 'predicted_value', 'signal_value'],
            "Forecasts vs MOERs DataFrame columns do not match expected columns.",
        )
        self.assertEqual(
            self.handler.forecasts_v_moers.index.names,
            ['generated_at', 'point_time'],
            "Forecasts vs MOERs DataFrame index names do not match expected names.",
        )
        self.assertEqual(
            self.handler.forecasts_v_moers.index.get_level_values('generated_at').tz.zone,
            self.handler.tz,
            "generated_at index level does not have the expected timezone.",
        )
        self.assertEqual(
            self.handler.forecasts_v_moers.index.get_level_values('point_time').tz.zone,
            self.handler.tz,
            "point_time index level does not have the expected timezone.",
        )

