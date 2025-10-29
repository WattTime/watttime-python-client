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
        generated_at_times = pd.date_range("2025-03-11 00:00:00", periods=4, freq="5min")
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


class TestBootstrapConfidenceInterval(unittest.TestCase):
    """Tests for the bootstrap_confidence_interval function."""

    def setUp(self):
        """Create sample daily data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        self.daily_data = pd.DataFrame({
            "date": dates,
            "emissions": np.random.normal(100, 20, 30),
            "count": np.random.randint(10, 50, 30),
        })

    def test_bootstrap_returns_dict_with_ci_keys(self):
        """Test that bootstrap returns a dict with CI lower/upper keys."""
        def metric_func(df):
            return {"mean_emissions": df["emissions"].mean()}
        
        result = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=1.0,
            n_bootstrap=100,
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("mean_emissions_ci_lower", result)
        self.assertIn("mean_emissions_ci_upper", result)

    def test_bootstrap_ci_bounds_sensible(self):
        """Test that CI lower < upper and bounds are reasonable."""
        def metric_func(df):
            return {"mean_emissions": df["emissions"].mean()}
        
        result = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=1.0,
            n_bootstrap=100,
        )
        
        self.assertLess(
            result["mean_emissions_ci_lower"],
            result["mean_emissions_ci_upper"],
            "CI lower bound should be less than upper bound"
        )
        
        # Check that bounds are within reasonable range of data
        mean_val = self.daily_data["emissions"].mean()
        self.assertLess(result["mean_emissions_ci_lower"], mean_val * 1.5)
        self.assertGreater(result["mean_emissions_ci_upper"], mean_val * 0.5)

    def test_bootstrap_multiple_metrics(self):
        """Test that bootstrap works with multiple metrics returned."""
        def metric_func(df):
            return {
                "mean": df["emissions"].mean(),
                "total": df["emissions"].sum(),
                "max": df["emissions"].max(),
            }
        
        result = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=1.0,
            n_bootstrap=100,
        )
        
        # Check all metrics have CIs
        self.assertIn("mean_ci_lower", result)
        self.assertIn("mean_ci_upper", result)
        self.assertIn("total_ci_lower", result)
        self.assertIn("total_ci_upper", result)
        self.assertIn("max_ci_lower", result)
        self.assertIn("max_ci_upper", result)
        
        # All should be valid
        for metric in ["mean", "total", "max"]:
            self.assertLess(result[f"{metric}_ci_lower"], result[f"{metric}_ci_upper"])

    def test_bootstrap_with_low_sample_rate(self):
        """Test that low sample rate inflates CIs (wider intervals)."""
        def metric_func(df):
            return {"mean": df["emissions"].mean()}
        
        # Full sample
        result_full = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=1.0,
            n_bootstrap=200,
            random_seed=42,
        )
        
        # Low sample rate (10%)
        result_sampled = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=0.1,
            n_bootstrap=200,
            random_seed=42,
        )
        
        width_full = result_full["mean_ci_upper"] - result_full["mean_ci_lower"]
        width_sampled = result_sampled["mean_ci_upper"] - result_sampled["mean_ci_lower"]
        
        self.assertGreater(
            width_sampled,
            width_full,
            "Low sample rate should produce wider confidence intervals"
        )

    def test_bootstrap_reproducible_with_seed(self):
        """Test that results are reproducible with same random seed."""
        def metric_func(df):
            return {"mean": df["emissions"].mean()}
        
        result1 = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=1.0,
            n_bootstrap=100,
            random_seed=123,
        )
        
        result2 = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=1.0,
            n_bootstrap=100,
            random_seed=123,
        )
        
        self.assertEqual(result1["mean_ci_lower"], result2["mean_ci_lower"])
        self.assertEqual(result1["mean_ci_upper"], result2["mean_ci_upper"])

    def test_bootstrap_raises_on_empty_data(self):
        """Test that bootstrap raises ValueError on empty DataFrame."""
        def metric_func(df):
            return {"mean": df["emissions"].mean()}
        
        empty_df = pd.DataFrame({"date": [], "emissions": []})
        
        with self.assertRaises(ValueError) as context:
            report.bootstrap_confidence_interval(
                df_with_date=empty_df,
                metric_func=metric_func,
                effective_forecast_samp_rate=1.0,
            )
        
        self.assertIn("empty", str(context.exception).lower())

    def test_bootstrap_confidence_level(self):
        """Test that different confidence levels produce different intervals."""
        def metric_func(df):
            return {"mean": df["emissions"].mean()}
        
        result_95 = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=1.0,
            confidence_level=0.95,
            n_bootstrap=200,
            random_seed=42,
        )
        
        result_90 = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=1.0,
            confidence_level=0.90,
            n_bootstrap=200,
            random_seed=42,
        )
        
        width_95 = result_95["mean_ci_upper"] - result_95["mean_ci_lower"]
        width_90 = result_90["mean_ci_upper"] - result_90["mean_ci_lower"]
        
        self.assertGreater(
            width_95,
            width_90,
            "95% CI should be wider than 90% CI"
        )

    def test_bootstrap_effective_sample_size_calculation(self):
        """Test that effective sample size is calculated correctly for different rates."""
        def metric_func(df):
            return {"count": len(df)}
        
        # With 30 days and 0.1 sample rate, effective n should be max(2, int(30*0.1)) = 3
        result = report.bootstrap_confidence_interval(
            df_with_date=self.daily_data,
            metric_func=metric_func,
            effective_forecast_samp_rate=0.1,
            n_bootstrap=50,
            random_seed=42,
        )
        
        # The count metric should reflect bootstrap sampling
        # With very low effective sample size, CIs should be quite wide
        self.assertIn("count_ci_lower", result)
        self.assertIn("count_ci_upper", result)


class TestCalcNormMaeWithCI(unittest.TestCase):
    """Test calc_norm_mae with confidence intervals enabled."""

    def setUp(self):
        """Create a mock AnalysisDataHandler with synthetic data."""
        # Create synthetic forecasts_v_moers data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="America/Los_Angeles")
        
        rows = []
        for date in dates:
            for hour in range(24):
                generated_at = date + pd.Timedelta(hours=hour)
                for horizon in [55, 115, 175]:  # 55min, 115min, 175min horizons
                    point_time = generated_at + pd.Timedelta(minutes=horizon)
                    predicted = 100 + np.random.normal(0, 10)
                    signal = 100 + np.random.normal(0, 10)
                    rows.append({
                        "generated_at": generated_at,
                        "point_time": point_time,
                        "horizon_mins": horizon,
                        "predicted_value": predicted,
                        "signal_value": signal,
                    })
        
        forecasts_v_moers = pd.DataFrame(rows).set_index(["generated_at", "point_time"])
        
        # Create a mock data handler
        self.mock_handler = type('MockHandler', (), {
            'forecasts_v_moers': forecasts_v_moers,
            'effective_forecast_sample_rate': 1.0,
        })()

    def test_calc_norm_mae_without_ci_returns_scalar(self):
        """Test that calc_norm_mae returns scalar when ci=False."""
        result = report.calc_norm_mae(
            self.mock_handler,
            horizon_mins=55,
            ci=False,
        )
        
        self.assertIsInstance(result, (int, float, np.floating))

    def test_calc_norm_mae_with_ci_returns_dict(self):
        """Test that calc_norm_mae returns dict with CIs when ci=True."""
        result = report.calc_norm_mae(
            self.mock_handler,
            horizon_mins=55,
            ci=True,
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("norm_mae", result)
        self.assertIn("norm_mae_ci_lower", result)
        self.assertIn("norm_mae_ci_upper", result)

    def test_calc_norm_mae_ci_bounds_sensible(self):
        """Test that MAE CIs are sensible (lower < value < upper)."""
        result = report.calc_norm_mae(
            self.mock_handler,
            horizon_mins=55,
            ci=True,
        )
        
        self.assertLessEqual(result["norm_mae_ci_lower"], result["norm_mae"])
        self.assertLessEqual(result["norm_mae"], result["norm_mae_ci_upper"])


class TestCalcRankCorrWithCI(unittest.TestCase):
    """Test calc_rank_corr with confidence intervals enabled."""

    def setUp(self):
        """Create a mock AnalysisDataHandler with synthetic data."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="America/Los_Angeles")
        
        rows = []
        for date in dates:
            for hour in range(24):
                generated_at = date + pd.Timedelta(hours=hour)
                # Create correlated predictions and signals
                signal_base = 100 + hour * 2  # Time-varying pattern
                for horizon in [55, 115, 175]:
                    point_time = generated_at + pd.Timedelta(minutes=horizon)
                    signal = signal_base + np.random.normal(0, 5)
                    predicted = signal_base + np.random.normal(0, 8)  # Correlated with signal
                    rows.append({
                        "generated_at": generated_at,
                        "point_time": point_time,
                        "horizon_mins": horizon,
                        "predicted_value": predicted,
                        "signal_value": signal,
                    })
        
        forecasts_v_moers = pd.DataFrame(rows).set_index(["generated_at", "point_time"])
        
        self.mock_handler = type('MockHandler', (), {
            'forecasts_v_moers': forecasts_v_moers,
            'effective_forecast_sample_rate': 1.0,
        })()

    def test_calc_rank_corr_without_ci_returns_scalar(self):
        """Test that calc_rank_corr returns scalar when ci=False."""
        result = report.calc_rank_corr(
            self.mock_handler,
            horizon_mins=175,
            ci=False,
        )
        
        self.assertIsInstance(result, (int, float, np.floating))

    def test_calc_rank_corr_with_ci_returns_dict(self):
        """Test that calc_rank_corr returns dict with CIs when ci=True."""
        result = report.calc_rank_corr(
            self.mock_handler,
            horizon_mins=175,
            ci=True,
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("rank_corr", result)
        self.assertIn("rank_corr_ci_lower", result)
        self.assertIn("rank_corr_ci_upper", result)

    def test_calc_rank_corr_ci_bounds_sensible(self):
        """Test that correlation CIs are sensible (within [-1, 1] and ordered)."""
        result = report.calc_rank_corr(
            self.mock_handler,
            horizon_mins=175,
            ci=True,
        )
        
        # CIs should be ordered
        self.assertLessEqual(result["rank_corr_ci_lower"], result["rank_corr"])
        self.assertLessEqual(result["rank_corr"], result["rank_corr_ci_upper"])
        
        # All values should be in valid correlation range [-1, 1]
        self.assertGreaterEqual(result["rank_corr_ci_lower"], -1)
        self.assertLessEqual(result["rank_corr_ci_upper"], 1)


