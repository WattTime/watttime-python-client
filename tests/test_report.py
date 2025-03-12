import unittest
from datetime import datetime, date
from watttime.evaluation import report
from typing import List, Union, Dict
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from watttime.evaluation.get_wt_api_forecast_evaluation_data import DataHandlerFactory

REGION_LIST: List[str] = ["WEM"]
MODEL_DATE_LIST: List[str] = ["2022-10-01", "2025-02-01"]
SIGNAL_TYPE: str = "co2_moer"
EVAL_START: Union[str, datetime, date] = "2023-10-29T00:00Z"
EVAL_END: Union[str, datetime, date] = "2023-11-03T00:00Z"
FORECAST_SAMPLE = 3


class TestPlotCreationMultiModel(unittest.TestCase):
    def setUp(self):

        self.factory = DataHandlerFactory(
            eval_start=EVAL_START,
            eval_end=EVAL_END,
            regions=REGION_LIST,
            model_dates=MODEL_DATE_LIST,
            signal_types=SIGNAL_TYPE,
            forecast_sample_size=FORECAST_SAMPLE
        )

    def validate_figure(self, fig_dict: Dict[str, go.Figure]):
        self.assertIsInstance(fig_dict, dict)
        self.assertIsInstance(fig_dict[REGION_LIST[0]], go.Figure)
        self.assertGreater(len(fig_dict[REGION_LIST[0]].data), 0)
        _ = fig_dict[REGION_LIST[0]].to_html(
            full_html=False, include_plotlyjs=False, include_mathjax=False, validate=True
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
            forecast_sample_size=FORECAST_SAMPLE
        )

    def validate_figure(self, fig_dict: Dict[str, go.Figure]):
        self.assertIsInstance(fig_dict, dict)
        self.assertIsInstance(fig_dict[REGION_LIST[0]], go.Figure)
        self.assertGreater(len(fig_dict[REGION_LIST[0]].data), 0)
        _ = fig_dict[REGION_LIST[0]].to_html(
            full_html=False, include_plotlyjs=False, include_mathjax=False, validate=True
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
            names=["point_time", "generated_at"]
        )

        data = {
            "window_start": [pd.Timestamp("2025-03-11 00:00:00")] * len(index),
            "forecast_value": np.random.randint(1, 20, len(index)),  # Random forecast values
            "sort_col": np.tile([1, 2, 3, 4], len(generated_at_times))  # Ranks within each generated_at
        }

        self.df = pd.DataFrame(data, index=index)

    def test_charge_status_output(self):
        """Ensure the function outputs a charge_status column with correct type."""
        charge_status = report.simulate_charge(self.df, sort_col="sort_col", charge_mins=30)
        self.assertIsInstance(charge_status, pd.Series)
        self.assertTrue(charge_status.dtype == bool)

    def test_correct_length(self):
        """Check that the charge_status output has the correct length."""
        charge_status = report.simulate_charge(self.df, sort_col="sort_col", charge_mins=30)
        self.assertEqual(len(charge_status), len(self.df))

    def test_all_point_times_gte_generated_at(self):
        """Ensure all point_time values are greater than or equal to generated_at."""
        self.assertTrue((self.df.index.get_level_values("point_time") >= self.df.index.get_level_values("generated_at")).all())

    def test_charging_fully_schedules(self):
        """Verify that a full charge is scheduled correctly."""
        charge_status = report.simulate_charge(self.df, sort_col="sort_col", charge_mins=30)
        charge_needed = 20 // 5  # Number of charge periods required
        charged_windows = self.df.loc[charge_status].groupby("window_start").size()
        self.assertTrue((charged_windows >= charge_needed - 1).all())
        self.assertTrue((charged_windows <= charge_needed).all())
