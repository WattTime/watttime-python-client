import unittest
from datetime import datetime, date
from watttime import report
from pathlib import Path
from typing import List, Union
from itertools import product

BA_LIST: List[str] = ["WEM"]
MODEL_DATE_LIST: List[str] = ["2022-10-01", "2024-10-16"]
SIGNAL_TYPE: str = "co2_moer"
EVAL_START: Union[str, datetime, date] = "2024-01-01T00:00Z"
EVAL_END: Union[str, datetime, date] = "2024-03-01T00:00Z"
FORECAST_SAMPLE_DAYS = 3


class TestReport(unittest.TestCase):
    def setUp(self):
        self.eval_days = report.parse_eval_days(
            eval_start=EVAL_START, eval_end=EVAL_END
        )
        self.sample_days = report.parse_forecast_sample_days(
            eval_days=self.eval_days, forecast_sample_days=FORECAST_SAMPLE_DAYS
        )

        self.jobs = list(
            report.ModelAnalysis(
                ba=i[0],
                model_date=i[1],
                signal_type=SIGNAL_TYPE,
                eval_start=EVAL_START,
                eval_end=EVAL_END,
                eval_days=self.eval_days,
                sample_days=self.sample_days,
            )
            for i in product(BA_LIST, MODEL_DATE_LIST)
        )

        # Compile forecast_v_moer for each job
        for job in self.jobs:
            job.compile_forecast_v_moer()

    def test_compile_forecast_v_moer(self):
        for job in self.jobs:
            self.assertFalse(job.moers.empty)
            self.assertFalse(job.forecasts.empty)
            self.assertFalse(job.forecasts_v_moers.empty)

    def test_plot_sample_moers(self):
        fig = report.plot_sample_moers(self.jobs)
        self.assertGreater(len(fig.data), 0)

    def test_plot_distribution_moers(self):
        fig = report.plot_distribution_moers(self.jobs)
        self.assertGreater(len(fig.data), 0)

    def test_plot_ba_heatmaps(self):
        fig = report.plot_ba_heatmaps(self.jobs)
        self.assertGreater(len(fig.data), 0)

    def test_plot_norm_mae(self):
        fig = report.plot_norm_mae(self.jobs)
        self.assertGreater(len(fig.data), 0)

    def test_plot_rank_correlation(self):
        fig = report.plot_rank_correlation(self.jobs)
        self.assertGreater(len(fig.data), 0)

    def test_plot_impact_forecast_metrics(self):
        fig = report.plot_impact_forecast_metrics(self.jobs)
        self.assertGreater(len(fig.data), 0)

    def test_plot_sample_moers_single_job(self):
        fig = report.plot_sample_moers([self.jobs[0]])
        self.assertGreater(len(fig.data), 0)

    def test_plot_distribution_moers_single_job(self):
        fig = report.plot_distribution_moers([self.jobs[0]])
        self.assertGreater(len(fig.data), 0)

    def test_plot_ba_heatmaps_single_job(self):
        fig = report.plot_ba_heatmaps([self.jobs[0]])
        self.assertGreater(len(fig.data), 0)

    def test_plot_norm_mae_single_job(self):
        fig = report.plot_norm_mae([self.jobs[0]])
        self.assertGreater(len(fig.data), 0)

    def test_plot_rank_correlation_single_job(self):
        fig = report.plot_rank_correlation([self.jobs[0]])
        self.assertGreater(len(fig.data), 0)

    def test_plot_impact_forecast_metrics_single_job(self):
        fig = report.plot_impact_forecast_metrics([self.jobs[0]])
        self.assertGreater(len(fig.data), 0)
