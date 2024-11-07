from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class EvalMetrics:
    baseline_col: str
    ideal_col: str
    forecast_col: str
    actuals_col: str

    def percent_diff(self, y, x):
        return ((y - x) / x) * 100

    def _calculate_diff(self, results_frame: pd.DataFrame, col1: str, col2: str, percent: bool):
        diff = results_frame[col1] - results_frame[col2]
        return self.percent_diff(results_frame[col1], results_frame[col2]) if percent else diff

    def forecast_less_baseline(self, results_frame: pd.DataFrame, percent=False):
        return self._calculate_diff(results_frame, self.forecast_col, self.baseline_col, percent)

    def forecast_less_ideal(self, results_frame: pd.DataFrame, percent=False):
        return self._calculate_diff(results_frame, self.forecast_col, self.ideal_col, percent)

    def actuals_less_forecast(self, results_frame: pd.DataFrame, percent=False):
        return self._calculate_diff(results_frame, self.actuals_col, self.forecast_col, percent)

    def actuals_less_baseline(self, results_frame: pd.DataFrame, percent=False):
        return self._calculate_diff(results_frame, self.actuals_col, self.baseline_col, percent)

    def calculate_results(self, results_frame: pd.DataFrame, keep_cols: List[str], percent=False):
        columns = ["emissions_avoided", "expected_avoidance", "nearness_to_ideal", "forecast_error"]
        
        df = pd.DataFrame({
            columns[0]: self.actuals_less_baseline(results_frame, percent=percent),
            columns[1]: self.forecast_less_baseline(results_frame, percent=percent),
            columns[2]: self.forecast_less_ideal(results_frame, percent=percent),
            columns[3]: self.actuals_less_forecast(results_frame, percent=percent)
        })

        if percent:
            df.columns = [f"{col}_percent" for col in columns]

        return pd.concat([results_frame[keep_cols], df], axis=1)