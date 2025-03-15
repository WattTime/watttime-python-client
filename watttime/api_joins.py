from typing import Optional, Union, Literal
from datetime import datetime
import random
import math
from dateutil.parser import parse
from functools import cached_property

from dataclasses import dataclass

import pandas as pd

from .api import WattTimeHistorical, WattTimeForecast


@dataclass
class WattTimeAnalysisData:
    """
    This dataclass performs ETL on WattTime signals including
    co2_moer, co2_health_damages, and more. Forecasts of these underlying
    signals are also loaded and merged with the signal for analysis.

    The signal, its forecast, and the merged dataframe are lazily loaded when needed.

    Args:
        region (str): A single watttime region abbreviation
        model_date (str | None): A string representing a released model,
                                 if None, will default to latest provided through API
        signal_type (str): Such as `co2_moer`, `co2_aoer`, `co2_health_damages`,
        eval_start (str or datetime): The date to begin pulling signal data for analysis
        eval_end (str or datetime): The date to stop pulling signal data for analysis
        forecast_sample_perc (float): A percentage of days of forecasts to sample in
        order to speed up data retrieval and analysis (between eval_start and eval_end).
        A value of 1.0 will get all forecasts between eval_start and eval_end.
        forecast_max_horizon (int): The maximum forecast horizon to pull in minutes
        wt_forecast (WattTimeForecast): A WattTimeForecast API accessor class
            may be passed in if you wish to handle login variables or other parameters.
            Otherwise, credentials are loaded from the environment variables.
        wt_hist (WattTimeHistorical): A WattTimeHistorical API accessor class
            may be passed in if you wish to handle login variables or other parameters.
            Otherwise, credentials are loaded from the environment variables.
    """

    region: str
    eval_start: Union[datetime, str]
    eval_end: Union[datetime, str]
    forecast_sample_perc: float = 1.0
    forecast_sample_seed: int = 42
    forecast_max_horizon: int = 60 * 24
    signal_type: Literal["co2_moer", "co2_aoer", "health_damage"] = ("co2_moer",)
    model_date: Optional[str] = None  # Default to latest provided through the API
    wt_forecast: Optional[WattTimeForecast] = WattTimeForecast(multithreaded=True)
    wt_hist: Optional[WattTimeHistorical] = WattTimeHistorical()

    def __post_init__(self):
        self.region = self.region.upper()

        if isinstance(self.eval_start, str):
            self.eval_start = parse(self.eval_start)
        if isinstance(self.eval_end, str):
            self.eval_end = parse(self.eval_end)
        self.eval_days = list(
            pd.date_range(start=self.eval_start, end=self.eval_end, freq="1D")
        )

        if self.forecast_sample_perc >= 1.0:
            self.sample_days = self.eval_days
        else:
            n_sample_days = round(len(self.eval_days) * self.forecast_sample_perc)
            random.seed(self.forecast_sample_seed)
            self.sample_days = random.sample(self.eval_days, n_sample_days)

    @cached_property
    def moers(self) -> pd.DataFrame:
        moers = (
            self.wt_hist.get_historical_pandas(
                start=self.eval_start,
                # When merging into forecasts, get true moers through the last horizon
                end=self.eval_end + pd.Timedelta(minutes=self.forecast_max_horizon),
                region=self.region,
                signal_type=self.signal_type,
                model=self.model_date,
            )
            .rename(columns={"value": "signal_value"})
            .set_index("point_time", drop=True)
        )

        self.returned_meta = self.wt_hist._last_request_meta
        self.returned_hist_model_date = self.wt_hist._last_request_meta["model"]["date"]
        return moers

    @cached_property
    def forecasts(self) -> pd.DataFrame:

        forecasts = self.wt_forecast.get_historical_forecast_pandas_list(
            list_of_dates=self.sample_days,
            region=self.region,
            signal_type=self.signal_type,
            model=self.model_date,
            horizon_hours=math.ceil(self.forecast_max_horizon / 60),
        )

        forecasts["point_time"] = pd.to_datetime(forecasts["point_time"])
        forecasts["horizon_mins"] = (
            forecasts["point_time"] - forecasts["generated_at"]
        ).dt.total_seconds() / 60
        forecasts.rename({"value": "predicted_value"}, axis="columns", inplace=True)
        self.returned_meta = self.wt_forecast._last_request_meta
        self.returned_forecast_model_date = self.wt_forecast._last_request_meta[
            "model"
        ]["date"]
        return forecasts

    @cached_property
    def forecast_v_moer(self) -> pd.DataFrame:
        return self.forecasts.merge(
            self.moers,
            how="inner",
            left_on="point_time",
            right_on="point_time",
        ).set_index(["generated_at", "point_time"], drop=True)
