from typing import Optional, List, Union, Dict
from datetime import datetime
import inspect
import warnings
import random
import math
from dateutil.parser import parse
from functools import cached_property
from itertools import product

from dataclasses import dataclass

import pandas as pd


from watttime import api


@dataclass
class AnalysisDataHandler:
    """
    This dataclass performs ETL on WattTime signals including
    co2_moer, co2_health_damages and more. Data is also loaded
    for forecasts of these underlying signals.

    Args:
        region (str): A single BA abbreviation
        model_date (str | None): A string representing a released model, if None, will default to latest provided through API
        signal_type (str): Such as `co2_moer`, `co2_aoer`, `co2_health_damages`
        eval_start (str or datetime): The starting point of the analysis to begin pulling signal data.
        eval_end (str or datetime): The ending point of the analysis to stop pulling signal data.
        forecast_sample_size (float or int): A float will represent a sample of days
            to pull forecasts across (between eval_start and eval_end) while an int
            greater than 1 will represent an absolute number of days. A value of 1
            will pull all forecasts between eval_start and eval_end, however
            this can be time consuming.
        forecast_max_horizon (int): The maximum forecast horizon to pull in minutes
        forecast_sample_seed (int): Used to ensure consistency in sample dates
            between multiple AnalysisDataHandler objects that share an eval_start
            and eval_end
        wt_forecast (api.WattTimeForecast): a WattTimeForecast API accessor class
            may be passed in if you wish to handle login variables or other parameters.
            Otherwise, these are loaded from the environment variables.
        wt_hist (api.WattTimeHistorical): a WattTimeHistorical API accessor class
            may be passed in if you wish to handle login variables or other parameters.
            Otherwise, these are loaded from the environment variables.

    Returns:
        _type_: _description_
    """

    region: str
    eval_start: Union[datetime, str]
    eval_end: Union[datetime, str]
    forecast_sample_size: Union[int, float] = 0.1
    forecast_max_horizon: int = 60 * 24
    forecast_sample_seed: int = 42
    signal_type: str = "co2_moer"
    model_date: Optional[
        str
    ] = None  # if None, will default to latest provided through API
    wt_forecast: Optional[api.WattTimeForecast] = api.WattTimeForecast(
        multithreaded=True
    )
    wt_hist: Optional[api.WattTimeHistorical] = api.WattTimeHistorical(
        multithreaded=True
    )
    wt_fuel_mix: Optional[api.WattTimeMarginalFuelMix] = api.WattTimeMarginalFuelMix(
        multithreaded=False, rate_limit=1
    )

    def __post_init__(self):
        self.region = self.region.upper()

        # parse eval_days
        if isinstance(self.eval_start, str):
            self.eval_start = parse(self.eval_start)
        if isinstance(self.eval_end, str):
            self.eval_end = parse(self.eval_end)
        assert self.eval_start < self.eval_end
        self.eval_days = list(
            pd.date_range(start=self.eval_start, end=self.eval_end, freq="1D")
        )

        # parse sample_days as int (day count), or float (fraction of eval_days)
        if isinstance(self.forecast_sample_size, float):
            k = round(len(self.eval_days) * self.forecast_sample_size)
        else:
            k = self.forecast_sample_size
        k = max(k, 1)
        random.seed(self.forecast_sample_seed)
        self.sample_days = random.sample(self.eval_days, k)

    @cached_property
    def moers(self) -> pd.DataFrame:
        moers = (
            self.wt_hist.get_historical_pandas(
                start=self.eval_start,
                end=self.eval_end
                + pd.Timedelta(
                    minutes=self.forecast_max_horizon
                ),  # include truth_values for final generated_at
                region=self.region,
                signal_type=self.signal_type,
                model=self.model_date,
            )
            .rename(columns={"value": "signal_value"})
            .set_index("point_time", drop=True)
        )

        moers = moers.sort_index()

        self.returned_moer_warnings = self.wt_hist.raised_warnings
        self.returned_meta = self.wt_hist._last_request_meta
        self.returned_hist_model_date = self.wt_hist._last_request_meta.get(
            "model", {}
        ).get("date", None)
        return moers

    @cached_property
    def forecasts(self) -> pd.DataFrame:
        # context manager to only return critical logs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

            self.returned_forecast_warnings = self.wt_forecast.raised_warnings
            self.returned_meta = self.wt_forecast._last_request_meta
            self.returned_forecast_model_date = self.wt_forecast._last_request_meta.get(
                "model", {}
            ).get("date", None)
            return forecasts

    @cached_property
    def fuel_mix(self) -> pd.DataFrame:
        fm = self.wt_fuel_mix.get_fuel_mix_pandas(
            start=self.eval_start,
            end=self.eval_end
            + pd.Timedelta(
                minutes=self.forecast_max_horizon
            ),  # include truth_values for final generated_at
            region=self.region,
            signal_type="marginal_fuel_mix",
            model=self.model_date,
        )

        self.returned_fuel_mix_warnings = self.wt_fuel_mix.raised_warnings
        self.returned_meta = self.wt_fuel_mix._last_request_meta
        self.returned_fuel_mix_model_date = self.wt_fuel_mix._last_request_meta.get(
            "model", {}
        ).get("date", None)
        return fm

    def moer_v_fuel_mix(self) -> pd.DataFrame:
        return self.moers.merge(
            self.fuel_mix,
            how="inner",
            left_on="point_time",
            right_on="point_time",
        ).set_index(["generated_at", "point_time"], drop=True)

    @cached_property
    def forecasts_v_moers(self) -> pd.DataFrame:
        return self.forecasts.merge(
            self.moers,
            how="inner",
            left_on="point_time",
            right_on="point_time",
        ).set_index(["generated_at", "point_time"], drop=True)

    def forecast_v_moer_filtered_by_rank(
        self, n_rows, filter_on="signal_value", use_highest=False
    ):
        forecast_ranks = self.forecast_v_moer.groupby("generated_at")[filter_on].rank(
            method="first"
        )
        if use_highest:
            return self.forecast_v_moer[forecast_ranks >= n_rows]
        return self.forecast_v_moer[forecast_ranks <= n_rows]


class DataHandlerFactory:
    """
    Create a list of AnalysisDataHandler objects for the permutations
    of regions, model_dates, and signal_types.

    Args:
        regions (List[str]): List of regions to analyze
        signal_types (List[str]): List of signal types to analyze
        model_dates (List[str]): List of model dates to analyze
    """

    def __init__(
        self,
        eval_start: Union[str, datetime],
        eval_end: Union[str, datetime],
        regions: Union[str, List[str]],
        signal_types: Union[str, List[str]] = "co2_moer",
        model_dates: List[str] = None,
        **kwargs,
    ):
        if isinstance(eval_start, str):
            eval_start = parse(eval_start)

        if isinstance(eval_end, str):
            eval_end = parse(eval_end)

        if not isinstance(regions, list):
            regions = [regions]

        if not isinstance(signal_types, list):
            signal_types = [signal_types]

        if not isinstance(model_dates, list):
            model_dates = [model_dates]

        permutations = list(product(regions, signal_types, model_dates))

        # only pass through relevant kwargs
        _args = inspect.signature(AnalysisDataHandler.__init__).parameters
        _kwargs = {k: v for k, v in kwargs.items() if k in _args}

        self.data_handlers = []
        for region, signal_type, model_date in permutations:
            dh = AnalysisDataHandler(
                eval_start=eval_start,
                eval_end=eval_end,
                region=region,
                signal_type=signal_type,
                model_date=model_date,
                **_kwargs,
            )
            self.data_handlers.append(dh)

    def yield_datahandlers(self) -> AnalysisDataHandler:
        for dh in self.data_handlers:
            yield dh

    @property
    def collected_model_meta(self):
        meta = []
        for dh in self.data_handlers:
            # Helper: safely convert warnings to list[dict]
            def serialize_warnings(warnings):
                if warnings is None:
                    return None
                return [w.to_dict() if hasattr(w, "to_dict") else w for w in warnings]

            # Build the warnings dict, excluding None values
            warnings = {
                k: serialize_warnings(v)
                for k, v in {
                    "moer": getattr(dh, "returned_moer_warnings", None),
                    "fuel_mix": getattr(dh, "returned_fuel_mix_warnings", None),
                    "forecast": getattr(dh, "returned_forecast_warnings", None),
                }.items()
                if v is not None
            }

            # Build the main meta dict, excluding None values and empty warnings
            m = {
                k: v
                for k, v in {
                    "region": dh.region,
                    "requested_model_date": dh.model_date,
                    "returned_hist_model_date": getattr(
                        dh, "returned_hist_model_date", None
                    ),
                    "returned_fuel_mix_model_date": getattr(
                        dh, "returned_fuel_mix_model_date", None
                    ),
                    "returned_forecast_model_date": getattr(
                        dh, "returned_forecast_model_date", None
                    ),
                    "returned_warnings": warnings if warnings else None,
                }.items()
                if v is not None
            }

            meta.append(m)
        return meta

    @property
    def data_handlers_by_region_dict(self) -> Dict[str, List[AnalysisDataHandler]]:
        unique_regions = list(set([dh.region for dh in self.data_handlers]))
        out = {
            region: [dh for dh in self.data_handlers if dh.region == region]
            for region in unique_regions
        }
        out = {
            k: sorted(v, key=lambda x: x.model_date, reverse=True)
            for k, v in out.items()
        }
        return out
