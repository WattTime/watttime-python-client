
import inspect
import math
import random
import os
import hashlib
import pickle
from functools import lru_cache

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from itertools import product
from typing import Dict, List, Optional, Union
from zoneinfo import ZoneInfo
from shapely.geometry import shape
from timezonefinder import TimezoneFinder

import pandas as pd
from dateutil.parser import parse
from watttime import api


def remap_model_date(
    signal_type: str, step_type: str, requested_model_date: Optional[str]
) -> Optional[str]:
    if len(signal_type) == 11 and 'tail' in signal_type and requested_model_date == "2023-03-01":
        return "2023-01-21"
    return requested_model_date

@lru_cache
def get_tz_from_centroid(region):
    wt_maps = api.WattTimeMaps()
    all_maps = wt_maps.get_maps_json()
    region = {f["properties"]["region"]: f["geometry"] for f in all_maps["features"]}[
        region
    ]
    centroid = shape(region).centroid
    tz = TimezoneFinder().certain_timezone_at(lat=centroid.y, lng=centroid.x)
    return tz

def localize_dataframe_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """
    Localize or convert all DatetimeIndex levels and columns named 'generated_at' or 'point_time' to the given timezone.
    """

    def localize(dtobj):
        # For DatetimeIndex
        if isinstance(dtobj, pd.DatetimeIndex):
            if dtobj.tz is None:
                return dtobj.tz_localize(tz)
            return dtobj.tz_convert(tz)
        # For Series with .dt accessor
        elif isinstance(dtobj, pd.Series) and pd.api.types.is_datetime64_any_dtype(dtobj):
            if dtobj.dt.tz is None:
                return dtobj.dt.tz_localize(tz)
            return dtobj.dt.tz_convert(tz)
        return dtobj

    # Localize index levels
    if isinstance(df.index, pd.MultiIndex):
        new_levels = [localize(level) if isinstance(level, pd.DatetimeIndex) else level for level in df.index.levels]
        df.index = pd.MultiIndex.from_arrays([
            new_levels[i][df.index.codes[i]] for i in range(len(df.index.levels))
        ], names=df.index.names)
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = localize(df.index)

    # Localize columns if present
    for col in ["generated_at", "point_time"]:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = localize(df[col])

    return df

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
    forecast_sample_size: Union[int, float] = 0.03
    forecast_max_horizon: int = 60 * 24
    forecast_sample_seed: int = 42
    forecast_data_horizon_days: int = 3
    signal_type: str = "co2_moer"
    model_date: Optional[str] = None  # if None, will default to latest provided through API
    wt_forecast: Optional[api.WattTimeForecast] = api.WattTimeForecast(multithreaded=True)
    wt_hist: Optional[api.WattTimeHistorical] = api.WattTimeHistorical(multithreaded=True)
    wt_fuel_mix: Optional[api.WattTimeMarginalFuelMix] = api.WattTimeMarginalFuelMix(multithreaded=False, rate_limit=1)
    use_disk_cache: bool = True
    def _get_cache_path(self, kind: str, params: dict) -> str:
        """
        Generate a unique cache file path in /tmp/watttime_cache/ for the given kind and parameters.
        """
        cache_dir = "/tmp/watttime_cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Use sorted keys for deterministic hash
        key_str = f"{kind}_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return os.path.join(cache_dir, f"{kind}_{key_hash}.pkl")

    def _load_or_compute(self, kind: str, params: dict, compute_fn):
        """
        Load a DataFrame from disk if present, otherwise compute and cache it.
        """
        if not self.use_disk_cache:
            return compute_fn()
        cache_path = self._get_cache_path(kind, params)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                # If loading fails, fall back to recompute
                pass
        df = compute_fn()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
        except Exception:
            pass
        return df

    def __post_init__(self):
        self.region = self.region.upper()
        self.tz = get_tz_from_centroid(self.region)

        # parse eval_days
        # Ensure eval_start and eval_end are datetime before timezone logic
        if isinstance(self.eval_start, str):
            self.eval_start = parse(self.eval_start)
        if isinstance(self.eval_end, str):
            self.eval_end = parse(self.eval_end)
        assert self.eval_start < self.eval_end

        # localize if no timezone info
        if getattr(self.eval_start, 'tzinfo', None) is None:
            self.eval_start = self.eval_start.replace(tzinfo=ZoneInfo(self.tz))
        if getattr(self.eval_end, 'tzinfo', None) is None:
            self.eval_end = self.eval_end.replace(tzinfo=ZoneInfo(self.tz))

        # convert tz if not matching
        if getattr(self.eval_start.tzinfo, "key", None) != self.tz:
            self.eval_start = self.eval_start.astimezone(ZoneInfo(self.tz))
        if getattr(self.eval_end.tzinfo, "key", None) != self.tz:
            self.eval_end = self.eval_end.astimezone(ZoneInfo(self.tz))

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
        sample_days_extended = set()
        for day in self.sample_days:
            sample_days_extended = sample_days_extended | set(
                pd.date_range(
                    start=day, periods=self.forecast_data_horizon_days, freq="1D"
                )
            )
        self.sample_days = list(sample_days_extended)

    @cached_property
    def moers(self) -> pd.DataFrame:
        model = remap_model_date(self.signal_type, "signal", self.model_date)
        params = {
            "region": self.region,
            "eval_start": str(self.eval_start),
            "eval_end": str(self.eval_end),
            "signal_type": self.signal_type,
            "model_date": model,
            "forecast_max_horizon": self.forecast_max_horizon,
        }
        def compute():
            moers = (
                self.wt_hist.get_historical_pandas(
                    start=self.eval_start,
                    end=self.eval_end + pd.Timedelta(minutes=self.forecast_max_horizon),
                    region=self.region,
                    signal_type=self.signal_type,
                    model=model,
                )
                .rename(columns={"value": "signal_value"})
                .set_index("point_time", drop=True)
            )
            moers = moers.sort_index()
            moers = localize_dataframe_index(moers, self.tz)
            self.returned_moer_warnings = self.wt_hist.raised_warnings
            self.returned_meta = self.wt_hist._last_request_meta
            self.returned_hist_model_date = self.wt_hist._last_request_meta.get("model", {}).get("date", None)
            return moers
        return self._load_or_compute("moers", params, compute)

    @cached_property
    def forecasts(self) -> pd.DataFrame:
        model = remap_model_date(self.signal_type, "forecast", self.model_date)
        params = {
            "region": self.region,
            "sample_days": str(sorted(self.sample_days)),
            "signal_type": self.signal_type,
            "model_date": model,
            "forecast_max_horizon": self.forecast_max_horizon,
        }
        def compute():
            forecasts = self.wt_forecast.get_historical_forecast_pandas_list(
                list_of_dates=self.sample_days,
                region=self.region,
                signal_type=self.signal_type,
                model=model,
                horizon_hours=math.ceil(self.forecast_max_horizon / 60),
            )
            forecasts["point_time"] = pd.to_datetime(forecasts["point_time"])
            forecasts["horizon_mins"] = (
                forecasts["point_time"] - forecasts["generated_at"]
            ).dt.total_seconds() / 60
            forecasts.rename({"value": "predicted_value"}, axis="columns", inplace=True)
            forecasts = localize_dataframe_index(forecasts, self.tz)
            self.returned_forecast_warnings = self.wt_forecast.raised_warnings
            self.returned_meta = self.wt_forecast._last_request_meta
            self.returned_forecast_model_date = self.wt_forecast._last_request_meta.get("model", {}).get("date", None)
            return forecasts[['point_time', 'generated_at', 'horizon_mins', 'predicted_value']]
        return self._load_or_compute("forecasts", params, compute)

    @cached_property
    def fuel_mix(self) -> pd.DataFrame:
        params = {
            "region": self.region,
            "eval_start": str(self.eval_start),
            "eval_end": str(self.eval_end),
            "model_date": self.model_date,
            "forecast_max_horizon": self.forecast_max_horizon,
        }
        def compute():
            fm = self.wt_fuel_mix.get_fuel_mix_pandas(
                start=self.eval_start,
                end=self.eval_end + pd.Timedelta(minutes=self.forecast_max_horizon),
                region=self.region,
                signal_type="marginal_fuel_mix",
                model=self.model_date,
            )
            fm = localize_dataframe_index(fm, self.tz)
            self.returned_fuel_mix_warnings = self.wt_fuel_mix.raised_warnings
            self.returned_meta = self.wt_fuel_mix._last_request_meta
            self.returned_fuel_mix_model_date = self.wt_fuel_mix._last_request_meta.get("model", {}).get("date", None)
            return fm
        return self._load_or_compute("fuel_mix", params, compute)

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
        ).set_index(["generated_at", "point_time"], drop=True)[['horizon_mins', 'predicted_value', 'signal_value']]

    def forecast_v_moer_filtered_by_rank(
        self, n_rows, filter_on="signal_value", use_highest=False
    ):
        forecast_ranks = self.forecast_v_moer.groupby("generated_at")[filter_on].rank(
            method="first"
        )
        if use_highest:
            return self.forecast_v_moer[forecast_ranks >= n_rows]
        return self.forecast_v_moer[forecast_ranks <= n_rows]
    
    @cached_property
    def effective_forecast_sample_rate(self) -> float:
        df_reset_for_sampling = self.forecasts_v_moers.reset_index()
        generated_at_dates = df_reset_for_sampling["generated_at"].dt.normalize()
        unique_forecast_days = generated_at_dates.nunique()
        date_range_days = (generated_at_dates.max() - generated_at_dates.min()).days + 1
        return unique_forecast_days / date_range_days if date_range_days > 0 else 1.0


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
