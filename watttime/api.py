import os
import time
import math
import math
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import requests
from dateutil.parser import parse
from pytz import UTC, timezone
from watttime.optimizer.alg import optCharger, moer
from watttime.optimizer.alg import optCharger, moer


class WattTimeBase:
    url_base = "https://api.watttime.org"

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initializes a new instance of the class.

        Parameters:
            username (Optional[str]): The username to use for authentication. If not provided, the value will be retrieved from the environment variable "WATTTIME_USER".
            password (Optional[str]): The password to use for authentication. If not provided, the value will be retrieved from the environment variable "WATTTIME_PASSWORD".
        """
        self.username = username or os.getenv("WATTTIME_USER")
        self.password = password or os.getenv("WATTTIME_PASSWORD")
        self.token = None
        self.token_valid_until = None

    def _login(self):
        """
        Login to the WattTime API, which provides a JWT valid for 30 minutes

        Raises:
            Exception: If the login fails and the credentials are incorrect.
        """

        url = f"{self.url_base}/login"
        rsp = requests.get(
            url,
            auth=requests.auth.HTTPBasicAuth(self.username, self.password),
            timeout=20,
        )
        rsp.raise_for_status()
        self.token = rsp.json().get("token", None)
        self.token_valid_until = datetime.now() + timedelta(minutes=30)
        if not self.token:
            raise Exception("failed to log in, double check your credentials")

    def _is_token_valid(self) -> bool:
        if not self.token_valid_until:
            return False
        return self.token_valid_until > datetime.now()

    def _parse_dates(
        self, start: Union[str, datetime], end: Union[str, datetime]
    ) -> Tuple[datetime, datetime]:
        """
        Parse the given start and end dates.

        Args:
            start (Union[str, datetime]): The start date to parse. It can be either a string or a datetime object.
            end (Union[str, datetime]): The end date to parse. It can be either a string or a datetime object.

        Returns:
            Tuple[datetime, datetime]: A tuple containing the parsed start and end dates as datetime objects.
        """
        if isinstance(start, str):
            start = parse(start)
        if isinstance(end, str):
            end = parse(end)

        if start.tzinfo:
            start = start.astimezone(UTC)
        else:
            start = start.replace(tzinfo=UTC)

        if end.tzinfo:
            end = end.astimezone(UTC)
        else:
            end = end.replace(tzinfo=UTC)

        return start, end

    def _get_chunks(
        self, start: datetime, end: datetime, chunk_size: timedelta = timedelta(days=30)
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate a list of tuples representing chunks of time within a given time range.

        Args:
            start (datetime): The start datetime of the time range.
            end (datetime): The end datetime of the time range.
            chunk_size (timedelta, optional): The size of each chunk. Defaults to timedelta(days=30).

        Returns:
            List[Tuple[datetime, datetime]]: A list of tuples representing the chunks of time.
        """
        chunks = []
        while start < end:
            chunk_end = min(end, start + chunk_size)
            chunks.append((start, chunk_end))
            start = chunk_end

        # API response is inclusive, avoid overlap in chunks
        chunks = [(s, e - timedelta(minutes=5)) for s, e in chunks[0:-1]] + [chunks[-1]]
        return chunks

    def register(self, email: str, organization: Optional[str] = None) -> None:
        """
        Register a user with the given email and organization.

        Parameters:
            email (str): The email of the user.
            organization (Optional[str], optional): The organization the user belongs to. Defaults to None.

        Returns:
            None: An error will be raised if registration was unsuccessful.
        """

        url = f"{self.url_base}/register"
        params = {
            "username": self.username,
            "password": self.password,
            "email": email,
            "org": organization,
        }

        rsp = requests.post(url, json=params, timeout=20)
        rsp.raise_for_status()
        print(
            f"Successfully registered {self.username}, please check {email} for a verification email"
        )

    @cache
    def region_from_loc(
        self,
        latitude: Union[str, float],
        longitude: Union[str, float],
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
    ) -> Dict[str, str]:
        """
        Retrieve the region information based on the given latitude and longitude.

        Args:
            latitude (Union[str, float]): The latitude of the location.
            longitude (Union[str, float]): The longitude of the location.
            signal_type (Optional[Literal["co2_moer", "co2_aoer", "health_damage"]], optional):
                The type of signal to be used for the region classification.
                Defaults to "co2_moer".

        Returns:
            Dict[str, str]: A dictionary containing the region information with keys "region" and "region_full_name".
        """
        if not self._is_token_valid():
            self._login()
        url = f"{self.url_base}/v3/region-from-loc"
        headers = {"Authorization": "Bearer " + self.token}
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "signal_type": signal_type,
        }
        rsp = requests.get(url, headers=headers, params=params)
        rsp.raise_for_status()
        return rsp.json()


class WattTimeHistorical(WattTimeBase):
    def get_historical_jsons(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
    ) -> List[dict]:
        """
        Base function to scrape historical data, returning a list of .json responses.

        Args:
            start (datetime): inclusive start, with a UTC timezone.
            end (datetime): inclusive end, with a UTC timezone.
            region (str): string, accessible through the /my-access endpoint, or use the free region (CAISO_NORTH)
            signal_type (str, optional): one of ['co2_moer', 'co2_aoer', 'health_damage']. Defaults to "co2_moer".
            model (Optional[Union[str, date]], optional): Optionally provide a model, used for versioning models.
                Defaults to None.

        Raises:
            Exception: Scraping failed for some reason

        Returns:
            List[dict]: A list of dictionary representations of the .json response object
        """
        if not self._is_token_valid():
            self._login()
        url = "{}/v3/historical".format(self.url_base)
        headers = {"Authorization": "Bearer " + self.token}
        responses = []
        params = {"region": region, "signal_type": signal_type}

        start, end = self._parse_dates(start, end)
        chunks = self._get_chunks(start, end)

        # No model will default to the most recent model version available
        if model is not None:
            params["model"] = model

        for c in chunks:
            params["start"], params["end"] = c
            rsp = requests.get(url, headers=headers, params=params)
            try:
                rsp.raise_for_status()
                j = rsp.json()
                responses.append(j)
            except Exception as e:
                raise Exception(
                    f"\nAPI Response Error: {rsp.status_code}, {rsp.text} [{rsp.headers.get('x-request-id')}]"
                )

            if len(j["meta"]["warnings"]):
                print("\n", "Warnings Returned:", params, j["meta"])

        # the API should not let this happen, but ensure for sanity
        unique_models = set([r["meta"]["model"]["date"] for r in responses])
        chosen_model = model or max(unique_models)
        if len(unique_models) > 1:
            responses = [
                r for r in responses if r["meta"]["model"]["date"] == chosen_model
            ]

        return responses

    def get_historical_pandas(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
        include_meta: bool = False,
    ):
        """
        Return a pd.DataFrame with point_time, and values.

        Args:
            See .get_hist_jsons() for shared arguments.
            include_meta (bool, optional): adds additional columns to the output dataframe,
                containing the metadata information. Note that metadata is returned for each API response,
                not for each point_time.

        Returns:
            pd.DataFrame: _description_
        """
        responses = self.get_historical_jsons(start, end, region, signal_type, model)
        df = pd.json_normalize(
            responses, record_path="data", meta=["meta"] if include_meta else []
        )

        df["point_time"] = pd.to_datetime(df["point_time"])

        return df

    def get_historical_csv(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
    ):
        """
        Retrieves historical data from a specified start date to an end date and saves it as a CSV file.
        CSV naming scheme is like "CAISO_NORTH_co2_moer_2022-01-01_2022-01-07.csv"

        Args:
            start (Union[str, datetime]): The start date for retrieving historical data. It can be a string in the format "YYYY-MM-DD" or a datetime object.
            end (Union[str, datetime]): The end date for retrieving historical data. It can be a string in the format "YYYY-MM-DD" or a datetime object.
            region (str): The region for which historical data is requested.
            signal_type (Optional[Literal["co2_moer", "co2_aoer", "health_damage"]]): The type of signal for which historical data is requested. Default is "co2_moer".
            model (Optional[Union[str, date]]): The date of the model for which historical data is requested. It can be a string in the format "YYYY-MM-DD" or a date object. Default is None.

        Returns:
            None, results are saved to a csv file in the user's home directory.
        """
        df = self.get_historical_pandas(start, end, region, signal_type, model)

        out_dir = Path.home() / "watttime_historical_csvs"
        out_dir.mkdir(exist_ok=True)

        start, end = self._parse_dates(start, end)
        fp = out_dir / f"{region}_{signal_type}_{start.date()}_{end.date()}.csv"
        df.to_csv(fp, index=False)
        print(f"file written to {fp}")


class WattTimeMyAccess(WattTimeBase):
    def get_access_json(self) -> Dict:
        """
        Retrieves the my-access/ JSON from the API, which provides information on
        the signal_types, regions, endpoints, and models that you have access to.

        Returns:
            Dict: The access JSON as a dictionary.

        Raises:
            Exception: If the token is not valid.
        """
        if not self._is_token_valid():
            self._login()
        url = "{}/v3/my-access".format(self.url_base)
        headers = {"Authorization": "Bearer " + self.token}
        rsp = requests.get(url, headers=headers)
        rsp.raise_for_status()
        return rsp.json()

    def get_access_pandas(self) -> pd.DataFrame:
        """
        Retrieves my-access data from a JSON source and returns it as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing access data with the following columns:
                - signal_type: The type of signal.
                - region: The abbreviation of the region.
                - region_name: The full name of the region.
                - endpoint: The endpoint.
                - model: The date identifier of the model.
                - Any additional columns from the model_dict.
        """
        j = self.get_access_json()
        out = []
        for sig_dict in j["signal_types"]:
            for reg_dict in sig_dict["regions"]:
                for end_dict in reg_dict["endpoints"]:
                    for model_dict in end_dict["models"]:
                        out.append(
                            {
                                "signal_type": sig_dict["signal_type"],
                                "region": reg_dict["region"],
                                "region_name": reg_dict["region_full_name"],
                                "endpoint": end_dict["endpoint"],
                                **model_dict,
                            }
                        )

        out = pd.DataFrame(out)
        out = out.assign(
            data_start=pd.to_datetime(out["data_start"]),
            train_start=pd.to_datetime(out["train_start"]),
            train_end=pd.to_datetime(out["train_end"]),
        )

        return out


class WattTimeForecast(WattTimeBase):
    def get_forecast_json(
        self,
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
        horizon_hours: int = 24,
    ) -> Dict:
        """
        Retrieves the most recent forecast data in JSON format based on the given region, signal type, and model date.
        This endpoint does not accept start and end as parameters, it only returns the most recent data!
        To access historical data, use the /v3/forecast/historical endpoint.
        https://docs.watttime.org/#tag/GET-Forecast/operation/get_historical_forecast_v3_forecast_historical_get

        Args:
            region (str): The region for which forecast data is requested.
            signal_type (str, optional): The type of signal to retrieve forecast data for. Defaults to "co2_moer".
                Valid options are "co2_moer", "co2_aoer", and "health_damage".
            model (str or date, optional): The date of the model version to use for the forecast data.
                If not provided, the most recent model version will be used.
            horizon_hours (int, optional): The number of hours to forecast. Defaults to 24. Minimum of 0 provides a "nowcast" created with the forecast, maximum of 72.

        Returns:
            List[dict]: A list of dictionaries representing the forecast data in JSON format.
        """
        if not self._is_token_valid():
            self._login()
        params = {
            "region": region,
            "signal_type": signal_type,
            "horizon_hours": horizon_hours,
        }

        # No model will default to the most recent model version available
        if model is not None:
            params["model"] = model

        url = "{}/v3/forecast".format(self.url_base)
        headers = {"Authorization": "Bearer " + self.token}
        rsp = requests.get(url, headers=headers, params=params)
        rsp.raise_for_status()
        return rsp.json()

    def get_forecast_pandas(
        self,
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
        include_meta: bool = False,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Return a pd.DataFrame with point_time, and values.

        Args:
            See .get_forecast_json() for shared arguments.
            include_meta (bool, optional): adds additional columns to the output dataframe,
                containing the metadata information. Note that metadata is returned for each API response,
                not for each point_time.

        Returns:
            pd.DataFrame: _description_
        """
        j = self.get_forecast_json(region, signal_type, model, horizon_hours)
        return pd.json_normalize(
            j, record_path="data", meta=["meta"] if include_meta else []
        )

    def get_historical_forecast_json(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
        horizon_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the historical forecast data from the API as a list of dictionaries.

        Args:
            start (Union[str, datetime]): The start date of the historical forecast. Can be a string or a datetime object.
            end (Union[str, datetime]): The end date of the historical forecast. Can be a string or a datetime object.
            region (str): The region for which to retrieve the forecast data.
            signal_type (Optional[Literal["co2_moer", "co2_aoer", "health_damage"]]): The type of signal to retrieve. Defaults to "co2_moer".
            model (Optional[Union[str, date]]): The date of the model version to use. Defaults to None.
            horizon_hours (int, optional): The number of hours to forecast. Defaults to 24. Minimum of 0 provides a "nowcast" created with the forecast, maximum of 72.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the forecast data.

        Raises:
            Exception: If there is an API response error.
        """
        if not self._is_token_valid():
            self._login()
        url = "{}/v3/forecast/historical".format(self.url_base)
        headers = {"Authorization": "Bearer " + self.token}
        responses = []
        params = {
            "region": region,
            "signal_type": signal_type,
            "horizon_hours": horizon_hours,
        }

        start, end = self._parse_dates(start, end)
        chunks = self._get_chunks(start, end, chunk_size=timedelta(days=1))

        # No model will default to the most recent model version available
        if model is not None:
            params["model"] = model

        for c in chunks:
            params["start"], params["end"] = c
            rsp = requests.get(url, headers=headers, params=params)
            try:
                rsp.raise_for_status()
                j = rsp.json()
                responses.append(j)
            except Exception as e:
                raise Exception(
                    f"\nAPI Response Error: {rsp.status_code}, {rsp.text} [{rsp.headers.get('x-request-id')}]"
                )

            if len(j["meta"]["warnings"]):
                print("\n", "Warnings Returned:", params, j["meta"])
            time.sleep(1)  # avoid rate limit

        return responses

    def get_historical_forecast_pandas(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Retrieves the historical forecast data as a pandas DataFrame.

        Args:
            start (Union[str, datetime]): The start date or datetime for the historical forecast.
            end (Union[str, datetime]): The end date or datetime for the historical forecast.
            region (str): The region for which the historical forecast data is retrieved.
            signal_type (Optional[Literal["co2_moer", "co2_aoer", "health_damage"]], optional):
                The type of signal for the historical forecast data. Defaults to "co2_moer".
            model (Optional[Union[str, date]], optional): The model date for the historical forecast data. Defaults to None.
            horizon_hours (int, optional): The number of hours to forecast. Defaults to 24. Minimum of 0 provides a "nowcast" created with the forecast, maximum of 72.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the historical forecast data.
        """
        json_list = self.get_historical_forecast_json(
            start, end, region, signal_type, model, horizon_hours
        )
        out = pd.DataFrame()
        for json in json_list:
            for entry in json["data"]:
                _df = pd.json_normalize(entry, record_path=["forecast"])
                _df = _df.assign(generated_at=pd.to_datetime(entry["generated_at"]))
                out = pd.concat([out, _df])
        return out

OPT_INTERVAL = 5
MAX_PREDICTION_HOURS = 72

class WattTimeOptimizer(WattTimeForecast):
    """
    This class inherits from WattTimeForecast, with additional methods to generate
    optimal usage plans for energy consumption based on various parameters and
    constraints.

    Additional Methods:
    --------
    get_optimal_usage_plan(region, usage_window_start, usage_window_end,
                           usage_time_required_minutes, usage_power_kw,
                           usage_time_uncertainty_minutes, optimization_method,
                           moer_data_override)
        Generates an optimal usage plan for energy consumption.
    """

    def get_optimal_usage_plan(
        self,
        region: str,
        usage_window_start: datetime,
        usage_window_end: datetime,
        usage_time_required_minutes: float,
        usage_power_kw: Union[int, float, pd.DataFrame],
        usage_time_uncertainty_minutes: Optional[float] = 0,
        optimization_method: Optional[
            Literal["baseline", "simple", "sophisticated", "auto"]
        ] = "baseline",
        moer_data_override: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generates an optimal usage plan for energy consumption based on given parameters.

        This method calculates the most efficient energy usage schedule within a specified
        time window, considering factors such as regional data, power requirements, and
        optimization methods.

        Parameters:
        -----------
        region : str
            The region for which forecast data is requested.
        usage_window_start : datetime
            Start time of the window when power consumption is allowed.
        usage_window_end : datetime
            End time of the window when power consumption is allowed.
        usage_time_required_minutes : float
            Required usage time in minutes.
        usage_power_kw : Union[int, float, pd.DataFrame]
            Power usage in kilowatts. Can be a constant value or a DataFrame for variable power.
        usage_time_uncertainty_minutes : Optional[float], default=0
            Uncertainty in usage time, in minutes.
        optimization_method : Optional[Literal["baseline", "simple", "sophisticated", "auto"]], default="baseline"
            The method used for optimization.
        moer_data_override : Optional[pd.DataFrame], default=None
            Pre-generated MOER (Marginal Operating Emissions Rate) DataFrame, if available.

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the optimal usage plan, including columns for
            predicted MOER, usage, CO2 emissions, and energy usage.

        Raises:
        -------
        AssertionError
            If input parameters do not meet specified conditions (e.g., timezone awareness,
            valid time ranges, supported optimization methods).

        Notes:
        ------
        - The method uses WattTime forecast data unless overridden by moer_data_override.
        - It supports various optimization methods and can handle both constant and variable power usage.
        - The resulting plan aims to minimize emissions while meeting the specified energy requirements.
        """
        def is_tz_aware(dt):
            return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

        assert is_tz_aware(usage_window_start), "Start time is not tz-aware"
        assert is_tz_aware(usage_window_end), "End time is not tz-aware"
        # Perform these checks if we are using live data
        if moer_data_override is None:
            datetime_now = datetime.now(UTC)
            assert (
                usage_window_end > datetime_now
            ), "Error, Window end is before current datetime"
            assert usage_window_end - datetime_now < timedelta(
                hours=MAX_PREDICTION_HOURS
            ), "End time is too far in the future"
        assert optimization_method in ("baseline", "simple", "sophisticated", "auto"), (
            "Unsupported optimization method:" + optimization_method
        )

        if moer_data_override is None:
            forecast_df = self.get_forecast_pandas(
                region=region,
                signal_type="co2_moer",
                horizon_hours=MAX_PREDICTION_HOURS,
            )
        else:
            forecast_df = moer_data_override.copy()
        forecast_df = forecast_df.set_index("point_time")
        forecast_df.index = pd.to_datetime(forecast_df.index)

        relevant_forecast_df = forecast_df[usage_window_start:usage_window_end]
        relevant_forecast_df = relevant_forecast_df.rename(
            columns={"value": "pred_moer"}
        )
        result_df = relevant_forecast_df[["pred_moer"]]
        moer_values = relevant_forecast_df["pred_moer"].values

        m = moer.Moer(
            mu=moer_values,
            isDiagonal=True,
            sig2=0.0,
        )

        model = optCharger.OptCharger(
            fixedChargeRate=1,
        )

        total_charge_units = usage_time_required_minutes // OPT_INTERVAL
        if optimization_method == "sophisticated":
            # Give a buffer time equal to the uncertainty
            buffer_time = usage_time_uncertainty_minutes
            buffer_periods = int(math.ceil(buffer_time / OPT_INTERVAL))
            # TODO: Check if there is any off-by-1 error here
            buffer_enforce_time = max(
                total_charge_units, len(moer_values) - buffer_periods
            )
            constraints = {buffer_enforce_time: (total_charge_units, None)}
        else:
            constraints = {}

        if type(usage_power_kw) in (int, float):
            # Convert to the MWh used in an optimization interval
            # expressed as a function to meet the parameter requirements for OptC function
            emission_multiplier_fn = (
                lambda sc, ec: float(usage_power_kw) * 0.001 * OPT_INTERVAL / 60.0
            )
        else:
            usage_power_kw["time_step"] = usage_power_kw["time"] / OPT_INTERVAL
            usage_power_kw_new_index = pd.DataFrame(
                index=list([float(x) for x in range(total_charge_units + 1)])
            )
            usage_power_kw = pd.merge_asof(
                usage_power_kw_new_index,
                usage_power_kw.set_index("time_step"),
                left_index=True,
                right_index=True,
                direction="backward",
                allow_exact_matches=True,
            )

            def emission_multiplier_fn(sc: float, ec: float) -> float:
                """
                Calculate the energy used for a given time range in the charging schedule.

                This gives us the MWh used per OPT_INTERVAL.

                Parameters:
                -----------
                sc : float
                    Start of the time range (in optimizer time units).
                ec : float
                    End of the time range (in optimizer time units).

                Returns:
                --------
                float
                    Energy used for a given time range

                """
                value = (
                    usage_power_kw[sc : max(sc, ec - 1e-12)]["power_kw"].mean()
                    * 0.001
                    * OPT_INTERVAL
                    / 60.0
                )
                return value

        model.fit(
            totalCharge=total_charge_units,
            totalTime=len(moer_values),
            moer=m,
            emission_multiplier_fn=emission_multiplier_fn,
            constraints=constraints,
            optimization_method=optimization_method,
        )

        optimizer_result = model.get_schedule()
        result_df["usage"] = [x * float(OPT_INTERVAL) for x in optimizer_result]
        result_df["emissions_co2e_lb"] = model.get_charging_emissions_over_time()
        result_df["energy_usage_mwh"] = model.get_energy_usage_over_time()

        return result_df


class WattTimeMaps(WattTimeBase):
    def get_maps_json(
        self,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
    ):
        """
        Retrieves JSON data for the maps API.

        Args:
            signal_type (Optional[str]): The type of signal to retrieve data for.
                Valid options are "co2_moer", "co2_aoer", and "health_damage".
                Defaults to "co2_moer".

        Returns:
            dict: The JSON response from the API.
        """
        if not self._is_token_valid():
            self._login()
        url = "{}/v3/maps".format(self.url_base)
        headers = {"Authorization": "Bearer " + self.token}
        params = {"signal_type": signal_type}
        rsp = requests.get(url, headers=headers, params=params)
        rsp.raise_for_status()
        return rsp.json()

class RecalculatingWattTimeOptimizer:
    def __init__(
        self, 
        watttime_username: str, 
        watttime_password: str,
        region: str, usage_time_required_minutes: float, 
        usage_power_kw:  Union[int, float, pd.DataFrame], 
        optimization_method: Optional[
            Literal["baseline", "simple", "sophisticated", "auto"]
        ],
    ) -> None:
        # Settings that stay consistent across calls to get_optimal_usage_plan
        self.region = region
        self.total_time_required = usage_time_required_minutes
        self.usage_power_kw = usage_power_kw
        self.optimization_method = optimization_method
        # Setup for us to track schedule/usage
        self.all_schedules = [] # (schedule, ctx)
        self.remaining_time_required = usage_time_required_minutes

        # Set up to query for fcsts
        self.forecast_generator = WattTimeForecast(watttime_username, watttime_password)
        self.wt_opt = WattTimeOptimizer(watttime_username, watttime_password)

        # Set up to query for actual data
        self.wt_hist = WattTimeHistorical(watttime_username, watttime_password)


    def get_new_schedule(self, new_start_time: datetime, new_end_time: datetime, curr_fcst_data: pd.DataFrame=None) -> pd.DataFrame:
        if curr_fcst_data is None:
            # Get new data
            curr_fcst_data = self.forecast_generator.get_historical_forecast_pandas(
                start=new_start_time - timedelta(minutes=OPT_INTERVAL), 
                end=new_start_time,
                region=self.region,
                signal_type="co2_moer",
                horizon_hours=MAX_PREDICTION_HOURS,
            )
        curr_fcst_data["point_time"] = pd.to_datetime(curr_fcst_data["point_time"])
        new_schedule_start_time = curr_fcst_data["point_time"].iloc[0]
        
        # If there a previously produced schedule, assume we followed that schedule until getting the new one
        if len(self.all_schedules) > 0:
            # Set end time of last ctx
            schedule, ctx = self.all_schedules[-1]
            self.all_schedules[-1] = (schedule, (ctx[0], new_schedule_start_time))
            assert ctx[0] < new_schedule_start_time

            # Calculate remaining time required
            usage = int(schedule[schedule.index < new_schedule_start_time]["usage"].sum())
            self.remaining_time_required -= usage

        # Generate new schedule
        new_schedule = self.wt_opt.get_optimal_usage_plan(
            self.region, 
            new_start_time - timedelta(minutes=OPT_INTERVAL), 
            new_end_time, 
            self.remaining_time_required, 
            self.usage_power_kw, 
            optimization_method=self.optimization_method,
            moer_data_override=curr_fcst_data,
        )
        new_schedule_ctx = (new_schedule_start_time, new_end_time)

        self.all_schedules.append((new_schedule, new_schedule_ctx))
        return new_schedule

    def get_combined_schedule(self, end_time: datetime = None) -> pd.DataFrame:
        schedule_segments = []
        for s, ctx in self.all_schedules:
            schedule_segments.append(s[s.index < ctx[1]])
        combined_schedule =  pd.concat(schedule_segments)

        if end_time:
            # Only keep segments that complete before end_time
            last_segment_start_time = end_time + timedelta(minutes=OPT_INTERVAL)
            combined_schedule = combined_schedule[combined_schedule.index <= last_segment_start_time]
        return combined_schedule

    # def get_predicted_combined_schedule_cost(self, end_time: datetime = None) -> float:
    #     schedule = self.get_combined_schedule(end_time=end_time)
    #     return schedule["emissions_co2e_lb"].sum()
    
    # def get_actual_combined_schedule_cost(self, end_time: datetime = None) -> float:
    #     schedule = self.get_combined_schedule(end_time=end_time)
    #     hist_data = self.wt_hist.get_historical_pandas(
    #         start=schedule.index[0], 
    #         end=schedule.index[-1],
    #         region=self.region,
    #         signal_type="co2_moer",
    #     ).set_index("point_time")
    #     merged_data = pd.merge(schedule["energy_usage_mwh"], hist_data["value"], left_index=True, right_index=True, how="left")
    #     assert not merged_data.isnull().values.any()
    #     return merged_data["energy_usage_mwh"].dot(merged_data["value"])
