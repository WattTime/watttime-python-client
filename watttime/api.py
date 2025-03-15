import os
import time
import threading
import time
from datetime import date, datetime, timedelta, time as dt_time
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from dateutil.parser import parse
from pytz import UTC


class WattTimeBase:
    url_base = "https://api.watttime.org"

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        multithreaded: bool = False,
        rate_limit: int = 10,
    ):
        """
        Initializes a new instance of the class.

        Parameters:
            username (Optional[str]): The username to use for authentication. If not provided, the value will be retrieved from the environment variable "WATTTIME_USER".
            password (Optional[str]): The password to use for authentication. If not provided, the value will be retrieved from the environment variable "WATTTIME_PASSWORD".
        """
        self.username = username or os.getenv("WATTTIME_USER")
        self.password = password or os.getenv("WATTTIME_PASSWORD")
        self.token = None
        self.headers = None
        self.token_valid_until = None

        self.multithreaded = multithreaded
        self.rate_limit = rate_limit
        self._last_request_times = []

        if self.multithreaded:
            self._rate_limit_lock = (
                threading.Lock()
            )  # prevent multiple threads from modifying _last_request_times simultaneously
            self._rate_limit_condition = threading.Condition(self._rate_limit_lock)

        self.session = requests.Session()

    def _login(self):
        """
        Login to the WattTime API, which provides a JWT valid for 30 minutes

        Raises:
            Exception: If the login fails and the credentials are incorrect.
        """

        url = f"{self.url_base}/login"
        rsp = self.session.get(
            url,
            auth=requests.auth.HTTPBasicAuth(self.username, self.password),
            timeout=20,
        )
        rsp.raise_for_status()
        self.token = rsp.json().get("token", None)
        self.token_valid_until = datetime.now() + timedelta(minutes=30)
        if not self.token:
            raise Exception("failed to log in, double check your credentials")
        self.headers = {"Authorization": "Bearer " + self.token}

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

        rsp = self.session.post(url, json=params, timeout=20)
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
        url = f"{self.url_base}/v3/region-from-loc"
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "signal_type": signal_type,
        }
        j = self._make_rate_limited_request(url, params=params)
        return j

    def _make_rate_limited_request(self, url: str, params: Dict[str, Any]) -> Dict:
        """
        Makes a single API request while respecting the rate limit.
        """
        if not self._is_token_valid() or not self.headers:
            self._login()

        ts = time.time()

        # apply rate limiting by either sleeping (single thread) or
        # waiting on a condition ()
        if self.multithreaded:
            with self._rate_limit_condition:
                self._apply_rate_limit(ts)
        else:
            self._apply_rate_limit(ts)

        try:
            rsp = self.session.get(url, headers=self.headers, params=params, timeout=60)
            rsp.raise_for_status()
            j = rsp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"API Request Failed: {e}\nURL: {url}\nParams: {params}"
            ) from e

        if j.get("meta", {}).get("warnings"):
            print("Warnings Returned: %s | Response: %s", params, j["meta"])

        self._last_request_meta = j.get("meta", {})

        return j

    def _apply_rate_limit(self, ts: float):
        """
        Rate limiting not allowing more than self.rate_limit requests per second.

        This is applied by checking is `self._last_request_times` has more than self.rate_limit entries.
        If so, it will wait until the oldest entry is older than 1 second.

        If multithreading, waiting is achieved by setting a "condition" on the thread.
        If single threading, we sleep for the remaining time.
        """
        self._last_request_times = [t for t in self._last_request_times if ts - t < 1.0]

        if len(self._last_request_times) >= self.rate_limit:
            earliest_request_age = ts - self._last_request_times[0]
            wait_time = 1.0 - earliest_request_age
            if wait_time > 0:
                if self.multithreaded:
                    self._rate_limit_condition.wait(timeout=wait_time)
                else:
                    time.sleep(wait_time)

        self._last_request_times.append(time.time())

        if self.multithreaded:
            self._rate_limit_condition.notify_all()

    def _fetch_data(
        self,
        url: str,
        param_chunks: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[Dict]:
        """
        Base method for fetching data without multithreading.
        If you are making a single request, you can call _make_rate_limited_request directly.
        This class is suited for making a series of requests in a for loop, with
        varying `param_chunks`.
        """

        if isinstance(param_chunks, dict):
            param_chunks = [param_chunks]

        responses = []
        if self.multithreaded:
            with ThreadPoolExecutor(
                max_workers=min(10, (os.cpu_count() or 1) * 2)
            ) as executor:
                futures = {
                    executor.submit(
                        self._make_rate_limited_request, url, params
                    ): params
                    for params in param_chunks
                }

                for future in as_completed(futures):
                    responses.append(future.result())
        else:
            for params in param_chunks:
                rsp = self._make_rate_limited_request(url, params)
                responses.append(rsp)

        return responses


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
        url = "{}/v3/historical".format(self.url_base)
        params = {"region": region, "signal_type": signal_type}

        start, end = self._parse_dates(start, end)
        chunks = self._get_chunks(start, end)

        # No model will default to the most recent model version available
        if model is not None:
            params["model"] = model

        param_chunks = [{**params, "start": c[0], "end": c[1]} for c in chunks]
        responses = self._fetch_data(url, param_chunks)

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
        url = "{}/v3/my-access".format(self.url_base)
        return self._make_rate_limited_request(url, params={})

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
    def _parse_historical_forecast_json(
        self, json_list: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Parses the JSON response from the historical forecast API into a pandas DataFrame.

        Args:
            json_list (List[Dict[str, Any]]): A list of JSON responses from the API.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the parsed historical forecast data.
        """
        out = pd.DataFrame()
        for json in json_list:
            for entry in json.get("data", []):
                _df = pd.json_normalize(entry, record_path=["forecast"])
                _df = _df.assign(generated_at=pd.to_datetime(entry["generated_at"]))
                out = pd.concat([out, _df], ignore_index=True)
        return out

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
        params = {
            "region": region,
            "signal_type": signal_type,
            "horizon_hours": horizon_hours,
        }

        # No model will default to the most recent model version available
        if model is not None:
            params["model"] = model

        url = "{}/v3/forecast".format(self.url_base)
        return self._make_rate_limited_request(url, params)

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
        url = f"{self.url_base}/v3/forecast/historical"
        params = {
            "region": region,
            "signal_type": signal_type,
            "horizon_hours": horizon_hours,
        }

        start, end = self._parse_dates(start, end)
        chunks = self._get_chunks(start, end, chunk_size=timedelta(days=1))

        if model is not None:
            params["model"] = model

        param_chunks = [{**params, "start": c[0], "end": c[1]} for c in chunks]
        return self._fetch_data(url, param_chunks)

    def get_historical_forecast_json_list(
        self,
        list_of_dates: List[date],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
        horizon_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Fetches historical forecast data for a list of specific dates.

        Args:
            list_of_dates (List[date]): A list of dates to retrieve historical forecasts for.
            region (str): The region for which the forecast is needed.
            signal_type (Optional[str]): The type of signal ("co2_moer", "co2_aoer", or "health_damage").
            model (Optional[Union[str, date]]): Model version or date.
            horizon_hours (int): Forecast horizon in hours.

        Returns:
            List[Dict[str, Any]]: A list of JSON responses for each requested date.
        """

        url = f"{self.url_base}/v3/forecast/historical"
        params = {
            "region": region,
            "signal_type": signal_type,
            "horizon_hours": horizon_hours,
        }

        if model is not None:
            params["model"] = model

        param_chunks = [
            # add timezone to dates
            {
                **params,
                "start": datetime.combine(d, dt_time(0, 0)).isoformat() + "Z",
                "end": datetime.combine(d, dt_time(23, 59)).isoformat() + "Z",
            }
            for d in list_of_dates
        ]
        return self._fetch_data(url, param_chunks)

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
            signal_type (Optional[str]): The type of signal for the historical forecast data.
            model (Optional[Union[str, date]]): The model date for the historical forecast data.
            horizon_hours (int): The number of hours to forecast.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the historical forecast data.
        """
        json_list = self.get_historical_forecast_json(
            start, end, region, signal_type, model, horizon_hours
        )
        return self._parse_historical_forecast_json(json_list)

    def get_historical_forecast_pandas_list(
        self,
        list_of_dates: List[date],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model: Optional[Union[str, date]] = None,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Retrieves the historical forecast data for a list of specific dates as a pandas DataFrame.

        Args:
            list_of_dates (List[date]): A list of dates to retrieve historical forecasts for.
            region (str): The region for which the forecast is needed.
            signal_type (Optional[str]): The type of signal.
            model (Optional[Union[str, date]]): The model version or date.
            horizon_hours (int): Forecast horizon in hours.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the historical forecast data.
        """
        json_list = self.get_historical_forecast_json_list(
            list_of_dates, region, signal_type, model, horizon_hours
        )
        return self._parse_historical_forecast_json(json_list)


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

        url = "{}/v3/maps".format(self.url_base)
        params = {"signal_type": signal_type}
        return self._make_rate_limited_request(url, params)
