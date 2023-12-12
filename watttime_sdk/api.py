from datetime import datetime, timedelta, date
from dateutil.parser import parse
from pytz import timezone, UTC
from typing import List, Tuple, Dict, Union, Optional, Literal, Any
import os
import time

import requests
import pandas as pd


class WattTimeBase:
    url_base = "https://api.watttime.org"

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or os.getenv("WATTTIME_USER")
        self.password = password or os.getenv("WATTTIME_PASSWORD")
        self.token = None
        self.token_valid_until = None

    def _login(self):
        """Login to the WattTime API, which provides a JWT valid for 30 minutes."""
        url = f"{self.url_base}/login"
        rsp = requests.get(
            url,
            auth=requests.auth.HTTPBasicAuth(self.username, self.password),
            timeout=20,
        )
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
        """Internal function turns a start and end into 30-day chunks"""
        chunks = []
        while start < end:
            chunk_end = min(end, start + chunk_size)
            chunks.append((start, chunk_end))
            start = chunk_end

        # API response is inclusive, avoid overlap in chunks
        chunks = [(s, e - timedelta(minutes=5)) for s, e in chunks[0:-1]] + [chunks[-1]]
        return chunks


class WattTimeHistorical(WattTimeBase):
    def get_historical_jsons(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model_date: Optional[Union[str, date]] = None,
    ) -> List[dict]:
        """
        Base function to scrape historical data, returning a list of .json responses.

        Args:
            start (datetime): inclusive start, with a UTC timezone.
            end (datetime): inclusive end, with a UTC timezone.
            region (str): string, accessible through the /my-access endpoint, or use the free region (CAISO_NORTH)
            signal_type (str, optional): one of ['co2_moer', 'co2_aoer', 'health_damage']. Defaults to "co2_moer".
            model_date (Optional[Union[str, date]], optional): Optionally provide a model_date, used for versioning models.
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

        # No model_date will default to the most recent model version available
        if model_date is not None:
            params["model_date"] = model_date

        for c in chunks:
            params["start"], params["end"] = c
            rsp = requests.get(url, headers=headers, params=params)
            if rsp.status_code == 200:
                j = rsp.json()
                responses.append(j)
            else:
                raise Exception(f"\nAPI Response Error: {rsp.status_code}, {rsp.text}")

            if len(j["meta"]["warnings"]):
                print("\n", "Warnings Returned:", params, j["meta"])

        return responses

    def get_historical_pandas(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model_date: Optional[Union[str, date]] = None,
        include_meta: bool = False,
    ):
        """Return a pd.DataFrame with point_time, and values.

        Args:
            See .get_hist_jsons() for shared arguments.
            include_meta (bool, optional): adds additional columns to the output dataframe,
                containing the metadata information. Note that metadata is returned for each response,
                not for each point_time.

        Returns:
            pd.DataFrame: _description_
        """
        responses = self.get_historical_jsons(
            start, end, region, signal_type, model_date
        )
        df = pd.json_normalize(
            responses, record_path="data", meta=["meta"] if include_meta else []
        )
        return df

    def get_historical_csv(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model_date: Optional[Union[str, date]] = None,
    ):
        """
                Retrieves historical data from a specified start date to an end date and saves it as a CSV file.
                CSV naming scheme is like "CAISO_NORTH_co2_moer_2022-01-01_2022-01-07.csv"
        +
                Args:
                    start (Union[str, datetime]): The start date for retrieving historical data. It can be a string in the format "YYYY-MM-DD" or a datetime object.
                    end (Union[str, datetime]): The end date for retrieving historical data. It can be a string in the format "YYYY-MM-DD" or a datetime object.
                    region (str): The region for which historical data is requested.
                    signal_type (Optional[Literal["co2_moer", "co2_aoer", "health_damage"]]): The type of signal for which historical data is requested. Default is "co2_moer".
                    model_date (Optional[Union[str, date]]): The date of the model for which historical data is requested. It can be a string in the format "YYYY-MM-DD" or a date object. Default is None.

                Returns:
                    None
        """
        df = self.get_historical_pandas(start, end, region, signal_type, model_date)
        df.to_csv(
            f"{region}_{signal_type}_{start.date()}_{end.date()}.csv", index=False
        )


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
                - model_date: The date of the model.
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
                                "model_date": model_dict["model"],
                                **model_dict,
                            }
                        )

        return pd.DataFrame(out)


class WattTimeForecast(WattTimeBase):
    def get_forecast_json(
        self,
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model_date: Optional[Union[str, date]] = None,
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
            model_date (str or date, optional): The date of the model version to use for the forecast data.
                If not provided, the most recent model version will be used.

        Returns:
            List[dict]: A list of dictionaries representing the forecast data in JSON format.
        """
        if not self._is_token_valid():
            self._login()
        params = {"region": region, "signal_type": signal_type}

        # No model_date will default to the most recent model version available
        if model_date is not None:
            params["model_date"] = model_date

        url = "{}/v3/forecast".format(self.url_base)
        headers = {"Authorization": "Bearer " + self.token}
        rsp = requests.get(url, headers=headers, params=params)
        return rsp.json()

    def get_forecast_pandas(
        self,
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model_date: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        j = self.get_forecast_json(region, signal_type, model_date)
        return pd.json_normalize(j, record_path="data", meta=["meta"])

    def get_historical_forecast_json(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[
            Literal["co2_moer", "co2_aoer", "health_damage"]
        ] = "co2_moer",
        model_date: Optional[Union[str, date]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._is_token_valid():
            self._login()
        url = "{}/v3/forecast/historical".format(self.url_base)
        headers = {"Authorization": "Bearer " + self.token}
        responses = []
        params = {"region": region, "signal_type": signal_type}

        start, end = self._parse_dates(start, end)
        chunks = self._get_chunks(start, end, chunk_size=timedelta(days=1))

        # No model_date will default to the most recent model version available
        if model_date is not None:
            params["model_date"] = model_date

        for c in chunks:
            params["start"], params["end"] = c
            rsp = requests.get(url, headers=headers, params=params)
            if rsp.status_code == 200:
                j = rsp.json()
                responses.append(j)
            else:
                raise Exception(f"\nAPI Response Error: {rsp.status_code}, {rsp.text}")

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
        model_date: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        json_list = self.get_historical_forecast_json(
            start, end, region, signal_type, model_date
        )
        out = pd.DataFrame()
        for json in json_list:
            for entry in json['data']:
                _df = pd.json_normalize(entry, record_path=["forecast"])
                _df = _df.assign(generated_at=pd.to_datetime(entry["generated_at"]))
                out = pd.concat([out, _df])
        return out
