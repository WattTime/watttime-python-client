from datetime import datetime, timedelta, date
from dateutil.parser import parse
from typing import List, Tuple, Dict, Union, Optional
import requests

import pandas as pd
​
​
class WattTimeBulkHistorical:
    url_base = "https://api.watttime.org"
    valid_signals = ["co2_moer", "co2_aoer", "health_damages"]
​
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
​
    def _get_chunks(
        self, start: datetime, end: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Internal function turns a start and end into 30-day chunks"""
        chunks = []
        while start < end:
            chunk_end = min(end, start + timedelta(days=30))
            chunks.append((start, chunk_end))
            start = chunk_end
​
        # API response is inclusive, avoid overlap in chunks
        chunks = [(s, e - timedelta(minutes=5)) for s, e in chunks[0:-1]] + [chunks[-1]]
        return chunks
​
    def login(self):
        """Login to the WattTime API, which provides a JWT valid for 30 minutes."""
        url = f"{self.url_base}/login"
        rsp = requests.get(
            url,
            auth=requests.auth.HTTPBasicAuth(self.username, self.password),
            timeout=20,
        )
        self.token = rsp.json().get("token", None)
        if not self.token:
            raise Exception("failed to log in, double check your credentials")
​
    def get_hist_jsons(
        self,
        start: datetime,
        end: datetime,
        region: str,
        signal_type: Optional[str] = "co2_moer",
        model_date: Optional[Union[str, date]] = None,
    ) -> List[dict]:
        """
        Base function to scrape historical data, returning a list of .json responses.
​
        Args:
            start (datetime): inclusive start, with a UTC timezone.
            end (datetime): inclusive end, with a UTC timezone.
            region (str): string, accessible through the /my-access endpoint, or use the free region (CAISO_NORTH)
            signal_type (str, optional): one of ['co2_moer', 'co2_aoer', 'health_damages']. Defaults to "co2_moer".
            model_date (Optional[Union[str, date]], optional): Optionally provide a model_date, used for versioning models.
                Defaults to None.
​
        Raises:
            Exception: Scraping failed for some reason
​
        Returns:
            List[dict]: A list of dictionary representations of the .json response object
        """
        self.login()
        url = "{}/v3/historical".format(self.url_base)
        headers = {"Authorization": "Bearer " + self.token}
        chunks = self._get_chunks(start, end)
        responses = []
        params = {"region": region, "signal_type": signal_type}
​
        # No model_date will default to the most recent model version available
        if model_date is not None:
            params["model_date"] = model_date
​
        assert (
            signal_type in self.valid_signals
        ), f"signal_type must be one of {self.valid_signals}"
​
        for c in chunks:
            params["start"], params["end"] = c
            rsp = requests.get(url, headers=headers, params=params)
            if rsp.status_code == 200:
                j = rsp.json()
                responses.append(j)
            else:
                raise Exception(f"\nAPI Response Error: {rsp.status_code}, {rsp.text}")
​
            if len(j["meta"]["warnings"]):
                print("\n", "Warnings Returned:", params, j["meta"])
​
        return responses
​
    def get_hist_pandas(
        self,
        start: datetime,
        end: datetime,
        region: str,
        signal_type: str = "co2_moer",
        include_meta: bool = False,
        model_date: Optional[Union[str, date]] = None,
    ):
        """Return a pd.DataFrame with point_time, and values.
​
        Args:
            See .get_hist_jsons() for shared arguments.
            include_meta (bool, optional): adds additional columns to the output dataframe,
                containing the metadata information. Note that metadata is returned for each response,
                not for each point_time.
​
        Returns:
            pd.DataFrame: _description_
        """​
        responses = self.get_hist_jsons(start, end, region, signal_type, model_date)
        df = pd.json_normalize(
            responses, record_path="data", meta=["meta"] if include_meta else []
        )
        return df
