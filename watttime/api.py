import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from datetime import time as dt_time
from datetime import timedelta
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import requests
from dateutil.parser import parse
from pytz import UTC
from requests.adapters import HTTPAdapter
import urllib3.exceptions
from urllib3.util.retry import Retry

try:
    VERSION = _package_version("watttime")
except PackageNotFoundError:
    # Package not installed (e.g. running from a source checkout without install)
    VERSION = "0.0.0"


def _is_timeout(exc: BaseException) -> bool:
    """
    True if `exc` is, or wraps, a connect/read timeout.

    Once the session's retries are exhausted, requests surfaces a read timeout as
    ConnectionError -> MaxRetryError -> ReadTimeoutError rather than as ReadTimeout,
    so the whole chain has to be walked, not just the top-level exception.
    """
    seen = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(
            e, (requests.exceptions.Timeout, urllib3.exceptions.TimeoutError)
        ):
            return True
        stack.extend([e.__cause__, e.__context__, getattr(e, "reason", None)])
        stack.extend(a for a in e.args if isinstance(a, BaseException))
    return False


class WattTimeRequestError(RuntimeError):
    """
    Raised when an API request fails after the client has done what it can about it.

    Subclasses RuntimeError so existing `except RuntimeError` handlers keep working.
    `kind` names the failure class so callers can branch on it instead of parsing
    the message: "timeout", "http_401", "http_4xx", "http_5xx", "connection" or "other".
    """

    def __init__(self, message: str, kind: str, url: str, params: Dict[str, Any]):
        super().__init__(message)
        self.kind = kind
        self.url = url
        self.params = params


def _classify_request_exception(exc: requests.exceptions.RequestException) -> str:
    """
    Sort a failed request into the class that decides how the client responds.

    A 504 is grouped with read timeouts: the gateway gave up waiting on the same
    slow upstream query, so it calls for the same response (ask for less at once).
    """
    if _is_timeout(exc):
        return "timeout"
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 504:
        return "timeout"
    if status == 401:
        return "http_401"
    if status is not None and 400 <= status < 500:
        return "http_4xx"
    if status is not None and 500 <= status < 600:
        return "http_5xx"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection"
    return "other"


class WattTimeAPIWarning:
    def __init__(self, url: str, params: Dict[str, Any], warning_message: str):
        self.url = url
        self.params = params
        self.warning_message = warning_message

    def __repr__(self):
        return f"<WattTimeAPIWarning url={self.url}, params={self.params}, warning={self.warning_message}>\n"

    def to_dict(self) -> Dict[str, Any]:
        def stringify(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.isoformat()
            return value

        return {
            "url": self.url,
            "params": {k: stringify(v) for k, v in self.params.items()},
            "warning_message": self.warning_message,
        }


def get_log():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-1s]  " "%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger()


LOG = get_log()


class WattTimeBase:
    url_base = os.getenv("WATTTIME_API_URL", "https://api.watttime.org")

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        multithreaded: bool = False,
        rate_limit: int = 10,
        worker_count: int = min(10, (os.cpu_count() or 1) * 2),
        *,
        timeout: Optional[Union[float, Tuple[float, float]]] = (10, 60),
    ):
        """
        Initializes a new instance of the class.

        Parameters:
            username (Optional[str]): The username to use for authentication. If not provided, the value will be retrieved from the environment variable "WATTTIME_USER".
            password (Optional[str]): The password to use for authentication. If not provided, the value will be retrieved from the environment variable "WATTTIME_PASSWORD".
            multithreaded (bool): Whether to use multithreading for requests. Default is False.
            rate_limit (int): The maximum number of requests to make per second. Default is 10 as this algins well with WattTime's API rate limiting policy.
            worker_count (int): The number of worker threads to use for multithreading. Default is min(10, (os.cpu_count() or 1) * 2).
            timeout (Optional[Union[float, Tuple[float, float]]]): The timeout passed to every HTTP request, in seconds, using the standard `requests` forms: a single value applied to both the connect and read phases, or a (connect, read) tuple. `None` disables timeouts entirely. Default is (10, 60). A request that times out is not retried as-is; on the chunked endpoints its time span is split in half and retried (up to twice), and elsewhere it fails with a hint. Connection errors and transient server errors are retried up to 3 times with backoff.

        """

        if username and os.getenv("WATTTIME_USER") is not None:
            LOG.warning(
                "Both a username argument and WATTTIME_USER are set; using the username argument value."
            )

        if username:
            os.environ["WATTTIME_USER"] = username
        if password:
            os.environ["WATTTIME_PASSWORD"] = password

        self.token = None
        self.headers = None
        self.token_valid_until = None

        self.multithreaded = multithreaded
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_times = []
        self.worker_count = worker_count
        self.raised_warnings: List[WattTimeAPIWarning] = []
        self._login_lock = threading.Lock()

        if self.multithreaded:
            self._rate_limit_lock = (
                threading.Lock()
            )  # prevent multiple threads from modifying _last_request_times simultaneously
            self._rate_limit_condition = threading.Condition(self._rate_limit_lock)

        # The transport retries what re-issuing the identical request can fix:
        # connection errors, transient 5xx, and 429 (honouring Retry-After).
        # It does NOT retry read timeouts or 504: those mean the API was slow to
        # answer *this much* data, so the client responds by asking for less at a
        # time instead (see _fetch_adaptive). Repeating a slow query four times
        # only multiplies the load on a backend that is already struggling.
        retry_strategy = Retry(
            total=3,
            read=0,
            status_forcelist=[429, 500, 502, 503],
            backoff_factor=1,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _login(self):
        """
        Login to the WattTime API, which provides a JWT valid for 30 minutes

        Raises:
            Exception: If the login fails and the credentials are incorrect.
        """

        url = f"{self.url_base}/login"
        rsp = self.session.get(
            url,
            auth=requests.auth.HTTPBasicAuth(
                os.getenv("WATTTIME_USER"), os.getenv("WATTTIME_PASSWORD")
            ),
            timeout=self.timeout,
        )
        rsp.raise_for_status()
        self.token = rsp.json().get("token", None)
        self.token_valid_until = datetime.now() + timedelta(minutes=30)
        if not self.token:
            raise Exception("failed to log in, double check your credentials")
        self.headers = {
            "Authorization": "Bearer " + self.token,
            "User-Agent": f"watttime-python-sdk-{VERSION}",
        }

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
        self,
        start: datetime,
        end: datetime,
        chunk_size: Optional[timedelta] = None,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate a list of tuples representing chunks of time within a given time range.

        Args:
            start (datetime): The start datetime of the time range.
            end (datetime): The end datetime of the time range.
            chunk_size (Optional[timedelta], optional): The size of each chunk. None means the default of 30 days.
                Must be longer than 5 minutes, since 5 minutes is trimmed from the end of every chunk but the last.

        Returns:
            List[Tuple[datetime, datetime]]: A list of tuples representing the chunks of time.
            If start == end, a single zero-length chunk is returned.

        Raises:
            ValueError: If start is after end, or chunk_size is not longer than 5 minutes.
        """
        if chunk_size is None:
            chunk_size = timedelta(days=30)
        if chunk_size <= timedelta(minutes=5):
            raise ValueError(
                f"chunk_size must be longer than 5 minutes, got {chunk_size}"
            )

        if start > end:
            raise ValueError(f"start ({start}) must not be after end ({end})")

        # A zero-length span is a valid request: it asks the API for the single
        # point (or single forecast run) at that instant. The loop below would
        # produce no chunks at all, so return the span itself as one chunk.
        if start == end:
            return [(start, end)]

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
            "username": os.getenv("WATTTIME_USER"),
            "password": os.getenv("WATTTIME_PASSWORD"),
            "email": email,
            "org": organization,
        }

        rsp = self.session.post(url, json=params, timeout=self.timeout)
        rsp.raise_for_status()
        LOG.info(
            f"Successfully registered {os.getenv('WATTTIME_USER')}, please check {email} for a verification email"
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

    def _ensure_logged_in(self):
        """
        Log in if there is no token or the local 30-minute clock has run out.
        Serialised so that concurrent workers reaching an expired token log in once.
        """
        if self._is_token_valid() and self.headers:
            return
        with self._login_lock:
            if not self._is_token_valid() or not self.headers:
                self._login()

    def _refresh_token(self, rejected_headers: Optional[Dict[str, str]]):
        """
        Log in again after the API rejected a token the client thought was valid.
        Workers that all saw the same rejected token refresh it once, not once each.
        """
        with self._login_lock:
            if self.headers is rejected_headers:
                self._login()

    def _rate_limited_get_json(self, url: str, params: Dict[str, Any]) -> Dict:
        """One GET, rate limited, raising requests exceptions untouched."""
        ts = time.time()

        # apply rate limiting by either sleeping (single thread) or
        # waiting on a condition ()
        if self.multithreaded:
            with self._rate_limit_condition:
                self._apply_rate_limit(ts)
        else:
            self._apply_rate_limit(ts)

        rsp = self.session.get(
            url, headers=self.headers, params=params, timeout=self.timeout
        )
        rsp.raise_for_status()
        return rsp.json()

    def _request_error(
        self, exc: requests.exceptions.RequestException, url: str, params: Dict
    ) -> WattTimeRequestError:
        kind = _classify_request_exception(exc)
        msg = f"API Request Failed: {exc}\nURL: {url}\nParams: {params}"
        if kind == "timeout":
            msg += (
                f"\nHint: the request exceeded the client timeout ({self.timeout}). "
                "This usually means the API was slow to respond, which is more likely "
                "for requests covering a large time span. Either pass a smaller "
                "`chunk_size` to the historical methods (e.g. timedelta(days=10)) so "
                "each request covers less time, or construct the client with a longer "
                "`timeout`."
            )
        return WattTimeRequestError(msg, kind=kind, url=url, params=params)

    def _make_rate_limited_request(self, url: str, params: Dict[str, Any]) -> Dict:
        """
        Makes a single API request while respecting the rate limit.

        A 401 on a token the client still considers valid is refreshed and retried
        once; a second 401 is a real authentication failure. Every other failure is
        raised as WattTimeRequestError with its `kind` set.
        """
        self._ensure_logged_in()
        headers_used = self.headers

        try:
            try:
                j = self._rate_limited_get_json(url, params)
            except requests.exceptions.HTTPError as e:
                if _classify_request_exception(e) != "http_401":
                    raise
                LOG.warning(
                    f"API returned 401 for a token the client considered valid; "
                    f"refreshing and retrying once | URL: {url} | Params: {params}"
                )
                self._refresh_token(headers_used)
                j = self._rate_limited_get_json(url, params)
        except requests.exceptions.RequestException as e:
            raise self._request_error(e, url, params) from e

        meta = j.get("meta", {})
        warnings = meta.get("warnings")
        if warnings:
            for warning_message in warnings:
                warning = WattTimeAPIWarning(
                    url=url, params=params, warning_message=warning_message
                )
                self.raised_warnings.append(warning)
                LOG.warning(
                    f"API Warning: {warning_message} | URL: {url} | Params: {params}"
                )

        self._last_request_meta = meta

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

    # How many times one chunk may be halved after timing out. A chunk becomes at
    # most 2**_MAX_TIMEOUT_SPLITS requests. Splitting is depth-first and the first
    # sub-span that still times out at the last level fails the call, so a chunk
    # the API never answers costs about (1 + _MAX_TIMEOUT_SPLITS) x read timeout.
    _MAX_TIMEOUT_SPLITS = 2

    def _fetch_adaptive(
        self, url: str, params: Dict[str, Any], depth: int = 0
    ) -> List[Dict]:
        """
        Fetch one set of params, halving its time span and retrying on timeout.

        A read timeout or 504 on a span request usually means the API was slow to
        answer that much data at once, so re-issuing the identical request tends to
        fail the same way. Asking for half the span instead is what actually helps.
        Only applies when `start` and `end` are datetimes (the chunked endpoints), and
        at most _MAX_TIMEOUT_SPLITS times per original chunk. Every split is logged.
        """
        try:
            return [self._make_rate_limited_request(url, params)]
        except WattTimeRequestError as e:
            start, end = params.get("start"), params.get("end")
            if not (
                e.kind == "timeout"
                and isinstance(start, datetime)
                and isinstance(end, datetime)
                and depth < self._MAX_TIMEOUT_SPLITS
            ):
                raise
            half = (end - start) / 2
            if half <= timedelta(minutes=5):  # the chunker's floor
                raise
            halves = self._get_chunks(start, end, chunk_size=half)
            LOG.warning(
                f"Request timed out (timeout={self.timeout}); splitting {start} -> {end} "
                f"into {len(halves)} requests of {half} and retrying "
                f"(split {depth + 1} of {self._MAX_TIMEOUT_SPLITS}) | URL: {url}"
            )

        responses = []
        for sub_start, sub_end in halves:
            sub_params = {**params, "start": sub_start, "end": sub_end}
            responses.extend(self._fetch_adaptive(url, sub_params, depth + 1))
        return responses

    def _fetch_data(
        self,
        url: str,
        param_chunks: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[Dict]:
        """
        Fetch a series of requests with varying `param_chunks`, sequentially or with
        a thread pool. Each chunk goes through _fetch_adaptive, so a chunk that times
        out is split into smaller spans rather than failing the whole call.
        If you are making a single request, you can call _make_rate_limited_request directly.
        """

        # first try to login before beginning multithreading
        self._ensure_logged_in()

        if isinstance(param_chunks, dict):
            param_chunks = [param_chunks]

        responses = []
        if self.multithreaded:
            with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                futures = {
                    executor.submit(self._fetch_adaptive, url, params): params
                    for params in param_chunks
                }

                for future in as_completed(futures):
                    responses.extend(future.result())
        else:
            for params in param_chunks:
                responses.extend(self._fetch_adaptive(url, params))

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
        include_imputed_marker: bool = False,
        *,
        chunk_size: Optional[timedelta] = None,
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
            chunk_size (Optional[timedelta], optional): The span of each request the date range is split into.
                Defaults to 30 days. Smaller spans mean more, faster requests; use this when the API is
                slow to answer large spans and requests are hitting the read timeout.

        Raises:
            Exception: Scraping failed for some reason

        Returns:
            List[dict]: A list of dictionary representations of the .json response object
        """
        url = "{}/v3/historical".format(self.url_base)
        params = {"region": region, "signal_type": signal_type}

        if include_imputed_marker:
            params["include_imputed_marker"] = "true"

        start, end = self._parse_dates(start, end)
        chunks = self._get_chunks(start, end, chunk_size=chunk_size)

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
        include_imputed_marker: bool = False,
        *,
        chunk_size: Optional[timedelta] = None,
    ):
        """
        Return a pd.DataFrame with point_time, and values.

        Args:
            See .get_historical_jsons() for shared arguments.
            include_meta (bool, optional): adds additional columns to the output dataframe,
                containing the metadata information. Note that metadata is returned for each API response,
                not for each point_time.

        Returns:
            pd.DataFrame: _description_
        """
        responses = self.get_historical_jsons(
            start,
            end,
            region,
            signal_type=signal_type,
            model=model,
            include_imputed_marker=include_imputed_marker,
            chunk_size=chunk_size,
        )
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
        include_imputed_marker: bool = False,
        *,
        chunk_size: Optional[timedelta] = None,
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
            chunk_size (Optional[timedelta]): See .get_historical_jsons(). Default is None (30 days).

        Returns:
            None, results are saved to a csv file in the user's home directory.
        """
        df = self.get_historical_pandas(
            start,
            end,
            region,
            signal_type=signal_type,
            model=model,
            include_imputed_marker=include_imputed_marker,
            chunk_size=chunk_size,
        )

        out_dir = Path.home() / "watttime_historical_csvs"
        out_dir.mkdir(exist_ok=True)

        start, end = self._parse_dates(start, end)
        fp = out_dir / f"{region}_{signal_type}_{start.date()}_{end.date()}.csv"
        df.to_csv(fp, index=False)
        LOG.info(f"file written to {fp}")


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
        data = []
        for j in json_list:
            for gen_at in j["data"]:
                for point_time in gen_at["forecast"]:
                    point_time["generated_at"] = gen_at["generated_at"]
                    data.append(point_time)
        df = pd.DataFrame.from_records(data)
        df["point_time"] = pd.to_datetime(df["point_time"])
        df["generated_at"] = pd.to_datetime(df["generated_at"])
        return df

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


class WattTimeMarginalFuelMix(WattTimeBase):
    def get_fuel_mix_jsons(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[Literal["marginal_fuel_mix"]] = "marginal_fuel_mix",
        model: Optional[Union[str, date]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._is_token_valid():
            self._login()
        url = f"{self.url_base}/v3/fuel-mix"
        responses = []
        params = {
            "region": region,
            "signal_type": signal_type,
        }

        start, end = self._parse_dates(start, end)
        chunks = self._get_chunks(start, end, chunk_size=timedelta(days=30))

        # No model will default to the most recent model version available
        if model is not None:
            params["model"] = model

        param_chunks = [{**params, "start": c[0], "end": c[1]} for c in chunks]

        try:
            responses = self._fetch_data(url, param_chunks)
        except RuntimeError as e:
            if "403 Client Error: Forbidden" in str(e):
                print(
                    f"The /v3/fuel-mix endpoint is a beta endpoint that provides *marginal* fuel mix data. This endpoint is not currently available to all users, please reach out to WattTime if you believe accessing marginal fuel mix data could be impactful for your usecase!"
                )
                return []
            raise
        return responses

    def get_fuel_mix_pandas(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        region: str,
        signal_type: Optional[Literal["marginal_fuel_mix"]] = "marginal_fuel_mix",
        model: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        json_list = self.get_fuel_mix_jsons(start, end, region, signal_type, model)
        out = defaultdict(dict)
        for json in json_list:
            for entry in json["data"]:
                for by_fuel in entry["values"]:
                    out[by_fuel["fuel_type"]][entry["point_time"]] = by_fuel["value"]

        df = pd.DataFrame.from_dict(out).fillna(0).sort_index()
        df.index = pd.to_datetime(df.index)
        df.index.name = "point_time"
        return df
