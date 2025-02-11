import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import requests


class RateLimitedRequesterMixin:
    """
    Mixin to handle rate-limited multi-threaded requests.
    """

    def __init__(self, rate_limit: int = 10):
        """
        Args:
            rate_limit (int): Maximum number of requests per second.
        """
        self._rate_limit_lock = threading.Lock()
        self._last_request_times = []
        self.rate_limit = rate_limit

    def _make_rate_limited_request(
        self, url: str, headers: Dict[str, str], params: Dict[str, Any]
    ) -> Dict:
        """
        Makes an API request with rate limiting.

        Args:
            url (str): API endpoint URL.
            headers (Dict[str, str]): Request headers.
            params (Dict[str, Any]): Query parameters.

        Returns:
            Dict: The JSON response.
        """
        while True:
            with self._rate_limit_lock:
                current_time = time.time()
                self._last_request_times = [
                    t for t in self._last_request_times if current_time - t < 1.0
                ]

                if len(self._last_request_times) < self.rate_limit:
                    self._last_request_times.append(current_time)
                    break

            time.sleep(0.1)

        rsp = requests.get(url, headers=headers, params=params)
        rsp.raise_for_status()
        return rsp.json()

    def _fetch_data_multithreaded(
        self, url: str, headers: Dict[str, str], param_chunks: List[Dict[str, Any]]
    ) -> List[Dict]:
        """
        Fetch data using multithreading with rate limiting.

        Args:
            url (str): API endpoint URL.
            headers (Dict[str, str]): Request headers.
            param_chunks (List[Dict[str, Any]]): List of parameter sets.

        Returns:
            List[Dict]: A list of JSON responses.
        """
        responses = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as executor:
            futures = {
                executor.submit(
                    self._make_rate_limited_request, url, headers, params
                ): params
                for params in param_chunks
            }

            for future in as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as e:
                    if hasattr(e, "response"):
                        raise Exception(
                            f"\nAPI Response Error: {e.response.status_code}, {e.response.text} "
                            f"[{e.response.headers.get('x-request-id')}]"
                        )
                    raise

        return responses
