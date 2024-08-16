import os
os.chdir(f"/home/{os.getlogin()}/watttime-python-client-aer-algo")

import argparse
import math
import numpy as np
import pandas as pd
import datetime
import pytz
from datetime import datetime, timedelta
import concurrent.futures
import pickle

from watttime import WattTimeForecast, WattTimeHistorical

import optimizer.s3 as s3u
import evaluation.eval_framework as efu


def main():
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument('region', type=str, help='Region for caching')
    args = parser.parse_args()
    region = args.region

    regions = [region]

    print(f"Received input: {args.region}")

    username = os.getenv("WATTTIME_USER")
    password = os.getenv("WATTTIME_PASSWORD")

    actual_data = WattTimeHistorical(username, password)
    hist_data = WattTimeForecast(username, password)

    s3 = s3u.s3_utils()
    key = '20240726_1k_synth_users_163_days.csv'
    synth_data = s3.load_csvdataframe(file=key)

    synth_data["plug_in_time"] = pd.to_datetime(synth_data["plug_in_time"])
    synth_data["unplug_time"] = pd.to_datetime(synth_data["unplug_time"])

    def precache_actual_data(synth_data, regions):
        distinct_dates = [
            datetime.strptime(date, "%Y-%m-%d").date()
            for date in synth_data["distinct_dates"].unique().tolist()
        ]
        all_dates_regions = [(date, region) for date in distinct_dates for region in regions]

        def get_actual_data_for_region_date(date, region):
            start = pd.to_datetime(date)
            end = start + pd.Timedelta("2d")  
            return (region, date, actual_data.get_historical_pandas(
                start - pd.Timedelta("9h"),
                end + pd.Timedelta("9h"),
                region,
            ))

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            result = executor.map(get_actual_data_for_region_date,
                [date for (date, region) in all_dates_regions], 
                [region for (date, region) in all_dates_regions]
                )
        result = list(result)

        return {(region, date): data for (region, date, data) in result}

    def precache_fcst_data(synth_data, regions):
        distinct_dates = [
            datetime.strptime(date, "%Y-%m-%d").date()
            for date in synth_data["distinct_dates"].unique().tolist()
        ]
        all_dates_regions = [(date, region) for date in distinct_dates for region in regions]

        def get_fsct_data_for_region_date(date, region):
            start = pd.to_datetime(date)
            end = (pd.to_datetime(date) + pd.Timedelta("1d"))      
            return  (region, date,  hist_data.get_historical_forecast_pandas(
                start - pd.Timedelta("9h"),
                end + pd.Timedelta("9h"),
                region,
            ))

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            result = executor.map(get_fsct_data_for_region_date,
                [date for (date, region) in all_dates_regions], 
                [region for (date, region) in all_dates_regions]
                )
        result = list(result)
        return {(region, date): data for (region, date, data) in result}

    HISTORICAL_ACTUAL_CACHE = precache_actual_data(synth_data, regions)
    pkl_actual = pickle.dumps(HISTORICAL_ACTUAL_CACHE)
    s3.store_dictionary(dictionary=pkl_actual, file = f"{region}_actual.pkl")

    HISTORICAL_FORECAST_CACHE = precache_fcst_data(synth_data, regions)
    pkl_fore = pickle.dumps(HISTORICAL_FORECAST_CACHE)
    s3.store_dictionary(dictionary=pkl_fore, file = f"{region}_fore.pkl")

if __name__ == "__main__":
    main()