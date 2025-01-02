import os

PATH = os.getenv("HOME")
os.chdir(f"{PATH}/watttime-python-client-aer-algo")

import pandas as pd
from datetime import timedelta
import concurrent.futures
from tqdm import tqdm
from watttime import WattTimeForecast, WattTimeHistorical
import data.s3 as s3u

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

actual_data = WattTimeHistorical(username, password)
hist_data = WattTimeForecast(username, password)

s3 = s3u.s3_utils()

regions = [
    "CAISO_NORTH",
    "SPP_TX",
    "ERCOT_EASTTX",
    "FPL",
    "SOCO",
    "PJM_CHICAGO",
    "LDWP",
    "PJM_DC",
    "NYISO_NYC",
]

dates_2023 = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

forecast_generator = WattTimeForecast(username, password)
historical_generator = WattTimeHistorical(username, password)


def get_daily_historical_data(date, region):
    daily_data = historical_generator.get_historical_pandas(
        start=date, end=date + timedelta(days=1), region=region, signal_type="co2_moer"
    )

    daily_data["region"] = region

    return daily_data


for region in regions:
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        print(f"Getting 2024 actuals data for {region}")

        result = list(
            tqdm(
                executor.map(
                    lambda date: get_daily_historical_data(date, region), dates_2023
                ),
                total=len(dates_2023),
                desc=f"Processing forecast data for region {region}",
            )
        )

    out = pd.concat(result, ignore_index=True)
    s3.store_parquetdataframe(out, f"complete_2023_actual_history/{region}.parquet")
    print("Wrote parquet file to s3")


def get_daily_forecast_data(date, region, horizon=24):
    daily_data = forecast_generator.get_historical_forecast_pandas(
        start=date,
        end=date + timedelta(days=1),
        region=region,
        signal_type="co2_moer",
        horizon_hours=horizon,
    )
    daily_data["region"] = region

    return daily_data


for region in regions:
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        print(f"Getting 2023 data for {region}")

        result = list(
            tqdm(
                executor.map(
                    lambda date: get_daily_forecast_data(date, region), dates_2023
                ),
                total=len(dates_2023),
                desc=f"Processing forecast data for region {region}",
            )
        )

    out = pd.concat(result, ignore_index=True)
    s3.store_parquetdataframe(out, f"complete_2024_forecast_history/{region}.parquet")
    print("Wrote parquet file to s3")
