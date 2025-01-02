import os
PATH = os.getenv("HOME")
os.chdir(f"{PATH}/watttime-python-client-aer-algo")

from datetime import timedelta
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from watttime import WattTimeForecast, WattTimeHistorical
import data.s3 as s3u
import evaluation.eval_framework as evu

s3 = s3u.s3_utils()

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

forecast_generator = WattTimeForecast(username, password)
historical_generator = WattTimeHistorical(username, password)

def add_previous_day(date):
    """
    Subtract one day to the given datetime object.

    Parameters:
    -----------
    date : datetime
        The datetime object to which one day will be substracted.

    Returns:
    --------
    datetime
        A new datetime object with one day less.
    """
    return date + timedelta(days=-1)

def add_next_day(date):
    """
    Subtract one day to the given datetime object.

    Parameters:
    -----------
    date : datetime
        The datetime object to which one day will be substracted.

    Returns:
    --------
    datetime
        A new datetime object with one day less.
    """
    return date + timedelta(days=1)

def get_daily_historical_data_in_utc(date, region):
    time_zone = evu.get_timezone_from_dict(region)
    date_utc = evu.convert_to_utc(date,time_zone)
    daily_data = historical_generator.get_historical_pandas(
        start=date_utc,
        end=date_utc + timedelta(days=1),
        region=region,
        signal_type="co2_moer"
    )

    daily_data["region"] = region

    return daily_data

def get_daily_forecast_data_in_utc(date, region, horizon=14):
    time_zone = evu.get_timezone_from_dict(region)
    date_utc = evu.convert_to_utc(date,time_zone)
    daily_data = forecast_generator.get_historical_forecast_pandas(
        start=date_utc,
        end=date_utc + timedelta(days=1),
        region=region,
        signal_type="co2_moer",
        horizon_hours=12,
    )
    daily_data["region"] = region

    return daily_data

df_req = s3.load_csvdataframe("requery_data/20241203_1k_synth_users_96_days.csv")
dates_2024_only = pd.to_datetime(df_req.distinct_dates.drop_duplicates()).tolist() # has no timezone. local time ambiguous

# gather forecasts across range of possible generated_at times
prev_days = []
next_days = []
for day in dates_2024_only:
    previous_day = add_previous_day(day)
    prev_days.append(previous_day)
    next_day = add_next_day(day)
    next_days.append(next_day)

dates_2024 = prev_days + dates_2024_only + next_days
dates_2024 = pd.to_datetime(dates_2024)

original_regions = [
    "SPP_TX",
    "ERCOT_EASTTX",
    "FPL",
    "SOCO",
    "PJM_CHICAGO",
    "LDWP",
    "PJM_DC",
    "NYISO_NYC",
]

#from evaluation.config import MOER_REGION_LIST
#random_regions = random.sample([elem for elem in MOER_REGION_LIST if elem not in regions], 9)

random_regions = [
    'PACE',
    'PNM',
    'MISO_INDIANAPOLIS',
    'WALC',
    'ERCOT_AUSTIN',
    'SPP_KANSAS',
    'ISONE_VT',
    'SPP_SIOUX',
    'SC'
]

regions = original_regions+random_regions

for region in regions:

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        result = list(
            tqdm(
                executor.map(
                    lambda date: get_daily_historical_data_in_utc(date, region), dates_2024
                ),
                total=len(dates_2024),
                desc=f"Getting 2024 actuals data for {region}",
            )
        )

    out = pd.concat(result, ignore_index=True)
    s3.store_parquetdataframe(out, f"complete_2024_actual_history/{region}.parquet")

for region in regions:
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        result = list(
            tqdm(
                executor.map(
                    lambda date: get_daily_forecast_data_in_utc(date, region), dates_2024
                ),
                total=len(dates_2024),
                desc=f"Getting 2024 data for {region}",
            )
        )

    out = pd.concat(result, ignore_index=True)
    s3.store_parquetdataframe(out, f"complete_2024_forecast_history/{region}.parquet")