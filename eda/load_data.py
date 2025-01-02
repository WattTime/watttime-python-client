import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from watttime import WattTimeHistorical
from tqdm import tqdm
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

login_url = "https://api.watttime.org/login"
rsp = requests.get(login_url, auth=HTTPBasicAuth(username, password))
TOKEN = rsp.json()["token"]
headers = {"Authorization": f"Bearer {TOKEN}"}


def get_historical_forecast_pandas(region, start, end):
    """
    Returns a dataframe with index named "point_time",
    which are five minute increments in [start, end + 24h).
    The columns is named "generated_at", and there is one
    column for each five minute incremenet in [start, end].

    The value in a cell is the historical forecast for the index time
    generated at the time the column time.
    So the upper triangle is NaN because forecasts are not issued
    for times in the past. Also each column only has 24 hours of data
    so there is a lower triangle which is also NaN.

    region = "CAISO_NORTH"
    start = "2023-07-01 00:00Z"
    end = "2023-07-02 00:00Z"
    """
    url = "https://api.watttime.org/v3/forecast/historical"
    params = {
        "region": region,
        "start": start,
        "end": end,
        "signal_type": "co2_moer",
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    forecast_json = response.json()

    df_list = []
    for generated_at_json in forecast_json["data"]:
        df = pd.DataFrame(generated_at_json["forecast"])
        df["generated_at"] = generated_at_json["generated_at"]
        df_list.append(df)
    df = pd.concat(df_list, axis=0)

    df_piv = (
        df.set_index(["generated_at", "point_time"])
        .unstack(level=0)
        .droplevel(axis=1, level=0)
    )
    df_piv.index = pd.to_datetime(df_piv.index)
    return df_piv


def get_historical(regions, start, end):
    """
    Returns a dataframe with index "point_time",
    which are five minute increments in [start, end].
    There is one column, named region},
    and the values are the actual MOER signal
    in lbs CO2/MWh

    example input:
    region = "CAISO_NORTH"
    start = "2023-07-01 00:00Z"
    end = "2023-07-02 00:00Z"
    """

    if isinstance(regions, str):  # there is only one region
        regions = [regions]

    wt_hist = WattTimeHistorical(username, password)
    df_region_list = []
    for region in tqdm(regions, desc="loading regions"):
        df_region = (
            wt_hist.get_historical_pandas(
                start=start, end=end, region=region, signal_type="co2_moer"
            )
            .set_index("point_time")
            .rename(columns={"value": region})
        )
        df_region_list.append(df_region)

    df_all_regions = pd.concat(df_region_list, axis=1)
    return df_all_regions


def get_region_geojson():
    """
    Returns a geojson of all of WattTime's regions.
    Each region is specified by a MultiPolygon
    """
    url = "https://api.watttime.org/v3/maps"
    params = {"signal_type": "co2_moer"}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()
