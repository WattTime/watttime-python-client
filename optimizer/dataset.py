from watttime import WattTimeForecast, WattTimeHistorical
import pandas as pd
from typing import List

us_regions = pd.read_csv("optimizer/us_region_meta_data.csv",index_col=0)
us_region_list_forecast = us_regions.query('endpoint == "v3/forecast"')["region"].to_list()

def get_forecast_multigrid_regions(
    username:str,
    password:str,
    horizon_hours:int = 24,
    region_list: List[str] = us_region_list_forecast,
    signal_type: str = 'co2_moer',
    include_meta = False
):
    wt_forecast = WattTimeForecast(username, password)
    dfs = []
    for region in region_list:
        df_r = wt_forecast.get_forecast_pandas(
            region = region,
            signal_type = signal_type,
            include_meta = include_meta
            )
        df_r["region"] = region
        dfs.append(df_r)
    df = pd.concat(dfs)
    return df