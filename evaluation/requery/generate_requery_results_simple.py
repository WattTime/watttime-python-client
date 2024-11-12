import os

os.chdir(f"/home/{os.getlogin()}/watttime-python-client-aer-algo")

import math
import numpy as np
import pandas as pd
import datetime
from pytz import UTC, timezone
import seaborn as sns
from datetime import datetime, timedelta
import concurrent.futures
import contextlib
import io

from watttime import WattTimeForecast, WattTimeHistorical, RecalculatingWattTimeOptimizer

import data.s3 as s3u
import evaluation.eval_framework as efu
from plotnine import *

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

actual_data = WattTimeHistorical(username, password)
hist_data = WattTimeForecast(username, password)

s3 = s3u.s3_utils()
pd.options.mode.chained_assignment = None  # default='warn'
import random
from tqdm import tqdm
import warnings


def random_date_with_time(start, end):

    time_between_dates = end - start
    random_number_of_seconds = random.randint(0, int(time_between_dates.total_seconds()))
    random_date = start + timedelta(seconds=random_number_of_seconds)
    
    random_hour = random.randint(0, 23)
    random_minute = random.randint(0, 59)
    
    random_date_with_time = random_date.replace(hour=random_hour, minute=random_minute)
    return pd.to_datetime(random_date_with_time, utc = True)


def full_requery_sim(region, full_forecast, full_history, increments, start_time, end_time, usage_power_kw, time_needed, method = "simple"):

    results = {}
    all_relevant_forecasts = full_forecast.set_index("generated_at")[start_time - timedelta(minutes = 5):end_time].reset_index()
    all_relevant_forecasts = all_relevant_forecasts.set_index("generated_at")[start_time - timedelta(minutes = 5):end_time]
    baseline_forecast = all_relevant_forecasts.loc[all_relevant_forecasts.index.min()].reset_index()
    schedules = []

    ideal = efu.get_schedule_and_cost_api_requerying(region = "SOCO",
                                        usage_power_kw = 2,
                                        time_needed = time_needed,
                                        start_time = start_time,
                                        end_time = end_time, 
                                        optimization_method="simple",
                                        moer_list = [full_history.set_index("point_time")[start_time - timedelta(minutes = 5):end_time].reset_index()]).reset_index().rename({"pred_moer" : "actual_moer"}, axis = 1)

    results["ideal_emissions"] = round(ideal["emissions_co2e_lb"].sum(), 2)
    ideal["increment"] = "Ideal"
    ideal["pred_moer"] = ideal["actual_moer"]
    ideal["actual_emissions"] = ideal["actual_moer"]*ideal["energy_usage_mwh"]
    schedules.append(ideal)


    baseline = efu.get_schedule_and_cost_api_requerying(region = region,
                                        usage_power_kw = usage_power_kw,
                                        time_needed = time_needed,
                                        start_time = start_time,
                                        end_time = end_time, 
                                        optimization_method="baseline",
                                        moer_list = [baseline_forecast]).reset_index()

    baseline = baseline.merge(ideal[["point_time", "actual_moer"]])

    baseline["increment"] = "Baseline"
    baseline["actual_emissions"] = baseline["actual_moer"]*baseline["energy_usage_mwh"]

    schedules.append(baseline)

    results["baseline_predicted_emissions"] = round(baseline["emissions_co2e_lb"].sum(), 2)
    results["baseline_actual_emissions"] = round((baseline["actual_moer"]*baseline["energy_usage_mwh"]).sum(), 2)

    no_requery = efu.get_schedule_and_cost_api_requerying(region = region,
                                        usage_power_kw = usage_power_kw,
                                        time_needed = time_needed,
                                        start_time = start_time,
                                        end_time = end_time, 
                                        optimization_method=method,
                                        moer_list = [baseline_forecast]).reset_index()

    no_requery = no_requery.merge(ideal[["point_time", "actual_moer"]])
    no_requery["increment"] = "No requery"
    no_requery["actual_emissions"] = no_requery["actual_moer"]*no_requery["energy_usage_mwh"]
    schedules.append(no_requery)

    results["no_requery_predicted_emissions"] = round(no_requery["emissions_co2e_lb"].sum(), 2)
    results["no_requery_actual_emissions"] = round((no_requery["actual_moer"]*no_requery["energy_usage_mwh"]).sum(), 2)



    for increment in increments:
        inc_times = pd.date_range(all_relevant_forecasts.index.min(), all_relevant_forecasts.index.max(), freq=timedelta(minutes=increment))
        moer_list = [all_relevant_forecasts.loc[timestamp].reset_index() for timestamp in inc_times]

        print(len(moer_list))

        schedule = efu.get_schedule_and_cost_api_requerying(region = region,
                                        usage_power_kw = 2,
                                        time_needed = 180,
                                        start_time = start_time,
                                        end_time = end_time, 
                                        optimization_method=method,
                                        moer_list = moer_list).reset_index()
        
        
        schedule = schedule.merge(ideal[["point_time", "actual_moer"]])
        schedule["actual_emissions"] = schedule["actual_moer"]*schedule["energy_usage_mwh"]
        schedule["increment"] = f"Requery {increment} minutes"
        schedules.append(schedule)


        results[f"schedule_predicted_emissions_requery_{increment}"] = round(schedule["emissions_co2e_lb"].sum(), 2)
        results[f"schedule_actual_emissions_requery_{increment}"] = round((schedule["actual_moer"]*schedule["energy_usage_mwh"]).sum(), 2)

    increment_order = [f"Requery {increment} minutes" for increment in increments]
    order = ["Ideal", "Baseline", "No requery"] + increment_order[::-1]
    full_schedules = pd.concat(schedules)
    full_schedules["increment"] = pd.Categorical(full_schedules["increment"], order, ordered = True)

    return full_schedules


# Some basic paramaters to get simple data. Will eventually be expanded to the synthetic users

increments = [5, 15, 30, 60, 120, 180, 240, 360]
start_time = random_date_with_time(datetime(2023, 1, 1), datetime(2023, 12, 31))
end_time = start_time + timedelta(hours = 12)
usage_power_kw = 2
time_needed = 180


regions = [
 'CAISO_NORTH',
 'SPP_TX',
 'ERCOT_EASTTX',
 'FPL',
 'SOCO',
 'PJM_CHICAGO',
 'LDWP',
 'PJM_DC',
 'NYISO_NYC'
]

dates = [pd.to_datetime(random_date_with_time(datetime(2023, 1, 1), datetime(2023, 12, 31))) for i in range(0, 1000)]


out = []
for region in regions:
    print(region)
    full_forecast = s3.load_parquetdataframe(f"complete_2023_forecast_history/{region}.parquet").drop_duplicates()
    full_forecast['point_time'] = pd.to_datetime(full_forecast['point_time'], utc=True)
    full_history = s3.load_parquetdataframe(f"complete_2023_actual_history/{region}.parquet").drop_duplicates()

    for date in tqdm(dates):
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  
                schedules = full_requery_sim(region, full_forecast, full_history, increments, date, date + timedelta(hours = 12), usage_power_kw, time_needed, method = "simple")
            schedules["init_time"] = date
            schedules["region"] = region
            out.append(schedules)
        except Exception as e:
            print(e)

out_df = pd.concat(out)
s3.store_parquetdataframe(out_df, f'historical_requery_sim_1000_simple_fit.parquet')
