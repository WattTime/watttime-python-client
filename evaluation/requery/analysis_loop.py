
import os
PATH = os.getenv("HOME")
os.chdir(f"{PATH}/watttime-python-client-aer-algo")

import pandas as pd
import evaluation.eval_framework as evu
from datetime import datetime

import seaborn as sns
import evaluation.metrics as m
from datetime import timedelta

import random
import math
from watttime import WattTimeForecast, WattTimeHistorical
import data.s3 as s3u
import importlib
import watttime.api as wt

import warnings
warnings.filterwarnings("ignore")

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

s3 = s3u.s3_utils()

sample_size = 5000

df_req = s3.load_csvdataframe("requery_data/20241203_1k_synth_users_96_days.csv")

def sanitize_time_needed(x,y):
    return int(math.ceil(min(x, y) / 300.0) * 5)

def sanitize_total_intervals(x):
    return math.ceil(x)

def load_forecast_file(region):
    full_forecast = s3.load_parquetdataframe(
    f"complete_2024_forecast_history/{region}.parquet"
    ).drop_duplicates()

    full_forecast["point_time"] = pd.to_datetime(
    full_forecast["point_time"], utc=True
    )

    return full_forecast

def load_history_file(region):
    return s3.load_parquetdataframe(f"complete_2024_actual_history/{region}.parquet").drop_duplicates()

def prepare_set_of_historic_actuals(
        full_history,
        start_time,
        end_time
        ):

        moer_list = full_history.loc[
                (full_history["point_time"] >= start_time - timedelta(minutes=5)) &
                (full_history["point_time"] <= end_time - timedelta(minutes=5))
                ]
        return moer_list

def prepare_set_of_forecasts(
                forecasts,
                start_time,
                end_time,
                increment
    ):
        inc_times = pd.date_range(
            start_time,
            end_time,
            freq=timedelta(minutes=increment),
        ).tolist()

        moer_list = [
            forecasts.loc[
            forecasts["generated_at"] == timestamp].sort_values(by=["point_time"], ascending=True)
            for timestamp in inc_times
        ]

        return moer_list

def get_recalculating_optimizer_results(
    region: str,
    moer_list: pd.DataFrame,
    start_time: datetime,
    end_time: datetime,
    usage_power_kw,
    time_needed,
    increment,
    charge_per_interval = None,
):
    
    if charge_per_interval is None:
        wt_opt_rc = wt.RecalculatingWattTimeOptimizer(
            region=region,
            watttime_username=username,
            watttime_password=password,
            usage_time_required_minutes=time_needed,
            usage_power_kw=usage_power_kw,
            optimization_method="auto"
            )
    else:
        wt_opt_rc = wt.RecalculatingWattTimeOptimizerWithContiguity(
            username,
            password,
            region,
            usage_time_required_minutes=time_needed,
            usage_power_kw=usage_power_kw,
            optimization_method='auto',
            charge_per_interval=charge_per_interval,
        )

    new_start_time = start_time
    for fcst_data in moer_list:
        new_start_time = pd.Timestamp(fcst_data["point_time"].min())
        wt_opt_rc.get_new_schedule(
                new_start_time=new_start_time,
                new_end_time=end_time,
                curr_fcst_data=fcst_data
                )
    print("combining schedules")
    usage_plan = wt_opt_rc.get_combined_schedule(end_time=end_time)
    usage_plan["requery_increment"] = increment

    return usage_plan

df_req["sanitize_intervals_plugged_in"] = df_req.apply(lambda x: sanitize_total_intervals(x.total_intervals_plugged_in), axis=1)
df_req["sanitize_time_needed"] = df_req.apply(lambda x: sanitize_time_needed(x.total_seconds_to_95, x.length_of_session_in_seconds), axis=1)

synth_data = df_req.sample(sample_size, random_state=42).copy()
synth_data.session_start_time = pd.to_datetime(synth_data.session_start_time)
synth_data.session_end_time = pd.to_datetime(synth_data.session_end_time)

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

#regions = original_regions+random_regions

regions = random_regions[-2:]

for region in regions:
    print(region)

    full_forecast = load_forecast_file(region)
    full_history = load_history_file(region)

    all_synth_users_list = []
    bad_dat = []
    increments = [5,15,30,60,90,120,180,240]
    for i in range(0,synth_data.shape[0]):
        try:
            loc_num = i
            time_zone = evu.get_timezone_from_dict(region)                    
            start_time_utc = pd.Timestamp(evu.convert_to_utc(synth_data.iloc[loc_num]['session_start_time'].round('5min') , time_zone))
            end_time_utc = pd.Timestamp(evu.convert_to_utc(synth_data.iloc[loc_num]['session_end_time'].round('5min'), time_zone))
            time_needed = synth_data.iloc[loc_num]["sanitize_time_needed"]
            total_intervals_plugged_in = synth_data.iloc[loc_num]["sanitize_intervals_plugged_in"]
            usage_power_kw = float(synth_data.iloc[loc_num]["power_output_rate"])
            user_type = synth_data.iloc[loc_num]["user_type"]
            optimization_method = "auto"
            uuid = user_type + "|" + str(start_time_utc) + "|" + region
            
            results_dfs = []
            for increment in increments:
                    moer_list = prepare_set_of_forecasts(
                    forecasts=full_forecast,
                    increment=increment,
                    start_time=start_time_utc,
                    end_time=end_time_utc
                    )
                    try:
                        results = get_recalculating_optimizer_results(
                        region=region,
                        moer_list = moer_list,
                        start_time=start_time_utc,
                        end_time=end_time_utc,
                        time_needed=time_needed,
                        usage_power_kw=usage_power_kw,
                        increment=increment
                        )
                        results_dfs.append(results)
                    except:
                        no_go = synth_data.iloc[i].copy(deep=True)
                        no_go["uuid"] = uuid
                        bad_dat.append(no_go)
                        continue
            requery_results = pd.concat(results_dfs)
            
            # baseline + ideal
            moer_list_actuals = prepare_set_of_historic_actuals(
                full_history=full_history,
                start_time=start_time_utc,
                end_time=end_time_utc
                )
            
            no_requery = evu.get_schedule_and_cost_api(
                total_time_horizon=total_intervals_plugged_in,
                usage_power_kw=usage_power_kw,
                time_needed=time_needed,
                optimization_method="auto",
                moer_data=moer_list[0],
                charge_per_interval=None
            )
            no_requery["requery_increment"] = "0"
            
            ideal = evu.get_schedule_and_cost_api(
                    total_time_horizon = total_intervals_plugged_in,
                    usage_power_kw=usage_power_kw,
                    time_needed=time_needed,
                    optimization_method="auto",
                    moer_data=moer_list_actuals,
                    charge_per_interval=None
                    )
            ideal["requery_increment"] = "ideal"

            baseline = evu.get_schedule_and_cost_api(
                    total_time_horizon = total_intervals_plugged_in,
                    usage_power_kw=usage_power_kw,
                    time_needed=time_needed,
                    optimization_method="baseline",
                    moer_data=moer_list_actuals,
                    charge_per_interval=None
                    )
            baseline["requery_increment"] = "baseline"

            complete_session_cycle = pd.concat([requery_results,baseline,ideal,no_requery]).merge(moer_list_actuals, on="point_time", how="left")
            complete_session_cycle["emissions_co2e_lb_actual"] = complete_session_cycle["value"]*complete_session_cycle["energy_usage_mwh"]
            complete_session_cycle["user_type"] = user_type
            complete_session_cycle["uuid"] = uuid
            all_synth_users_list.append(complete_session_cycle)
        except:
            print("bad data")
            continue
    region_users_df = pd.concat(all_synth_users_list)
    s3.store_csvdataframe(region_users_df,f"results/analysis_requery_20241212_v3_{region}.csv")
    print(f"File stored")

if len(bad_dat) > 0:
    bad_dat_df = pd.concat(bad_dat, axis=1).T
    s3.store_csvdataframe(bad_dat_df,f"results/bad_dat_20241212_{region}.csv")