import os
os.chdir("/home/jennifer.badolato/watttime-python-client-aer-algo")

import math
import numpy as np
import pandas as pd
import datetime
import pytz
from datetime import datetime, timedelta
from watttime import WattTimeForecast

import optimizer.s3 as s3u
import evaluation.eval_framework as efu

import watttime.shared_anniez.alg.optCharger as optC
import watttime.shared_anniez.alg.moer as Moer


region = "PJM_NJ"
username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

actual_data = WattTimeHistorical(username, password)
hist_data = WattTimeForecast(username, password)

s3 = s3u.s3_utils()
key = '20240713_1k_synth_users.csv'
generated_data = s3.load_csvdataframe(file=key)


# Get per-row historical fcsts at 'plug in time'
def get_historical_fcst_data(
    plug_in_time, 
    horizon,
    region
    ):

    time_zone = efu.get_timezone_from_dict(region)
    plug_in_time = pd.Timestamp(
        evu.convert_to_utc(
            plug_in_time,
            time_zone
            )
        )
    horizon = math.ceil(horizon / 12)
    return hist_data.get_historical_forecast_pandas(
        start=plug_in_time - pd.Timedelta(minutes=5), 
        end=plug_in_time, 
        horizon_hours=horizon,
        region=region
    )

# Set up OptCharger based on moer fcsts and get info on projected schedule
def get_schedule_and_cost(
    charge_rate_per_window, 
    charge_needed, 
    total_time_horizon, 
    moer_data,
    asap = False
    ):
    charger = optC.OptCharger(charge_rate_per_window) # charge rate needs to be an int
    moer = Moer.Moer(moer_data['value'])

    charger.fit(
        totalCharge=charge_needed, # also currently an int value
        totalTime = total_time_horizon,
        moer=moer,
        asap=asap
    )
    return charger

# generated_data["plug_in_time" ]= pd.to_datetime(generated_data["plug_in_time"])
# generated_data["unplug_time" ]= pd.to_datetime(generated_data["unplug_time"])

synth_data['moer_data'] = synth_data.apply(
    lambda x: get_historical_fcst_data(
    x.plug_in_time,
    x.total_intervals_plugged_in,
    region = region
    ), axis = 1
)

synth_data['charger']= synth_data.apply(
    lambda x: get_schedule_and_cost(
        x.MWh_fraction,
        x.charge_MWh_needed,
        x.total_intervals_plugged_in, # will throw an error if the plug in time is too shart to reach full charge, should soften to a warning
        x.moer_data,
        asap = True
        ), 
        axis = 1
        )


synth_data['charging_schedule'] = synth_data['charger'].apply(
    lambda x: x.get_schedule()
    )
synth_data['projected_charging_cost_to_full'] = synth_data['charger'].apply(
    lambda  x: x.get_total_cost()
    )