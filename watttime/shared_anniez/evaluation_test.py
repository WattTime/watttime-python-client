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

import watttime.shared_anniez.alg.optCharger as optC
import watttime.shared_anniez.alg.moer as Moer



region = "PJM_NJ"
username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

actual_data = WattTimeHistorical(username, password)
hist_data = WattTimeForecast(username, password)

s3 = s3u.s3_utils()
key = '.csv'
generated_data = s3.load_csvdataframe(file=key)

def intervalize_power_rate(kW_value: float, convert_to_MW = True):
    five_min_rate = kW_value / 12
    if convert_to_MW:
        five_min_rate = five_min_rate / 1000
    else:
        five_min_rate
    return five_min_rate

# Get per-row historical fcsts at 'plug in time'
def get_historical_fcst_data(
    plug_in_time, 
    horizon,
    region
    ):
    plug_in_time = pd.Timestamp(plug_in_time) # this needs to be in utc from local
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

generated_data["MWh_fraction"] = generated_data["power_output_rate"].apply(intervalize_power_rate)

generated_data["plug_in_time" ]= pd.to_datetime(generated_data["plug_in_time"])
generated_data["unplug_time" ]= pd.to_datetime(generated_data["unplug_time"])

generated_data["charge_MWh_needed"] = generated_data["total_capacity"] * (0.95 - generated_data["initial_charge"]) / 1000

generated_data["total_intervals_plugged_in"] = generated_data["length_plugged_in"] / 300 # number of seconds in 5 minutes


synth_data['moer_data'] = synth_data.apply(
    lambda x: get_historical_fcst_data(
    x.plug_in_time, # this needs to be converted to utc from local time
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