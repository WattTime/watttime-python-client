
import os
os.chdir("/home/jennifer.badolato/watttime-python-client-aer-algo")

import math
import numpy as np
import pandas as pd
import datetime
import pytz
from datetime import datetime, timedelta
from watttime import WattTimeForecast, WattTimeHistorical

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
key = '20240715_1k_synth_users.csv'
generated_data = s3.load_csvdataframe(file=key)


synth_data = generated_data.copy(deep=True)
synth_data = synth_data.head(1)

synth_data["plug_in_time"] = pd.to_datetime(synth_data["plug_in_time"])
synth_data["unplug_time"] = pd.to_datetime(synth_data["unplug_time"])


synth_data['moer_data'] = synth_data.apply(
    lambda x: efu.get_historical_fcst_data(
    x.plug_in_time,
    math.ceil(x.total_intervals_plugged_in),
    region = region
    ), axis = 1
)


synth_data['charger_simple']= synth_data.apply(
    lambda x: efu.get_schedule_and_cost(
        x.MWh_fraction,
        x.charged_MWh_actual,
        math.ceil(x.total_intervals_plugged_in), # will throw an error if the plug in time is too shart to reach full charge, should soften to a warning
        x.moer_data,
        asap = False
        ), 
        axis = 1
        )


synth_data['simple_fit_results'] = synth_data['charger_simple'].apply(
    lambda  x: x.get_total_cost()
    )


synth_data['simple_fit_results']

# TO DO - actuals for comparison against greedy / simple
# Idea: pass the charging schedules, filter, matrix multiplication (np.dot) to get actuals comparison

# make this a function
'''
moer_actuals = actual_data.get_historical_pandas(
    start=generated_data.plug_in_time.min(),
    end=generated_data.unplug_time.max(),
    region=region
)

# adapt this function
def sum_moer_actuals(
    moer_data,
    time_zone,
    MWh_fraction,
    plug_in_time,
    number_conseq_intervals
    ):
    plug_in_time_utc=evu.convert_to_utc(plug_in_time, time_zone)
    index_lower_limit = moer_data[moer_data.point_time >= plug_in_time_utc].index[0]
    index_upper_limit = index_lower_limit + int(number_conseq_intervals)
    return sum(moer_data[index_lower_limit: index_upper_limit]["value"] * MWh_fraction)
'''