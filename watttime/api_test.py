import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
from dateutil.parser import parse
from pytz import UTC, timezone
import matplotlib.pyplot as plt

from watttime import WattTimeMyAccess, WattTimeHistorical, WattTimeForecast, WattTimeOptimizer

region = 'PJM_NJ'
username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

wt_opt = WattTimeOptimizer(username,password)

now = datetime.now(UTC)
window_start_test = now + timedelta(minutes=10)
window_end_test = now + timedelta(minutes=720)
usage_power_kw = 12

print("Using Baseline Plan")
basic_usage_plan = wt_opt.get_optimal_usage_plan(
region = region,
usage_window_start = window_start_test,
usage_window_end = window_end_test,
usage_time_required_minutes = 240,
usage_power_kw = usage_power_kw,
optimization_method = "baseline"
)


print("Using Simple Plan")
dp_usage_plan = wt_opt.get_optimal_usage_plan(
region = region,
usage_window_start = window_start_test,
usage_window_end = window_end_test,
usage_time_required_minutes = 240,
usage_power_kw = usage_power_kw,
optimization_method = "simple"
)


print("Using DP Plan")
dp_usage_plan = wt_opt.get_optimal_usage_plan(
region = region,
usage_window_start = window_start_test,
usage_window_end = window_end_test,
usage_time_required_minutes = 240,
usage_power_kw = usage_power_kw,
optimization_method = "sophisticated"
)

print("Using DP Plan with charging uncertainty")
dp_usage_plan_2 = wt_opt.get_optimal_usage_plan(
region = region,
usage_window_start = window_start_test,
usage_window_end = window_end_test,
usage_time_required_minutes = 240,
usage_power_kw = usage_power_kw,
usage_time_uncertainty_minutes = 180,
optimization_method = "sophisticated"
)

print("Using DP Plan with variable power rate")
usage_power_kw_df = pd.DataFrame(
    [[0, 12], [20, 12], [40, 12], [100, 12], [219, 12], [220, 2.4], [320, 2.4]],
    columns=["time", "power_kwh"]
)

dp_usage_plan_3 = wt_opt.get_optimal_usage_plan(
region = region,
usage_window_start = window_start_test,
usage_window_end = window_end_test,
usage_time_required_minutes = 320,
usage_power_kw = usage_power_kw_df,
optimization_method = "sophisticated"
)

# print(basic_usage_plan.index[0], basic_usage_plan.index[-1], window_start_test, window_end_test)