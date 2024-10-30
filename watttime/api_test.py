import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
from dateutil.parser import parse
from pytz import UTC, timezone
import matplotlib.pyplot as plt

from watttime import (
    # WattTimeMyAccess,
    # WattTimeHistorical,
    # WattTimeForecast,
    WattTimeOptimizer,
)

region = "PJM_NJ"
username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

wt_opt = WattTimeOptimizer(username, password)

now = datetime.now(UTC)
window_start_test = now + timedelta(minutes=10)
window_end_test = now + timedelta(minutes=720)
usage_power_kw = 12

print("Using Baseline Plan")
basic_usage_plan = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    optimization_method="baseline",
)
print(basic_usage_plan["emissions_co2e_lb"].sum())

print("Using DP Plan w/ fixed power rate (kW)")
dp_usage_plan = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    optimization_method="sophisticated",
)
print(dp_usage_plan["emissions_co2e_lb"].sum())

print("Using DP Plan w/ fixed power rate and charging uncertainty")
dp_usage_plan_2 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    usage_time_uncertainty_minutes=180,
    optimization_method="sophisticated",
)
print(dp_usage_plan_2["emissions_co2e_lb"].sum())


print("Using DP Plan w/ variable power rate (kW)")
usage_power_kw_df = pd.DataFrame(
    [[0, 12], [20, 12], [40, 12], [100, 12], [219, 12], [220, 2.4], [320, 2.4]],
    columns=["time", "power_kw"],
)
dp_usage_plan_3 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=320,
    usage_power_kw=usage_power_kw_df,
    optimization_method="auto",
)
print(dp_usage_plan_3["emissions_co2e_lb"].sum())


print("Using auto mode, but with a non-round usage time minutes")
dp_usage_plan_4 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=7,
    usage_power_kw=usage_power_kw,
    optimization_method="auto",
)
print(dp_usage_plan_4)
print(dp_usage_plan_4.sum())


print("Using auto mode, but constrained to a single contiguous interval")
dp_usage_plan_5 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    charge_per_interval=[(160,160)],
    optimization_method="auto",
)
print(dp_usage_plan_5["usage"].tolist())
print(dp_usage_plan_5.sum())

print("Using auto mode, but constrained to two contiguous intervals with length constraints")
dp_usage_plan_6 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    charge_per_interval=[(60,100),(60,100)],
    optimization_method="auto",
)
print(dp_usage_plan_6["usage"].tolist())
print(dp_usage_plan_6.sum())

print("Using auto mode, but constrained to two contiguous intervals with length constraints")
dp_usage_plan_7 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    charge_per_interval=[60,100],
    optimization_method="auto",
)
print(dp_usage_plan_7["usage"].tolist())
print(dp_usage_plan_7.sum())