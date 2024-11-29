# /bin/python3 -m watttime.api_test /home/annie.zhu/watttime-python-client-aer-algo/watttime/api_test.py 
import os
from datetime import datetime, timedelta

from dateutil.parser import parse
from pytz import UTC, timezone

from watttime import (
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


print("Using auto mode, but constrained to two contiguous intervals with fixed length constraints")
dp_usage_plan_6 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    charge_per_interval=[60,100],
    optimization_method="auto",
)
print(dp_usage_plan_6["usage"].tolist())
print(dp_usage_plan_6.sum())

print("Using auto mode, but constrained to two contiguous intervals with variable length constraints")
dp_usage_plan_7 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    charge_per_interval=[(60,100),(60,100)],
    optimization_method="auto",
)
print(dp_usage_plan_7["usage"].tolist())
print(dp_usage_plan_7.sum())

print("Using auto mode, but constrained to two contiguous intervals with variable length constraints, and doesn't need to use all intervals")
dp_usage_plan_8 = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start_test,
    usage_window_end=window_end_test,
    usage_time_required_minutes=160,
    usage_power_kw=usage_power_kw,
    charge_per_interval=[(40,None)]*4,
    use_all_intervals=False,
    optimization_method="auto",
)
print(dp_usage_plan_8["usage"].tolist())
print(dp_usage_plan_8.sum())