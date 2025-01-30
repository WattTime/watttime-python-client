import threading
import time
from watttime.api import WattTimeOptimizer, Recalculator
import os
from datetime import datetime, timedelta
from pytz import UTC

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

window_start = datetime.now(UTC) + timedelta(minutes=10)
window_end = datetime.now(UTC) + timedelta(minutes=720)

initial_plan = wt_opt.get_optimal_usage_plan(
            region='CAISO_NORTH',
            usage_window_start= datetime.now(UTC) + timedelta(minutes=10),
            usage_window_end=window_end,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="baseline",
        )

recalculator = Recalculator(
            initial_schedule=initial_plan,
            start_time=window_start,
            end_time=window_end,
            total_time_required=240
        )


def timer(timer_runs):
    # (4) The code runs while the boolean is true.
    while timer_runs.is_set():
        new_time_required = recalculator.get_remaining_time_required(new_window_start)
        usage_plan = wt_opt.get_optimal_usage_plan(
            region='CAISO_NORTH',
            usage_window_start= datetime.now(UTC) + timedelta(minutes=10),
            usage_window_end=window_end,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="baseline",
        )
        print(usage_plan.head())
        time.sleep(10)

# (1) Create the boolean which indicates whether the secondary
# thread runs or not.
timer_runs = threading.Event()
# (2) Initialize it as true.
timer_runs.set()
# (3) Pass it as an argument so it can be read by the timer.
t = threading.Thread(target=timer, args=(timer_runs,))
t.start()
timer_runs.clear()