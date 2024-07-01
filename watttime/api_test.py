import os
import time
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from dateutil.parser import parse
from pytz import UTC, timezone

from watttime import WattTimeMyAccess, WattTimeHistorical, WattTimeForecast, WattTimeOptimizer

username = "annie"
password = "" # Replace this with actual password, but do not commit
region = 'PJM_NJ'

now = datetime.now(UTC)
window_start_test = now + timedelta(minutes=10)
window_end_test = now + timedelta(minutes=720)

print("Using Basic Plan")
wt_opt = WattTimeOptimizer(username,password)
optimal_usage_plan = wt_opt.get_optimal_usage_plan(
region = region,
usage_window_start = window_start_test,
usage_window_end = window_end_test,
usage_time_required_minutes = 20,
optimization_method = "basic"
)
print(optimal_usage_plan)


print("Using Basic Plan")
wt_opt = WattTimeOptimizer(username,password)
optimal_usage_plan = wt_opt.get_optimal_usage_plan(
region = region,
usage_window_start = window_start_test,
usage_window_end = window_end_test,
usage_time_required_minutes = 20,
optimization_method = "dp"
)
print(optimal_usage_plan)
