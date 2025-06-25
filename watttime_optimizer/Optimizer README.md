# About the Optimizer Module Project

WattTime data users use WattTime electricity grid-related data for real-time, evidence-based emissions reduction strategies. These data, served programmatically via API, support automation strategies that minimize carbon emissions and human health impacts. In particular, the Marginal Operating Emissions Rate (MOER) can be used to avoid emissions via time- or place-based optimizations, and to calculate the reductions achieved by project-level interventions in accordance with GHG Protocol Scope 4.

Energy generation sources meet different energy demands throughout the day, and the WattTime forecast anticipates the order in which the generators dispatch energy based on anticipated changes in demand. So, the MOER data signal represents the emissions rate of the electricity generator(s) that dispatch energy in direct response to changes in load on the grid.

# Using the Optimizer Class

`WattTimeOptimizer` produces a proposed power usage schedule that minimizes carbon emissions subject to user and device constraints.

The `WattTimeOptimizer` class requires 4 things:

- Watttime’s forecast of marginal emissions (MOER)
- device capacity and energy needs
- region
- window start
- window end

| optimization\_method | ASAP | Charging curve | Time constraint | Contiguous |
| :---- | :---- | :---- | :---- | :---- |
| auto | No | Chooses the fastest algorithm that can still process all inputs |  |  |
| baseline | Yes | Constant | No | No |
| simple | No | Constant | No | No |
| sophisticated | No | Variable | Yes | No |
| contiguous | No | Variable | Yes | Segments at fixed lengths |
| Variable contiguous | No | Variable | Yes | Segments at variable lengths |

Click any of the thumbnails below to see the notebook that generated it.

1.![Naive Smart device charging](https://github.com/jbadsdata/watttime-python-client/edit/optimizer/watttime_optimizer/Optimizer%20README.md#:~:text=ev.-,ipynb,-ev_variable_charge.ipynb))]: needs 30 minutes to reach full charge, expected plug out time within the next 4 hours. Simple use case.
2. Requery: update usage plan every 20 minutes using new forecast for the next 4 hours. Simple use case with recalculation
3. Partial charging guarantee: charge 75% by 8am. User constraint
4. Data center workloads 1:  estimated runtime is 2 hours and it needs to complete by 12pm Contiguous (single period, fixed length)
5. Data center workload 2: needs to run over two usage intervals of lengths 80 min and 40 min. They must complete in that order. Contiguous (multiple periods, fixed length)
6. Compressor: needs to run 120 minutes over the next 12 hours; each cycle needs to be at least 20 minutes long, and any number of contiguous intervals (from one to six) is okay. Contiguous (multiple periods, variable length)

**Naive Smart Device Charging [EV or pluggable battery-powered device]**

```py
from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime_optimizer import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

# 12 hour charge window (720/60 = 12)
now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)

usage_plan = wt_opt.get_optimal_usage_plan(
    region="CAISO_NORTH",
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=240,
    usage_power_kw=12,
    optimization_method="auto",
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

**Partial Charging Guarantee - Introducing Constraints**
  * Sophisticated - total charge window 12 hours long, 75% charged by hour 8.

```py
from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime_optimizer import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

# 12 hour charge window (720/60 = 12)
# Minute 480 is time context when the constraint, i.e. 75% charge, must be satisfied
# 75% of 240 (required charge expressed in minutes) is 180

now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)
usage_time_required_minutes = 240
constraint_time = now + timedelta(minutes=480)
constraint_usage_time_required_minutes = 180
usage_power_kw = 12.0

# map the constraint to the time context
constraints = {constraint_time:constraint_usage_time_required_minutes}

usage_plan = wt_opt.get_optimal_usage_plan(
    region="CAISO_NORTH",
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=240,
    usage_power_kw=12,
    constraints=constraints,
    optimization_method="auto",
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

**Variable Charging Curve - EV**
I know the model of my vehicle and want to match device characteristics. If we have a 10 kWh battery which initially charges at 20kW, the charge rate then linearly decreases to 10kW as the battery is 50%
charged, and then remains at 10kW for the rest of the charging. This is the charging curve.

```py
from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime_optimizer import WattTimeOptimizer
from watttime_optimizer.battery import Battery
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)

battery = Battery(
    initial_soc=0.0,
    charging_curve=pd.DataFrame(
        columns=["SoC", "kW"],
        data=[
            [0.0, 20.0],
            [0.5, 10.0],
            [1.0, 10.0],
        ]
    ),
    capacity_kWh=10.0,
)

variable_usage_power = battery.get_usage_power_kw_df()

usage_plan = wt_opt.get_optimal_usage_plan(
    region="CAISO_NORTH",
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=240,
    usage_power_kw=variable_usage_power,
    optimization_method="auto",
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

* **Data Center Workload 1**:  
  * (single segment, fixed length) - charging schedule to be composed of a single contiguous, i.e. "block" segment of fixed length
		
```py
## AI model training - estimated runtime is 2 hours and it needs to complete within 12 hours

from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime_optimizer import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)

usage_power_kw = 12.0
region = "CAISO_NORTH"

# by passing a single interval of 120 minutes to charge_per_segment, the Optimizer will know to fit call the fixed contigous modeling function.
usage_plan = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=120,
    usage_power_kw=12,
    charge_per_segment=[120],
    optimization_method="auto",
    verbose = False
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

**Data Center Workload 2**: 
  * (multiple segments, fixed length) - runs over two usage periods of lengths 80 min and 40 min. The order of the segments is immutable.

```py
## there are two cycles of length 80 min and 40 min each, and they must be completed in that order. 

from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime_optimizer import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

# Suppose that the time now is 12 midnight
now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)

# Pass two values to charge_per_segment instead of one.
usage_plan = wt_opt.get_optimal_usage_plan(
    region="CAISO_NORTH",
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=120, # 80 + 40
    usage_power_kw=12,
    charge_per_segment=[80,40],
    optimization_method="auto",
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```
