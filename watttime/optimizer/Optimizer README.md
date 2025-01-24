# Optimizer README

## Overview

This code is built to implement and evaluate an algorithm to produce a charging schedule for devices that minimizes carbon emissions subject to a set of constraints. It is based on Watttime’s forecast of marginal emissions combined with inputs related to device capacity and energy needs. The project presents a few optimization algorithms that operate under different assumptions and produce different results. This optionality is part of the API and the results of different algorithms presented are evaluated using actual and forecasted data from power grids in the US. The evaluation section of the project includes a suite of functions to generate synthetic user data with a few behavioral assumptions that can serve to understand the benefits and limitations of our algorithms and evaluate the magnitude of emissions that would be saved if the algorithm were used. 

* **Running the model with constraints:**:  
  * Contiguous (single period, fixed length):

		

```py
## AI model training - estimated runtime is 2 hours and it needs to complete by 12pm

from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime.api import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

# Suppose that the time now is 12 midnight
now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)

usage_time_required_minutes=120
usage_power_kw = 12.0
region = "CAISO_NORTH"

usage_plan = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=usage_time_required_minutes,
    usage_power_kw=usage_power_kw,
    charge_per_interval=[usage_time_required_minutes],
    optimization_method="auto",
    verbose = False
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

* Contiguous (multiple periods, fixed length):

```py
## Dishwasher - there are two cycles of length 80 min and 40 min each, and they must be completed in that order. 

from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

# Suppose that the time now is 12 midnight
now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)

usage_time_required_minutes=120
usage_power_kw = 12.0
region = "CAISO_NORTH"

usage_plan = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=usage_time_required_minutes,
    usage_power_kw=usage_power_kw,
    charge_per_interval=[80,40],
    optimization_method="auto",
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

  * Contiguous (multiple periods, variable length):

		

```py
## Compressor - needs to run 120 minutes over the next 12 hours; each cycle needs to be at least 20 minutes long, and any number of contiguous intervals (from one to six) is okay.

from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

# Suppose that the time now is 12 midnight
now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)

usage_time_required_minutes=120
usage_power_kw = 12.0
region = "CAISO_NORTH"

usage_plan = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=usage_time_required_minutes,
    usage_power_kw=usage_power_kw,
    # Here _None_ implies that there is no upper bound, and replacing None by 120 would have the exact same effect.
    charge_per_interval=[(20,None),(20,None),(20,None),(20,None),(20,None),(20,None)],
    optimization_method="auto",
    use_all_intervals=False
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

* Partial charging guarantee:

```py
## I would like to charge 75% by 8am in case of any emergencies (airport, kid bus, roadtrip)

from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

# Suppose that the time now is 12 midnight
now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)
usage_time_required_minutes = 240
constraint_time = now + timedelta(minutes=480)
constraint_usage_time_required_minutes = 180
constraints = {constraint_time:constraint_usage_time_required_minutes}
usage_power_kw = 12.0
region = "CAISO_NORTH"

usage_plan = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=240,
    usage_power_kw=usage_power_kw,
    constraints=constraints,
    optimization_method="auto",
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())

```
## Optimizer: basic principles and options

The **basic intuition of the algorithm** is that when a device is plugged in for longer than the time required to fully charge it there exist ways to pick charging vs. non-charging time intervals such that the device draws power from the grid during cleaner intervals and thus minimizes emissions. **The algorithm takes as inputs** user and device parameters such as: the plug-in and plug-out times of the device, as well as the charging curve that determines the time it takes to charge it as well as the power that it needs to draw. **As an output, it produces** a charging schedule that divides the time between plug-in and plug-out time into charging and non-charging intervals such that emissions are minimized. Watttime’s forecast provides the basic building block for these algorithms as it forecasts when those relatively cleaner grid periods occur. 

There are **three different optimization algorithms** that are implemented in the API (alongside a baseline algorithm that just charges the device from the moment it’s plugged in to when it is fully charged, which is what devices do out of the box). We first start with **a simple algorithm** that, under full information about plug out time, uses the forecast to find the lowest possible emission interval that charges the device and outputs a charge schedule based on that. We then follow with a **sophisticated** version of the algorithm which takes into account variable charging curves and implements a dynamic optimization algorithm to adjust for the fact that device charging curves are non-linear. We provide additional functionality in the **fixed contiguous** and **contiguous** versions of the algorithms, which can enforce the charging schedule to be composed of several contiguous intervals; the length of each interval is either fixed or falls in a provided range.

| optimization\_method | ASAP | Charging curve | Time constraint | Contiguous |
| :---- | :---- | :---- | :---- | :---- |
| baseline | Yes | Constant | No | No |
| simple | No | Constant | No | No |
| sophisticated | No | Variable | Yes | No |
| contiguous | No | Variable | Yes | Intervals at fixed lengths |
| Variable contiguous | No | Variable | Yes | Intervals at variable lengths |
| auto | No | Chooses the fastest algorithm that can still process all inputs |  |  |

Evaluating the effectiveness of the algorithm, as well as the conditions that maximize emissions savings, we have implemented a suite of functions that generate synthetic user data that can be evaluated on data from the largest electrical grids in the US. The code here also contains these functions, which can be modified and are meant to capture behavioral assumptions of how users charge devices. 

A final note on device types (this is focused for now on EVs, but altering some of the behavioral assumptions of usage \+ the device charging curves can extend this functionality to other devices.)

### Raw Inputs

*What we simulate for each use case*

- Capacity C  
  - Might also need init battery capacity if we don’t start from 0%   
  - Unit: kWh  
  - Type: energy  
- Power usage curve from capacity Q:cp   
  - Marginal power usage to charge battery when it’s currently at capacity c  
  - Unit: kW  
  - Type: power  
- Marginal emission rate M:tm  
  - Unit: lb/MWh  
  - Type: emission per energy  
  - We convert this to lb/kWh by multiplying M by 0.001  
- OPT\_INTERVAL   
  - Smallest interval on which we have constant charging behavior and emissions.   
  - Currently set to 5 minutes  
  - We have now discretized time into L=T/ intervals of length . The l-th interval is lt\<(l+1).  
- Contiguity   
- Constraints

### API Inputs 

*When calling the API*

- usage\_time\_required\_minutes Tr  
  - We compute this using C and Q. See example below.   
  - Unit: mins  
- usage\_power\_kw P:tp  
  - Marginal power usage to charge battery when it has been charged for t minutes. Converted from Q. 	  
  - Unit: kW  
- usage\_window\_start, usage\_window\_end  
  - These are timestamps to specify the charging window 

### Algorithm

Find schedule s0,...,sL-1 that minimizes total emission 60l=0L-1slPl'=0l-1sl' Ml1000subject to 

* sl0,1  
  * sl=1 if we charge on interval l  
  * sl=0 if we do not charge on interval l  
  * As an extension for future use cases, suppose we can supercharge the battery by consuming up to K times as much power. The DP algorithm will also be able to handle this optimization and output a schedule with sl0,1,...,K  
* l=0L-1sl=Tr  
  * This just means that we charge for a total of Tr minutes according to this schedule.

### API Output

 A data frame with \-spaced timestamp index and charging usage. For example, 

| time | usage (min) | energe\_use\_mwh | emissions\_co2e\_lb |
| :---- | :---- | :---- | :---- |
| 2024-7-26 15:00:00+00:00 | 5 | 0.0001 | 1.0 |
| 2024-7-26 15:05:00+00:00 | 5 | 0.0001 | 1.0 |
| 2024-7-26 15:10:00+00:00 | 0 | 0\. | 0.0 |
| 2024-7-26 15:15:00+00:00 | 5 | 0.00005 | 0.5 |

This would mean that we charge from 15:00-15:10 and then from 15:15-15:20. Note that the last column reflects **forecast emissions** based on forecast emission rate rather than the actuals. To compute actual emissions, you can take the dot product of energey\_use\_mwh and actual emission rates. 

In mathematical terms, we compute these three columns as follows. For the l-th interval tl,l+1, we have

* usage: sl  
* energy\_use\_mwh: 1100060slPl'=0l-1sl', where 60 reflects interval length in hours and Pl'=0l-1sl' is the power. 11000 is a conversion factor from kWh to mWh  
* emissions\_co2e\_lb: energy\_us\_mwh \* Mt.

# FAQs

## How much co2 can we expect to avoid by using the optimizer?

The amount of emission you can avoid will vary significantly based on a range of factors. For example:

* The grid where the charging is occurring.  
* The amount of “slack time” available, that is, possible charging time beyond the minimum amount required for charging.