# About
This SDK is meant to help users with basic queries to WattTime’s API (version 3), and to get data returned in specific formats (e.g., JSON, pandas, csv).

Users may register for access to the WattTime API through this client, however the basic user scoping given will only allow newly registered users to access data for the `CAISO_NORTH` region. Additionally, data may not be available for all signal types for newly registered users.

Full documentation of WattTime's API, along with response samples and information about [available endpoints is also available](https://docs.watttime.org/).

# Configuration
The SDK can be installed as a python package from the PyPi repository, we recommend using an environment manager such as [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [venv](https://docs.python.org/3/library/venv.html).
```
pip install watttime
```

If you are not registered for the WattTime API, you can do so using the SDK:
```
from watttime import WattTimeMyAccess

wt = WattTimeMyAccess(username=<USERNAME>, password=<PASSWORD>)
wt.register(email=<EMAIL>, organization=<ORGANIZATION>)

```

If you are already registered for the WattTime API, you may set your credentials as environment variables to avoid passing these during class initialization:
```
# linux or mac
export WATTTIME_USER=<your WattTime API username>
export WATTTIME_PASSWORD=<your WattTime API password>
```

Once you have set your credentials as environment variables, you can omit passing `username` and `password` when instantiating sdk objects. For instance, in the example below, you could replace the second line with

```python
wt_myaccess = WattTimeMyAccess()
```

# Using the SDK
Users may first want to query the `/v3/my-access` endpoint using the `WattTimeMyAccess` class to get a dataframe of regions and signal types available to them:

```python
from watttime import WattTimeMyAccess

wt_myaccess = WattTimeMyAccess(username, password)

# return a nested json describing signals and regions you have access to
wt_myaccess.get_access_json()

# return a pandas dataframe describing signals and regions you have access to
wt_myaccess.get_access_pandas()
```

### Accessing Historical Data

Once you confirm your access, you may wish to request data for a particular region:

```python
from watttime import WattTimeHistorical

wt_hist = WattTimeHistorical(username, password)

# get data as a pandas dataframe
moers = wt_hist.get_historical_pandas(
    start = '2022-01-01 00:00Z', # ISO 8601 format, UTC
    end = '2023-01-01 00:00Z', # ISO 8601 format, UTC
    region = 'CAISO_NORTH',
    signal_type = 'co2_moer' # ['co2_moer', 'co2_aoer', 'health_damage', etc.]
)

# save data as a csv -> ~/watttime_historical_csvs/<region>_<signal_type>_<start>_<end>.csv
wt_hist.get_historical_csv(
    start = '2022-01-01 00:00Z', # ISO 8601 format, UTC
    end = '2023-01-01 00:00Z', # ISO 8601 format, UTC
    region = 'CAISO_NORTH',
    signal_type = 'co2_moer' # ['co2_moer', 'co2_aoer', 'health_damage', etc.]
)
```

You could also combine these classes to iterate through all regions where you have access to data:

```python
from watttime import WattTimeMyAccess, WattTimeHistorical
import pandas as pd

wt_myaccess = WattTimeMyAccess(username, password)
wt_hist = WattTimeHistorical(username, password)

access_df = wt_myaccess.get_access_pandas()

moers = pd.DataFrame()
moer_regions = access_df.loc[access_df['signal_type'] == 'co2_moer', 'region'].unique()
for region in moer_regions:
    region_df = wt_hist.get_historical_pandas(
        start = '2022-01-01 00:00Z',
        end = '2023-01-01 00:00Z',
        region = region,
        signal_type = 'co2_moer'
    )
    moers = pd.concat([moers, region_df], axis='rows')
```

### Accessing Real-Time and Historical Forecasts
You can also use the SDK to request a current forecast for some signal types, such as co2_moer and health_damage:

```python
from watttime import WattTimeForecast

wt_forecast = WattTimeForecast(username, password)
forecast = wt_forecast.get_forecast_json(
    region = 'CAISO_NORTH',
    signal_type = 'health_damage'
)

```
We recommend using the `WattTimeForecast` class to access data for real-time optimization. The first item of the response from this call is always guaranteed to be an estimate of the signal_type for the current five minute period, and forecasts extend at least 24 hours at a five minute granularity, which is useful for scheduling utilization during optimal times.

Methods also exist to request historical forecasts, however these responses may be slower as the volume of data can be significant:
```python
hist_forecasts = wt_forecast.get_historical_forecast_json(
    start = '2022-12-01 00:00+00:00',
    end = '2022-12-31 23:59+00:00',
    region = 'CAISO_NORTH',
    signal_type = 'health_damage'
)
```

### Accessing Location Data
We provide two methods to access location data:

1) The `region_from_loc()` method allows users to provide a latitude and longitude coordinates in order to receive the valid region for a given signal type.

2) the `WattTimeMaps` class provides a `get_maps_json()` method which returns a [GeoJSON](https://en.wikipedia.org/wiki/GeoJSON) object with complete boundaries for all regions available for a given signal type. Note that access to this endpoint is only available for Pro and Analyst subscribers. 

```python
from watttime import WattTimeMaps

wt = WattTimeMaps()

# get BA region for a given location
wt.region_from_loc(
    latitude=39.7522,
    longitude=-105.0,
    signal_type='co2_moer'
)

# get shape files for all regions of a signal type
wt.get_maps_json('co2_moer')
```

# Using the Optimizer Class
`WattTime.Optimizer` produces a power consumption schedule that minimizes carbon emissions subject to user and device constraints.

The `WattTime.Optimizer` class requires 4 things:

- Watttime’s forecast of marginal emissions
- device capacity and energy needs
- region
- window start
- window end


| optimization\_method | ASAP | Charging curve | Time constraint | Contiguous |
| :---- | :---- | :---- | :---- | :---- |
| baseline | Yes | Constant | No | No |
| simple | No | Constant | No | No |
| sophisticated | No | Variable | Yes | No |
| contiguous | No | Variable | Yes | Intervals at fixed lengths |
| Variable contiguous | No | Variable | Yes | Intervals at variable lengths |
| auto | No | Chooses the fastest algorithm that can still process all inputs |  |  |

Click any of the thumbnails below to see the notebook that generated it.

1. Naive Smart device charging: needs 30 minutes to reach full charge, expected plug out time within the next 4 hours. Simple use case.
2. Requery: update usage plan every 20 minutes using new forecast for the next 4 hours. Simple use case with recalculation
3. Partial charging guarantee: charge 75% by 8am. User constraint
4. Data center workloads:  estimated runtime is 2 hours and it needs to complete by 12pm Contiguous (single period, fixed length)
5. Dishwasher: needs to run over two usage intervals of lengths 80 min and 40 min. They must complete in that order. Contiguous (multiple periods, fixed length)
6. Compressor: needs to run 120 minutes over the next 12 hours; each cycle needs to be at least 20 minutes long, and any number of contiguous intervals (from one to six) is okay. Contiguous (multiple periods, variable length)

**Naive Smart Device Charging**
  * Simple - uses the MOER forecast at window start to find the set of intervals that minimize emissions, and outputs a charge schedule based on that.

```py
from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime import WattTimeOptimizer
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
from watttime import WattTimeOptimizer
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
  * Sophisticated - total charge window 12 hours long, 75% charged by hour 8.

```py
from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime import WattTimeOptimizer
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
variable_usage_power = ''

usage_plan = wt_opt.get_optimal_usage_plan(
    region="CAISO_NORTH",
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=240,
    usage_power_kw=variable_usage_power,
    constraints=constraints,
    optimization_method="auto",
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```


* **Data Center Workload**:  
  * Fixed Contiguous (single period, fixed length) - charging schedule to be composed of contiguous interval(s) of fixed length
		
```py
## AI model training - estimated runtime is 2 hours and it needs to complete within 12 hours

from datetime import datetime, timedelta
import pandas as pd
from pytz import UTC
from watttime.api import WattTimeOptimizer
import os

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

now = datetime.now(UTC)
window_start = now
window_end = now + timedelta(minutes=720)

usage_power_kw = 12.0
region = "CAISO_NORTH"

# by passing a single interval of 120 minutes to charge_per_interval, the Optimizer will know to fit call the fixed contigous modeling function.
usage_plan = wt_opt.get_optimal_usage_plan(
    region=region,
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=120,
    usage_power_kw=12,
    charge_per_interval=[120],
    optimization_method="auto",
    verbose = False
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

**Dishwasher**: 
  * Fixed Contiguous (multiple periods, fixed length) - runs over two usage intervals of lengths 80 min and 40 min. The order of the intervals is immutable.

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

# Pass two values to charge_per_interval instead of one.
usage_plan = wt_opt.get_optimal_usage_plan(
    region="CAISO_NORTH",
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=120, # 80 + 40
    usage_power_kw=12,
    charge_per_interval=[80,40],
    optimization_method="auto",
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```

**Compressor**:
  * Contiguous (multiple periods, variable length) - runs 120 minutes over the next 12 hours; each cycle needs to be at least 20 minutes long, and any number of intervals (from one to six) is okay.
		
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

usage_plan = wt_opt.get_optimal_usage_plan(
    region="CAISO_NORTH",
    usage_window_start=window_start,
    usage_window_end=window_end,
    usage_time_required_minutes=120,
    usage_power_kw=12,
    # Here _None_ implies that there is no upper bound, and replacing None by 120 would have the exact same effect.
    charge_per_interval=[(20,None),(20,None),(20,None),(20,None),(20,None),(20,None)],
    optimization_method="auto",
    use_all_intervals=False
)

print(usage_plan.head())
print(usage_plan["usage"].tolist())
print(usage_plan.sum())
```