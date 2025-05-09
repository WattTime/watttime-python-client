from watttime_optimizer.battery import Battery, CARS
import pandas as pd

tesla_charging_curve = pd.DataFrame(
        columns=["SoC", "kW"],
        data = CARS['tesla']
    )

capacity_kWh = 70
initial_soc = .50

batt = Battery(tesla_charging_curve)

df = batt.get_usage_power_kw_df(capacity_kWh=capacity_kWh, initial_soc=initial_soc)

print(df.head())