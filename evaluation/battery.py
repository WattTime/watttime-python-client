# encode the variable power curves
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class Battery:
    capacity_kWh: float
    charging_curve: pd.DataFrame # columns SoC and kW
    initial_soc: float = 0.2

    def plot_changing_curve(self):
        self.charging_curve.set_index("SoC").plot(
            grid=True,
            figsize=(4, 2),
            ylabel="kW",
            legend=False,
            title=f"battery capacity {self.capacity_kWh} kWh"
        )

    def get_usage_power_kw_df(self, max_capacity_fraction=0.95):
        """
        Output the variable charging curve in the format that optimizer accepts.
        That is, dataframe with index "time" in minutes and "power_kw" which
        tells us the average power consumption in a five minute interval
        after an elapsed amount of time of charging.
        """
        capacity_kWh = self.capacity_kWh
        initial_soc = self.initial_soc
        charging_curve = self.charging_curve

        # Convert SoC column to numpy array for faster access
        soc_array = self.charging_curve["SoC"].values
        kW_array = self.charging_curve["kW"].values

        def get_kW_at_SoC(soc):
            """Linear interpolation to get charging rate at any SoC."""
            idx = np.searchsorted(soc_array, soc)
            if idx == 0:
                return kW_array[0]
            elif idx >= len(soc_array):
                return kW_array[-1]
            m1, m2 = soc_array[idx - 1], soc_array[idx]
            p1, p2 = kW_array[idx - 1], kW_array[idx]
            return p1 + (soc - m1) / (m2 - m1) * (p2 - p1)

        # iterate over seconds
        result = []
        secs_elapsed = 0
        charged_kWh = capacity_kWh * initial_soc
        kW_by_second = []
        while charged_kWh < capacity_kWh * max_capacity_fraction:
            secs_elapsed += 1
            curr_soc = charged_kWh / capacity_kWh
            curr_kW = get_kW_at_SoC(charging_curve, curr_soc)
            kW_by_second.append(curr_kW)
            charged_kWh += curr_kW / 3600

            if secs_elapsed % 300 == 0:
                result.append((int(secs_elapsed / 60 - 5), pd.Series(kW_by_second).mean()))
                kW_by_second = []

        return pd.DataFrame(columns=["time", "power_kw"], data=result)