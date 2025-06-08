# encode the variable power curves
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Battery:
    capacity_kWh: float
    charging_curve: pd.DataFrame # columns SoC and kW
    initial_soc: float = 0.2

    def plot_charging_curve(self, ax=None):
        """Plot the variabel charging curve of the battery"""
        ax = self.charging_curve.set_index("SoC").plot(
            ax=ax,
            grid=True,
            ylabel="kW",
            legend=False,
            title=f"Charging curve \nBattery capacity: {self.capacity_kWh} kWh"
        )
        if ax is None:
            plt.show()

    def get_usage_power_kw_df(self, max_capacity_fraction=0.95):
        """
        Output the variable charging curve in the format that optimizer accepts.
        That is, dataframe with index "time" in minutes and "power_kw" which
        tells us the average power consumption in a five minute interval
        after an elapsed amount of time of charging.
        """
        capacity_kWh = self.capacity_kWh
        initial_soc = self.initial_soc
        # convert SoC column to numpy array for faster access
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
            curr_kW = get_kW_at_SoC(curr_soc)
            kW_by_second.append(curr_kW)
            charged_kWh += curr_kW / 3600

            if secs_elapsed % 300 == 0:
                result.append((int(secs_elapsed / 60 - 5), pd.Series(kW_by_second).mean()))
                kW_by_second = []

        return pd.DataFrame(columns=["time", "power_kw"], data=result)   

CARS_L3 = {
    # pulled data from https://www.fastnedcharging.com/en/brands-overview
    # this is a subset of the cars 
    "audi": [ # 71kWh, https://www.fastnedcharging.com/en/brands-overview/audi
        [0.0, 120.0],
        [0.6, 120.0],
        [1.0, 30.0],
    ],
    "bmw": [ # 42.2kWh, https://www.fastnedcharging.com/en/brands-overview/bmw
        [0.0, 40.0],
        [0.85, 50.0],
        [1.0, 5.0],
    ],
    'bolt':[
            [0.0, 50.0],
            [0.5, 50.0],
            [0.93, 20.0],
            [1.0, 0.5],
        ],
    "honda": [ # 35.5kWh, https://www.fastnedcharging.com/en/brands-overview/honda
        [0.0, 40.0],
        [0.4, 40.0],
        [0.41, 30.0],
        [0.70, 30.0],
        [0.71, 20.0],
        [0.95, 20.0],
        [1.0, 10.0],
    ],
    "lucid": [ # 112kWh https://www.fastnedcharging.com/en/brands-overview/lucid
        [0.0, 300.0],
        [1.0, 50.0],
    ],
    "mazda": [ #35.5kWh https://www.fastnedcharging.com/en/brands-overview/mazda
        [0.0, 50.0],
        [0.2, 50.0],
        [0.21, 40.0],
        [1.0, 10.0],
    ],
    "subaru": [ # 75kWh https://www.fastnedcharging.com/en/brands-overview/subaru
        [0.0, 150.0],
        [0.25, 150.0],
        [0.85, 30.0],
        [1.00, 30.0],
    ],
    "tesla": [ # ??kWh https://www.fastnedcharging.com/en/brands-overview/tesla
        [0.0, 180.0],
        [0.4, 190.0],
        [0.9, 40.0],
        [1.0, 40.0],
    ],
    "volkswagen": [ # 24.2kWh https://www.fastnedcharging.com/en/brands-overview/volkswagen?model=e-Golf
        [0.0, 40.0],
        [0.1, 40.0],
        [0.75, 45.0],
        [0.81, 23.0],
        [0.92, 17.0],
        [0.95, 9.0],
        [1.0, 9.0],
    ]
}

TZ_DICTIONARY = {
    "AECI": "America/Chicago",
    "AVA": "America/Los_Angeles",
    "AZPS": "America/Phoenix",
    "BANC": "America/Los_Angeles",
    "BPA": "America/Los_Angeles",
    "CAISO_ESCONDIDO": "America/Los_Angeles",
    "CAISO_LONGBEACH": "America/Los_Angeles",
    "CAISO_NORTH": "America/Los_Angeles",
    "CAISO_PALMSPRINGS": "America/Los_Angeles",
    "CAISO_REDDING": "America/Los_Angeles",
    "CAISO_SANBERNARDINO": "America/Los_Angeles",
    "CAISO_SANDIEGO": "America/Los_Angeles",
    "CHPD": "America/Los_Angeles",
    "CPLE": "America/New_York",
    "CPLW": "America/New_York",
    "DOPD": "America/Los_Angeles",
    "DUK": "America/New_York",
    "ELE": "America/Denver",
    "ERCOT_AUSTIN": "America/Chicago",
    "ERCOT_COAST": "America/Chicago",
    "ERCOT_EASTTX": "America/Chicago",
    "ERCOT_HIDALGO": "America/Chicago",
    "ERCOT_NORTHCENTRAL": "America/Chicago",
    "ERCOT_PANHANDLE": "America/Chicago",
    "ERCOT_SANANTONIO": "America/Chicago",
    "ERCOT_SECOAST": "America/Chicago",
    "ERCOT_SOUTHTX": "America/Chicago",
    "ERCOT_WESTTX": "America/Chicago",
    "FMPP": "America/New_York",
    "FPC": "America/New_York",
    "FPL": "America/New_York",
    "GVL": "America/New_York",
    "IID": "America/Los_Angeles",
    "IPCO": "America/Boise",
    "ISONE_CT": "America/New_York",
    "ISONE_ME": "America/New_York",
    "ISONE_NEMA": "America/New_York",
    "ISONE_NH": "America/New_York",
    "ISONE_RI": "America/New_York",
    "ISONE_SEMA": "America/New_York",
    "ISONE_VT": "America/New_York",
    "ISONE_WCMA": "America/New_York",
    "JEA": "America/New_York",
    "LDWP": "America/Los_Angeles",
    "LGEE": "America/New_York",
    "MISO_INDIANAPOLIS": "America/Indiana/Indianapolis",
    "MISO_N_DAKOTA": "America/North_Dakota/Center",
    "MPCO": "America/Denver",
    "NEVP": "America/Los_Angeles",
    "NYISO_NYC": "America/New_York",
    "PACE": "America/Denver",
    "PACW": "America/Los_Angeles",
    "PGE": "America/Los_Angeles",
    "PJM_CHICAGO": "America/Chicago",
    "PJM_DC": "America/New_York",
    "PJM_EASTERN_KY": "America/New_York",
    "PJM_EASTERN_OH": "America/New_York",
    "PJM_ROANOKE": "America/New_York",
    "PJM_NJ": "America/New_York",
    "PJM_SOUTHWEST_OH": "America/New_York",
    "PJM_WESTERN_KY": "America/New_York",
    "PNM": "America/Denver",
    "PSCO": "America/Denver",
    "PSEI": "America/Los_Angeles",
    "SC": "America/New_York",
    "SCEG": "America/New_York",
    "SCL": "America/Los_Angeles",
    "SEC": "America/New_York",
    "SOCO": "America/Chicago",
    "SPA": "America/Chicago",
    "SPP_FORTPECK": "America/Denver",
    "SPP_KANSAS": "America/Chicago",
    "SPP_KC": "America/Chicago",
    "SPP_MEMPHIS": "America/Chicago",
    "SPP_ND": "America/North_Dakota/Beulah",
    "SPP_OKCTY": "America/Chicago",
    "SPP_SIOUX": "America/Chicago",
    "SPP_SPRINGFIELD": "America/Chicago",
    "SPP_SWOK": "America/Chicago",
    "SPP_TX": "America/Chicago",
    "SPP_WESTNE": "America/Chicago",
    "SRP": "America/Phoenix",
    "TAL": "America/New_York",
    "TEC": "America/New_York",
    "TEPC": "America/Phoenix",
    "TID": "America/Los_Angeles",
    "TPWR": "America/Los_Angeles",
    "TVA": "America/Chicago",
    "WACM": "America/Denver",
    "WALC": "America/Phoenix",
    "WAUW": "America/Denver",
}

MOER_REGION_LIST = [
    "AECI",
    "AVA",
    "AZPS",
    "BANC",
    "BPA",
    "CAISO_ESCONDIDO",
    "CAISO_LONGBEACH",
    "CAISO_NORTH",
    "CAISO_PALMSPRINGS",
    "CAISO_REDDING",
    "CAISO_SANBERNARDINO",
    "CAISO_SANDIEGO",
    "CHPD",
    "CPLE",
    "CPLW",
    "DOPD",
    "DUK",
    "ELE",
    "ERCOT_AUSTIN",
    "ERCOT_COAST",
    "ERCOT_EASTTX",
    "ERCOT_HIDALGO",
    "ERCOT_NORTHCENTRAL",
    "ERCOT_PANHANDLE",
    "ERCOT_SANANTONIO",
    "ERCOT_SECOAST",
    "ERCOT_SOUTHTX",
    "ERCOT_WESTTX",
    "FMPP",
    "FPC",
    "FPL",
    "GVL",
    "IID",
    "IPCO",
    "ISONE_CT",
    "ISONE_ME",
    "ISONE_NEMA",
    "ISONE_NH",
    "ISONE_RI",
    "ISONE_SEMA",
    "ISONE_VT",
    "ISONE_WCMA",
    "JEA",
    "LDWP",
    "LGEE",
    "MISO_INDIANAPOLIS",
    "MISO_N_DAKOTA",
    "MPCO",
    "NEVP",
    "NYISO_NYC",
    "PACE",
    "PACW",
    "PGE",
    "PJM_CHICAGO",
    "PJM_DC",
    "PJM_EASTERN_KY",
    "PJM_EASTERN_OH",
    "PJM_NJ",
    "PJM_SOUTHWEST_OH",
    "PJM_WESTERN_KY",
    "PNM",
    "PSCO",
    "PSEI",
    "SC",
    "SCEG",
    "SCL",
    "SEC",
    "SOCO",
    "SPA",
    "SPP_FORTPECK",
    "SPP_KANSAS",
    "SPP_KC",
    "SPP_MEMPHIS",
    "SPP_ND",
    "SPP_OKCTY",
    "SPP_SIOUX",
    "SPP_SPRINGFIELD",
    "SPP_SWOK",
    "SPP_TX",
    "SPP_WESTNE",
    "SRP",
    "TAL",
    "TEC",
    "TEPC",
    "TID",
    "TPWR",
    "TVA",
    "WACM",
    "WALC",
    "WAUW",
]