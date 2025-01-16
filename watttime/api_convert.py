import pandas as pd
import numpy as np


# This file contains utility functions for converting formats for now
def convert_soc_to_soe(soc_power_df, voltage_curve, battery_capacity_coulombs):
    """
    Convert State of Charge (SoC) to State of Energy (SoE) by integrating voltage over SoC.

    Parameters:
    soc_power_df (pd.DataFrame): DataFrame with 'SoC' and 'power_kw' columns.
    voltage_curve (function): Voltage as a function of SoC.
    battery_capacity_coulombs (float): Maximum current capacity of the battery in coulombs.

    Returns:
    pd.DataFrame: DataFrame with 'SoE' and 'power_kw' columns.
    """
    soc = soc_power_df["SoC"]

    # Voltage at each SoC
    voltage = voltage_curve(soc)

    # Calculate differential SoC for numerical integration
    delta_soc = np.diff(soc, prepend=0)
    charge_per_interval = delta_soc * battery_capacity_coulombs
    # Energy is voltage * charge
    energy_kwh = np.cumsum(voltage * charge_per_interval * 0.001 / 3600)

    # Normalize so that State of energy goes from 0 to 1
    soe_array = energy_kwh / energy_kwh.iloc[-1]

    # Create a new DataFrame with 'SoE' and 'power_kw'
    soe_power_df = pd.DataFrame(
        {"SoE": soe_array, "power_kw": soc_power_df["power_kw"]}
    )

    return soe_power_df


def convert_soe_to_time(soe_power_df, battery_capacity):
    """
    Convert Power vs SoE DataFrame to a Power vs Time DataFrame.

    Parameters:
    soe_power_df (pd.DataFrame): DataFrame with 'SoE' and 'power_kw' columns.
    battery_capacity (float): Maximum energy capacity of the battery in kWh.

    Returns:
    pd.DataFrame: DataFrame with 'time' (in minutes) and 'power_kw' columns.
    """
    time_list = [0]  # Starting at t = 0 minutes
    previous_time = 0

    for i in range(len(soe_power_df) - 1):
        # Calculate the delta SoE
        delta_soe = soe_power_df["SoE"].iloc[i + 1] - soe_power_df["SoE"].iloc[i]

        # Energy transferred for this delta SoE
        delta_energy = delta_soe * battery_capacity  # in kWh

        # Power to use during this step
        power_to_use = soe_power_df["power_kw"].iloc[i]

        # Time step for this segment
        delta_time_minutes = delta_energy / power_to_use * 60

        # Add the time to the previous time to get cumulative time
        current_time = previous_time + delta_time_minutes
        time_list.append(current_time)
        previous_time = current_time

    # Convert SoE dataframe to Time dataframe
    time_power_df = pd.DataFrame(
        {"time": time_list, "power_kw": soe_power_df["power_kw"]}
    )

    return time_power_df


def get_usage_power_kw_df(soe_power_df, capacity_kWh):
    """
    Output the variable charging curve in the format that optimizer accepts.
    That is, dataframe with index "time" in minutes and "power_kw" which
    tells us the average power consumption in a five minute interval
    after an elapsed amount of time of charging.

    Assumes df is sorted by SoE
    """

    def get_kW_at_SoE(df, soe):
        """Linear interpolation to get charging rate at any SoE"""
        before_df = df[df["SoE"] <= soe]
        # print("Before_df", before_df)
        prev_row = before_df.iloc[-1] if len(before_df) > 0 else None
        after_df = df[df["SoE"] >= soe]
        # print("After_df", after_df)
        next_row = after_df.iloc[0] if len(after_df) > 0 else None
        if prev_row is None:
            return next_row["power_kw"]
        if next_row is None:
            return prev_row["power_kw"]

        m1 = prev_row["SoE"]
        p1 = prev_row["power_kw"]
        m2 = next_row["SoE"]
        p2 = next_row["power_kw"]

        if m1 == m2:
            return 0.5 * (p1 + p2)

        return p1 + (soe - m1) / (m2 - m1) * (p2 - p1)

    # iterate over seconds
    result = []
    secs_elapsed = 0
    sub_interval_seconds = 60
    # For now, we assume the starting capacity is 0.0
    charged_kWh = 0.0
    kW_by_second = []
    while charged_kWh < capacity_kWh:
        secs_elapsed += sub_interval_seconds
        curr_soe = charged_kWh / capacity_kWh
        curr_kW = get_kW_at_SoE(soe_power_df, curr_soe)
        # print("Debug:", curr_kW, curr_soe, secs_elapsed)
        kW_by_second.append(curr_kW)
        charged_kWh += curr_kW * sub_interval_seconds / 3600

        if secs_elapsed % 300 == 0:
            result.append((int(secs_elapsed / 60 - 5), pd.Series(kW_by_second).mean()))
            kW_by_second = []

    return pd.DataFrame(columns=["time", "power_kw"], data=result)


# Example usage:
soe_power_df = pd.DataFrame(
    {
        "SoE": np.linspace(0.0, 1.0, 11),  # SoE from 0% to 100%
        "power_kw": [
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
        ],  # Example power values in kW
    }
)

battery_capacity = 100  # Max energy capacity in kWh
result_df = convert_soe_to_time(soe_power_df, battery_capacity)

print("Old:", result_df)
print("New:", get_usage_power_kw_df(soe_power_df, battery_capacity))


# Example voltage curve for testing
def voltage_curve_test(soc):
    return 3.0 + 0.5 * soc


# Example SoC dataframe (with SoC ranging from 0.1 to 1.0)
soc_power_df = pd.DataFrame(
    {
        "SoC": np.linspace(0.0, 1.0, 11),
        "power_kw": [
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
        ],  # Example power values in kW
    }
)

battery_capacity_coulombs = 1_000_000  # Max energy capacity in kWh

# Convert SoC to SoE
soe_power_df = convert_soc_to_soe(
    soc_power_df, voltage_curve_test, battery_capacity_coulombs
)

# print(soe_power_df)
