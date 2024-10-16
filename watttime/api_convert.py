import pandas as pd

# This file contains utility functions for converting formats for now

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
        delta_soe = soe_power_df['SoE'].iloc[i+1] - soe_power_df['SoE'].iloc[i]

        # Energy transferred for this delta SoE
        delta_energy = delta_soe * battery_capacity  # in kWh

        # Power to use during this step
        power_to_use = soe_power_df['power_kw'].iloc[i]

        # Time step for this segment
        delta_time_minutes = delta_energy / power_to_use * 60

        # Add the time to the previous time to get cumulative time
        current_time = previous_time + delta_time_minutes
        time_list.append(current_time)
        previous_time = current_time

    # Convert SoE dataframe to Time dataframe
    time_power_df = pd.DataFrame({
        'time': time_list,
        'power_kw': soe_power_df['power_kw']
    })

    return time_power_df

# Example usage:
soe_power_df = pd.DataFrame({
    'SoE': np.linspace(0.1, 1.0, 10),  # SoE from 10% to 100%
    'power_kw': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]  # Example power values in kW
})

battery_capacity = 100  # Max energy capacity in kWh
result_df = convert_soe_to_time(soe_power_df, battery_capacity)

print(result_df)