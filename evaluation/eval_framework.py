#!/usr/bin/env python
# coding: utf-8

import os
from typing import List, Any, Optional
import numpy as np
import pandas as pd
import random
import pytz
from tqdm import tqdm
from datetime import datetime, timedelta, date
from watttime import WattTimeHistorical, WattTimeForecast, WattTimeOptimizer
import math
from typing import Union

from evaluation.config import TZ_DICTIONARY

import watttime.optimizer.alg.optCharger as optC
import watttime.optimizer_v0.alg.optCharger as optC_v0
import watttime.optimizer.alg.moer as Moer

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

start = "2024-02-15 00:00Z"
end = "2024-02-16 00:00Z"
distinct_date_list = [
    pd.Timestamp(date) for date in pd.date_range(start, end, freq="d").values
]


def intervalize_power_rate(kW_value: float, convert_to_MWh=True) -> float:
    """
    Calculate the energy used in an interval from a power rate in kilowatts
    This will return a value in units of MWh by default.
    If convert_to_MWh is false, it will convert to kWh units instead.
    """
    five_min_rate = kW_value / 12
    if convert_to_MWh:
        five_min_rate = five_min_rate / 1000
    return five_min_rate


def convert_to_utc(local_time_str, local_tz_str):
    """
    Convert a time expressed in any local time to UTC.

    Parameters:
    -----------
    local_time_str : str
        The local time as a pd.Timestamp.
    local_tz_str : str
        The timezone of the local time as a string, e.g., 'America/New_York'.

    Returns:
    --------
    str
        The time in UTC as a datetime object in the format 'YYYY-MM-DD HH:MM:SS'.

    Example:
    --------
    >>> convert_to_utc(pd.Timestamp('2023-08-29 14:30:00'), 'America/New_York')
    '2023-08-29 18:30:00'
    """
    local_time = datetime.strptime(
        local_time_str.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"
    )
    local_tz = pytz.timezone(local_tz_str)
    local_time = local_tz.localize(local_time)
    return local_time.astimezone(pytz.utc)


def generate_random_session_start_time(
    date, start_hour: str = "17:00:00", end_hour: str = "21:00:00"
):
    """
    Generate a random datetime on the given date, uniformly distributed between two times.

    Parameters:
    -----------
    date : datetime.date
        The date for which to generate the random time.
    start_hour: string
        The earliest possible random start time generated. Formatted as HH:MM:SS.
    end_hour:
        The latest possible random start time generated. Formatted as HH:MM:SS.

    Returns:
    --------
    datetime
        A datetime object representing the generated random time on the given date.

    Example:
    --------
    >>> generate_random_session_start_time(datetime.date(2023, 8, 29),"17:00:00","21:00:00")
    datetime.datetime(2023, 8, 29, 19, 45, 30)  # Example output
    """
    start_time = datetime.combine(
        date, datetime.strptime(start_hour, "%H:%M:%S").time()
    )
    end_time = datetime.combine(date, datetime.strptime(end_hour, "%H:%M:%S").time())
    total_seconds = int((end_time - start_time).total_seconds())
    random_seconds = random.randint(0, total_seconds)
    random_datetime = start_time + timedelta(seconds=random_seconds)
    return random_datetime


def generate_random_session_end_time(random_start_time, mean, stddev):
    """
    Adds a number of seconds drawn from a normal distribution to the given datetime.

    Parameters:
    -----------
    random_start_time : datetime
        The initial plug-in time.
    mean : float
        The mean of the normal distribution for generating random seconds.
    stddev : float
        The standard deviation of the normal distribution for generating random seconds.

    Returns:
    --------
    pd.Timestamp
        The new datetime after adding the random seconds.

    Example:
    --------
    >>> plug_time = datetime(2023, 8, 29, 19, 0, 0)
    >>> generate_random_session_end_time(random_start_time, 3600, 900)
    Timestamp('2023-08-29 20:01:23.456789')  # Example output
    """
    random_seconds = abs(np.random.normal(loc=mean, scale=stddev))
    random_timedelta = timedelta(seconds=random_seconds)
    new_datetime = random_start_time + random_timedelta
    if not isinstance(new_datetime, pd.Timestamp):
        new_datetime = pd.Timestamp(new_datetime)
    return new_datetime


def generate_synthetic_user_data(
    distinct_date_list: List[Any],
    max_percent_capacity: float = 0.95,
    user_charge_tolerance: float = 0.8,
    power_output_efficiency: float = 0.75,
    average_battery_starting_capacity: float = 0.2,
    start_hour="17:00:00",
    end_hour="21:00:00",
    power_output_max_rates = [11, 7.4, 22]
) -> pd.DataFrame:
    """
    Generate synthetic user data for electric vehicle charging sessions.

    This function creates a DataFrame with synthetic data for EV charging sessions,
    including plug-in times, unplug times, initial charge, and other relevant metrics.

    Parameters:
    -----------
    distinct_date_list : List[Any]
        A list of distinct dates for which to generate charging sessions.
    max_percent_capacity : float, optional
        The maximum percentage of battery capacity to charge to (default is 0.95).
    average_battery_starting_capacity: float
        The average percent charged at session start.
    user_charge_tolerance : float, optional
        The minimum acceptable charge percentage for users (default is 0.8).
    power_output_efficiency : float, optional
        The efficiency of power output (default is 0.75).
    start_hour: string
        The earliest possible random start time generated. Formatted as HH:MM:SS.
    end_hour:
        The latest possible random start time generated. Formatted as HH:MM:SS.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing synthetic user data for EV charging sessions.
    """

    power_output_efficiency = round(random.uniform(0.5, 0.9), 3)
    power_output_max_rate = random.choice(power_output_max_rates) * power_output_efficiency
    rate_per_second = np.divide(power_output_max_rate, 3600)
    total_capacity = round(random.uniform(21, 123))
    mean_length_charge = round(random.uniform(20000, 30000))
    std_length_charge = round(random.uniform(6800, 8000))

    # print(
    #   f"working on user with {total_capacity} total_capacity, {power_output_max_rate} rate of charge, and ({mean_length_charge/3600},{std_length_charge/3600}) charging behavior."
    # )

    # This generates a dataset with a unique date per user
    user_df = (
        pd.DataFrame(distinct_date_list, columns=["distinct_dates"])
        .sort_values(by="distinct_dates")
        .copy()
    )

    # Unique user type given by the convo of 4 variables.
    user_df["user_type"] = (
        "r"
        + str(power_output_max_rate)
        + "_tc"
        + str(total_capacity)
        + "_avglc"
        + str(mean_length_charge)
        + "_sdlc"
        + str(std_length_charge)
    )

    user_df["session_start_time"] = user_df["distinct_dates"].apply(
        generate_random_session_start_time, args=(start_hour, end_hour)
    )
    user_df["session_end_time"] = user_df["session_start_time"].apply(
        lambda x: generate_random_session_end_time(
            x, mean_length_charge, std_length_charge
        )
    )

    # Another random parameter, this time at the session level, 
    # it's the initial charge of the battery as a percentage.
    user_df["initial_charge"] = user_df.apply(
        lambda _: random.uniform(average_battery_starting_capacity, 0.6), axis=1
    )
    user_df["total_seconds_to_95"] = user_df["initial_charge"].apply(
        lambda x: total_capacity
        * (max_percent_capacity - x)
        / power_output_max_rate
        * 3600
    )

    # What time will the battery reach 95%
    user_df["full_charge_time"] = user_df["session_start_time"] + pd.to_timedelta(
        user_df["total_seconds_to_95"], unit="s"
    )
    user_df["length_of_session_in_seconds"] = (
        user_df.session_end_time - user_df.session_start_time
    ) / pd.Timedelta(seconds=1)

    # what happened first? did the user unplug or did it reach 95%
    user_df["charged_kWh_actual"] = user_df[
        ["total_seconds_to_95", "length_of_session_in_seconds"]
    ].min(axis=1) * (rate_per_second)
    user_df["final_perc_charged"] = user_df.charged_kWh_actual.apply(
        lambda x: x / total_capacity
    )
    user_df["final_perc_charged"] = user_df.final_perc_charged + user_df.initial_charge
    user_df["final_charge_time"] = user_df[
        ["full_charge_time", "session_end_time"]
    ].min(axis=1)
    user_df["uncharged"] = np.where(user_df["final_perc_charged"] < 0.80, True, False)
    user_df["total_capacity"] = total_capacity
    user_df["power_output_rate"] = power_output_max_rate

    user_df["total_intervals_plugged_in"] = (
        user_df["length_of_session_in_seconds"] / 300
    )  # number of seconds in 5 minutes
    user_df["charge_MWh_needed"] = (
        user_df["total_capacity"] * (0.95 - user_df["initial_charge"]) / 1000
    )
    user_df["charged_MWh_actual"] = user_df["charged_kWh_actual"] / 1000
    user_df["MWh_fraction"] = user_df["power_output_rate"].apply(intervalize_power_rate)

    return user_df


def execute_synth_data_process(
    distinct_date_list: List[Any], number_of_users: int = 1000, **kwargs
):
    """
    Execute the synthetic data generation process for multiple users.

    This function generates synthetic charging data for a specified number of users
    across the given distinct dates.

    Parameters:
    -----------
    distinct_date_list : List[Any]
        A list of distinct dates for which to generate charging sessions.
    number_of_users : int, optional
        The number of users to generate data for (default is 1000).

    Returns:
    --------
    pd.DataFrame
        A concatenated DataFrame containing synthetic data for all users.
    """
    dfs = []
    for i in tqdm(range(number_of_users)):
        df_temp = generate_synthetic_user_data(
            distinct_date_list=distinct_date_list, **kwargs
        )
        dfs.append(df_temp)
    df_all = pd.concat(dfs)
    df_all.reset_index(inplace=True)
    return df_all


def add_one_day(date):
    """
    Add one day to the given datetime object.

    Parameters:
    -----------
    date : datetime
        The datetime object to which one day will be added.

    Returns:
    --------
    datetime
        A new datetime object with one day added.
    """
    return date + timedelta(days=1)


def get_date_from_week_and_day(year, week_number, day_number):
    """
    Return the date corresponding to the year, week number, and day number provided.

    This function calculates the date based on the ISO week date system. It assumes
    the first week of the year is the first week that fully starts that year, and
    the last week of the year can spill into the next year.

    Parameters:
    -----------
    year : int
        The year for which to calculate the date.
    week_number : int
        The week number (1-52).
    day_number : int
        The day number (1-7 where 1 is Monday).

    Returns:
    --------
    datetime.date
        The corresponding date as a datetime.date object.

    Notes:
    ------
    The function checks that all returned dates are before today and cannot
    return dates in the future.
    """
    # Calculate the first day of the year
    first_day_of_year = date(year, 1, 1)

    # Calculate the first Monday of the eyar (ISO calendar)
    first_monday = first_day_of_year + timedelta(
        days=(7 - first_day_of_year.isoweekday()) + 1
    )

    # Calculate the target date
    target_date = first_monday + timedelta(weeks=week_number - 1, days=day_number - 1)

    # if the first day of the year is Monday, adjust the target date
    if first_day_of_year.isoweekday() == 1:
        target_date -= timedelta(days=7)

    return target_date


def generate_random_dates(year):
    """
    Generate a list containing two random dates from each week in the given year.

    Parameters:
    -----------
    year : int
        The year for which to generate the random dates.

    Returns:
    --------
    list
        A list of dates, with two random dates from each week of the specified year.
    """
    random_dates = []
    for i in range(1, 53):
        days = random.sample(range(1, 8), 2)
        days.sort()
        random_dates.append(get_date_from_week_and_day(year, i, days[0]))
        random_dates.append(get_date_from_week_and_day(year, i, days[1]))
    random_dates = [date for date in random_dates if date < date.today()]
    random_dates = remove_duplicates(random_dates)

    return random_dates


def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while maintaining the order of the first occurrences.

    Parameters:
    -----------
    input_list : list
        List of items that may contain duplicates.

    Returns:
    --------
    list
        A new list with duplicates removed, maintaining the order of first occurrences.
    """
    seen = set()
    output_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            output_list.append(item)
    return output_list


def get_timezone_from_dict(key, dictionary=TZ_DICTIONARY):
    """
    Retrieve the timezone value from the dictionary based on the given key.

    Parameters:
    -----------
    key : str
        The key whose corresponding timezone value is to be retrieved.
    dictionary : dict, optional
        The dictionary from which to retrieve the value (default is TZ_DICTIONARY).

    Returns:
    --------
    str or None
        The timezone value corresponding to the given key if it exists, otherwise None.
    """
    return dictionary.get(key)


# Get per-row historical fcsts at 'plug in time'
def get_historical_fcst_data(session_start_time, horizon, region):
    """
    Retrieve historical forecast data for a specific plug-in time, horizon, and region.

    Parameters:
    -----------
    session_start_time : datetime
        The time at which the EV was plugged in.
    horizon : int
        The number of hours to forecast ahead.
    region : str
        The region for which to retrieve the forecast data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing historical forecast data.
    """

    time_zone = get_timezone_from_dict(region)
    session_start_time = pd.Timestamp(convert_to_utc(session_start_time, time_zone))
    horizon = math.ceil(horizon / 12)

    hist_data = WattTimeForecast(username, password)
    return hist_data.get_historical_forecast_pandas(
        start=session_start_time - pd.Timedelta(minutes=5),
        end=session_start_time,
        horizon_hours=horizon,
        region=region,
    )


def get_historical_actual_data(session_start_time, horizon, region):
    """
    Retrieve historical actual data for a specific plug-in time, horizon, and region.

    Parameters:
    -----------
    session_start_time : datetime
        The time at which the EV was plugged in.
    horizon : int
        The number of hours to retrieve data for.
    region : str
        The region for which to retrieve the actual data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing historical actual data.
    """

    time_zone = get_timezone_from_dict(region)
    session_start_time = pd.Timestamp(convert_to_utc(session_start_time, time_zone))
    horizon = math.ceil(horizon / 12)

    hist_data = WattTimeHistorical(username, password)
    return hist_data.get_historical_pandas(
        start=session_start_time - pd.Timedelta(minutes=5),
        end=session_start_time + pd.Timedelta(hours=horizon),
        region=region,
    )


# Set up OptCharger based on moer fcsts and get info on projected schedule
def get_schedule_and_cost(
    charge_rate_per_window, charge_needed, total_time_horizon, moer_data, asap=False
):
    """
    Generate an optimal charging schedule and associated cost based on MOER forecasts.

    Parameters:
    -----------
    charge_rate_per_window : int
        The charge rate per time window.
    charge_needed : int
        The total charge needed.
    total_time_horizon : int
        The total time horizon for scheduling.
    moer_data : pd.DataFrame
        MOER (Marginal Operating Emissions Rate) forecast data.
    asap : bool, optional
        Whether to charge as soon as possible (default is False).

    Returns:
    --------
    OptCharger
        An OptCharger object containing the optimal charging schedule and cost.
    """
    charger = optC_v0.OptCharger(
        charge_rate_per_window
    )  # charge rate needs to be an int
    moer = Moer.Moer(moer_data["value"])

    charger.fit(
        totalCharge=charge_needed,  # also currently an int value
        totalTime=total_time_horizon,
        moer=moer,
        asap=asap,
    )
    return charger

def get_time_needed(
    total_capacity_kWh: float,
    usage_power_kW: Union[float, pd.DataFrame],
    initial_capacity_fraction: float,
    max_capacity_fraction: float = 0.95,
) -> int:
    """
    Get the number of minutes needed to charge

    Parameters:
    -----------
    total_capacity_kWh : float
        The total capcity of the battery in kilowatts hours
    usage_power_kW : float or pd.DataFrame
        the charging rate in kW, either constant (float)
        or variable charging curve (DataFrame) 
    initial_capacity_fraction : float
        The battery capacity when it is plugged in,
        as a fraction of total capacity
    max_capacity_fraction : float
        The percentage of capacity at which we stop charging.
        Defaults to 95% 

    Returns:
    --------
    int
        The number of minutes that the battery needs to charge for 

    Notes:
    ------
    This is then able to be fed into get_schedule_and_cost_api
    as the time_needed parameter
    """
    needed_kWh = (max_capacity_fraction - initial_capacity_fraction) * total_capacity_kWh

    if isinstance(usage_power_kW, float):
        needed_minutes = math.ceil(needed_kWh / usage_power_kW * 60)
    
    elif isinstance(usage_power_kW, pd.DataFrame):
        OPT_INTERVAL = 5 # units: minutes
        df = usage_power_kW.copy()
        df["kWh_per_interval"] = df["power_kw"] / 60 * OPT_INTERVAL
        df["kWh_cumsum"] = df["kWh_per_interval"].cumsum()
        if df["kWh_cumsum"].max() < needed_kWh:
            needed_minutes = df["time"].max() + OPT_INTERVAL
        else:
            needed_minutes = df[df["kWh_cumsum"] > needed_kWh]["time"].values[0] + OPT_INTERVAL

    else:
        raise ValueError(f"usage_power_kW should be type float or DataFrame but got {type(usage_power_kW)}")

    return needed_minutes

# Set up OptCharger based on moer fcsts and get info on projected schedule
def get_schedule_and_cost_api(
    usage_power_kw,
    time_needed,
    total_time_horizon,
    moer_data,
    optimization_method="sophisticated",
    charge_per_interval: list = []
):
    """
    Generate an optimal charging schedule and associated cost using WattTimeOptimizer.

    Parameters:
    -----------
    usage_power_kw : float or pd.Series
        The power usage in kilowatts.
    time_needed : float
        The time needed for charging in minutes.
    total_time_horizon : int
        The total time horizon for scheduling in number of intervalss.
    moer_data : pd.DataFrame
        MOER (Marginal Operating Emissions Rate) forecast data.
    optimization_method : str, optional
        The optimization method to use (default is "sophisticated").

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the optimal usage plan.

    Notes:
    ------
    This function uses the WattTimeOptimizer to generate an optimal charging schedule.
    It prints a warning if the resulting emissions are 0.0 lb of CO2e.
    """
    wt_opt = WattTimeOptimizer(username, password)
    usage_window_start = pd.to_datetime(moer_data["point_time"].iloc[0])
    usage_window_end = pd.to_datetime(
        moer_data["point_time"].iloc[total_time_horizon - 1]
    )

    # if we need to charge for more minutes than given in between
    # plugin time and plugout then we charge for the entire period 
    time_needed = min(time_needed, total_time_horizon * wt_opt.OPT_INTERVAL)

    dp_usage_plan = wt_opt.get_optimal_usage_plan(
        region=None,
        usage_window_start=usage_window_start,
        usage_window_end=usage_window_end,
        usage_time_required_minutes=time_needed,
        usage_power_kw=usage_power_kw,
        optimization_method=optimization_method,
        moer_data_override=moer_data,
        charge_per_interval= charge_per_interval
    )



    if dp_usage_plan["emissions_co2e_lb"].sum() == 0.0:
        print(
            "Warning using 0.0 lb of CO2e:",
            usage_power_kw,
            usage_power_kw,
            time_needed,
            dp_usage_plan["usage"].sum(),
        )

    return dp_usage_plan


def get_total_emission(moer, schedule):
    x = np.array(schedule).flatten()
    return np.dot(moer[: x.shape[0]], x)

def create_car_model_feature(car):
    if random.random()<= 0.5:
        car == 'chevy'
    else:
        car == 'tesla'
    return car


