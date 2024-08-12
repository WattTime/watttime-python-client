#!/usr/bin/env python
# coding: utf-8

import os
from typing import List, Any
import numpy as np
import pandas as pd
import random
import pytz
from tqdm import tqdm
from datetime import datetime, timedelta, date
from watttime import WattTimeHistorical, WattTimeForecast, WattTimeOptimizer
import math

from evaluation.config import TZ_DICTIONARY

import watttime.shared_anniez.alg.optCharger as optC
import watttime.shared_anniez.alg.moer as Moer

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

start = "2024-02-15 00:00Z"
end = "2024-02-16 00:00Z"
distinct_date_list = [
    pd.Timestamp(date) for date in pd.date_range(start, end, freq="d").values
]


def intervalize_power_rate(kW_value: float, convert_to_MW=True):
    five_min_rate = kW_value / 12
    if convert_to_MW:
        five_min_rate = five_min_rate / 1000
    else:
        five_min_rate
    return five_min_rate


def convert_to_utc(local_time_str, local_tz_str):
    """
    Convert a time expressed in any local time to UTC.

    Parameters:
    local_time_str (str): The local time as a pd.Timestamp.
    local_tz_str (str): The timezone of the local time as a string, e.g., 'America/New_York'.

    Returns:
    str: The time in UTC as a datetime object in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    # Parse the local time string to a datetime object
    local_time = datetime.strptime(
        local_time_str.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"
    )

    # Set the local timezone
    local_tz = pytz.timezone(local_tz_str)

    # Localize the local time to the local timezone
    local_time = local_tz.localize(local_time)

    # Return the UTC time as a datetime
    return local_time.astimezone(pytz.utc)


def generate_random_plug_time(date):
    """
    Generate a random datetime on the given date, uniformly distributed between 5 PM and 9 PM.

    Parameters:
    date (datetime.date): The date for which to generate the random time.

    Returns:
    datetime: A datetime object representing the generated random time on the given date.
    """
    # Define the start and end times for the interval (5 PM to 9 PM)
    start_time = datetime.combine(
        date, datetime.strptime("17:00:00", "%H:%M:%S").time()
    )
    end_time = datetime.combine(date, datetime.strptime("21:00:00", "%H:%M:%S").time())

    # Calculate the total number of seconds between start and end times
    total_seconds = int((end_time - start_time).total_seconds())

    # Generate a random number of seconds within the interval
    random_seconds = random.randint(0, total_seconds)

    # Add the random seconds to the start time to get the random datetime
    random_datetime = start_time + timedelta(seconds=random_seconds)

    return random_datetime


def generate_random_unplug_time(random_plug_time, mean, stddev):
    """
    Adds a number of sconds drawn from a normal distribution to the given datetime.

    Parameters:
    -datetime_obj
    -mean
    -stddev

    REturns
    -pd.Timestamp: the new datetime after adding the random seconds
    """
    random_seconds = abs(
        np.random.normal(loc=mean, scale=stddev)
    )  # ensure the delta is positive

    # convert to timedelta
    random_timedelta = timedelta(seconds=random_seconds)
    new_datetime = random_plug_time + random_timedelta

    if not isinstance(new_datetime, pd.Timestamp):
        new_datetime = pd.Timestamp(new_datetime)
    return new_datetime


def generate_synthetic_user_data(
    distinct_date_list: List[Any],
    max_percent_capacity: float = 0.95,
    user_charge_tolerance: float = 0.8,
    power_output_efficiency: float = 0.75,
) -> pd.DataFrame:

    power_output_efficiency = round(random.uniform(0.5, 0.9), 3)
    power_output_max_rate = random.choice([11, 7.4, 22]) * power_output_efficiency
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

    user_df["plug_in_time"] = user_df["distinct_dates"].apply(generate_random_plug_time)
    user_df["unplug_time"] = user_df["plug_in_time"].apply(
        lambda x: generate_random_unplug_time(x, mean_length_charge, std_length_charge)
    )

    # Another random parameter, this time at the session level, it's the initial charge of the battery.
    user_df["initial_charge"] = user_df.apply(
        lambda _: random.uniform(0.2, 0.6), axis=1
    )
    user_df["total_seconds_to_95"] = user_df["initial_charge"].apply(
        lambda x: total_capacity
        * (max_percent_capacity - x)
        / power_output_max_rate
        * 3600
    )

    # What time will the battery reach 95%
    user_df["full_charge_time"] = user_df["plug_in_time"] + pd.to_timedelta(
        user_df["total_seconds_to_95"], unit="s"
    )
    user_df["length_plugged_in"] = (
        user_df.unplug_time - user_df.plug_in_time
    ) / pd.Timedelta(seconds=1)

    # what happened first? did the user unplug or did it reach 95%
    user_df["charged_kWh_actual"] = user_df[
        ["total_seconds_to_95", "length_plugged_in"]
    ].min(axis=1) * (rate_per_second)
    user_df["final_perc_charged"] = user_df.charged_kWh_actual.apply(
        lambda x: x / total_capacity
    )
    user_df["final_perc_charged"] = user_df.final_perc_charged + user_df.initial_charge
    user_df["final_charge_time"] = user_df[["full_charge_time", "unplug_time"]].min(
        axis=1
    )
    user_df["uncharged"] = np.where(user_df["final_perc_charged"] < 0.80, True, False)
    user_df["total_capacity"] = total_capacity
    user_df["power_output_rate"] = power_output_max_rate

    user_df["total_intervals_plugged_in"] = (
        user_df["length_plugged_in"] / 300
    )  # number of seconds in 5 minutes
    user_df["charge_MWh_needed"] = (
        user_df["total_capacity"] * (0.95 - user_df["initial_charge"]) / 1000
    )
    user_df["charged_MWh_actual"] = user_df["charged_kWh_actual"] / 1000
    user_df["MWh_fraction"] = user_df["power_output_rate"].apply(intervalize_power_rate)

    return user_df


def execute_synth_data_process(
    distinct_date_list: List[Any], number_of_users: int = 1000
):
    dfs = []
    for i in tqdm(range(number_of_users)):
        df_temp = generate_synthetic_user_data(distinct_date_list=distinct_date_list)
        dfs.append(df_temp)
    df_all = pd.concat(dfs)
    df_all.reset_index(inplace=True)
    return df_all


def add_one_day(date):
    """
    Add one day to the given datetime object.

    Parameters:
    date (datetime): The datetime object to which one day will be added.

    Returns:
    datetime: A new datetime object with one day added.
    """
    return date + timedelta(days=1)

def get_date_from_week_and_day(year,week_number,day_number):
    """
    Return the date corresponding to the year, week number,
    and day number provided. It assumes the first week of 
    the year is the first week that fully starts that year; 
    and the last week of the year can splill into next year 
    (i.e. if monday is dec 31, then the week goes all the 
    way to Sunday January 6th of the next year. 

    The function also checks that all the dates returned 
    are before today. I.e. it cannot return dates in 
    the future. 

    Arguments:
    year -- the year we want sampled
    week_number -- The week number (1-52)
    day_number -- The day number (1-7 where 1 is Monday)

    Returns:
    The corresponding date as a datetime.date object
    """
    # Calculate the first day of the year
    first_day_of_year = date(year,1,1)

    #Calculate the first Monday of the eyar (ISO calendar)
    first_monday = first_day_of_year + timedelta(days=(7- first_day_of_year.isoweekday()) + 1)

    #Calculate the target date
    target_date = first_monday + timedelta(weeks=week_number -1, days=day_number -1)

    #if the first day of the year is Monday, adjust the target date
    if first_day_of_year.isoweekday() ==1:
        target_date -= timedelta(days=7)

    return target_date

def generate_random_dates(year):
    """
    Generate a list of containing two random dates from each week in the given year.

    Parameters:
    year (int): The year for which to generate the random dates.

    Returns:
    list: A list of dates.
    """
    random_dates = []
    for i in range(1,53):
        days = random.sample(range(1,8),2)
        days.sort()
        random_dates.append(get_date_from_week_and_day(year,i,days[0]))
        random_dates.append(get_date_from_week_and_day(year,i,days[1]))  
    random_dates = [date for date in random_dates if date < date.today()]
    random_dates = remove_duplicates(random_dates)
    
    return random_dates


def remove_duplicates(input_list):
    """
    Removes duplicate items from a list while maintaining the order of the first occurrences.

    :param input_list: List of items that may contain duplicates.
    :return: A new list with duplicates removed.
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
    Returns the value from the dictionary based on the given key.

    Parameters:
    - dictionary: The dictionary from which to retrieve the value.
    - key: The key whose corresponding value is to be retrieved.

    Returns:
    - The value corresponding to the given key if the key exists, otherwise None.
    """
    return dictionary.get(key)


# Get per-row historical fcsts at 'plug in time'
def get_historical_fcst_data(plug_in_time, horizon, region):

    time_zone = get_timezone_from_dict(region)
    plug_in_time = pd.Timestamp(convert_to_utc(plug_in_time, time_zone))
    horizon = math.ceil(horizon / 12)

    hist_data = WattTimeForecast(username, password)
    return hist_data.get_historical_forecast_pandas(
        start=plug_in_time - pd.Timedelta(minutes=5),
        end=plug_in_time,
        horizon_hours=horizon,
        region=region,
    )

def get_historical_actual_data(plug_in_time, horizon, region):

    time_zone = get_timezone_from_dict(region)
    plug_in_time = pd.Timestamp(convert_to_utc(plug_in_time, time_zone))
    horizon = math.ceil(horizon / 12)

    hist_data = WattTimeHistorical(username, password)
    return hist_data.get_historical_pandas(
        start=plug_in_time - pd.Timedelta(minutes=5),
        end=plug_in_time + pd.Timedelta(hours=horizon),
        region=region,
    )


# Set up OptCharger based on moer fcsts and get info on projected schedule
def get_schedule_and_cost(
    charge_rate_per_window, charge_needed, total_time_horizon, moer_data, asap=False
):
    charger = optC.OptCharger(charge_rate_per_window)  # charge rate needs to be an int
    moer = Moer.Moer(moer_data["value"])

    charger.fit(
        totalCharge=charge_needed,  # also currently an int value
        totalTime=total_time_horizon,
        moer=moer,
        asap=asap,
    )
    return charger

# Set up OptCharger based on moer fcsts and get info on projected schedule
def get_schedule_and_cost_v2(
    usage_power_kw, time_needed, moer_data, optimization_method="sophisticated"
):
    wt_opt = WattTimeOptimizer(username,password)
    usage_window_start = pd.to_datetime(moer_data["point_time"].iloc[0])
    usage_window_end = pd.to_datetime(moer_data["point_time"].iloc[-1])
    # print(usage_window_start, usage_window_end, usage_power_kw, time_needed)

    dp_usage_plan = wt_opt.get_optimal_usage_plan(
                    region=None,
                    usage_window_start = usage_window_start,
                    usage_window_end = usage_window_end,
                    usage_time_required_minutes = time_needed,
                    usage_power_kw = usage_power_kw,
                    optimization_method = optimization_method,
                    moer_data_override = moer_data
                    )

    if dp_usage_plan["emissions_co2e_lb"].sum() == 0.0:
        print("Warning using 0.0 lb of CO2e:", usage_power_kw, usage_power_kw, time_needed, dp_usage_plan["usage"].sum())

    return dp_usage_plan