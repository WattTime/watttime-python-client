#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import random
import pytz
from tqdm import tqdm
from datetime import datetime, timedelta
import random
from watttime import (
    WattTimeMyAccess,
    WattTimeHistorical,
    WattTimeForecast,
    WattTimeMaps,
)
import os
import optimizer.dataset as od
from typing import List, Any
from evaluation.config import MOER_REGION_LIST

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")

start = "2024-02-15 00:00Z"
end = "2024-02-16 00:00Z"
distinct_date_list = [
    pd.Timestamp(date).replace(tzinfo=pytz.UTC)
    for date in pd.date_range(start, end, freq="d", tz=pytz.UTC).values
]


def generate_random_plug_time(date):
    """
    Generate a random datetime ona the given date, uniformly distributed between 5pm and 9 pm.

    Parameters:
    date (datetime.date): The date on which to generate the random time.

    Returns:
    - datetime: A datetime object for the given date with a random time between 5 PM and 9 PM
    """
    #  Define the start and end times for the interval (5 PM to 9PM)
    start_time = datetime.combine(
        date, datetime.strptime("17:00:00", "%H:%M:%S").time()
    )
    end_time = datetime.combine(date, datetime.strptime("21:00:00", "%H:%M:%S").time())

    # Calculate the total number of seconds between start and end times
    total_seconds = int((end_time - start_time).total_seconds())

    # Generate a random number of seconds within the interval
    total_seconds = random.randint(0, total_seconds)

    # Add the random seconds to the start time to get the random datetime
    random_datetime = start_time + timedelta(seconds=total_seconds)

    random_datetime_utc = pytz.utc.localize(random_datetime)

    return random_datetime_utc


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
    random_seconds = np.random.normal(loc=mean, scale=stddev)

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
    power_output_max_rate = random.choice([11, 7.4, 22]) / power_output_efficiency
    power_output_max_rate = power_output_max_rate / power_output_efficiency
    rate_per_second = np.divide(power_output_max_rate, 3600)
    total_capacity = round(random.uniform(21, 123))
    mean_length_charge = round(random.uniform(20000, 30000))
    std_length_charge = round(random.uniform(6800, 8000))

    print(
        f"working on user with {total_capacity} total_capacity, {power_output_max_rate} rate of charge, and ({mean_length_charge/3600},{std_length_charge/3600}) charging behavior."
    )

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
    user_df["session_charge"] = user_df[
        ["total_seconds_to_95", "length_plugged_in"]
    ].min(axis=1) * (rate_per_second)
    user_df["final_perc_charged"] = user_df.session_charge.apply(
        lambda x: x / total_capacity
    )
    user_df["final_perc_charged"] = user_df.final_perc_charged + user_df.initial_charge
    user_df["uncharged"] = np.where(user_df["final_perc_charged"] < 0.80, True, False)
    user_df["total_capacity"] = total_capacity
    user_df["power_output_max_rate"] = power_output_max_rate

    return user_df


def execute_synth_data_process(
    distinct_date_list: List[Any], number_of_users: int = 1000
) -> pd.DataFrame:
    dfs = []
    for i in tqdm(range(number_of_users)):
        df_temp = generate_synthetic_user_data(distinct_date_list=distinct_date_list)
        dfs.append(df_temp)
    df_all = pd.concat(dfs)
    df_all.reset_index(inplace=True)
    return df_all
