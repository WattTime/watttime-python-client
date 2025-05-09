from typing import List, Any
from datetime import datetime, timedelta, date
import pandas as pd
import random
import numpy as np
import tqdm

class SessionsGenerator:
    def __init__(
            self, 
            max_percent_capacity: float = 0.95,
            power_output_efficiency: float = 0.75,
            minimum_battery_starting_capacity: float = 0.2,
            minimum_usage_window_start_time: str = "17:00:00",
            maximum_usage_window_start_time: str = "21:00:00",
            max_power_output_rates: List[Any] = [11, 7.4, 22]
            ):
        """ Initialize with user behavior + device characteristics"""
        self.max_percent_capacity = max_percent_capacity
        self.power_output_efficiency = power_output_efficiency
        self.minimum_battery_starting_capacity = minimum_battery_starting_capacity
        self.minimum_usage_window_start_time = minimum_usage_window_start_time
        self.maximum_usage_window_start_time = maximum_usage_window_start_time
        self.max_power_output_rates = max_power_output_rates
        self.distinct_dates = None

    def return_kwargs(self):
        return self.__dict__

    def generate_start_time(self, date, start_hour: str, end_hour: str) -> datetime:
        """
        Generate a random datetime on the given date between two times.
        
        Parameters:
        -----------
        date : datetime.date
            The date for which to generate the random time.
        start_hour: string
            The earliest possible start time (HH:MM:SS format).
        end_hour: string
            The latest possible start time (HH:MM:SS format).
            
        Returns:
        --------
        datetime
            Generated random datetime on the given date.
        """
        start_time = datetime.combine(
            date, 
            datetime.strptime(start_hour, "%H:%M:%S").time()
        )
        end_time = datetime.combine(
            date, 
            datetime.strptime(end_hour, "%H:%M:%S").time()
        )
        
        total_seconds = int((end_time - start_time).total_seconds())
        random_seconds = random.randint(0, total_seconds)
        return start_time + timedelta(seconds=random_seconds)

    def generate_end_time(
        self,
        start_time: datetime,
        mean: float = None,
        stddev: float = None,
    ) -> pd.Timestamp:
        """
        Generate session end time based on start time using specified distribution.
        
        Parameters:
        -----------
        start_time : datetime
            Initial plug-in time.
        mean : float, optional
            Normal distribution mean (required if method='normal').
        stddev : float, optional
            Normal distribution standard deviation (required if method='normal').
        elements : List[Any], optional
            Options for uniform distribution in seconds (required if method='random_choice').
            
        Returns:
        --------
        pd.Timestamp
            Generated end time.
        """
        random_seconds = abs(np.random.normal(loc=mean, scale=stddev))
        random_timedelta = timedelta(seconds=random_seconds)
        new_datetime = start_time + random_timedelta
        if not isinstance(new_datetime, pd.Timestamp):
            new_datetime = pd.Timestamp(new_datetime)
        return new_datetime

    def synthetic_user_data(self, distinct_date_list, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data for a single user.

        This function creates a DataFrame with synthetic data for EV charging sessions,
        including plug-in times, unplug times, initial charge, and other relevant metrics.

        Parameters:
        -----------
        distinct_date_list : List[Any]
            A list of distinct dates for which to generate charging sessions.
        max_percent_capacity : float, optional
            The maximum percentage of battery capacity to charge to (default is 0.95).
        minimum_battery_starting_capacity: float
            The minimum percent charged at session start.
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
        power_output_max_rate = random.choice(self.max_power_output_rates) * power_output_efficiency
        total_capacity = round(random.uniform(21, 123))
        mean_length_charge = round(random.uniform(20000, 30000))
        std_length_charge = round(random.uniform(6800, 8000))
        
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

        user_df["usage_window_start"] = user_df["distinct_dates"].apply(
            self.generate_start_time, args=(self.minimum_usage_window_start_time, self.maximum_usage_window_start_time)
        )
        user_df["usage_window_end"] = user_df["usage_window_start"].apply(
            lambda x: self.generate_end_time(
                x, mean_length_charge, std_length_charge
            )
        )

        user_df["usage_window_start"] = user_df["usage_window_start"].dt.round('5min')
        user_df["usage_window_end"] = user_df["usage_window_end"].dt.round('5min')


        # Another random parameter, this time at the session level, 
        # it's the initial charge of the battery as a percentage.
        user_df["initial_charge"] = user_df.apply(
            lambda _: random.uniform(self.minimum_battery_starting_capacity, 0.8), axis=1
        )
        user_df["time_needed"] = user_df["initial_charge"].apply(
            lambda x: total_capacity
            * (self.max_percent_capacity - x)
            / power_output_max_rate
            * 60
        )

        # What time will the battery reach max capacity
        user_df["expected_baseline_charge_complete_timestamp"] = user_df["usage_window_start"] + pd.to_timedelta(
            user_df["time_needed"], unit="m"
        )
        user_df["window_length_in_minutes"] = (
            user_df.usage_window_end - user_df.usage_window_start
        ) / pd.Timedelta(seconds=60)

        user_df["final_charge_time"] = user_df[
            ["expected_baseline_charge_complete_timestamp", "usage_window_end"]
        ].min(axis=1)

        user_df["total_capacity"] = total_capacity
        user_df["usage_power_kw"] = power_output_max_rate

        user_df["total_intervals_plugged_in"] = (
            user_df["window_length_in_minutes"] / 5
        )

        user_df["MWh_fraction"] = user_df["usage_power_kw"] / 12 / 1000

        user_df["early_session_stop"] = user_df["usage_window_end"] < user_df["expected_baseline_charge_complete_timestamp"]

        return user_df
    
    def generate_synthetic_dataset(
        self, distinct_date_list: List[Any], number_of_users: int = 1
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
            The number of users to generate data for (default is 1).

        Returns:
        --------
        pd.DataFrame
            A concatenated DataFrame containing synthetic data for all users.
        """
        dfs = []
        for i in tqdm.tqdm(range(number_of_users)):
            df_temp = self.synthetic_user_data(
                distinct_date_list=distinct_date_list, **self.__dict__
                )
            dfs.append(df_temp)
        df_all = pd.concat(dfs)
        df_all.reset_index(inplace=True)
        return df_all
    
    def assign_random_dates(self, years: List[int]):
        all_dates = []
        for year in years:
            y = self.generate_random_dates(year)
            all_dates = all_dates + y
        return all_dates
            
    def _get_date_from_week_and_day(self, year, week_number, day_number):
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

    def generate_random_dates(self, year):
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
            random_dates.append(self._get_date_from_week_and_day(year, i, days[0]))
            random_dates.append(self._get_date_from_week_and_day(year, i, days[1]))
        random_dates = [date for date in random_dates if date < date.today()]
        random_dates = self._remove_duplicates(random_dates)

        return random_dates

    def _remove_duplicates(self, input_list):
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