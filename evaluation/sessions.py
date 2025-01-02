class RandomSessionGenerator:
    def __init__(self):
        """Initialize the RandomSessionGenerator class."""
        self.datetime = datetime
        self.random = random
        self.timedelta = timedelta

    def generate_start_time(self, date, start_hour: str = "17:00:00", end_hour: str = "21:00:00") -> datetime:
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
        start_time = self.datetime.combine(
            date, 
            self.datetime.strptime(start_hour, "%H:%M:%S").time()
        )
        end_time = self.datetime.combine(
            date, 
            self.datetime.strptime(end_hour, "%H:%M:%S").time()
        )
        
        total_seconds = int((end_time - start_time).total_seconds())
        random_seconds = self.random.randint(0, total_seconds)
        return start_time + self.timedelta(seconds=random_seconds)

    def generate_end_time(
        self,
        start_time: datetime,
        method: str = "normal",
        mean: float = None,
        stddev: float = None,
        elements: List[Any] = None
    ) -> pd.Timestamp:
        """
        Generate session end time based on start time using specified distribution.
        
        Parameters:
        -----------
        start_time : datetime
            Initial plug-in time.
        method : str
            Distribution type ('normal' or 'random choice').
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
        if method == "normal":
            if mean is None or stddev is None:
                raise ValueError("Mean and standard deviation required for normal distribution")
            random_seconds = abs(np.random.normal(loc=mean, scale=stddev))
        elif method == "random_choice":
            if elements is None:
                raise ValueError("Elements list required for random choice")
            random_seconds = self.random.choice(elements)
        else:
            raise ValueError("Method must be 'normal' or 'random choice'")
            
        new_datetime = start_time + self.timedelta(seconds=random_seconds)
        return pd.Timestamp(new_datetime) if not isinstance(new_datetime, pd.Timestamp) else new_datetime
    

    def generate_synthetic_user_data(
    distinct_date_list: List[Any],
    max_percent_capacity: float = 0.95,
    user_charge_tolerance: float = 0.8,
    power_output_efficiency: float = 0.75,
    average_battery_starting_capacity: float = 0.2,
    start_hour="17:00:00",
    end_hour="21:00:00",
    power_output_max_rates=[11, 7.4, 22],
    proportion_contiguous=0,
    session_lengths: List[Any] = None,
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
    session_lengths : List[Any], optional
        Selection of session lengths.
        Required for generate_random_session_end_time() method='random_choice'.


    Returns:
    --------
    pd.DataFrame
        A DataFrame containing synthetic user data for EV charging sessions.
    """

    power_output_efficiency = round(random.uniform(0.5, 0.9), 3)
    power_output_max_rate = (
        random.choice(power_output_max_rates) * power_output_efficiency
    )
    rate_per_second = np.divide(power_output_max_rate, 3600)
    total_capacity = round(random.uniform(21, 123))
    mean_length_charge = round(random.uniform(20000, 30000))
    std_length_charge = round(random.uniform(6800, 8000))
    contiguous = random.uniform(0, 1) < proportion_contiguous

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
        + "_cont"
        + str(contiguous)
    )

    user_df["session_start_time"] = user_df["distinct_dates"].apply(
        generate_random_session_start_time, args=(start_hour, end_hour)
    )

    if session_lengths is None:
        user_df["session_end_time"] = user_df["session_start_time"].apply(
            lambda x: generate_random_session_end_time(
                x, mean_length_charge, std_length_charge
            )
        )
    else:
        user_df["session_end_time"] = user_df["session_start_time"].apply(
            lambda x: generate_random_session_end_time(
                x, method="random_choice", elements=session_lengths
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

    user_df["usage_time_required_minutes"] = np.ceil(
        np.minimum(
            user_df["total_seconds_to_95"], user_df["length_of_session_in_seconds"]
        )
        / 60
    )
    user_df["contiguous_block"] = contiguous

    user_df["charge_per_interval"] = user_df.apply(
        lambda row: (
            [row["usage_time_required_minutes"]] if row["contiguous_block"] else None
        ),
        axis=1,
    )

    return user_dfwedsd