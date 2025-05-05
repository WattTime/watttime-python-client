import os
import math
from datetime import datetime, timedelta
from typing import Any, Literal, Optional, Union

import pandas as pd
from dateutil.parser import parse
from pytz import UTC, timezone
from watttime_optimizer.alg import optCharger, moer
from itertools import accumulate
import bisect

from watttime.api import WattTimeForecast


OPT_INTERVAL = 5
MAX_PREDICTION_HOURS = 72


class WattTimeOptimizer(WattTimeForecast):
    """
    This class inherits from WattTimeForecast, with additional methods to generate
    optimal usage plans for energy consumption based on various parameters and
    constraints.

    Additional Methods:
    --------
    get_optimal_usage_plan(region, usage_window_start, usage_window_end,
                           usage_time_required_minutes, usage_power_kw,
                           usage_time_uncertainty_minutes, optimization_method,
                           moer_data_override)
        Generates an optimal usage plan for energy consumption.
    """

    OPT_INTERVAL = 5
    MAX_PREDICTION_HOURS = 72
    MAX_INT = 99999999999999999

    def get_optimal_usage_plan(
        self,
        region: str,
        usage_window_start: datetime,
        usage_window_end: datetime,
        usage_time_required_minutes: Optional[Union[int, float]] = None,
        usage_power_kw: Optional[Union[int, float, pd.DataFrame]] = None,
        energy_required_kwh: Optional[Union[int, float]] = None,
        usage_time_uncertainty_minutes: Optional[Union[int, float]] = 0,
        charge_per_interval: Optional[list] = None,
        use_all_intervals: bool = True,
        constraints: Optional[dict] = None,
        optimization_method: Optional[
            Literal["baseline", "simple", "sophisticated", "auto"]
        ] = "baseline",
        moer_data_override: Optional[pd.DataFrame] = None,
        verbose=True,
    ) -> pd.DataFrame:
        """
        Generates an optimal usage plan for energy consumption based on given parameters.

        This method calculates the most efficient energy usage schedule within a specified
        time window, considering factors such as regional data, power requirements, and
        optimization methods.

        You should pass in exactly 2 of 3 parameters of (usage_time_required_minutes, usage_power_kw, energy_required_kwh)

        Parameters:
        -----------
        region : str
            The region for which forecast data is requested.
        usage_window_start : datetime
            Start time of the window when power consumption is allowed.
        usage_window_end : datetime
            End time of the window when power consumption is allowed.
        usage_time_required_minutes : Optional[Union[int, float]], default=None
            Required usage time in minutes.
        usage_power_kw : Optional[Union[int, float, pd.DataFrame]], default=None
            Power usage in kilowatts. Can be a constant value or a DataFrame for variable power.
        energy_required_kwh : Optional[Union[int, float]], default=None
            Energy required in kwh
        usage_time_uncertainty_minutes : Optional[Union[int, float]], default=0
            Uncertainty in usage time, in minutes.
        charge_per_interval : Optional[list], default=None
            Either a list of length-2 tuples representing minimium and maximum (inclusive) charging minutes per interval,
            or a list of ints representing both the min and max. [180] OR [(180,180)]
        use_all_intervals : Optional[bool], default=False
            If true, use all intervals provided by charge_per_interval; if false, can use the first few intervals and skip the rest.
        constraints : Optional[dict], default=None
            A dictionary containing contraints on how much usage must be used before the given time point
        optimization_method : Optional[Literal["baseline", "simple", "sophisticated", "auto"]], default="baseline"
            The method used for optimization.
        moer_data_override : Optional[pd.DataFrame], default=None
            Pre-generated MOER (Marginal Operating Emissions Rate) DataFrame, if available.
        verbose : default = True
            If false, suppresses print statements in the opt charger class.

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the optimal usage plan, including columns for
            predicted MOER, usage, CO2 emissions, and energy usage.

        Raises:
        -------
        AssertionError
            If input parameters do not meet specified conditions (e.g., timezone awareness,
            valid time ranges, supported optimization methods).

        Notes:
        ------
        - The method uses WattTime forecast data unless overridden by moer_data_override.
        - It supports various optimization methods and can handle both constant and variable power usage.
        - The resulting plan aims to minimize emissions while meeting the specified energy requirements.
        """

        def is_tz_aware(dt):
            return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

        def minutes_to_units(x, floor=False):
            """Converts minutes to forecase intervals. Rounds UP by default."""
            if x:
                if floor:
                    return int(x // self.OPT_INTERVAL)
                else:
                    return int(math.ceil(x / self.OPT_INTERVAL))
            return x

        assert is_tz_aware(usage_window_start), "Start time is not tz-aware"
        assert is_tz_aware(usage_window_end), "End time is not tz-aware"

        if constraints is None:
            constraints = {}
        else:
            # Convert constraints to a standardized format
            raw_constraints = constraints.copy()
            constraints = {}

            for (
                constraint_time_clock,
                constraint_usage_minutes,
            ) in raw_constraints.items():
                constraint_time_minutes = (
                    constraint_time_clock - usage_window_start
                ).total_seconds() / 60
                constraint_time_units = minutes_to_units(constraint_time_minutes)
                constraint_usage_units = minutes_to_units(constraint_usage_minutes)

                constraints.update(
                    {constraint_time_units: (constraint_usage_units, None)}
                )

        num_inputs = 0
        for input in (usage_time_required_minutes, usage_power_kw, energy_required_kwh):
            if input is not None:
                num_inputs += 1
        assert (
            num_inputs == 2
        ), "Exactly 2 of 3 inputs in (usage_time_required_minutes, usage_power_kw, energy_required_kwh) required"
        if usage_power_kw is None:
            usage_power_kw = energy_required_kwh / usage_time_required_minutes * 60
            print("Implied usage_power_kw =", usage_power_kw)
        if usage_time_required_minutes is None:
            if type(usage_power_kw) in (float, int) and type(energy_required_kwh) in (
                float,
                int,
            ):
                usage_time_required_minutes = energy_required_kwh / usage_power_kw * 60
                print("Implied usage time required =", usage_time_required_minutes)
            else:
                # TODO: Implement and test
                raise NotImplementedError(
                    "When usage_time_required_minutes is None, only float or int usage_power_kw and energy_required_kwh is supported."
                )

        # Perform these checks if we are using live data
        if moer_data_override is None:
            datetime_now = datetime.now(UTC)
            assert (
                usage_window_end > datetime_now
            ), "Error, Window end is before current datetime"
            assert usage_window_end - datetime_now < timedelta(
                hours=self.MAX_PREDICTION_HOURS
            ), "End time is too far in the future"
        assert optimization_method in ("baseline", "simple", "sophisticated", "auto"), (
            "Unsupported optimization method:" + optimization_method
        )
        if moer_data_override is None:
            forecast_df = self.get_forecast_pandas(
                region=region,
                signal_type="co2_moer",
                horizon_hours=self.MAX_PREDICTION_HOURS,
            )
        else:
            forecast_df = moer_data_override.copy()
        forecast_df = forecast_df.set_index("point_time")
        forecast_df.index = pd.to_datetime(forecast_df.index)

        # relevant_forecast_df = forecast_df[usage_window_start:usage_window_end]
        relevant_forecast_df = forecast_df[forecast_df.index >= usage_window_start]
        relevant_forecast_df = relevant_forecast_df[
            relevant_forecast_df.index < usage_window_end
        ]
        relevant_forecast_df = relevant_forecast_df.rename(
            columns={"value": "pred_moer"}
        )
        result_df = relevant_forecast_df[["pred_moer"]]
        moer_values = relevant_forecast_df["pred_moer"].values

        m = moer.Moer(mu=moer_values)

        model = optCharger.OptCharger(verbose=verbose)

        total_charge_units = minutes_to_units(usage_time_required_minutes)
        if optimization_method in ("sophisticated", "auto"):
            # Give a buffer time equal to the uncertainty
            buffer_time = usage_time_uncertainty_minutes
            buffer_periods = minutes_to_units(buffer_time) if buffer_time else 0
            buffer_enforce_time = max(
                total_charge_units, len(moer_values) - buffer_periods
            )
            constraints.update({buffer_enforce_time: (total_charge_units, None)})
        else:
            assert (
                usage_time_uncertainty_minutes == 0
            ), "usage_time_uncertainty_minutes is only supported in optimization_method='sophisticated' or 'auto'"

        if type(usage_power_kw) in (int, float):
            # Convert to the MWh used in an optimization interval
            # expressed as a function to meet the parameter requirements for OptC function
            emission_multiplier_fn = (
                lambda sc, ec: float(usage_power_kw) * 0.001 * self.OPT_INTERVAL / 60.0
            )
        else:
            usage_power_kw = usage_power_kw.copy()
            # Resample usage power dataframe to an OPT_INTERVAL frequency
            usage_power_kw["time_step"] = usage_power_kw["time"] / self.OPT_INTERVAL
            usage_power_kw_new_index = pd.DataFrame(
                index=[float(x) for x in range(total_charge_units + 1)]
            )
            usage_power_kw = pd.merge_asof(
                usage_power_kw_new_index,
                usage_power_kw.set_index("time_step"),
                left_index=True,
                right_index=True,
                direction="backward",
                allow_exact_matches=True,
            )

            def emission_multiplier_fn(sc: float, ec: float) -> float:
                """
                Calculate the approximate mean power in the given time range,
                in units of MWh used per optimizer time unit.

                sc and ec are float values representing the start and end time of
                the time range, in optimizer time units.
                """
                value = (
                    usage_power_kw[sc : max(sc, ec - 1e-12)]["power_kw"].mean()
                    * 0.001
                    * self.OPT_INTERVAL
                    / 60.0
                )
                return value

        if charge_per_interval:
            # Handle the charge_per_interval input by converting it from minutes to units, rounding up
            converted_charge_per_interval = []
            for c in charge_per_interval:
                if isinstance(c, int):
                    converted_charge_per_interval.append(minutes_to_units(c))
                else:
                    assert (
                        len(c) == 2
                    ), "Length of tuples in charge_per_interval is not 2"
                    interval_start_units = minutes_to_units(c[0]) if c[0] else 0
                    interval_end_units = (
                        minutes_to_units(c[1]) if c[1] else self.MAX_INT
                    )
                    converted_charge_per_interval.append(
                        (interval_start_units, interval_end_units)
                    )
        else:
            converted_charge_per_interval = None
        model.fit(
            total_charge=total_charge_units,
            total_time=len(moer_values),
            moer=m,
            constraints=constraints,
            charge_per_interval=converted_charge_per_interval,
            use_all_intervals=use_all_intervals,
            emission_multiplier_fn=emission_multiplier_fn,
            optimization_method=optimization_method,
        )

        optimizer_result = model.get_schedule()
        result_df = self._reconcile_constraints(
            optimizer_result,
            result_df,
            model,
            usage_time_required_minutes,
            charge_per_interval,
        )

        return result_df

    def _reconcile_constraints(
        self,
        optimizer_result,
        result_df,
        model,
        usage_time_required_minutes,
        charge_per_interval,
    ):
        # Make a copy of charge_per_interval if necessary
        if charge_per_interval is not None:
            charge_per_interval = charge_per_interval[::]
            for i in range(len(charge_per_interval)):
                if type(charge_per_interval[i]) == int:
                    charge_per_interval[i] = (
                        charge_per_interval[i],
                        charge_per_interval[i],
                    )
                assert len(charge_per_interval[i]) == 2
                processed_start = (
                    charge_per_interval[i][0]
                    if charge_per_interval[i][0] is not None
                    else 0
                )
                processed_end = (
                    charge_per_interval[i][1]
                    if charge_per_interval[i][1] is not None
                    else self.MAX_INT
                )

                charge_per_interval[i] = (processed_start, processed_end)

        if not charge_per_interval:
            # Handle case without charge_per_interval constraints
            total_usage_intervals = sum(optimizer_result)
            current_usage_intervals = 0
            usage_list = []
            for to_charge_binary in optimizer_result:
                current_usage_intervals += to_charge_binary
                if current_usage_intervals < total_usage_intervals:
                    usage_list.append(to_charge_binary * float(self.OPT_INTERVAL))
                else:
                    # Partial interval
                    minutes_to_trim = (
                        total_usage_intervals * self.OPT_INTERVAL
                        - usage_time_required_minutes
                    )
                    usage_list.append(
                        to_charge_binary * float(self.OPT_INTERVAL - minutes_to_trim)
                    )
            result_df["usage"] = usage_list
        else:
            # Process charge_per_interval constraints
            result_df["usage"] = [
                x * float(self.OPT_INTERVAL) for x in optimizer_result
            ]
            usage = result_df["usage"].values
            sections = []
            interval_ids = model.get_interval_ids()

            def get_min_max_indices(lst, x):
                # Find the first occurrence of x
                min_index = lst.index(x)
                # Find the last occurrence of x
                max_index = len(lst) - 1 - lst[::-1].index(x)
                return min_index, max_index

            for interval_id in range(0, max(interval_ids) + 1):
                assert (
                    interval_id in interval_ids
                ), "interval_id not found in interval_ids"
                sections.append(get_min_max_indices(interval_ids, interval_id))

            # Adjust sections to satisfy charge_per_interval constraints
            for i, (start, end) in enumerate(sections):
                section_usage = usage[start : end + 1]
                total_minutes = section_usage.sum()

                # Get the constraints for this section
                if isinstance(charge_per_interval[i], int):
                    min_minutes, max_minutes = (
                        charge_per_interval[i],
                        charge_per_interval[i],
                    )
                else:
                    min_minutes, max_minutes = charge_per_interval[i]

                # Adjust the section to fit the constraints
                if total_minutes < min_minutes:
                    raise ValueError(
                        f"Cannot meet the minimum charging constraint of {min_minutes} minutes for section {i}."
                    )
                elif total_minutes > max_minutes:
                    # Reduce usage to fit within the max_minutes
                    excess_minutes = total_minutes - max_minutes
                    for j in range(len(section_usage)):
                        if section_usage[j] > 0:
                            reduction = min(section_usage[j], excess_minutes)
                            section_usage[j] -= reduction
                            excess_minutes -= reduction
                            if excess_minutes <= 0:
                                break
                    usage[start : end + 1] = section_usage
            result_df["usage"] = usage

        # Recalculate these values approximately, based on the new "usage" column
        # Note: This is approximate since it assumes that
        # the charging emissions over time of the unrounded values are similar to the rounded values
        result_df["emissions_co2_lb"] = (
            model.get_charging_emissions_over_time()
            * result_df["usage"]
            / self.OPT_INTERVAL
        )
        result_df["energy_usage_mwh"] = (
            model.get_energy_usage_over_time() * result_df["usage"] / self.OPT_INTERVAL
        )

        return result_df


class WattTimeRecalculator:
    """A class to manage and update charging schedules over time.

    This class maintains a list of charging schedules and their associated time contexts,
    allowing for updates and recalculations of remaining charging time required.

    Attributes:
        all_schedules (list): List of tuples containing (schedule, time_context) pairs
        total_time_required (int): Total charging time needed in minutes
        end_time (datetime): Final deadline for the charging schedule
        charge_per_interval (list): List of charging durations per interval
        is_contiguous (bool): Flag indicating if charging must be contiguous
        sleep_delay(bool): Flag indicating if next query time must be delayed
        contiguity_values_dict (dict): Dictionary storing contiguity-related values
    """

    def __init__(
        self,
        initial_schedule: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        total_time_required: int,
        contiguous=False,
        charge_per_interval: Optional[list] = None,
    ) -> None:
        """Initialize the Recalculator with an initial schedule.

        Args:
            initial_schedule (pd.DataFrame): Starting charging schedule
            start_time (datetime): Start time for the schedule
            end_time (datetime): End time for the schedule
            total_time_required (int): Total charging time needed in minutes
            charge_per_interval (list): List of charging durations per interval
        """
        self.OPT_INTERVAL = 5
        self.all_schedules = [(initial_schedule, (start_time, end_time))]
        self.end_time = end_time
        self.total_time_required = total_time_required
        self.charge_per_interval = charge_per_interval
        self.is_contiguous = contiguous
        self.sleep_delay = False
        self.contiguity_values_dict = {
            "delay_usage_window_start": None,
            "delay_in_minutes": None,
            "delay_in_intervals": None,
            "remaining_time_required": None,
            "remaining_units_required": None,
            "num_segments_complete": None,
        }

        self.total_available_units = self.minutes_to_units(
            int(int((self.end_time - start_time).total_seconds()) / 60)
        )

    def is_tz_aware(dt):
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

    def minutes_to_units(self, x, floor=False):
        """Converts minutes to forecase intervals. Rounds UP by default."""
        if x:
            if floor:
                return int(x // self.OPT_INTERVAL)
            else:
                return int(math.ceil(x / self.OPT_INTERVAL))
        return x

    def get_remaining_units_required(self, next_query_time):
        _minutes = self.get_remaining_time_required(next_query_time)
        return self.minutes_to_units(_minutes)

    def get_remaining_time_required(self, next_query_time: datetime):
        """Calculate remaining charging time needed at a given query time.

        Args:
            next_query_time (datetime): Time from which to calculate remaining time

        Returns:
            int: Remaining charging time required in minutes
        """
        if len(self.all_schedules) == 0:
            return self.total_time_required

        combined_schedule = self.get_combined_schedule()
        t = next_query_time - timedelta(minutes=5)

        usage_in_minutes = combined_schedule.loc[:t]["usage"].sum()

        return self.total_time_required - usage_in_minutes

    def set_last_schedule_end_time(self, next_query_time: datetime):
        """Update the end time of the most recent schedule.

        Args:
            next_query_time (datetime): New end time for the last schedule

        Raises:
            AssertionError: If new end time is before start time
        """
        if len(self.all_schedules) > 0:
            schedule, ctx = self.all_schedules[-1]
            self.all_schedules[-1] = (schedule, (ctx[0], next_query_time))
            assert ctx[0] < next_query_time

    def update_charging_schedule(
        self,
        next_query_time: datetime,
        next_new_schedule_start_time=None,
        new_schedule: Optional[pd.DataFrame] = None,
    ):
        """
        Update charging schedule and contiguity values.

        Args:
            next_query_time: Current query time
            next_new_schedule_start_time: Start time for next schedule
            new_schedule: New charging schedule to add
        """

        def _protocol_no_new_schedule(next_new_schedule_start_time):
            """
            1. Confirm that charging is not in progress and sleep delay is not required
            """
            if self.is_contiguous is True:
                self.sleep_delay = self.check_if_contiguity_sleep_required(
                    self.all_schedules[0][0], next_new_schedule_start_time
                )
            else:
                pass

        def _protocol_new_schedule(
            new_schedule, next_query_time, next_new_schedule_start_time
        ):
            """
            1. Modify previous schedule to end at "next_query_time"
            2. Append new schedule to record of existing schedules
            3. Confirm that charging is not in progress and sleep delay is not required
            """
            self.set_last_schedule_end_time(next_query_time)
            self.all_schedules.append((new_schedule, (next_query_time, self.end_time)))
            if self.is_contiguous is True:
                self.sleep_delay = self.check_if_contiguity_sleep_required(
                    new_schedule, next_new_schedule_start_time
                )

        def _protocol_sleep_delay(next_new_schedule_start_time):
            print("sleep protocol activated...")
            assert (
                next_new_schedule_start_time is not None
            ), "Sleep delay next new time is None"
            s = (
                self.get_combined_schedule().loc[next_new_schedule_start_time:]["usage"]
                == 0
            )
            delay_time = (
                self.end_time
                if s[s == True].empty == True
                else s[s == True].index.min()
            )
            self.contiguity_values_dict = {
                "delay_usage_window_start": delay_time,
                "delay_in_minutes": len(s[s == False]) * 5,
                "delay_in_intervals": len(s[s == False]),
                "remaining_units_required": self.get_remaining_units_required(
                    delay_time
                ),
                "remaining_time_required": self.get_remaining_time_required(delay_time),
            }

            self.contiguity_values_dict["num_segments_complete"] = (
                self.number_segments_complete(
                    next_query_time=self.contiguity_values_dict[
                        "delay_usage_window_start"
                    ]
                )
            )

        if new_schedule is None:
            _protocol_no_new_schedule(next_new_schedule_start_time)
        else:
            _protocol_new_schedule(
                new_schedule, next_query_time, next_new_schedule_start_time
            )

        if self.sleep_delay is True:
            _protocol_sleep_delay(next_new_schedule_start_time)
        else:
            self.contiguity_values_dict = {
                "delay_usage_window_start": None,
                "delay_in_minutes": None,
                "delay_in_intervals": None,
                "remaining_units_required": self.get_remaining_units_required(
                    next_query_time
                ),
                "remaining_time_required": self.get_remaining_time_required(
                    next_query_time
                ),
                "num_segments_complete": self.number_segments_complete(
                    next_query_time=next_query_time
                ),
            }

    def get_combined_schedule(self, end_time: datetime = None) -> pd.DataFrame:
        """Combine all schedules into a single DataFrame.

        Args:
            end_time (datetime, optional): Optional cutoff time for the combined schedule

        Returns:
            pd.DataFrame: Combined schedule of all charging segments
        """
        schedule_segments = []
        for s, ctx in self.all_schedules:
            schedule_segments.append(s[s.index < ctx[1]])
        combined_schedule = pd.concat(schedule_segments)

        if end_time:
            last_segment_start_time = end_time
            combined_schedule = combined_schedule.loc[:last_segment_start_time]

        return combined_schedule

    def check_if_contiguity_sleep_required(self, usage_plan, next_query_time):
        """Check if charging needs to be paused for contiguity.

        Args:
            usage_plan (pd.DataFrame): Planned charging schedule
            next_query_time (datetime): Time of next schedule update

        Returns:
            bool: True if charging needs to be paused
        """
        return bool(
            usage_plan.loc[(next_query_time - timedelta(minutes=5))]["usage"] > 0
        )

    def number_segments_complete(self, next_query_time: datetime = None):
        """Calculate number of completed charging segments.

        Args:
            next_query_time (datetime, optional): Time to check completion status

        Returns:
            int: Number of completed charging segments
        """
        if self.is_contiguous is True:
            combined_schedule = self.get_combined_schedule()
            completed_schedule = combined_schedule.loc[:next_query_time]
            charging_indicator = completed_schedule["usage"].astype(bool).sum()
            return bisect.bisect_right(
                list(accumulate(self.charge_per_interval)), (charging_indicator * 5)
            )
        else:
            return None


class RequerySimulator:
    def __init__(
        self,
        moers_list,
        requery_dates,
        region="CAISO_NORTH",
        window_start=datetime(2025, 1, 1, hour=21, second=1, tzinfo=UTC),
        window_end=datetime(2025, 1, 2, hour=8, second=1, tzinfo=UTC),
        usage_time_required_minutes=240,
        usage_power_kw=2,
        charge_per_interval=None,
    ):
        self.moers_list = moers_list
        self.requery_dates = requery_dates
        self.region = region
        self.window_start = window_start
        self.window_end = window_end
        self.usage_time_required_minutes = usage_time_required_minutes
        self.usage_power_kw = usage_power_kw
        self.charge_per_interval = charge_per_interval

        self.username = os.getenv("WATTTIME_USER")
        self.password = os.getenv("WATTTIME_PASSWORD")
        self.wt_opt = WattTimeOptimizer(self.username, self.password)

    def _get_initial_plan(self):
        return self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start,
            usage_window_end=self.window_end,
            usage_time_required_minutes=self.usage_time_required_minutes,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=self.charge_per_interval,
            optimization_method="simple",
            moer_data_override=self.moers_list[0][["point_time", "value"]],
        )

    def simulate(self):
        initial_plan = self._get_initial_plan()
        recalculator = WattTimeRecalculator(
            initial_schedule=initial_plan,
            start_time=self.window_start,
            end_time=self.window_end,
            total_time_required=self.usage_time_required_minutes,
            charge_per_interval=self.charge_per_interval,
        )

        # check to see the status of my segments to know if I should requery at all
        # if I do need to requery, then I need time required + segments remaining
        # if I don't then I store the state of my recalculator as is

        for i, new_window_start in enumerate(self.requery_dates[1:], 1):
            new_time_required = recalculator.get_remaining_time_required(
                new_window_start
            )
            if new_time_required > 0.0:
                next_plan = self.wt_opt.get_optimal_usage_plan(
                    region=self.region,
                    usage_window_start=new_window_start,
                    usage_window_end=self.window_end,
                    usage_time_required_minutes=new_time_required,
                    usage_power_kw=self.usage_power_kw,
                    charge_per_interval=self.charge_per_interval,
                    optimization_method="simple",
                    moer_data_override=self.moers_list[i][["point_time", "value"]],
                )
                recalculator.update_charging_schedule(
                    new_schedule=next_plan,
                    next_query_time=new_window_start,
                    next_new_schedule_start_time=None,
                )
            else:
                return recalculator
