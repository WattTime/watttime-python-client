import os
from datetime import datetime, timedelta
import unittest
import pandas as pd
from pytz import UTC
import pytz
from watttime.api import RecalculatingWattTimeOptimizer, WattTimeOptimizer, WattTimeForecast, RecalculatingWattTimeOptimizerWithContiguity


def get_usage_plan_mean_power(usage_plan):
    usage_plan_when_active = usage_plan[usage_plan["usage"] != 0].copy()
    usage_plan_when_active["power_kw"] = (
        usage_plan_when_active["energy_usage_mwh"]
        / (usage_plan_when_active["usage"] / 60)
        * 1000
    )

    return usage_plan_when_active["power_kw"].mean()


def get_contiguity_info(usage_plan):
    """
    Extract contiguous non-zero components from a DataFrame column 'usage'
    and compute the sum for each component.

    Args:
        usage_plan (pd.DataFrame): DataFrame with a column named 'usage'.

    Returns:
        List[Dict]: A list of dictionaries, each containing the indices and sum
                    of a contiguous non-zero component.
    """
    components = []
    current_component = []
    current_sum = 0

    for index, value in usage_plan["usage"].items():
        if value != 0:
            current_component.append(index)
            current_sum += value
        else:
            if current_component:
                components.append({"indices": current_component, "sum": current_sum})
                current_component = []
                current_sum = 0

    # Add the last component if the dataframe ends with a non-zero sequence
    if current_component:
        components.append({"indices": current_component, "sum": current_sum})

    return components


def pretty_format_usage(usage_plan):
    return "".join(["." if usage == 0 else "E" for usage in usage_plan["usage"]])


class TestWattTimeOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize WattTimeOptimizer before running any tests."""
        username = os.getenv("WATTTIME_USER")
        password = os.getenv("WATTTIME_PASSWORD")
        cls.wt_opt = WattTimeOptimizer(username, password)
        cls.region = "PJM_NJ"
        cls.usage_power_kw = 12
        now = datetime.now(UTC)
        cls.window_start_test = now + timedelta(minutes=10)
        cls.window_end_test = now + timedelta(minutes=720)

    def test_baseline_plan(self):
        """Test the baseline plan."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=240,
            usage_power_kw=self.usage_power_kw,
            optimization_method="baseline",
        )
        print("Using Baseline Plan\n", pretty_format_usage(usage_plan))

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 240)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw / 60
        )
        # Check number of components (1 for baseline)
        self.assertEqual(len(get_contiguity_info(usage_plan)), 1)

    def test_simple_plan(self):
        """Test the simple plan."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=240,
            usage_power_kw=self.usage_power_kw,
            optimization_method="simple",
        )
        print("Using Simple Plan\n", pretty_format_usage(usage_plan))

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 240)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw / 60
        )

    def test_dp_fixed_power_rate(self):
        """Test the sophisticated plan with a fixed power rate."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=240,
            usage_power_kw=self.usage_power_kw,
            optimization_method="sophisticated",
        )
        print("Using DP Plan w/ fixed power rate\n", pretty_format_usage(usage_plan))

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 240)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw / 60
        )

    def test_dp_fixed_power_rate_with_uncertainty(self):
        """Test the sophisticated plan with fixed power rate and time uncertainty."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=240,
            usage_power_kw=self.usage_power_kw,
            usage_time_uncertainty_minutes=180,
            optimization_method="sophisticated",
        )
        print("Using DP Plan w/ fixed power rate and charging uncertainty")
        print(usage_plan["emissions_co2e_lb"].sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 240)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw / 60
        )

    def test_dp_variable_power_rate(self):
        """Test the plan with variable power rate."""
        usage_power_kw_df = pd.DataFrame(
            [[0, 12], [20, 12], [40, 12], [100, 12], [219, 12], [220, 2.4], [320, 2.4]],
            columns=["time", "power_kw"],
        )
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=320,
            usage_power_kw=usage_power_kw_df,
            optimization_method="auto",
        )
        print("Using DP Plan w/ variable power rate")
        print(usage_plan["emissions_co2e_lb"].sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 320)
        # Check power
        usage_plan_nonzero_entries = usage_plan[usage_plan["usage"] > 0]
        power_kwh_array = (
            usage_plan_nonzero_entries["energy_usage_mwh"].values * 1e3 * 60 / 5
        )
        self.assertAlmostEqual(power_kwh_array[: 220 // 5].mean(), 12.0)
        self.assertAlmostEqual(power_kwh_array[220 // 5 :].mean(), 2.4)
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 220 * 12 / 60 + 100 * 2.4 / 60
        )

    def test_dp_non_round_usage_time(self):
        """Test auto mode with non-round usage time minutes."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=7,
            usage_power_kw=self.usage_power_kw,
            optimization_method="auto",
        )
        print("Using auto mode, but with a non-round usage time minutes")
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 7)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 7 * self.usage_power_kw / 60
        )

    def test_dp_input_time_energy(self):
        """Test auto mode with a usage time and energy required."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=120,
            energy_required_kwh=17,
            optimization_method="auto",
        )
        print("Using auto mode, with energy required in kWh")
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 120)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), 8.5)
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 120 * 8.5 / 60
        )

    def test_dp_input_constant_power_energy(self):
        """Test auto mode with a constant power and energy required."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_power_kw=5,
            energy_required_kwh=15,
            optimization_method="auto",
        )
        print("Using auto mode, with energy required in kWh")
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 180)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), 5)
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 180 * 5 / 60
        )

    def test_dp_two_intervals_unbounded(self):
        """Test auto mode with two intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(0, 999999), (0, 999999)],
            optimization_method="auto",
        )
        print(
            "Using auto mode with two unbounded intervals\n",
            pretty_format_usage(usage_plan),
        )
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60
        )
        # Check number of components
        self.assertLessEqual(len(get_contiguity_info(usage_plan)), 2)

    def test_dp_two_intervals_flexible_length(self):
        """Test auto mode with two variable length intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(60, 100), (60, 100)],
            optimization_method="auto",
        )
        print(
            "Using auto mode with two flexible intervals\n",
            pretty_format_usage(usage_plan),
        )
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60
        )

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertLessEqual(len(contiguity_info), 2)
        if len(contiguity_info) == 2:
            # Check first component length
            self.assertGreaterEqual(contiguity_info[0]["sum"], 60)
            self.assertLessEqual(contiguity_info[0]["sum"], 100)
            # Check second component length
            self.assertGreaterEqual(contiguity_info[1]["sum"], 60)
            self.assertLessEqual(contiguity_info[1]["sum"], 100)
        else:
            # Check combined component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 160)

    def test_dp_two_intervals_one_sided_length(self):
        """Test auto mode with two variable length intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(30, None), (30, None), (30, None), (30, None)],
            optimization_method="auto",
        )
        print(
            "Using auto mode with one-sided intervals\n",
            pretty_format_usage(usage_plan),
        )
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60
        )

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertLessEqual(len(contiguity_info), 4)
        for i in range(len(contiguity_info)):
            # Check component length
            self.assertGreaterEqual(contiguity_info[i]["sum"], 30)

    def test_dp_two_intervals_one_sided_length_use_all_false(self):
        """Test auto mode with two variable length intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(40, None), (40, None), (40, None), (40, None)],
            use_all_intervals=False,
            optimization_method="auto",
        )
        print(
            "Using auto mode with one-sided intervals\n",
            pretty_format_usage(usage_plan),
        )
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60
        )

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertLessEqual(len(contiguity_info), 4)
        for i in range(len(contiguity_info)):
            # Check component length
            self.assertGreaterEqual(contiguity_info[i]["sum"], 40)

    def test_dp_two_intervals_exact_input_a(self):
        """Test auto mode with two intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(60, 60), (100, 100)],
            optimization_method="auto",
        )
        print(
            "Using auto mode with two exact intervals\n",
            pretty_format_usage(usage_plan),
        )
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60
        )

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertLessEqual(len(contiguity_info), 2)
        if len(contiguity_info) == 2:
            # Check first component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 60)
            # Check second component length
            self.assertAlmostEqual(contiguity_info[1]["sum"], 100)
        else:
            # Check combined component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 160)

    def test_dp_two_intervals_exact_input_b(self):
        """Test auto mode with two intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[60, 100],
            optimization_method="auto",
        )
        print("Using auto mode, but with two intervals")
        print(pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60
        )

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertLessEqual(len(contiguity_info), 2)
        if len(contiguity_info) == 2:
            # Check first component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 60)
            # Check second component length
            self.assertAlmostEqual(contiguity_info[1]["sum"], 100)
        else:
            # Check combined component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 160)

    def test_dp_two_intervals_exact_unround(self):
        """Test auto mode with two intervals, specified via list of tuple."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(67, 67), (93, 93)],
            optimization_method="auto",
        )
        print(
            "Using auto mode with two exact unround intervals\n",
            pretty_format_usage(usage_plan),
        )
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60
        )

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertLessEqual(len(contiguity_info), 2)
        if len(contiguity_info) == 2:
            # Check first component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 67)
            # Check second component length
            self.assertAlmostEqual(contiguity_info[1]["sum"], 93)
        else:
            # Check combined component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 160)

    def test_dp_two_intervals_exact_unround_alternate_input(self):
        """Test auto mode with two intervals, specified via list of ints."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[67, 93],
            optimization_method="auto",
        )
        print(
            "Using auto mode with two exact unround intervals\n",
            pretty_format_usage(usage_plan),
        )
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60
        )

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertLessEqual(len(contiguity_info), 2)
        if len(contiguity_info) == 2:
            # Check first component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 67)
            # Check second component length
            self.assertAlmostEqual(contiguity_info[1]["sum"], 93)
        else:
            # Check combined component length
            self.assertAlmostEqual(contiguity_info[0]["sum"], 160)

    def test_dp_two_intervals_exact_inconsistent_b(self):
        """Test auto mode with one interval that is inconsistent with usage_time_required."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(65, 65)],
            optimization_method="auto",
        )
        print("Using auto mode, but with two intervals")
        print(pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 65)
        # Check power
        self.assertAlmostEqual(
            get_usage_plan_mean_power(usage_plan), self.usage_power_kw
        )
        # Check energy required
        self.assertAlmostEqual(
            usage_plan["energy_usage_mwh"].sum() * 1000, 65 * self.usage_power_kw / 60
        )

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertEqual(len(contiguity_info), 1)

def convert_to_utc(local_time_str, local_tz_str):
    local_time = datetime.strptime(
        local_time_str.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"
    )
    local_tz = pytz.timezone(local_tz_str)
    local_time = local_tz.localize(local_time)
    return local_time.astimezone(pytz.utc)


class TestRecalculatingOptimizer(unittest.TestCase):
    def setUp(self):
        self.region = "PJM_NJ"
        self.username = os.getenv("WATTTIME_USER")
        self.password = os.getenv("WATTTIME_PASSWORD")
        self.static_start_time = convert_to_utc(
            datetime(2024, 1, 1, hour=20, second=1), local_tz_str="America/New_York"
        )
        self.static_end_time = convert_to_utc(
            datetime(2024, 1, 2, hour=8, second=1), local_tz_str="America/New_York"
        )

        self.wth = WattTimeForecast(self.username, self.password)
        self.curr_fcst_data = self.wth.get_historical_forecast_pandas(
            start=self.static_start_time - timedelta(minutes=5),
            end=self.static_end_time,
            region=self.region,
            signal_type="co2_moer",
            horizon_hours=72,
        )
        self.data_times = self.curr_fcst_data["generated_at"]

    def test_init_recalculating_optimizer(self) -> None:
        """Test init"""
        fcst_data = self.curr_fcst_data[
            self.curr_fcst_data["generated_at"] < self.static_start_time
        ]
        basic_schedule = WattTimeOptimizer(
            self.username, self.password
        ).get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.static_start_time,
            usage_window_end=self.static_end_time,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="auto",
            moer_data_override=fcst_data,
        )

        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="auto",
        )

        starting_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time, self.static_end_time, curr_fcst_data=fcst_data
        )

        self.assertEqual(
            basic_schedule["usage"].tolist(), starting_schedule["usage"].tolist()
        )
        self.assertEqual(basic_schedule["usage"].sum(), 240)

    def test_get_single_combined_schedule(self) -> None:
        """Test get_combined with single schedule"""
        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="auto",
        )

        newest_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time,
            self.static_end_time,
        )
        combined_schedule = recalculating_optimizer.get_combined_schedule()

        self.assertEqual(
            newest_schedule["usage"].tolist(), combined_schedule["usage"].tolist()
        )
        self.assertEqual(combined_schedule["usage"].sum(), 240)

    def test_multiple_schedules_combined(self) -> None:
        """Test combining two schedules"""
        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="auto",
        )
        first_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time,
            self.static_end_time,
        )
        first_combined_schedule = recalculating_optimizer.get_combined_schedule()
        second_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time + timedelta(hours=7),
            self.static_end_time,
        )
        second_combined_schedule = recalculating_optimizer.get_combined_schedule()

        self.assertNotEqual(
            first_combined_schedule["usage"].tolist(),
            second_combined_schedule["usage"].tolist(),
        )
        self.assertEqual(
            first_combined_schedule["usage"].tolist()[: 12 * 7],
            second_combined_schedule["usage"].tolist()[: 12 * 7],
        )
        self.assertEqual(first_combined_schedule["usage"].sum(), 240)
        self.assertEqual(second_combined_schedule["usage"].sum(), 240)

    def test_schedule_times(self) -> None:
        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=30,
            usage_power_kw=2,
            optimization_method="auto",
        )

        start_time = self.static_start_time
        end_time = self.static_end_time + timedelta(hours=2)

        for i in range(2 * 2):
            start_time = start_time + timedelta(minutes=30)
            schedule = recalculating_optimizer.get_new_schedule(start_time, end_time)
            self.assertTrue(schedule.index.is_unique)
            self.assertEquals(
                schedule.index[0].to_pydatetime(),
                start_time + timedelta(minutes=4, seconds=59),
            )

        self.assertTrue(recalculating_optimizer.get_combined_schedule().index.is_unique)

    def test_override_data_behavior(self) -> None:
        """Test combining schedules with overriden data"""
        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="auto",
        )
        last_data_time = self.data_times[self.data_times < self.static_start_time].max()
        first_query_time_data = self.curr_fcst_data[
            self.curr_fcst_data["generated_at"] == last_data_time
        ]
        first_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time, self.static_end_time, first_query_time_data
        )
        first_combined_schedule = recalculating_optimizer.get_combined_schedule()

        last_data_time = self.data_times[
            self.data_times < self.static_start_time + timedelta(hours=7)
        ].max()
        second_query_time_data = self.curr_fcst_data[
            self.curr_fcst_data["generated_at"] == last_data_time
        ]
        second_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time + timedelta(hours=7),
            self.static_end_time,
            second_query_time_data,
        )

        second_combined_schedule = recalculating_optimizer.get_combined_schedule()
        self.assertNotEqual(
            first_combined_schedule["usage"].tolist(),
            second_combined_schedule["usage"].tolist(),
        )
        self.assertEqual(
            first_combined_schedule["usage"].tolist()[: 12 * 7],
            second_combined_schedule["usage"].tolist()[: 12 * 7],
        )

        self.assertEqual(first_combined_schedule["usage"].sum(), 240)
        self.assertEqual(second_combined_schedule["usage"].sum(), 240)


def check_num_intervals(schedule: pd.DataFrame) -> int:
    charging_indicator = schedule["usage"].apply(lambda x: 1 if x > 0 else 0)
    intervals = charging_indicator.diff().value_counts().get(1, 0)
    if charging_indicator[0] > 0:
        intervals += 1
    return intervals


class TestRecalculatingOptimizerWithConstraints(unittest.TestCase):
    def setUp(self):
        self.region = "PJM_NJ"
        self.username = os.getenv("WATTTIME_USER")
        self.password = os.getenv("WATTTIME_PASSWORD")

        self.static_start_time = convert_to_utc(
            datetime(2024, 1, 1, hour=20, second=1), local_tz_str="America/New_York"
        )
        self.static_end_time = convert_to_utc(
            datetime(2024, 1, 2, hour=8, second=1), local_tz_str="America/New_York"
        )

        self.wth = WattTimeForecast(self.username, self.password)
        self.curr_fcst_data = self.wth.get_historical_forecast_pandas(
            start=self.static_start_time - timedelta(minutes=5),
            end=self.static_end_time,
            region=self.region,
            signal_type="co2_moer",
            horizon_hours=72,
        )
        self.data_times = self.curr_fcst_data["generated_at"]

    def test_recalculating_optimizer_adjust_num_intervals(self) -> None:
        recalculating_optimizer = RecalculatingWattTimeOptimizerWithContiguity(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="sophisticated",
            charge_per_interval=[140, 100],
        )

        initial_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time,
            self.static_end_time,
        )
        self.assertTrue(check_num_intervals(initial_schedule) <= 2)

        first_interval_end_time = initial_schedule[
            initial_schedule["usage"].diff() < 0
        ].index[0]

        next_schedule = recalculating_optimizer.get_new_schedule(
            first_interval_end_time,
            self.static_end_time,
        )

        self.assertTrue(check_num_intervals(next_schedule) == 1)
        self.assertEqual(
            recalculating_optimizer.get_combined_schedule()["usage"].sum(), 240
        )

    def test_recalculating_optimizer_mid_interval(self) -> None:
        recalculating_optimizer = RecalculatingWattTimeOptimizerWithContiguity(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="sophisticated",
            charge_per_interval=[120, 120],
        )

        initial_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time,
            self.static_end_time,
        )
        self.assertTrue(check_num_intervals(initial_schedule) <= 2)

        mid_interval_time = initial_schedule[
            initial_schedule["usage"].diff() < 0
        ].index[0] - timedelta(minutes=10)

        next_schedule = recalculating_optimizer.get_new_schedule(
            mid_interval_time,
            self.static_end_time,
        )

        # Check that remaining schedule before interval end is the same
        self.assertTrue(
            initial_schedule[initial_schedule.index >= mid_interval_time]
            .head(2)
            .equals(next_schedule.head(2))
        )
        self.assertEqual(next_schedule.index[0], mid_interval_time)
        self.assertEqual(
            recalculating_optimizer.get_combined_schedule()["usage"].sum(), 240
        )

    def test_init_recalculating_contiguity_optimizer(self) -> None:
        """Test init"""

        recalculating_optimizer = RecalculatingWattTimeOptimizerWithContiguity(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="sophisticated",
            charge_per_interval=[100, 140],
        )

        for i in range(12):
            schedule = recalculating_optimizer.get_new_schedule(
                self.static_start_time + timedelta(hours=i),
                self.static_end_time,
            )

        self.assertTrue(
            check_num_intervals(recalculating_optimizer.get_combined_schedule()) <= 2
        )
        self.assertEqual(
            recalculating_optimizer.get_combined_schedule()["usage"].sum(), 240
        )

    def test_frequent_recalculating_with_contiguity(self) -> None:
        recalculating_optimizer = RecalculatingWattTimeOptimizerWithContiguity(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=30,
            usage_power_kw=2,
            optimization_method="sophisticated",
            charge_per_interval=[15, 15],
        )
        start_time = self.static_start_time
        end_time = self.static_end_time + timedelta(hours=2)

        for i in range(12 * 2):
            start_time = start_time + timedelta(minutes=5)
            schedule = recalculating_optimizer.get_new_schedule(start_time, end_time)

        self.assertTrue(
            check_num_intervals(recalculating_optimizer.get_combined_schedule()) <= 2
        )
        self.assertEqual(
            recalculating_optimizer.get_combined_schedule()["usage"].sum(), 30
        )

    def test_schedule_times(self) -> None:
        recalculating_optimizer = RecalculatingWattTimeOptimizerWithContiguity(
            region=self.region,
            watttime_username=self.username,
            watttime_password=self.password,
            usage_time_required_minutes=30,
            usage_power_kw=2,
            optimization_method="sophisticated",
            charge_per_interval=[15, 15],
        )

        start_time = self.static_start_time
        end_time = self.static_end_time + timedelta(hours=2)

        for i in range(2 * 2):
            start_time = start_time + timedelta(minutes=30)
            schedule = recalculating_optimizer.get_new_schedule(start_time, end_time)
            self.assertTrue(schedule.index.is_unique)
            self.assertEqual(
                schedule.index[0].to_pydatetime(),
                start_time + timedelta(minutes=4, seconds=59),
            )

        self.assertTrue(recalculating_optimizer.get_combined_schedule().index.is_unique)

if __name__ == "__main__":
    unittest.main()
    # TestWattTimeOptimizer.setUpClass()
    # TestWattTimeOptimizer().test_dp_non_round_usage_time()
