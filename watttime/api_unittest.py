# /bin/python3 -m watttime.api_test /home/annie.zhu/watttime-python-client-aer-algo/watttime/api_test.py 
import os
import time
from datetime import datetime, timedelta
import unittest
import pandas as pd
from pytz import UTC
from watttime import WattTimeOptimizer

def get_usage_plan_mean_power(usage_plan):
    usage_plan_when_active = usage_plan[usage_plan["usage"]!=0].copy()
    usage_plan_when_active["power_kw"] = usage_plan_when_active["energy_usage_mwh"] / (usage_plan_when_active["usage"] / 60) * 1000

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

    for index, value in usage_plan['usage'].items():
        if value != 0:
            current_component.append(index) 
            current_sum += value
        else:
            if current_component:
                components.append({'indices': current_component, 'sum': current_sum})
                current_component = []
                current_sum = 0

    # Add the last component if the dataframe ends with a non-zero sequence
    if current_component:
        components.append({'indices': current_component, 'sum': current_sum})

    return components

def pretty_format_usage(usage_plan):
    return "".join(["." if usage==0 else "E" for usage in usage_plan["usage"]])

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
        # cls.window_end_test = now + timedelta(minutes=240)

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
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw / 60)        
        # Check number of components (1 for baseline)
        self.assertEqual(len(get_contiguity_info(usage_plan)), 1)
    
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
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw / 60)


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
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw / 60)


    def test_dp_variable_power_rate(self):
        """Test the sophisticated plan with variable power rate."""
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
        ### TODO: Maybe implement way of checking power
        # Check energy required
        ### TODO: Maybe implement way of checking energy

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
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 7 * self.usage_power_kw / 60)


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
        # TODO: Add assert on the energy_usage_mwh per interval
        print("Using auto mode, with energy required in kWh")
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 120)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), 8.5)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 120 * 8.5 / 60)

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
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 180 * 5 / 60)

    def test_dp_two_intervals_unbounded(self):
        """Test auto mode with two intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(0,999999), (0,999999)],
            optimization_method="auto",
        )
        print("Using auto mode with two unbounded intervals\n", pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60)
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
            charge_per_interval=[(60,100), (60,100)],
            optimization_method="auto",
        )
        print("Using auto mode with two flexible intervals\n", pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60)
        
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
            charge_per_interval=[(30,None), (30,None), (30,None), (30,None)],
            optimization_method="auto",
        )
        print("Using auto mode with one-sided intervals\n", pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60)
        
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
            charge_per_interval=[(40,None), (40,None), (40,None), (40,None)],
            use_all_intervals=False,
            optimization_method="auto",
        )
        print("Using auto mode with one-sided intervals\n", pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60)
        
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
            charge_per_interval=[(60,60), (100,100)],
            optimization_method="auto",
        )
        print("Using auto mode with two exact intervals\n", pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60)

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
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60)

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
        """Test auto mode with two intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            charge_per_interval=[(67,67), (93,93)],
            optimization_method="auto",
        )
        print("Using auto mode with two exact unround intervals\n", pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 160)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 160 * self.usage_power_kw / 60)

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
            charge_per_interval=[(65,65)],
            optimization_method="auto",
        )
        print("Using auto mode, but with two intervals")
        print(pretty_format_usage(usage_plan))
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 65)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 65 * self.usage_power_kw / 60)

        contiguity_info = get_contiguity_info(usage_plan)
        # Check number of components
        self.assertEqual(len(contiguity_info), 1)
            

if __name__ == "__main__":
    unittest.main()
    # TestWattTimeOptimizer.setUpClass()
    # TestWattTimeOptimizer().test_dp_non_round_usage_time()