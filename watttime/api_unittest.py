import os
import time
from datetime import datetime, timedelta
import unittest
import pandas as pd
from pytz import UTC
from watttime import WattTimeOptimizer

def get_usage_plan_mean_power(usage_plan):
    usage_plan_when_active = usage_plan[usage_plan["usage"]!=0].copy()
    usage_plan_when_active["power_kw"] = usage_plan_when_active["energy_usage_mwh"]/ (usage_plan_when_active["usage"] / 60) * 1000

    return usage_plan_when_active["power_kw"].mean()

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
        print("Using Baseline Plan")
        print(usage_plan["emissions_co2e_lb"].sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 240)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw)
    
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
        print("Using DP Plan w/ fixed power rate (kW)")
        print(usage_plan["emissions_co2e_lb"].sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 240)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw)


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
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 240 * self.usage_power_kw)


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
        print("Using DP Plan w/ variable power rate (kW)")
        print(usage_plan["emissions_co2e_lb"].sum())

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
        print(usage_plan)
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 7)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), self.usage_power_kw)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 7 * self.usage_power_kw)


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
        print(usage_plan)
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 120)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), 8.5)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 17)

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
        print(usage_plan)
        print(usage_plan.sum())

        # Check time required
        self.assertAlmostEqual(usage_plan["usage"].sum(), 180)
        # Check power
        self.assertAlmostEqual(get_usage_plan_mean_power(usage_plan), 5)
        # Check energy required
        self.assertAlmostEqual(usage_plan["energy_usage_mwh"].sum() * 1000, 15)

    def test_dp_two_intervals(self):
        """Test auto mode with two intervals."""
        usage_plan = self.wt_opt.get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.window_start_test,
            usage_window_end=self.window_end_test,
            usage_time_required_minutes=160,
            usage_power_kw=self.usage_power_kw,
            total_intervals=2,
            optimization_method="auto",
        )
        print("Using auto mode, but with two intervals")
        print(usage_plan["usage"].tolist())
        print(usage_plan.sum())



if __name__ == "__main__":
    unittest.main()