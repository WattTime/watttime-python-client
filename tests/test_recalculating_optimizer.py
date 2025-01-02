import unittest
from watttime.api import (
    RecalculatingWattTimeOptimizer,
    WattTimeOptimizer,
    WattTimeForecast,
    RecalculatingWattTimeOptimizerWithContiguity,
)
from datetime import datetime, timedelta
from pytz import UTC
import pandas as pd
import os


class TestRecalculatingOptimizer(unittest.TestCase):
    def setUp(self):
        self.region = "PJM_NJ"
        self.username = os.getenv("WATTTIME_USER")
        self.password = os.getenv("WATTTIME_PASSWORD")
        now = datetime.now(UTC)
        self.static_start_time = now - timedelta(minutes=720)
        self.static_end_time = now - timedelta(minutes=10)

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
    intervals = charging_indicator.diff().value_counts()[1]
    if charging_indicator[0] > 0:
        intervals += 1
    return intervals


class TestRecalculatingOptimizerWithConstraints(unittest.TestCase):
    def setUp(self):
        self.region = "PJM_NJ"
        self.username = os.getenv("WATTTIME_USER")
        self.password = os.getenv("WATTTIME_PASSWORD")
        now = datetime.now(UTC)
        self.static_start_time = now - timedelta(minutes=720)
        self.static_end_time = now - timedelta(minutes=10)

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
            self.assertEquals(
                schedule.index[0].to_pydatetime(),
                start_time + timedelta(minutes=4, seconds=59),
            )

        self.assertTrue(recalculating_optimizer.get_combined_schedule().index.is_unique)


if __name__ == "__main__":
    unittest.main()
    # TestWattTimeOptimizer.setUpClass()
    # TestWattTimeOptimizer().test_dp_non_round_usage_time()
