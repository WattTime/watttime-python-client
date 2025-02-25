import unittest
import os
from pytz import UTC
from watttime.api import WattTimeForecast, WattTimeOptimizer, WattTimeRecalculator
from datetime import timedelta, datetime
import pandas as pd

class TestRecalculatingOptimizer(unittest.TestCase):
    def setUp(self):
        self.region = "CAISO_NORTH"
        self.username = os.getenv("WATTTIME_USER")
        self.password = os.getenv("WATTTIME_PASSWORD")
        self.static_start_time = datetime(2025, 1, 1, hour=20, second=0, tzinfo=UTC)
        self.static_end_time = datetime(2025, 1, 2, hour=8, second=0, tzinfo=UTC)
        self.wt_hist = WattTimeForecast(self.username, self.password)
        self.wt_opt = WattTimeOptimizer(self.username, self.password)

        self.initial_usage_plan = self.wt_opt.get_optimal_usage_plan(
            region = self.region,
            usage_window_start=self.static_start_time,
            usage_window_end=self.static_end_time,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="auto",
            moer_data_override = self.moer_data_override(self.static_start_time,self.static_end_time,self.region)
        )

        self.recalculating_optimizer = WattTimeRecalculator(
            initial_schedule=self.initial_usage_plan,
            start_time=self.static_start_time,
            end_time=self.static_end_time,
            total_time_required=240
        )

    def moer_data_override(self, start_time,end_time,region):
        df = self.wt_hist.get_historical_forecast_pandas(
            start=start_time,
            end=end_time,
            region=region
            )
        return df[df.generated_at == df.generated_at.min()]
    
    def next_query_time(self,time,interval:int = 60):
        return time + timedelta(minutes=interval)
      
    # test initializing the recalculator class
    def test_init_recalculator_class(self) -> None:

        starting_schedule = self.recalculating_optimizer.get_combined_schedule()
        
        self.assertEqual(
            self.initial_usage_plan["usage"].tolist(), starting_schedule["usage"].tolist()
        )

        self.assertEqual(len(self.recalculating_optimizer.all_schedules), 1)

        self.assertEqual(self.initial_usage_plan["usage"].sum(), 240)
        self.assertEqual(starting_schedule["usage"].sum(), 240)

    def test_multiple_schedules_combined(self) -> None:
        """Test combining two schedules"""

        new_window_start = self.next_query_time(time=self.static_start_time)
        new_time_required =  self.recalculating_optimizer.get_remaining_time_required(new_window_start)
        new_usage_plan = self.wt_opt.get_optimal_usage_plan(
                region=self.region,
                usage_window_start=new_window_start,
                usage_window_end=self.static_end_time,
                usage_time_required_minutes=new_time_required,
                usage_power_kw=2,
                optimization_method="auto",
                moer_data_override=self.moer_data_override(new_window_start,self.static_end_time,self.region)
            )

        first_combined_schedule = self.recalculating_optimizer.get_combined_schedule()

        self.recalculating_optimizer.update_charging_schedule(
            new_schedule = new_usage_plan, 
            next_query_time=new_window_start
        )

        second_combined_schedule = self.recalculating_optimizer.get_combined_schedule()

        self.assertNotEqual(
            first_combined_schedule["usage"].tolist(),
            second_combined_schedule["usage"].tolist(),
        )
        self.assertEqual(
            first_combined_schedule["usage"].tolist()[: 12],
            second_combined_schedule["usage"].tolist()[: 12],
        )
        self.assertEqual(first_combined_schedule["usage"].sum(), 240)
        self.assertEqual(second_combined_schedule["usage"].sum(), 240)
    

    def test_schedules_date_index(self) -> None:
        idx = self.recalculating_optimizer.get_combined_schedule().index

        self.assertTrue(idx.is_unique)
        self.assertEqual(idx, pd.date_range(idx.min(),idx.max()),freq=timedelta(minutes=5))

'''
def check_num_intervals(schedule: pd.DataFrame) -> int:
    charging_indicator = schedule["usage"].apply(lambda x: 1 if x > 0 else 0)
    intervals = charging_indicator.diff().value_counts().get(1, 0)
    if charging_indicator.iloc[0] > 0:
        intervals += 1
    return intervals

class TestRecalculatingOptimizerWithConstraints(unittest.TestCase):
    def setUp(self):
        self.region = REGION
        self.username = os.getenv("WATTTIME_USER")
        self.password = os.getenv("WATTTIME_PASSWORD")

        self.static_start_time = datetime(2024, 1, 1, hour=20, second=1, tzinfo=UTC)
        self.static_end_time = datetime(2024, 1, 2, hour=8, second=1, tzinfo=UTC)

        self.wth = WattTimeForecast(self.username, self.password)
        self.curr_fcst_data = self.wth.get_historical_forecast_pandas(
            start=self.static_start_time - timedelta(minutes=5),
            end=self.static_end_time,
            region=self.region,
            signal_type="co2_moer",
            horizon_hours=12,
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
'''

if __name__ == "__main__":
    unittest.main()