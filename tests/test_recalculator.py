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
        self.assertListEqual(list(idx), list(pd.date_range(idx.min(),idx.max(),freq=timedelta(minutes=5))))

if __name__ == "__main__":
    unittest.main()