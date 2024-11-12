import unittest
from watttime import RecalculatingWattTimeOptimizer, WattTimeOptimizer, WattTimeForecast
from evaluation import eval_framework as efu
from datetime import datetime, timedelta

class TestRecalculatingOptimizer(unittest.TestCase):
    def setUp(self):
        self.region = "PJM_NJ"
        self.username = ""
        self.password = ""
        
        # Seems that the watttime API considers both start and end to be inclusive 
        self.static_start_time = efu.convert_to_utc(datetime(2024, 1, 1, hour=20, second=1), local_tz_str="America/New_York")
        self.static_end_time = efu.convert_to_utc(datetime(2024, 1, 2, hour=8, second=1), local_tz_str="America/New_York")
        
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
        fcst_data = self.curr_fcst_data[self.curr_fcst_data["generated_at"] < self.static_start_time]
        basic_schedule = WattTimeOptimizer(self.username, self.password).get_optimal_usage_plan(
            region=self.region,
            usage_window_start=self.static_start_time,
            usage_window_end=self.static_end_time,
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="simple",
            moer_data_override=fcst_data
        )

        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region, 
            watttime_username=self.username, 
            watttime_password=self.password, 
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="simple"
        )

        starting_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time, 
            self.static_end_time, 
            curr_fcst_data=fcst_data
        )

        self.assertEqual(basic_schedule["usage"].tolist(), starting_schedule["usage"].tolist())


    def test_get_single_combined_schedule(self) -> None:
        """Test get_combined with single schedule"""
        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region, 
            watttime_username=self.username, 
            watttime_password=self.password, 
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="simple"
        )

        newest_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time, 
            self.static_end_time, 
        )
        combined_schedule = recalculating_optimizer.get_combined_schedule()

        self.assertEqual(newest_schedule["usage"].tolist(), combined_schedule["usage"].tolist())

    def test_multiple_schedules_combined(self) -> None: 
        """Test combining two schedules"""
        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region, 
            watttime_username=self.username, 
            watttime_password=self.password, 
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="simple"
        )
        first_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time, 
            self.static_end_time, 
        )
        first_combined_schedule = recalculating_optimizer.get_combined_schedule()
        second_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time + timedelta(hours=1), 
            self.static_end_time, 
        )
        
        second_combined_schedule = recalculating_optimizer.get_combined_schedule()
        self.assertNotEqual(first_combined_schedule["usage"].tolist(), second_combined_schedule["usage"].tolist())
        self.assertEqual(first_combined_schedule["usage"].tolist()[:12], second_combined_schedule["usage"].tolist()[:12])


    def test_override_data_behavior(self) -> None:
        """Test combining schedules with overriden data"""
        recalculating_optimizer = RecalculatingWattTimeOptimizer(
            region=self.region, 
            watttime_username=self.username, 
            watttime_password=self.password, 
            usage_time_required_minutes=240,
            usage_power_kw=2,
            optimization_method="simple"
        )
        last_data_time = self.data_times[self.data_times < self.static_start_time].max()
        curr_data = self.curr_fcst_data[self.curr_fcst_data["generated_at"] == last_data_time]
        first_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time, 
            self.static_end_time, 
            curr_data
        )
        first_combined_schedule = recalculating_optimizer.get_combined_schedule()

        last_data_time = self.data_times[self.data_times < self.static_start_time + timedelta(hours=1)].max()
        curr_data = self.curr_fcst_data[self.curr_fcst_data["generated_at"] == last_data_time]
        second_schedule = recalculating_optimizer.get_new_schedule(
            self.static_start_time + timedelta(hours=1), 
            self.static_end_time, 
            curr_data
        )
        
        second_combined_schedule = recalculating_optimizer.get_combined_schedule()
        self.assertNotEqual(first_combined_schedule["usage"].tolist(), second_combined_schedule["usage"].tolist())
        self.assertEqual(first_combined_schedule["usage"].tolist()[:12], second_combined_schedule["usage"].tolist()[:12])


    
