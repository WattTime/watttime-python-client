from watttime.api import Recalculator, WattTimeForecast, WattTimeOptimizer
import os
from datetime import datetime, timedelta
from pytz import UTC

import sched, time

username = os.getenv("WATTTIME_USER")
password = os.getenv("WATTTIME_PASSWORD")
wt_opt = WattTimeOptimizer(username, password)

region = "CAISO_NORTH"
usage_power_kw = 12
now = datetime.now(UTC)
window_start = now + timedelta(minutes=10)
window_end = now + timedelta(minutes=720)
total_time_required = 240
total_intervals = (window_end - window_start).seconds // 60 // 20

initial_schedule = wt_opt.get_optimal_usage_plan(
            region=region,
            usage_window_start=window_start,
            usage_window_end=window_end,
            usage_time_required_minutes=total_time_required,
            usage_power_kw=usage_power_kw,
            optimization_method="simple",
        )

recalculator = Recalculator(
    initial_schedule=initial_schedule,
    start_time=window_start,
    end_time=window_end,
    total_time_required = total_time_required
)

new_window_start = now + timedelta(minutes=10)

def func(recalculator):
    new_window_start += timedelta(minutes=20)
    new_time_required = recalculator.get_remaining_time_required(next_query_time=new_window_start)

    next_plan = wt_opt.get_optimal_usage_plan(
            region=region,
            usage_window_start=window_start,
            usage_window_end=window_end,
            usage_time_required_minutes=new_time_required,
            usage_power_kw=usage_power_kw,
            optimization_method="simple",
        )
    recalculator.update_charging_schedule(
        new_schedule=next_plan,
        new_schedule_start_time=window_start
    )

    return next_plan["usage"].head()

s = sched.scheduler(time.time, time.sleep) #timefunc, delayfunc
s.enter(5*60,1,func(recalculator)) # delay, priority, action
s.run()








class RequeryScheduler:
    def __init__(self,
                 region="CAISO_NORTH",
                 window_start=datetime(2025, 1, 1, hour=20, second=1, tzinfo=UTC),
                 window_end=datetime(2025, 1, 2, hour=8, second=1, tzinfo=UTC),
                 usage_time_required_minutes=240,
                 usage_power_kw=2
                 ):
        self.region = region
        self.window_start = window_start
        self.window_end = window_end
        self.usage_time_required_minutes = usage_time_required_minutes
        self.usage_power_kw = usage_power_kw
        
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
            charge_per_interval=None,
            optimization_method="simple"
        )
    
    def func_to_trigger_new_run():

#https://pythonassets.com/posts/executing-code-every-certain-time/

    def simulate(self):
        initial_plan = self._get_initial_plan()
        recalculator = Recalculator(
            initial_schedule=initial_plan,
            start_time=self.window_start,
            end_time=self.window_end,
            total_time_required=self.usage_time_required_minutes
        )
        
        for i, new_window_start in enumerate(self.requery_dates[1:], 1):
            print(i)
            new_time_required = recalculator.get_remaining_time_required(new_window_start)
            next_plan = self.wt_opt.get_optimal_usage_plan(
                region=self.region,
                usage_window_start=new_window_start,
                usage_window_end=self.window_end,
                usage_time_required_minutes=new_time_required,
                usage_power_kw=self.usage_power_kw,
                charge_per_interval=None,
                optimization_method="simple",
                moer_data_override=self.moers_list[i][["point_time","value"]]
            )
            recalculator.update_charging_schedule(
                new_schedule=next_plan,
                new_schedule_start_time=new_window_start
            )
            
        return recalculator
