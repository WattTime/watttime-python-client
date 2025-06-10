from watttime.api import WattTimeForecast, WattTimeHistorical
from watttime_optimizer.api_opt import WattTimeOptimizer, WattTimeRecalculator
import pandas as pd
from watttime_optimizer.evaluator.utils import convert_to_utc, get_timezone_from_dict
import numpy as np
from typing import Optional
from datetime import timedelta
import matplotlib.pyplot as plt

class ImpactEvaluator:
    def __init__(self, username:str, password:str, obj: pd.DataFrame, region:Optional[str] = 'CAISO_NORTH'):
        """
        Evaluates the impact of a charging schedule.

        Parameters:
        -----------
        username : str
            API username
        password : str
            API password
        obj: pd.DataFrame
            Watttime Optimizer results frame.
        """
        self.actuals = WattTimeHistorical(username,password)
        self.obj = obj
        self.region=region

    def get_historical_actual_data(self, region:str = None):
        """
        Retrieve historical actual data for a specific plug-in time, horizon, and region.

        Parameters:
        -----------
        region : str
            The region for which to retrieve the actuals data.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing historical actuals data.
        """

        if region is None:
            region = self.region

        session_start_time = self.obj.index[0]
        session_end_time = self.obj.index[-1]

        return self.actuals.get_historical_pandas(
            start=session_start_time,
            end=session_end_time,
            region=region,
        )
    
    def get_historical_forecast_data(self):
        """
        Retrieve historical actual data for a specific plug-in time, horizon, and region.

        Parameters:
        -----------
        region : str
            The region for which to retrieve the actuals data.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing historical actuals data.
        """

        return self.obj["pred_moer"]
    
    def get_charging_schedule(self):
        """
        Extract and flatten usage values from input data
        Args:
            x: Input dictionary containing 'usage' key
        Returns:
            Flattened array of usage values
        """
        return self.obj["usage"].values.flatten()
    
    def get_energy_usage(self):
        """
        Extract and flatten usage values from input data
        Args:
            x: Input dictionary containing 'usage' key
        Returns:
            Flattened array of usage values
        """
        return self.obj["energy_usage_mwh"].values.flatten()

    def get_actual_emissions(self,region:str):
        """
        Calculate total CO2 emissions in pounds
        Args:
            region: eGrid region for API
        Returns:
            Sum of CO2 emissions
        """
        moer = self.get_historical_actual_data(region)['value'].values
        energy_usage_mwh = self.get_energy_usage()
        
        return np.multiply(moer, energy_usage_mwh)
    
    def get_forecast_emissions(self):
        """
        Calculate total CO2 emissions in pounds
        Args:
            x: Input dictionary containing 'emissions_co2_lb' key
        Returns:
            Sum of CO2 emissions
        """
        return self.obj["emissions_co2_lb"]
    
    def get_baseline_emissions(self,region:str):
        """
        Calculate total CO2 emissions in pounds.
        Assumes device did not follow an optimized schedule.
        """
        energy_usage_mwh = self.get_energy_usage()
        N = len(energy_usage_mwh[energy_usage_mwh<=0])
        moer = self.get_historical_actual_data(region)['value'].values

        return np.multiply(moer, np.pad(energy_usage_mwh[energy_usage_mwh>0], (0, N), 'constant'))
    
    def get_all_emissions_metrics(self,region:str):
        return {
            'baseline': self.get_baseline_emissions(region).sum(),
            'forecast': self.get_forecast_emissions().sum(),
            'actual':self.get_actual_emissions(region).sum()
        }
    
    def get_all_emissions_values(self,region:str):
        df = pd.DataFrame(self.get_forecast_emissions()).rename({"emissions_co2_lb":"forecast"},axis=1)
        df["baseline"] = self.get_baseline_emissions(region)
        df["actual"] = self.get_actual_emissions(region)
        return df
    
    def plot_predicated_moer(self):
        self.obj["pred_moer"].plot()
    
    def plot_usage_schedule(self):
        self.obj['usage'].plot()
    
    def get_timeseries_stats(self,df: pd.DataFrame, col:str = "pred_moer"):
        ''' Dispersion, slope, and intercept of the moer forecast'''
        m, b = np.polyfit(np.arange(len(df[col].values)),df[col].values, 1)
        stddev = df[col].std()
        mean = df[col].mean()
        return {
            'm':m,
            'b':b,
            'stddev':stddev,
            'mean': mean
        }

    def plot_impact(self, region:str):
        act = self.get_historical_actual_data(region=region).set_index("point_time")
        df = self.get_all_emissions_values(region=region)

        x = df.index
        y0 = (df['actual'] > 0).astype(int).values
        y1 = (df['baseline'] > 0).astype(int).values
        y2 = act.value.values

        # Create the main plot
        fig, ax1 = plt.subplots()
        
        # Plot the first data set
        ax1.plot(x, y1, 'b-', alpha=.2, label="ASAP Schedule")
        ax1.plot(x,y0,'g-',alpha=.2, label="Optimized Schedule")
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Usage Fraction (5 minute interval)', color='blue')
        ax1.tick_params('y', colors='b')
        ax1.fill_between(x,y0,0, where=y0>0,color='green', alpha=.2)
        ax1.fill_between(x,y1,0, where=y1>0,color='blue', alpha=.2)

        # Create the second y-axis
        ax2 = ax1.twinx()

        # Plot the second data set
        ax2.plot(x, y2, 'r-')
        ax2.set_ylabel('Actual Historic MOER (co2/lb)', color='r')
        ax2.tick_params('y', colors='r')

        # Display the plot
        ax1.legend(loc = "best", bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        plt.show()


class OptChargeEvaluator(WattTimeOptimizer):
    """
    This class inherits from WattTimeOptimizer

    Additional Methods:
    []
    """
    def moer_data_override(self, start_time, end_time, region, local_tz = None):
        if local_tz:
            time_zone = get_timezone_from_dict(local_tz)
            start_time = pd.Timestamp(convert_to_utc(start_time, time_zone))
            end_time = pd.Timestamp(convert_to_utc(end_time, time_zone))

        forecast_history = WattTimeForecast(self.username,self.password)
        df = forecast_history.get_historical_forecast_pandas(
            start=start_time,
            end=end_time,
            region=region      
        )
        return df[df.generated_at == df.generated_at.min()]
    
    def tz_conversion(self,time,region):
            return pd.Timestamp(convert_to_utc(time,get_timezone_from_dict(region)))
    
    def get_schedule_and_cost_api(
        self,
        usage_window_start: pd.Timestamp,
        usage_window_end: pd.Timestamp,
        usage_power_kw: float,
        time_needed: float,
        region:str = 'CAISO_NORTH',
        optimization_method: str = "auto",
        constraints: Optional[dict] = None,
        charge_per_segment: Optional[list] = None,
        tz_convert: bool = False,
        verbose:bool=False
    ) -> pd.DataFrame:
        """
        Generate optimal charging schedule based on MOER forecasts.
        
        Parameters:
        -----------
        usage_power_kw : float
            Power usage in kilowatts
        time_needed : float
            Required charging time in minutes
        total_time_horizon : int
            Total scheduling horizon in intervals
        moer_data : pd.DataFrame
            MOER forecast data
        optimization_method : str, optional
            Optimization method (default: "auto")
        charge_per_segment : list, optional
            List of charging constraints per interval
            
        Returns:
        --------
        pd.DataFrame
            Optimal charging schedule with emissions data
        """

        if tz_convert is True:
           usage_window_start = self.tz_conversion(usage_window_start,region)
           usage_window_end = self.tz_conversion(usage_window_end, region)

        # Generate optimal usage plan
        schedule = self.get_optimal_usage_plan(
            region=None,
            usage_window_start=usage_window_start,
            usage_window_end=usage_window_end,
            usage_time_required_minutes=time_needed,
            usage_power_kw=usage_power_kw,
            optimization_method=optimization_method,
            moer_data_override=self.moer_data_override(start_time = usage_window_start, end_time = usage_window_end, region=region),
            charge_per_segment=charge_per_segment,
            constraints=constraints,
            verbose=verbose
        )
        
        # Validate emissions data
        if schedule["emissions_co2_lb"].sum() == 0.0:
            self._log_zero_emissions_warning(
                usage_power_kw,
                time_needed,
                schedule["usage"].sum()
            )
            
        return schedule
            
    def _log_zero_emissions_warning(
        self,
        power: float,
        time_needed: float,
        total_usage: float
    ) -> None:
        """Log warning when zero emissions are detected."""
        print(
            "Warning using 0.0 lb of CO2e:",
            power,
            time_needed,
            total_usage
        )
class RecalculationOptChargeEvaluator(OptChargeEvaluator):
    '''
    TODO add notes on compatibility.
    '''

    def next_query_time(self,time,interval):
        return time + timedelta(minutes=interval)

    def fit_recalculator(
        self,
        usage_window_start: pd.Timestamp,
        usage_window_end: pd.Timestamp,
        usage_power_kw: float,
        time_needed: float,
        interval: int = 60,
        region:str = 'CAISO_NORTH',
        optimization_method: str = "auto",
        constraints: Optional[dict] = None,
        charge_per_segment: Optional[list] = None,
        tz_convert: bool = False,
        verbose:bool=False,
        contiguous:bool=False
    ):
        if tz_convert is True:
            print('tz converting...')
            usage_window_start = self.tz_conversion(usage_window_start,region)
            usage_window_end = self.tz_conversion(usage_window_end, region)

        initial_usage_plan = self.get_schedule_and_cost_api(
            region = region,
            usage_window_start=usage_window_start,
            usage_window_end=usage_window_end,
            time_needed=time_needed,
            usage_power_kw=usage_power_kw,
            charge_per_segment=charge_per_segment,
            optimization_method=optimization_method,
            constraints=constraints,
            verbose=verbose,
            tz_convert=False
        )

        recalculator = WattTimeRecalculator(
            initial_schedule = initial_usage_plan,
            start_time=usage_window_start,
            end_time=usage_window_end,
            total_time_required=time_needed,
            charge_per_segment=charge_per_segment,
            contiguous=contiguous
        )

        recalculator.update_charging_schedule(
                next_query_time=usage_window_start, 
                next_new_schedule_start_time=self.next_query_time(usage_window_start, interval)
            )
        
        optimization_outcomes = recalculator.contiguity_values_dict
        start_time = self.next_query_time(usage_window_start, interval)
        while optimization_outcomes["remaining_units_required"] >= recalculator.total_available_units:
            new_usage_plan = self.get_optimal_usage_plan(
                region = region,
                usage_window_start=start_time,
                usage_window_end=usage_window_end,
                usage_time_required_minutes=optimization_outcomes["remaining_time_required"],
                usage_power_kw=usage_power_kw,
                charge_per_segment=[int(optimization_outcomes["remaining_time_required"])] if recalculator.is_contiguous else None,
                optimization_method=optimization_method,
                moer_data_override=self.moer_data_override(start_time,usage_window_end,region),
                verbose=verbose
            )

            recalculator.update_charging_schedule(
                new_schedule = new_usage_plan, 
                next_query_time=start_time,
                next_new_schedule_start_time = self.next_query_time(start_time,interval)
            )

            optimization_outcomes = recalculator.contiguity_values_dict
            start_time = self.next_query_time(start_time,interval)

        return recalculator