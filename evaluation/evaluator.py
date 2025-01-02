from watttime.api import WattTimeOptimizer
import pandas as pd

class OptChargeEvaluator(WattTimeOptimizer):
    def __init__(self, username: str, password: str):
        """
        Initialize OptimalCharger with API credentials.
        
        Parameters:
        -----------
        username : str
            API username
        password : str
            API password
        """
        self.username = username
        self.password = password
        self.optimizer = WattTimeOptimizer(username, password)
        
    def get_optimal_schedule(
        self,
        usage_power_kw: float,
        time_needed: float,
        total_time_horizon: int,
        moer_data: pd.DataFrame,
        optimization_method: str = "auto",
        charge_per_interval: list = []
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
        charge_per_interval : list, optional
            List of charging constraints per interval
            
        Returns:
        --------
        pd.DataFrame
            Optimal charging schedule with emissions data
        """
        # Get time window from MOER data
        usage_window_start = pd.to_datetime(moer_data["point_time"].iloc[0])
        usage_window_end = pd.to_datetime(
            moer_data["point_time"].iloc[total_time_horizon - 1]
        )
        
        # Adjust time needed if it exceeds available window
        adjusted_time = min(
            time_needed, 
            total_time_horizon * self.optimizer.OPT_INTERVAL
        )
        
        # Generate optimal usage plan
        schedule = self.optimizer.get_optimal_usage_plan(
            region=None,
            usage_window_start=usage_window_start,
            usage_window_end=usage_window_end,
            usage_time_required_minutes=adjusted_time,
            usage_power_kw=usage_power_kw,
            optimization_method=optimization_method,
            moer_data_override=moer_data,
            charge_per_interval=charge_per_interval
        )
        
        # Validate emissions data
        if schedule["emissions_co2e_lb"].sum() == 0.0:
            self._log_zero_emissions_warning(
                usage_power_kw,
                adjusted_time,
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