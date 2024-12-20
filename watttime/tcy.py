from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import pytz
from typing import Optional, List
from .api import WattTimeHistorical
import os

@dataclass
class TCYConfig:
    """Configuration for TCY calculation"""
    region: str
    timezone: str
    holidays: Optional[List[str]] = None

class TCYCalculator:
    """Calculate Typical Carbon Year profiles from historical MOER data using a 3-year lookback period"""
    
    def __init__(
        self,
        region: str,
        timezone: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        holidays: Optional[List[str]] = None
    ):
        # Handle credentials like other API classes
        self.username = username or os.getenv("WATTTIME_USER")
        self.password = password or os.getenv("WATTTIME_PASSWORD")
        
        # Initialize API client with credentials
        self.wt_client = WattTimeHistorical(self.username, self.password)
        
        # Store configuration
        self.region = region
        self.timezone = pytz.timezone(timezone)
        
        # Set default holidays if none provided
        if not holidays:
            self.holidays = [
                "2023-01-01", "2023-05-29", "2023-07-04", "2023-09-04",
                "2023-11-23", "2023-12-25",
                "2024-01-01", "2024-05-27", "2024-07-04", "2024-09-02",
                "2024-11-28", "2024-12-25"
            ]
        else:
            self.holidays = holidays

    def _get_historical_data(self) -> pd.DataFrame:
        """Fetch 3 years of historical MOER data"""
        end = datetime.now(pytz.UTC)
        # Always use exactly 3 years of historical data
        start = end - timedelta(days=365 * 3)
        
        df = self.wt_client.get_historical_pandas(
            start=start.strftime('%Y-%m-%d %H:%MZ'),
            end=end.strftime('%Y-%m-%d %H:%MZ'),
            region=self.region,  # Changed from self.config.region
            signal_type='co2_moer'
        )
        
        # Set point_time as index and convert timezone
        df = df.set_index('point_time')
        df.index = df.index.tz_convert(self.timezone)  # Changed from self.config.timezone
        
        return df
    
    def _is_weekday(self, date: pd.Timestamp) -> bool:
        """Determine if a given date is a weekday (not weekend or holiday)"""
        date_str = date.strftime('%Y-%m-%d')
        return date.weekday() < 5 and date_str not in self.holidays

    def _create_reference_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create reference table of typical values for each month/hour/day type combination"""
        # Add helper columns
        df = df.assign(
            month=df.index.month,
            hour=df.index.hour,
            is_weekday=df.index.map(self._is_weekday)
        )
        
        # Group and calculate averages
        reference = df.groupby(['month', 'hour', 'is_weekday'])['value'].mean().reset_index()
        return reference

    def _generate_hourly_profile(self, year: int, reference: pd.DataFrame) -> pd.DataFrame:
        """Generate hourly profile for specified year using reference data"""
        # Create datetime index for entire year
        start = pd.Timestamp(year=year, month=1, day=1, tz=self.timezone)
        end = pd.Timestamp(year=year+1, month=1, day=1, tz=self.timezone)
        dates = pd.date_range(start=start, end=end, freq='h', inclusive='left')
        
        # Create profile DataFrame
        profile = pd.DataFrame(index=dates)
        profile['month'] = profile.index.month
        profile['hour'] = profile.index.hour
        profile['is_weekday'] = profile.index.map(self._is_weekday)
        
        # Merge with reference data to get typical values
        profile = profile.merge(
            reference,
            on=['month', 'hour', 'is_weekday'],
            how='left'
        )
        
        # Forward fill any missing values
        profile['value'] = profile['value'].ffill()
        
        return profile.set_index(dates)['value']

    def calculate_tcy(self, target_year: int) -> pd.DataFrame:
        """
        Calculate Typical Carbon Year profile for the specified year
        using 3 years of historical data to generate the profile.
        """
        # Get historical data (3 years)
        historical_data = self._get_historical_data()
        
        # Create reference table
        reference_table = self._create_reference_table(historical_data)
        
        # Generate hourly profile
        tcy_profile = self._generate_hourly_profile(target_year, reference_table)
        
        return tcy_profile