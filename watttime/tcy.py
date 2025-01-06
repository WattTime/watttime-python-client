from datetime import datetime, timedelta
import pandas as pd
import pytz
from typing import Optional
from .api import WattTimeHistorical
import holidays
import os

class TCYCalculator:
    """Calculate Typical Carbon Year profiles from historical MOER data using a 3-year lookback period"""
    
    def __init__(
        self,
        region: str,
        timezone: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        # Initialize API client with credentials
        self.wt_client = WattTimeHistorical(username, password)
        
        # Store configuration
        self.region = region
        self.timezone = pytz.timezone(timezone)
        
        # Initialize US holidays
        self.holidays = holidays.US()

    def _get_historical_data(self) -> pd.DataFrame:
        """Fetch most recent 3 years of historical MOER data"""
        end = datetime.now(pytz.UTC)
        # Always use exactly 3 years of historical data
        start = end - timedelta(days=365 * 3)
        
        df = self.wt_client.get_historical_pandas(
            start=start.strftime('%Y-%m-%d %H:%MZ'),
            end=end.strftime('%Y-%m-%d %H:%MZ'),
            region=self.region,
            signal_type='co2_moer'
        )
        
        # Set point_time as index and convert timezone
        df = df.set_index('point_time')
        df.index = df.index.tz_convert(self.timezone)
        
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
        Calculate Typical Carbon Year profile for the target year using recent MOER data
        but weekday/weekend/holiday patterns from the target year.
        """
        # Get historical data (most recent 3 years)
        historical_data = self._get_historical_data()
        
        # Create reference table from recent data
        reference_table = self._create_reference_table(historical_data)
        
        # Generate hourly profile using target year's calendar
        tcy_profile = self._generate_hourly_profile(target_year, reference_table)
        
        return tcy_profile