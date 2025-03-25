import math
from datetime import datetime
import pytz
import pandas as pd

import glob
import os



TZ_DICTIONARY = {
    "AECI": "America/Chicago",
    "AVA": "America/Los_Angeles",
    "AZPS": "America/Phoenix",
    "BANC": "America/Los_Angeles",
    "BPA": "America/Los_Angeles",
    "CAISO_ESCONDIDO": "America/Los_Angeles",
    "CAISO_LONGBEACH": "America/Los_Angeles",
    "CAISO_NORTH": "America/Los_Angeles",
    "CAISO_PALMSPRINGS": "America/Los_Angeles",
    "CAISO_REDDING": "America/Los_Angeles",
    "CAISO_SANBERNARDINO": "America/Los_Angeles",
    "CAISO_SANDIEGO": "America/Los_Angeles",
    "CHPD": "America/Los_Angeles",
    "CPLE": "America/New_York",
    "CPLW": "America/New_York",
    "DOPD": "America/Los_Angeles",
    "DUK": "America/New_York",
    "ELE": "America/Denver",
    "ERCOT_AUSTIN": "America/Chicago",
    "ERCOT_COAST": "America/Chicago",
    "ERCOT_EASTTX": "America/Chicago",
    "ERCOT_HIDALGO": "America/Chicago",
    "ERCOT_NORTHCENTRAL": "America/Chicago",
    "ERCOT_PANHANDLE": "America/Chicago",
    "ERCOT_SANANTONIO": "America/Chicago",
    "ERCOT_SECOAST": "America/Chicago",
    "ERCOT_SOUTHTX": "America/Chicago",
    "ERCOT_WESTTX": "America/Chicago",
    "FMPP": "America/New_York",
    "FPC": "America/New_York",
    "FPL": "America/New_York",
    "GVL": "America/New_York",
    "IID": "America/Los_Angeles",
    "IPCO": "America/Boise",
    "ISONE_CT": "America/New_York",
    "ISONE_ME": "America/New_York",
    "ISONE_NEMA": "America/New_York",
    "ISONE_NH": "America/New_York",
    "ISONE_RI": "America/New_York",
    "ISONE_SEMA": "America/New_York",
    "ISONE_VT": "America/New_York",
    "ISONE_WCMA": "America/New_York",
    "JEA": "America/New_York",
    "LDWP": "America/Los_Angeles",
    "LGEE": "America/New_York",
    "MISO_INDIANAPOLIS": "America/Indiana/Indianapolis",
    "MISO_N_DAKOTA": "America/North_Dakota/Center",
    "MPCO": "America/Denver",
    "NEVP": "America/Los_Angeles",
    "NYISO_NYC": "America/New_York",
    "PACE": "America/Denver",
    "PACW": "America/Los_Angeles",
    "PGE": "America/Los_Angeles",
    "PJM_CHICAGO": "America/Chicago",
    "PJM_DC": "America/New_York",
    "PJM_EASTERN_KY": "America/New_York",
    "PJM_EASTERN_OH": "America/New_York",
    "PJM_ROANOKE": "America/New_York",
    "PJM_NJ": "America/New_York",
    "PJM_SOUTHWEST_OH": "America/New_York",
    "PJM_WESTERN_KY": "America/New_York",
    "PNM": "America/Denver",
    "PSCO": "America/Denver",
    "PSEI": "America/Los_Angeles",
    "SC": "America/New_York",
    "SCEG": "America/New_York",
    "SCL": "America/Los_Angeles",
    "SEC": "America/New_York",
    "SOCO": "America/Chicago",
    "SPA": "America/Chicago",
    "SPP_FORTPECK": "America/Denver",
    "SPP_KANSAS": "America/Chicago",
    "SPP_KC": "America/Chicago",
    "SPP_MEMPHIS": "America/Chicago",
    "SPP_ND": "America/North_Dakota/Beulah",
    "SPP_OKCTY": "America/Chicago",
    "SPP_SIOUX": "America/Chicago",
    "SPP_SPRINGFIELD": "America/Chicago",
    "SPP_SWOK": "America/Chicago",
    "SPP_TX": "America/Chicago",
    "SPP_WESTNE": "America/Chicago",
    "SRP": "America/Phoenix",
    "TAL": "America/New_York",
    "TEC": "America/New_York",
    "TEPC": "America/Phoenix",
    "TID": "America/Los_Angeles",
    "TPWR": "America/Los_Angeles",
    "TVA": "America/Chicago",
    "WACM": "America/Denver",
    "WALC": "America/Phoenix",
    "WAUW": "America/Denver",
}

def sanitize_time_needed(x,y):
    return int(math.ceil(min(x, y) / 300.0) * 5)

def sanitize_total_intervals(x):
    return math.ceil(x)

def intervalize_power_rate(kW_value: float, convert_to_MWh=True) -> float:
    """
    Calculate the energy used in an interval from a power rate in kilowatts
    This will return a value in units of MWh by default.
    If convert_to_MWh is false, it will convert to kWh units instead.
    """
    five_min_rate = kW_value / 12
    if convert_to_MWh:
        five_min_rate = five_min_rate / 1000
    return five_min_rate


def get_timezone_from_dict(key, dictionary=TZ_DICTIONARY):
    """
    Retrieve the timezone value from the dictionary based on the given key.

    Parameters:
    -----------
    key : str
        The key whose corresponding timezone value is to be retrieved.
    dictionary : dict, optional
        The dictionary from which to retrieve the value (default is TZ_DICTIONARY).

    Returns:
    --------
    str or None
        The timezone value corresponding to the given key if it exists, otherwise None.
    """
    return dictionary.get(key)


def convert_to_utc(local_time_str, local_tz_str):
    """
    Convert a time expressed in any local time to UTC.

    Parameters:
    -----------
    local_time_str : str
        The local time as a pd.Timestamp.
    local_tz_str : str
        The timezone of the local time as a string, e.g., 'America/New_York'.

    Returns:
    --------
    str
        The time in UTC as a datetime object in the format 'YYYY-MM-DD HH:MM:SS'.

    Example:
    --------
    >>> convert_to_utc(pd.Timestamp('2023-08-29 14:30:00'), 'America/New_York')
    '2023-08-29 18:30:00'
    """
    local_time = datetime.strptime(
        local_time_str.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"
    )
    local_tz = pytz.timezone(local_tz_str)
    local_time = local_tz.localize(local_time)
    return local_time.astimezone(pytz.utc)