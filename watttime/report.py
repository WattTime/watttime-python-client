import argparse
import calendar
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse
from functools import cached_property
from operator import attrgetter
from pathlib import Path
from typing import List, Optional, Union
from zoneinfo import ZoneInfo
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import scipy.stats as stats
from nbconvert.exporters import HTMLExporter

import papermill as pm
from watttime import api

# hacky way to allow running this script locally
sys.path.append(str(Path(__file__).parents[1].resolve()))


def convert_to_timezone(dt: datetime, tz: Union[str, ZoneInfo]) -> datetime:
    """
    Converts datetime to specified timezone. If datetime is naive, assumes UTC.
    """
    # Handle string timezone
    tz = ZoneInfo(tz) if isinstance(tz, str) else tz

    # Replace the complex tzinfo checking with simple built-in
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).astimezone(tz)


def parse_datetime(date_str: str) -> datetime:
    """Parses datetime string and converts to UTC."""
    return convert_to_timezone(parse(date_str), timezone.utc)


def round_time(dt: datetime, minutes: int = 5) -> datetime:
    """
    Rounds datetime down to nearest interval.
    Uses integer division which is more efficient than timedelta operations.
    """
    return dt.replace(minute=(dt.minute // minutes) * minutes, second=0, microsecond=0)


def parse_eval_days(eval_start, eval_end):
    if isinstance(eval_start, str):
        eval_start = parse(eval_start)
    if isinstance(eval_end, str):
        eval_end = parse(eval_end)

    return [
        i.strftime("%Y-%m-%d")
        for i in pd.date_range(start=eval_start, end=eval_end, freq="1D")
    ]


def parse_forecast_sample_days(eval_days, forecast_sample_days):
    try:
        return random.sample(eval_days, forecast_sample_days)
    except ValueError as e:
        return eval_days


def _fetch_forecast(
    day, ba, new_model_date_str, wt_forecast: Optional[api.WattTimeForecast] = None
):
    """
    Fetch the historical forecast for a specific day and balancing authority (ba).
    """

    if not wt_forecast:
        wt_forecast = api.WattTimeForecast()

    return wt_forecast.get_historical_forecast_pandas(
        start=parse(day),
        end=parse(day) + pd.Timedelta(hours=23, minutes=55),
        region=ba,
        signal_type="co2_moer",
        model=new_model_date_str,
        horizon_hours=72,
    )


def _fetch_forecasts(
    sample_days: List[str],
    ba: str,
    model: str,
    signal_type: str,
    wt_forecast: Optional[api.WattTimeForecast] = None,
):

    _forecasts = []
    out = pd.DataFrame()

    # Use ThreadPoolExecutor for concurrent calls which are I/O bound
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as executor:
        future_to_day = {
            executor.submit(_fetch_forecast, day, ba, model, wt_forecast): day
            for day in sample_days
        }
        for future in as_completed(future_to_day):
            try:
                forecast = future.result()
                _forecasts.append(forecast)
            except Exception as e:
                print(
                    f"Error fetching forecast for {future_to_day[future]} in {ba}: {e}"
                )

    # Combine and process the results
    if _forecasts:  # Ensure there's data before concatenation
        out = pd.concat(_forecasts, ignore_index=True)
        out["point_time"] = pd.to_datetime(out["point_time"])
        out["horizon_mins"] = (
            out["point_time"] - out["generated_at"]
        ).dt.total_seconds() / 60
        out.rename({"value": "predicted_value"}, axis="columns", inplace=True)

    if out.empty:
        raise ValueError(
            f"Empty dataframe returned for {ba} - {model} - {signal_type} forecast API request"
        )
    return out


@dataclass
class ModelAnalysis:
    ba: str
    model_date: str
    signal_type: str
    eval_start: str
    eval_end: str
    eval_days: List[str]
    sample_days: List[str]
    wt_forecast: Optional[api.WattTimeForecast] = api.WattTimeForecast()
    wt_hist: Optional[api.WattTimeHistorical] = api.WattTimeHistorical()

    def __post_init__(self):
        self.ba = self.ba.upper()

    @cached_property
    def moers(self):
        return (
            self.wt_hist.get_historical_pandas(
                start=self.eval_start,
                end=self.eval_end,
                region=self.ba,
                signal_type=self.signal_type,
                model=self.model_date,
            )
            .rename(columns={"value": "signal_value"})
            .set_index("point_time", drop=True)
        )

    @cached_property
    def forecasts(self):
        return _fetch_forecasts(
            sample_days=self.sample_days,
            ba=self.ba,
            model=self.model_date,
            signal_type=self.signal_type,
            wt_forecast=self.wt_forecast,
        )

    def compile_forecast_v_moer(self):
        self.forecasts_v_moers = self.forecasts.merge(
            self.moers,
            how="left",
            left_on="point_time",
            right_on="point_time",
        )


def get_random_overlapping_period(dfs, max_period="7D"):
    """
    Find a random overlapping time period between multiple DataFrames' datetime indices,
    maximizing the period up to `max_period`.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames with datetime indices.
        max_period (str): Maximum time period as a string (e.g., '7D' for 7 days).

    Returns:
        pd.DatetimeIndex: List of overlapping datetimes within the defined time period.
    """
    if not dfs or len(dfs) < 2:
        raise ValueError("You must provide at least two DataFrames.")

    # Find the overlapping range across all DataFrames
    start_overlap = max(df.index.min() for df in dfs)
    end_overlap = min(df.index.max() for df in dfs)

    if start_overlap > end_overlap:
        raise ValueError(
            "No overlapping time period found among the provided DataFrames."
        )

    total_overlap_duration = end_overlap - start_overlap
    max_timedelta = pd.Timedelta(max_period)

    # If the total overlap duration is less than or equal to max_period, return the full range
    if total_overlap_duration <= max_timedelta:
        overlap_range = pd.date_range(start=start_overlap, end=end_overlap, freq="5T")
        # Check if all indices exist in both DataFrames
        if all(df.index.isin(overlap_range).all() for df in dfs):
            return overlap_range
        else:
            raise ValueError("No common overlap found among the provided DataFrames.")

    # Otherwise, select a random period within the overlapping range
    max_attempts = 10
    attempts = 0
    while attempts < max_attempts:
        random_start_time = pd.Timestamp(
            np.random.choice(
                pd.date_range(start_overlap, end_overlap - max_timedelta, freq="5T")
            )
        )
        random_end_time = random_start_time + max_timedelta
        overlap_range = pd.date_range(
            start=random_start_time, end=random_end_time, freq="5T"
        )
        # Check if all indices exist in both DataFrames
        if all(df.index.isin(overlap_range).all() for df in dfs):
            return overlap_range
        attempts += 1
    raise ValueError(f"No common overlap found after {max_attempts} attempts.")


def plot_sample_moers(jobs: List[ModelAnalysis], max_sample_period="7D"):
    """
    Plot a sample of old and new MOER values over time, creating a subplot for each unique BA.

    Args:
        jobs (list): List of ModelAnalysis objects with a .moers atribute.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with timeseries of values.
    """

    # Create a figure with a subplot for each BA, stacked vertically
    unique_bas = set([j.ba for j in jobs])
    fig = sp.make_subplots(
        rows=len(unique_bas),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"{ba.upper()}" for ba in unique_bas],
    )

    for i, ba_abbrev in enumerate(unique_bas, start=1):
        ba_abbrev = ba_abbrev.upper()
        _jobs = [j for j in jobs if j.ba == ba_abbrev]

        times = get_random_overlapping_period(
            [j.moers for j in _jobs], max_sample_period
        )

        for _job in _jobs:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=_job.moers.loc[times, "signal_value"].to_list(),
                    mode="lines",
                    name=_job.model_date,
                    line=dict(width=2),
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

    # Update layout for the figure
    fig.update_layout(
        height=300
        * len(unique_bas),  # Adjust height dynamically based on the number of BAs
        title="Data Sample Comparisons by Region",
        yaxis_title=f"{jobs[0].signal_type}",
        showlegend=True,
        margin=dict(l=150, r=20, t=60, b=80),
        xaxis=dict(
            type="date",
            tickformat="%Y-%m-%d %H:%M",
            tickangle=45,
            showgrid=True,
        ),
    )

    return fig


def plot_distribution_moers(jobs: List[ModelAnalysis]):
    """
    Plot the distribution of old and new MOER values for each BA, creating a stacked subplot.

    Args:
        jobs (list): List of ModelAnalysis objects with a .moers atribute.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with stacked subplots of distributions.
    """
    # Create a figure with a subplot for each BA, stacked vertically
    unique_bas = set([j.ba for j in jobs])
    fig = sp.make_subplots(
        rows=len(unique_bas),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=[f"{ba.upper()}" for ba in unique_bas],
    )

    for i, ba_abbrev in enumerate(unique_bas, start=1):
        ba_abbrev = ba_abbrev.upper()
        _jobs = [j for j in jobs if j.ba == ba_abbrev]
        for _job in _jobs:

            # Add a histogram trace for the new MOER distribution
            fig.add_trace(
                go.Box(
                    x=_job.moers["signal_value"].dropna(),
                    name=_job.model_date,
                    opacity=0.6,
                ),
                row=i,
                col=1,
            )

    # Update layout for the figure
    fig.update_layout(
        height=300
        * len(unique_bas),  # Adjust height dynamically based on the number of BAs
        title=f"{jobs[0].signal_type} Distribution Comparisons by Region",
        xaxis_title=f"{jobs[0].signal_type} Values",
        showlegend=True,
        margin=dict(l=150, r=20, t=60, b=80),
        xaxis=dict(
            type="linear",
            tickangle=45,
            showgrid=True,
        ),
        yaxis_visible=False,
        yaxis_showticklabels=False,
    )

    return fig


def plot_ba_heatmaps(jobs: List[ModelAnalysis], colorscale="Cividis"):
    """
    Generate vertically stacked heatmaps for each BA in the ba_list.

    Args:
        jobs (list): List of ModelAnalysis objects with a .moers atribute.

    Returns:
        plotly.graph_objects.Figure: Figure containing stacked heatmaps.
    """

    jobs = sorted(jobs, key=attrgetter("ba", "model_date"))

    # Initialize a subplot with one row per BA
    fig = sp.make_subplots(
        rows=len(jobs),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{j.ba} - {j.model_date}" for j in jobs],
        vertical_spacing=0.2,
    )

    # share upper and lower bounds between plots
    zmin = 1e6
    zmax = 0
    for job in jobs:
        jmin = job.moers["signal_value"].quantile(0.01)
        jmax = job.moers["signal_value"].quantile(0.99)

        if jmin < zmin:
            zmin = jmin
        if jmax > zmax:
            zmax = jmax

    for i, _job in enumerate(jobs, start=1):
        heat = _job.moers.assign(
            month=_job.moers.index.month, hour=_job.moers.index.hour
        )
        heat = heat.dropna(subset=["signal_value"])
        heat = (
            heat.groupby(["month", "hour"])["signal_value"]
            .mean()
            .unstack(fill_value=np.nan)
        )
        heat.index = [
            calendar.month_abbr[m] for m in heat.index
        ]  # Map month numbers to month abbreviations

        # Create a heatmap trace
        heatmap_trace = go.Heatmap(
            z=heat.values,
            x=heat.columns,  # Hours
            y=heat.index,  # Month abbreviations
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            colorbar_title="MOER",
            showscale=(i == 1),  # Show color scale only on the first heatmap
        )

        # Add the heatmap trace to the subplot
        fig.add_trace(heatmap_trace, row=i, col=1)

    # Update layout
    fig.update_layout(
        height=500 * len(jobs),  # Adjust the height based on the number of BAs
        title=f"Average {jobs[0].signal_type} by Month & Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Month",
        xaxis=dict(title="Hour of Day", tickmode="linear"),
    )

    return fig


def calc_norm_mae(
    df, horizon_mins, pred_col="predicted_value", truth_col="signal_value"
):
    """Returns normalized MAE in lbs/Mwh"""

    filtered_df = df[df["horizon_mins"] == horizon_mins].dropna()
    filtered_df["abs_error"] = (filtered_df[truth_col] - filtered_df[pred_col]).abs()
    # truth_mean = filtered_df[truth_col].mean()

    norm_mae = (filtered_df["abs_error"].mean() / filtered_df[truth_col].mean()) * 100.0

    return round(norm_mae, 1)


def calc_rank_corr(
    df, horizon_mins, pred_col="predicted_value", truth_col="signal_value"
):
    """Returns mean daily Rank Correlation"""

    filtered_df = df = df[df["horizon_mins"] <= horizon_mins].dropna()
    corr = filtered_df.groupby("generated_at").apply(
        lambda group_df: stats.spearmanr(
            group_df[truth_col], group_df[pred_col]
        ).statistic
    )

    return np.nanmean(corr).round(3)


def calc_rank_compare_metrics(
    in_df,
    charge_mins,
    window_mins,
    window_starts=None,
    pred_col="predicted_value",
    truth_col="signal_value",
):
    """
    Calculate rank_compare metrics: co2_reduction, co2_potential, co2_pct.

    Parameters:
        in_df (pd.DataFrame): DataFrame containing the 'truth' and 'predicted' MOER values.
        charge_hours (int): Charge window in minutes.
        window_hours (int): Comparison window in minutes.
        window_starts (list or None): Start times for windows (e.g., ["09:00"]).
        truth_col (str): Column name for the 'truth' MOER values.
        pred_col (str): Column name for the 'predicted' MOER values.

    Returns:
        dict: co2_reduction, co2_potential, co2_pct metrics.
    """

    df = in_df.copy()
    df.set_index("generated_at", inplace=True)
    df.dropna(inplace=True)

    # Rank of truth and predicted columns within each window
    if window_starts:
        # Generate window ranges based on start times and duration
        unique_dates = df.index.normalize().unique()  # Extract unique dates
        window_ranges = []
        for date in unique_dates:
            for start_time in window_starts:
                start = pd.Timestamp(f"{date} {start_time}")
                end = start + pd.Timedelta(minutes=window_mins)
                window_ranges.append((start, end))

        # Assign each row to a window
        def assign_window(row):
            generated_at = row.name
            point_time = row["point_time"]  # point_time column
            for start, end in window_ranges:
                if start <= generated_at < end and point_time < end:
                    return start  # Label window by its start time
            return pd.NaT

        df["window"] = df.apply(assign_window, axis=1)
    else:
        # Default behavior: use rolling windows
        def assign_rolling_window(row):
            ts = row.name  # Index (e.g., generated_at)
            point_time = row["point_time"]  # point_time column
            window_start = ts.floor(f"{window_mins}T")
            window_end = window_start + pd.Timedelta(minutes=window_mins)

            # Only assign window if point_time is within the window range
            if point_time < window_end:
                return window_start
            return pd.NaT

        df["window"] = df.apply(assign_rolling_window, axis=1)

    # Drop rows outside any window
    df = df.dropna(subset=["window"])

    df["y_rank"] = df.groupby(["window", "generated_at"])[truth_col].rank(
        method="first"
    )
    df["y_pred_rank"] = df.groupby(["window", "generated_at"])[pred_col].rank(
        method="first"
    )

    # TODO: remove this filter and replace with while loop simulation for max savings
    max_rank = df["y_rank"].max()
    df = df.groupby(["generated_at", "window"]).filter(
        lambda group: group["y_rank"].max() == max_rank
    )

    # Calculate total C02 emissions in lbs/Mwh of each scenario
    y_actual_total = (
        df[df["y_pred_rank"] <= (charge_mins / 5)]
        .groupby(["window", "generated_at"])[truth_col]
        .mean()
    )
    y_best_total = (
        df[df["y_rank"] <= (charge_mins / 5)]
        .groupby(["window", "generated_at"])[truth_col]
        .mean()
    )
    y_avg_total = df.groupby(["window", "generated_at"])[truth_col].mean()

    # Calculate savings: co2_reduction, co2_potential
    co2_reduction = (y_avg_total - y_actual_total).mean()  # actual savings
    co2_potential = (y_avg_total - y_best_total).mean()  # ideal savings
    # co2_pct = (co2_reduction / co2_potential) * 100. # actual / ideal; % savings realized

    # if co2_potential != 0:
    #     co2_pct = (co2_reduction / co2_potential) * 100.
    # else:
    #     co2_pct = 100. if co2_reduction == 0 else 0  # Handle the edge case when moers are flat

    return {
        "co2_reduction": round(co2_reduction, 1),
        "co2_potential": round(co2_potential, 1),
    }


def plot_norm_mae(jobs: List[ModelAnalysis], horizons_hr=[1, 12, 24, 48, 72]):
    """
    Create a Plotly bar chart for rank correlation by horizon with one subplot per BA (abbrev).
    """

    # Create subplots
    unique_bas = set([j.ba for j in jobs])
    fig = sp.make_subplots(
        rows=len(unique_bas),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=list(unique_bas),
    )

    y_min = y_max = 0

    # Iterate through each BA and create a bar plot
    for i, ba_abbrev in enumerate(unique_bas, start=1):
        ba_abbrev = ba_abbrev.upper()
        _jobs = [j for j in jobs if j.ba == ba_abbrev]

        x_values = [f"{h}hr" for h in horizons_hr]  # Add horizon labels

        for _job in _jobs:
            y_values = [
                calc_norm_mae(_job.forecasts_v_moers, (h * 60) - 5) for h in horizons_hr
            ]
            y_max = max(y_values + [y_max])
            y_min = min(y_values + [y_min])

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=_job.model_date,
                    text=[f"{y:.1f}%" for y in y_values],  # Add text labels on bars
                    textposition="outside",
                ),
                row=i,
                col=1,
            )
    # Set uniform y-axis range for all subplots
    fig.update_yaxes(range=[y_min - (0.25 * y_max), y_max + (0.25 * y_max)])

    fig.update_layout(
        height=300
        * len(unique_bas),  # Adjust figure height based on the number of subplots
        title_text="Normalized MAE by Horizon",
        xaxis_title="Horizon",
        yaxis_title="Normalized MAE (%)",
        showlegend=True,  # Legends appear in individual subplot titles
        margin=dict(l=50, r=50, t=50, b=50),
        barmode="group",  # Ensure bars for each BA are grouped
    )
    fig.update_xaxes(title_text="Horizon (Hours)", row=len(unique_bas), col=1)

    return fig


def plot_rank_corr(jobs: List[ModelAnalysis], horizons_hr=[12, 24, 48, 72]):
    """
    Create a Plotly line plot for rank correlation by horizon with one subplot per BA (abbrev).

    Parameters:
        df (pd.DataFrame): DataFrame containing the rank correlation data.
                           Columns: ['abbrev', 'name', '24hr', '48hr', '72hr'].
        metric_name (str): 'Rank Correlation' or 'Normalized MAE'

    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure with subplots.

    """

    # Create subplots
    unique_bas = set([j.ba for j in jobs])
    fig = sp.make_subplots(
        rows=len(unique_bas),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=list(unique_bas),
    )

    y_min = y_max = 0

    # Iterate through each BA and create a line plot
    for i, ba_abbrev in enumerate(unique_bas, start=1):

        ba_abbrev = ba_abbrev.upper()
        _jobs = [j for j in jobs if j.ba == ba_abbrev]

        # Extract data for the line plot
        x_values = [f"{h}hr" for h in horizons_hr]  # Add horizon labels

        for _job in _jobs:
            y_values = [
                calc_rank_corr(_job.forecasts_v_moers, (h * 60) - 5)
                for h in horizons_hr
            ]
            y_max = max(y_values + [y_max])
            y_min = min(y_values + [y_min])

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=f"{_job.ba} - {_job.model_date}",
                    text=[f"{y:.3f}" for y in y_values],  # Add text labels on bars
                    textposition="outside",
                ),
                row=i,
                col=1,
            )

    # Set uniform y-axis range for all subplots
    fig.update_yaxes(range=[y_min - (0.2 * y_max), y_max + (0.2 * y_max)])

    # Update layout
    fig.update_layout(
        height=300
        * len(unique_bas),  # Adjust figure height based on the number of subplots
        title_text=f"Rank Correlation by Horizon",
        xaxis_title="Horizon",
        yaxis_title="Rank Correlation",
        showlegend=True,  # Legends appear in individual subplot titles
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Update x-axis for all subplots
    fig.update_xaxes(title_text="Horizon (Hours)", row=len(unique_bas), col=1)

    return fig


# TODO: translate to local time for start windows?
AER_SCENARIOS = {
    "EV-night": {
        "charge_mins": 3 * 60,
        "window_mins": 12 * 60,
        "window_starts": ["19:00"],
    },
    "EV-day": {
        "charge_mins": 2 * 60,
        "window_mins": 8 * 60,
        "window_starts": ["09:00"],
    },
    "Thermostat": {
        "charge_mins": 30,
        "window_mins": 60,
        "window_starts": None,
    },
}


def plot_impact_forecast_metrics(
    jobs: List[ModelAnalysis], scenarios=["EV-night", "EV-day", "Thermostat"]
):

    # Create subplots
    fig = sp.make_subplots(
        rows=len(jobs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=[f"{j.ba} - {j.model_date}" for j in jobs],
    )

    # Iterate through each BA and create bar plots
    for i, _job in enumerate(jobs, start=1):

        _metrics = [
            {
                **calc_rank_compare_metrics(_job.forecasts_v_moers, **AER_SCENARIOS[s]),
                "scenario": s,
            }
            for s in scenarios
        ]

        _metrics = pd.DataFrame(_metrics)

        # Add the 'co2_potential' bar trace
        fig.add_trace(
            go.Bar(
                x=_metrics["scenario"],
                y=_metrics["co2_potential"],
                name=f"CO2 Potential Savings",  # Legend only in the first subplot
                marker=dict(
                    color="rgba(200, 200, 200, 0.8)"
                ),  # Light gray for potential
                hovertemplate="%{x}: %{y:.1f} lbs CO2/MWh<extra></extra>",
            ),
            row=i,
            col=1,
        )

        # Add the 'co2_reduction' bar trace
        fig.add_trace(
            go.Bar(
                x=_metrics["scenario"],
                y=_metrics["co2_reduction"],
                name="Forecast Achieved CO2 Savings",
                text=[
                    f"{(r / p) * 100:.1f}%"
                    for r, p in zip(
                        _metrics["co2_reduction"], _metrics["co2_potential"]
                    )
                ],
                textposition="outside",
                marker=dict(color="rgba(0, 128, 0, 0.8)"),  # Green for reduction
                hovertemplate="%{x}: %{y:.1f} lbs CO2/MWh<extra></extra>",
            ),
            row=i,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=400 * len(jobs),  # Adjust figure height based on the number of subplots
        title_text="Simulated CO2 Savings by AER Scenario and Region",
        xaxis_title="Scenario",
        yaxis_title="CO2 Savings (lbs/MWh)",
        barmode="group",  # Grouped bars (side by side)
        showlegend=True,  # Show legend
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Update axes for all subplots
    for i in range(1, len(jobs) + 1):
        fig.update_xaxes(title_text="Scenario", row=i, col=1)
        fig.update_yaxes(title_text="CO2 Savings (lbs/MWh)", row=i, col=1)

    return fig


def parse_report_command_line_args(sys_args):
    parser = argparse.ArgumentParser(
        description="Parse command line arguments to report_moers script"
    )

    parser.add_argument("-ba", "--ba_list", nargs="+", help="List of ba abbrevs.")

    parser.add_argument(
        "-d",
        "--signal_type",
        type=str,
        default="co2_moer",
        choices=["co2_moer", "co2_aoer", "health_damages"],
        help="signal_type / Signal Type. Default is 'co2_moer'.",
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        help="The name of the new model version.",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=parse_datetime,
        help="Evaluation Start timestamp YYYY-MM-DD HH:MM±HH:MM",
    )

    parser.add_argument(
        "-e",
        "--end",
        type=parse_datetime,
        help="Evaluation End timestamp YYYY-MM-DD HH:MM±HH:MM",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent / "analysis"),
        help="Top level directory to save model report. Default is '<project_root>/research/moer_reports'.",
    )

    parser.add_argument(
        "-u",
        "--username",
        type=str,
        default=None,
        help="WattTime Username can be provided as a CLI arg or set as an environment variable (WATTTIME_USER)",
    )

    parser.add_argument(
        "-p",
        "--password",
        type=str,
        default=None,
        help="WattTime Password can be provided as a CLI arg or set as an environment variable (WATTTIME_PASSWORD)",
    )

    parser.add_argument(
        "-f",
        "--forecast_sample_days",
        type=int,
        default=12,
        help="Number of days to sample from forecast. Default is 12.",
    )

    args = parser.parse_args(sys_args)

    now = datetime.now(timezone.utc)
    if args.end is None:
        args.end = round_time(now - timedelta(days=7))
    if args.start is None:
        args.start = args.end - timedelta(days=7)

    return args


def run_report_notebook(
    ba_list: List[str],
    model_date_list: List[str],
    signal_type: str,
    eval_start: datetime,
    eval_end: datetime,
    output_dir: Path,
    api_username: Optional[str] = None,
    api_password: Optional[str] = None,
    forecast_sample_days: Optional[int] = None,
):
    filename = f"{signal_type}_{model}_model_stats"

    # run notebook
    output_notebook = output_dir / f"{filename}.ipynb"
    output_html = output_dir / f"{filename}.html"
    pm.execute_notebook(
        Path(__file__).parents[0] / "new_moer_model_report.ipynb",
        str(output_notebook),
        parameters=dict(
            ba_list=ba_list,
            model_date_list=model_date_list,
            signal_type=signal_type,
            eval_start=eval_start.isoformat(),
            eval_end=eval_end.isoformat(),
            api_username=api_username,
            api_password=api_password,
            forecast_sample_days=forecast_sample_days,
        ),
    )

    # HTML exporter with the custom template
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True  # hide code cells

    # Convert the notebook to HTML + write to file
    (body, resources) = html_exporter.from_filename(str(output_notebook))
    with open(str(output_html), "w", encoding="utf-8") as f:
        f.write(body)


if __name__ == "__main__":

    cli_args = parse_report_command_line_args(sys.argv[1:])

    run_report_notebook(
        ba_list=cli_args.ba_list,
        model_date_list=cli_args.model_date_list,
        signal_type=cli_args.signal_type,
        eval_start=cli_args.start,
        eval_end=cli_args.end,
        output_dir=Path(cli_args.output_dir),
        api_username=cli_args.username,
        api_password=cli_args.password,
        forecast_sample_days=cli_args.forecast_sample_days,
    )
