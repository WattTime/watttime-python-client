import argparse
import calendar
import sys
import json
import warnings
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse
from operator import attrgetter
from pathlib import Path
from typing import List, Optional, Union, Literal, Dict, Tuple
from zoneinfo import ZoneInfo
import inspect
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.colors as pc
import scipy.stats as stats
from shapely.geometry import shape
from timezonefinder import TimezoneFinder

from jinja2 import Template
from watttime.evaluation.get_wt_api_forecast_evaluation_data import DataHandlerFactory
from watttime.evaluation.fuels_cp import fuel_cp
from watttime import api


@lru_cache
def get_tz_from_centroid(region):
    wt_maps = api.WattTimeMaps()
    all_maps = wt_maps.get_maps_json()
    region = {f["properties"]["region"]: f["geometry"] for f in all_maps["features"]}[
        region
    ]
    centroid = shape(region).centroid
    tz = TimezoneFinder().certain_timezone_at(lat=centroid.y, lng=centroid.x)
    return tz


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


def get_random_overlapping_period(
    dfs, max_period="365D", resample_freq="1h", first_week_of_month_only=False
):
    """
    Find a random overlapping time period between multiple DataFrames' datetime indices,
    maximizing the period up to `max_period`.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames with datetime indices.
        max_period (str): Maximum time period as a string (e.g., '7D' for 7 days).
        resample_freq (str): Frequency for resampling the datetime range.
        first_week_of_month_only (bool): If True, filters the results to only include
                                         the first full week (Monday-Sunday) of every month.

    Returns:
        pd.DatetimeIndex: List of overlapping datetimes within the defined time period.
    """
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
        overlap_range = pd.date_range(
            start=start_overlap, end=end_overlap, freq=resample_freq
        )
    else:
        overlap_range = pd.date_range(
            start=start_overlap, end=start_overlap + max_timedelta, freq=resample_freq
        )

    # Filter to only include the first full week (Monday-Sunday) of each month if specified
    if first_week_of_month_only:
        overlap_df = pd.DataFrame({"date": overlap_range})
        overlap_df["year_month"] = overlap_df["date"].dt.to_period("M")
        overlap_df["weekday"] = overlap_df["date"].dt.weekday

        first_monday = overlap_df.groupby("year_month")["date"].transform(
            lambda x: x[x.dt.weekday == 0].min()
        )
        first_sunday = first_monday + pd.Timedelta(days=6)

        overlap_range = overlap_df.loc[
            (overlap_df["date"] >= first_monday) & (overlap_df["date"] <= first_sunday),
            "date",
        ]

    return overlap_range


def plot_sample_moers(
    factory: DataHandlerFactory, max_sample_period="365D", first_week_of_month_only=True
) -> Dict[str, go.Figure]:
    """
    Plot a sample of old and new MOER values over time, creating a subplot for each unique region.

    Args:
        jobs (list): List of AnalysisDataHandler objects with a .moers atribute.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with timeseries of values.
    """

    figs = {}
    times = get_random_overlapping_period(
        [j.moers for j in factory.data_handlers],
        max_sample_period,
        first_week_of_month_only=first_week_of_month_only,
    )
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():

        fig = go.Figure()

        tz = get_tz_from_centroid(region_abbrev)

        for model_job in region_models:

            _df = model_job.moers.reindex(times)
            _df = _df.tz_convert(tz)

            fig.add_trace(
                go.Scatter(
                    x=_df.index,
                    y=_df.signal_value,
                    mode="lines",
                    name=model_job.model_date,
                    line=dict(width=2, shape="hv"),
                    showlegend=True,
                    connectgaps=False,
                )
            )

            # Update layout for the figure
            fig.update_layout(
                height=300,
                yaxis=dict(
                    title=f"{model_job.signal_type}",
                    fixedrange=True,  # Disable y-axis panning
                ),
                showlegend=True,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(
                    type="date",
                    tickformat="%Y-%m-%d %H:%M",
                    tickangle=45,
                    showgrid=True,
                    range=[
                        times.min(),
                        times.min() + pd.Timedelta("7D"),
                    ],  # Default 1-week view
                ),
            )
        figs[region_abbrev] = fig

    return figs


def plot_distribution_moers(
    factory: DataHandlerFactory, subsample_size=500
) -> Dict[str, go.Figure]:
    """
    Plot the distribution of MOER values for each region using horizontal Plotly violin plots.

    Args:
        factory: DataHandlerFactory that exposes .data_handlers_by_region_dict

    Returns:
        dict mapping region abbreviation to a Plotly Figure
    """

    figs = {}

    # Use Plotly's qualitative color palette
    palette = pc.qualitative.Plotly

    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        fig = go.Figure()

        for ix, model_job in enumerate(region_models):
            series = model_job.moers["signal_value"].dropna()
            color = palette[ix % len(palette)]

            # Calculate exact min/max for span
            min_val = series.min()
            max_val = series.max()

            fig.add_trace(
                go.Violin(
                    x=series.values,  # horizontal violin
                    name=str(model_job.model_date),
                    box=dict(visible=True),
                    meanline=dict(visible=False),
                    points="outliers",
                    scalemode="width",
                    spanmode="manual",  # Manually set the span
                    span=[min_val, max_val],  # Don't extend beyond data range
                    opacity=0.5,  # Fill opacity
                    line=dict(
                        color="black",
                        width=1.5,  # Slightly thicker line for better visibility
                    ),
                    fillcolor=color,
                )
            )

        if not fig.data:
            figs[region_abbrev] = go.Figure()
            continue

        fig.update_layout(
            height=350,
            xaxis_title=f"{region_models[0].signal_type} Values",  # x axis is now the value
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50),
            violingap=0.2,
            violingroupgap=0.1,
            violinmode="group",
        )

        figs[region_abbrev] = fig

    return figs


def plot_heatmaps(
    factory: DataHandlerFactory, colorscale="Oranges"
) -> Dict[str, go.Figure]:
    """
    Generate vertically stacked heatmaps for each region, with each heatmap having its own colorscale.
    Args:
        jobs (list): List of AnalysisDataHandler objects with a .moers attribute.
    Returns:
        Dict[str, go.Figure]: A dictionary mapping region abbreviations to their heatmap figures.
    """
    figs = {}
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():

        # Initialize a subplot with one row per job
        fig = sp.make_subplots(
            rows=len(region_models),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[j.model_date for j in region_models],
            vertical_spacing=0.1,  # Reduced spacing for better alignment
        )
        tz = get_tz_from_centroid(region_abbrev)

        for j, job in enumerate(region_models, start=1):
            moers = job.moers.tz_convert(tz)
            heat = moers.assign(month=moers.index.month, hour=moers.index.hour)
            heat = heat.dropna(subset=["signal_value"])
            heat = (
                heat.groupby(["month", "hour"])["signal_value"]
                .mean()
                .unstack(fill_value=np.nan)
            )
            heat.index = [calendar.month_abbr[m] for m in heat.index]

            heatmap_trace = go.Heatmap(
                z=heat.values,
                x=heat.columns,
                y=heat.index,
                colorscale=colorscale,
                colorbar_title=None,
                showscale=True,
                colorbar=dict(
                    yanchor="middle",
                    y=0.5,
                    len=0.9,
                    thickness=20,
                    title=dict(side="right"),
                ),
            )

            fig.add_trace(heatmap_trace, row=j, col=1)

            fig.update_yaxes(title_text="Month", row=j, col=1)
            if j == len(region_models):  # Only add x-axis title to bottom subplot
                fig.update_xaxes(title_text="Hour of Day", row=j, col=1)

        fig.update_layout(
            height=300 * len(region_models),  # Adjust height per subplot
            margin=dict(r=100),  # Add right margin for colorbars
            # title=f"{region_abbrev} Heat Maps"
        )

        # Update colorbar positions to align with subplots
        for model_ix, trace in enumerate(fig.data):
            trace.colorbar.y = 1 - (2 * model_ix + 1) / (2 * len(region_models))
            trace.colorbar.len = 1 / len(region_models) * 0.9

        figs[region_abbrev] = fig

    return figs


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
    """Compute mean daily Spearman rank correlation up to a given forecast horizon."""

    filtered_df = df[df["horizon_mins"] <= horizon_mins].dropna(
        subset=[pred_col, truth_col]
    )
    if filtered_df.empty:
        return np.nan

    def safe_spearmanr(group_df):
        if group_df[truth_col].nunique() < 2 or group_df[pred_col].nunique() < 2:
            return np.nan
        return stats.spearmanr(group_df[truth_col], group_df[pred_col]).statistic

    corr = filtered_df.groupby("generated_at").apply(safe_spearmanr)

    return float(np.nanmean(corr).round(3))


def pick_k_optimal_charge(in_df: pd.DataFrame, truth_col, charge_mins: int):
    df = in_df.copy()

    specify_generated_at = "generated_at" in df.index.names

    charge_needed = charge_mins // 5
    df["charge_status"] = False
    df = df.sort_index(ascending=True)

    for window_start, w_df in df.reset_index().groupby("window_start"):
        # Pick the `charge_needed` lowest truth_col values in this window
        # If there are ties, 'first' ensures deterministic selection
        if specify_generated_at:
            w_df = w_df[w_df["generated_at"] == w_df["point_time"]]
        selected_points = w_df.sort_values(truth_col, ascending=True).head(
            charge_needed
        )

        # Mark these point_times as charge periods in the original dataframe
        if specify_generated_at:
            df.loc[
                [(p, p) for p in selected_points["point_time"]], "charge_status"
            ] = True
        else:
            df.loc[selected_points["point_time"], "charge_status"] = True

    return df["charge_status"]


def simulate_charge(in_df: pd.DataFrame, sort_col: str, charge_mins: int):

    df = in_df.copy()

    assert sort_col in df.columns

    charge_needed = charge_mins // 5
    df = df.assign(charge_status=False)
    df = df.sort_index(ascending=True)
    for w in df.reset_index().groupby("window_start"):
        window_start, w_df = w
        w_df = w_df.sort_values(["generated_at", "point_time"], ascending=True)
        charge_periods = []
        _charge_needed = charge_needed
        generated_at_groups = list(w_df.groupby("generated_at"))
        total_groups = len(generated_at_groups)

        for processed_count, (generated_at, g_df) in enumerate(generated_at_groups):
            n_below_now = g_df[sort_col].rank().iloc[0]
            remaining_generated_at = total_groups - processed_count
            should_charge_now = (n_below_now <= _charge_needed) or (
                remaining_generated_at <= _charge_needed
            )

            if should_charge_now:

                if generated_at != g_df.iloc[0]["point_time"]:
                    warnings.warn(
                        "The generated_at and first point_time for a forecast do not match, look closely to understand why!",
                        RuntimeWarning,
                    )
                    continue

                charge_periods.append((generated_at, generated_at))
                _charge_needed -= 1

            if _charge_needed == 0:
                break

        df.loc[charge_periods, "charge_status"] = True

    assert (
        df["charge_status"].sum() <= (len(df["window_start"].unique())) * charge_needed
    ), "Charge status is too high, check the logic for simulate_charge"
    assert df["charge_status"].sum() >= (
        len(df["window_start"].unique()) * 0.97 * charge_needed
    ), "Charge status is too low, check the logic for simulate_charge"

    return df["charge_status"]


def assign_windows(
    generated_ats: List[pd.Timestamp], window_size: pd.Timedelta
) -> List[pd.Timestamp]:

    if isinstance(window_size, int):
        window_size = pd.Timedelta(minutes=window_size)

    n = len(generated_ats)
    if n == 0:
        return []

    # Create array of (timestamp, original_index) and sort by timestamp
    ts_with_idx = np.array(
        [(ts.value, i) for i, ts in enumerate(generated_ats)],
        dtype=[("ts", "i8"), ("idx", "i4")],
    )
    ts_with_idx.sort(order="ts")

    # Convert window_size to nanoseconds for direct comparison
    window_ns = window_size.value

    # Assign signposts using vectorized operations where possible
    signposts = np.empty(n, dtype="i8")
    current_start_ns = ts_with_idx[0]["ts"]
    signposts[0] = current_start_ns

    for i in range(1, n):
        ts_ns = ts_with_idx[i]["ts"]
        if ts_ns < current_start_ns + window_ns:
            signposts[i] = current_start_ns
        else:
            current_start_ns = ts_ns
            signposts[i] = current_start_ns

    # Map back to original order using fancy indexing
    result = np.empty(n, dtype="i8")
    result[ts_with_idx["idx"]] = signposts

    # Convert back to Timestamps
    return [pd.Timestamp(ns, tz="UTC") for ns in result]


def calc_rank_compare_metrics(
    in_df,
    charge_mins,
    window_mins,
    tz=None,
    window_start_time=None,
    pred_col="predicted_value",
    truth_col="signal_value",
    load_kw=1000,
):
    df = in_df.copy()

    if isinstance(window_mins, int):
        window_mins = pd.Timedelta(minutes=window_mins)

    window_starts = assign_windows(
        df.index.get_level_values("generated_at"), window_mins
    )

    if window_start_time:
        utc_offset = pd.to_timedelta(window_start_time + ":00")
        local_dt = (pd.Timestamp("1999-01-01T00:00Z") + utc_offset).tz_convert(tz)
        local_offset = (
            pd.to_timedelta(local_dt.hour, unit="h")
            + pd.to_timedelta(local_dt.minute, unit="m")
            + pd.to_timedelta(local_dt.second, unit="s")
        )
        adj_window_starts = [i.normalize() + local_offset for i in window_starts]
    else:
        adj_window_starts = window_starts

    df = df.assign(
        window_start=adj_window_starts,
        window_end=[i + window_mins for i in adj_window_starts],
    )

    # Filter out rows that do not fall within the valid window
    df.loc[
        (df.index.get_level_values("point_time") >= df["window_end"])
        | (df.index.get_level_values("generated_at") >= df["window_end"])
        | (df.index.get_level_values("generated_at") < df["window_start"]),
        "window_start",
    ] = pd.NaT
    df = df.dropna(subset=["window_start", "window_end"])

    # Filter out rows where there aren't enough generated_ats to fulfill charge_mins
    n_generated_at_in_window = df.groupby("window_start").transform(
        lambda x: x.reset_index()["generated_at"].nunique()
    )[pred_col]
    df = df.loc[n_generated_at_in_window >= (charge_mins // 5) - 1]

    # Filter out rows where generated_at != first point_time in the window
    df = df.loc[
        df.index.get_level_values("generated_at")
        == df.reset_index().groupby("generated_at")["point_time"].transform("first")
    ]

    df = df.assign(
        truth_charge_status=pick_k_optimal_charge(df, truth_col, charge_mins),
        pred_charge_status=simulate_charge(df, pred_col, charge_mins),
    )

    df = df.assign(
        truth_charge_emissions=df[truth_col]
        * df["truth_charge_status"]
        * (load_kw / 1000)
        * (5 / 60),  # normalize hourly MOER to 5-minute intervals,
        pred_charge_emissions=df[truth_col]
        * df["pred_charge_status"]
        * (load_kw / 1000)
        * (5 / 60),
    )

    # baseline: immediate charging rather than AER
    df = df.sort_index()
    df = df.assign(sequential_rank=df.groupby("window_start").cumcount())
    df = df.assign(baseline_charge_status=df["sequential_rank"] < charge_mins // 5)
    df = df.assign(
        baseline_charge_emissions=df[truth_col]
        * df["baseline_charge_status"]
        * (load_kw / 1000)
        * (5 / 60)
    )

    assert (
        df["baseline_charge_status"].sum()
        == (len(df["window_start"].unique())) * charge_mins // 5
    )
    assert (
        df["baseline_charge_status"].sum()
        <= (len(df["window_start"].unique()) + 1) * charge_mins // 5
    )

    # Calculate total CO2 emissions ("truth")
    y_best_emissions = df["truth_charge_emissions"].sum()
    y_pred_emissions = df["pred_charge_emissions"].sum()
    y_base_emissions = df["baseline_charge_emissions"].sum()

    assert y_pred_emissions >= y_best_emissions
    assert y_base_emissions >= y_best_emissions

    dates = df.index.get_level_values("generated_at")
    n_days = len(set(dates.dropna().date))

    if n_days == 0:
        reduction = np.nan
        potential = np.nan
    else:
        reduction = (y_base_emissions - y_pred_emissions) / n_days
        potential = (y_base_emissions - y_best_emissions) / n_days

    return {
        "reduction": round(reduction, 1),
        "potential": round(potential, 1),
    }


def plot_norm_mae(
    factory: DataHandlerFactory, horizons_hr=[1, 6, 12, 18, 24, 72]
) -> Dict[str, go.Figure]:
    """
    Create a Plotly bar chart for rank correlation by horizon with one subplot per region (abbrev).
    """

    y_min = y_max = 0
    figs = {}

    # Iterate through each region and create a bar plot
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        fig = go.Figure()
        x_values = [f"{h}hr" for h in horizons_hr]  # Add horizon labels

        for model_job in region_models:
            y_values = [
                calc_norm_mae(model_job.forecasts_v_moers, (h * 60) - 5)
                for h in horizons_hr
            ]
            y_max = max(y_values + [y_max])
            y_min = min(y_values + [y_min])

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=model_job.model_date,
                    text=[f"{y:.1f}%" for y in y_values],  # Add text labels on bars
                    textposition="outside",
                )
            )

            fig.update_layout(
                height=300,
                xaxis_title="Horizon (Hours)",
                yaxis_title="Normalized MAE (%)",
                showlegend=True,  # Legends appear in individual subplot titles
                margin=dict(l=50, r=50, t=50, b=50),
                barmode="group",  # Ensure bars for each region are grouped
            )

            figs[region_abbrev] = fig

        for region, fig in figs.items():
            figs[region] = fig.update_yaxes(
                range=[y_min - (0.25 * y_max), y_max + (0.25 * y_max)]
            )

    return figs


def plot_rank_corr(
    factory: DataHandlerFactory, horizons_hr=[1, 6, 12, 18, 24, 72]
) -> Dict[str, go.Figure]:
    """
    Create a Plotly line plot for rank correlation by horizon with one subplot per region (abbrev).

    Parameters:
        df (pd.DataFrame): DataFrame containing the rank correlation data.
                           Columns: ['abbrev', 'name', '24hr', '48hr', '72hr'].
        metric_name (str): 'Rank Correlation' or 'Normalized MAE'

    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure with subplots.

    """

    y_min = y_max = 0
    figs = {}

    # Iterate through each region and create a line plot
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        fig = go.Figure()

        # Extract data for the line plot
        x_values = [f"{h}hr" for h in horizons_hr]  # Add horizon labels

        for model_job in region_models:
            y_values = [
                calc_rank_corr(model_job.forecasts_v_moers, (h * 60) - 5)
                for h in horizons_hr
            ]
            y_max = max(y_values + [y_max])
            y_min = min(y_values + [y_min])

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=model_job.model_date,
                    text=[f"{y:.3f}" for y in y_values],  # Add text labels on bars
                    textposition="outside",
                )
            )

        # Update layout
        fig.update_layout(
            height=300,
            xaxis_title="Horizon (Hours)",
            yaxis_title="Rank Correlation",
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50),
        )

        figs[region_abbrev] = fig

    for region, fig in figs.items():
        # Set uniform y-axis range for all subplots
        figs[region] = fig.update_yaxes(
            range=[y_min - (0.25 * y_max), y_max + (0.25 * y_max)]
        )

    return figs


AER_SCENARIOS = {
    "EV-night": {
        "charge_mins": 3 * 60,
        "window_mins": 12 * 60,
        "window_start_time": "19:00",
        "load_kw": 19,
    },
    "EV-day": {
        "charge_mins": 2 * 60,
        "window_mins": 8 * 60,
        "window_start_time": "09:00",
        "load_kw": 19,
    },
    "Thermostat": {
        "charge_mins": 30,
        "window_mins": 60,
        "load_kw": 3,  # typical AC
    },
    "10 kW / 24hr / 25% Duty Cycle": {
        "charge_mins": int(24 * 60 * 0.25),
        "window_mins": 24 * 60,
        "load_kw": 10,
    },
    "10 kW / 72hr / 25% Duty Cycle": {
        "charge_mins": int(24 * 3 * 60 * 0.25),
        "window_mins": 24 * 3 * 60,
        "load_kw": 10,
    },
    "10 kW / 24hr / 50% Duty Cycle": {
        "charge_mins": int(24 * 60 * 0.5),
        "window_mins": 24 * 60,
        "load_kw": 10,
    },
    "10 kW / 72hr / 50% Duty Cycle": {
        "charge_mins": (int(24 * 3 * 60 * 0.5)),
        "window_mins": 24 * 3 * 60,
        "load_kw": 10,
    },
}


def plot_impact_forecast_metrics(
    factory: DataHandlerFactory,
    scenarios=[
        "10 kW / 24hr / 25% Duty Cycle",
        "10 kW / 72hr / 25% Duty Cycle",
        "10 kW / 24hr / 50% Duty Cycle",
        "10 kW / 72hr / 50% Duty Cycle",
        "EV-night",
        "EV-day",
        "Thermostat",
    ],
):

    figs = {}
    y_max = 0

    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():

        fig = sp.make_subplots(
            rows=len(region_models),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[j.model_date for j in region_models],
            vertical_spacing=0.2,
        )

        tz = get_tz_from_centroid(region_abbrev)

        for model_ix, model_job in enumerate(region_models, start=1):

            _metrics = [
                {
                    **calc_rank_compare_metrics(
                        model_job.forecasts_v_moers, **AER_SCENARIOS[s], tz=tz
                    ),
                    "scenario": s,
                }
                for s in scenarios
            ]

            _metrics = pd.DataFrame(_metrics)

            fig.add_trace(
                go.Bar(
                    x=_metrics["scenario"],
                    y=_metrics["potential"],
                    name=f"CO2 Potential Savings",  # Legend only in the first subplot
                    marker=dict(
                        color="rgba(200, 200, 200, 0.8)"
                    ),  # Light gray for potential
                    hovertemplate="%{x}: %{y:.1f} lbs CO2<extra></extra>",
                    legendgroup="Potential Savings",  # Group legend for potential savings
                    showlegend=(
                        model_ix == 1
                    ),  # Show legend only for the first subplot
                ),
                row=model_ix,
                col=1,
            )

            fig.add_trace(
                go.Bar(
                    x=_metrics["scenario"],
                    y=_metrics["reduction"],
                    name="Forecast Achieved",
                    text=[
                        f"{(r / (p + 1e-6)) * 100:.1f}%"
                        for r, p in zip(_metrics["reduction"], _metrics["potential"])
                    ],
                    textposition="outside",
                    marker=dict(color="rgba(0, 128, 0, 0.8)"),  # Green for reduction
                    hovertemplate="%{x}: %{y:.1f} lbs CO2<extra></extra>",
                    legendgroup="Potential Savings",  # Group legend for potential savings
                    showlegend=(
                        model_ix == 1
                    ),  # Show legend only for the first subplot
                ),
                row=model_ix,
                col=1,
            )

            y_max = max(_metrics["reduction"].max(), _metrics["potential"].max(), y_max)

        # Update layout
        fig.update_layout(
            height=300 * len(region_models),
            xaxis_title="Scenario",
            yaxis_title="CO2 Savings (lbs per day)",
            barmode="group",  # Grouped bars (side by side)
            showlegend=True,  # Show legend
            margin=dict(l=50, r=50, t=50, b=50),
        )

        # Update axes for all subplots
        for model_ix in range(1, len(region_models) + 1):
            fig.update_xaxes(title_text="Scenario", row=model_ix, col=1)
            fig.update_yaxes(
                title_text="CO2 Savings (lbs per day)", row=model_ix, col=1
            )

        figs[region_abbrev] = fig

    # Set uniform y-axis range for all subplots
    for region, fig in figs.items():
        figs[region] = fig.update_yaxes(range=[0, y_max + (0.25 * y_max)])

    return figs


def plot_sample_fuelmix(
    factory: DataHandlerFactory, max_sample_period="365D"
) -> Dict[str, go.Figure]:

    figs = {}
    times = get_random_overlapping_period(
        [j.fuel_mix for j in factory.data_handlers],
        max_sample_period,
        first_week_of_month_only=False,
    )
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():

        # Initialize a subplot with one row per region
        fig = sp.make_subplots(
            rows=len(region_models),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[j.model_date for j in region_models],
            vertical_spacing=0.2,
        )

        tz = get_tz_from_centroid(region_abbrev)

        # Track which fuel types have been added to legend
        fuels_in_legend = set()

        for model_ix, model_job in enumerate(region_models, start=1):

            stacked_values = model_job.fuel_mix.reindex(times)

            stacked_values = stacked_values.tz_convert(tz)

            # Create cumulative values for stacking
            for fuel_ix in range(1, len(stacked_values.columns)):
                stacked_values.iloc[:, fuel_ix] += stacked_values.iloc[:, fuel_ix - 1]

            # Add each fuel type as an area
            for fuel_ix, fuel in enumerate(model_job.fuel_mix.columns):
                # Only show in legend if not already added
                show_in_legend = fuel not in fuels_in_legend
                if show_in_legend:
                    fuels_in_legend.add(fuel)

                fig.add_trace(
                    go.Scatter(
                        x=stacked_values.index,
                        y=stacked_values.iloc[:, fuel_ix],
                        fill="tonexty" if fuel_ix > 0 else "tozeroy",
                        mode="none",
                        name=fuel,
                        fillcolor=fuel_cp[fuel],
                        connectgaps=False,
                        legendgroup=fuel,
                        showlegend=show_in_legend,
                        line=dict(shape="hv"),
                    ),
                    row=model_ix,
                    col=1,
                )

        # Update layout for the figure
        fig.update_layout(
            height=300 * len(region_models),
            yaxis=dict(
                title=f"{region_models[0].signal_type}",
                fixedrange=True,  # Disable y-axis panning
            ),
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
                type="date",
                tickformat="%Y-%m-%d %H:%M",
                tickangle=45,
                showgrid=True,
                range=[times.min(), times.min() + pd.Timedelta("7D")],
            ),
        )
        figs[region_abbrev] = fig

    return figs


def calc_max_potential(
    in_df,
    charge_mins,
    window_mins,
    window_start_time=None,
    tz=None,
    truth_col="signal_value",
    load_kw=1000,
):
    """Predict the maximum potential CO2 Savings, without any forecast to estimate the upper limit of impact from a given signal value.
    This version is resilient to partial windows (e.g. where the data does not fully cover the expected number of 5-min intervals).
    """
    df = in_df.copy()

    needed_rows = charge_mins // 5
    load_factor = load_kw / 1000

    if isinstance(window_mins, int):
        window_mins = pd.Timedelta(minutes=window_mins)

    window_starts = assign_windows(df.index.get_level_values("point_time"), window_mins)

    if window_start_time:
        utc_offset = pd.to_timedelta(window_start_time + ":00")
        local_dt = (pd.Timestamp("1999-01-01T00:00Z") + utc_offset).tz_convert(tz)
        local_offset = (
            pd.to_timedelta(local_dt.hour, unit="h")
            + pd.to_timedelta(local_dt.minute, unit="m")
            + pd.to_timedelta(local_dt.second, unit="s")
        )
        adj_window_starts = [i.normalize() + local_offset for i in window_starts]
    else:
        adj_window_starts = window_starts

    df = df.assign(
        window_start=adj_window_starts,
        window_end=[i + window_mins for i in adj_window_starts],
    )

    # Filter out rows that do not fall within the valid window
    df.loc[
        (df.index.get_level_values("point_time") >= df["window_end"]),
        "window_start",
    ] = pd.NaT
    df = df.dropna(subset=["window_start", "window_end"])

    # Filter out rows where there aren't enough generated_ats to fulfill charge_mins
    n_generated_at_in_window = df.groupby("window_start").transform(
        lambda x: x.reset_index()["point_time"].nunique()
    )[truth_col]
    df = df.loc[n_generated_at_in_window >= (charge_mins // 5) - 1]

    if df.empty:
        return {"potential": 0.0}

    df["charge_status"] = pick_k_optimal_charge(df, truth_col, charge_mins)

    truth_values = df[truth_col]
    charge_emissions = truth_values * df["charge_status"] * load_factor

    df["sequential_rank"] = df.groupby("window_start").cumcount()
    baseline_charge_status = df["sequential_rank"] < needed_rows
    baseline_charge_emissions = truth_values * baseline_charge_status * load_factor

    y_best_total = charge_emissions.sum()
    y_baseline_total = baseline_charge_emissions.sum()

    assert (
        y_best_total <= y_baseline_total
    ), f"Best case ({y_best_total}) should not be worse than baseline ({y_baseline_total})"

    num_unique_dates = df.index.get_level_values("point_time").normalize().nunique()
    potential = (y_baseline_total - y_best_total) / num_unique_dates

    return {"potential": round(potential, 1)}


def plot_max_impact_potential(
    factory: DataHandlerFactory,
    scenarios: List[str] = ["EV-night", "EV-day", "Thermostat"],
):
    figs = {}
    y_max = 0
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        fig = go.Figure()
        tz = get_tz_from_centroid(region_abbrev)

        for model_job in region_models:

            _metrics = [
                {
                    **calc_max_potential(model_job.moers, **AER_SCENARIOS[s], tz=tz),
                    "scenario": s,
                }
                for s in scenarios
            ]

            _metrics = pd.DataFrame(_metrics)

            # Add the 'potential' bar trace
            fig.add_trace(
                go.Bar(
                    x=_metrics["scenario"],
                    y=_metrics["potential"],
                    name=model_job.model_date,  # Legend only in the first subplot
                    hovertemplate="%{x}: %{y:.1f} lbs CO2<extra></extra>",
                )
            )

            y_max = max(_metrics["potential"].max(), y_max)

            fig.update_layout(
                height=300 * len(region_models),
                yaxis=dict(title="lbs CO2", fixedrange=True),  # Disable y-axis panning
            )

        figs[region_abbrev] = fig

    return figs


def plot_fuelmix_heatmap(factory: DataHandlerFactory):
    def create_pivot_table(df, column):
        df = df.assign(month=df.index.month, hour=df.index.hour)

        if column not in df.columns:
            # Create a zero-filled DataFrame with the expected shape
            pivot = pd.DataFrame(
                0, index=range(1, 13), columns=range(24)  # Months 1-12  # Hours 0-23
            )
        else:
            grouped = df.groupby(["month", "hour"])[column].mean().reset_index()
            pivot = grouped.pivot(index="month", columns="hour", values=column)

        return pivot

    def create_buttons(column_names, num_subplots):
        buttons = []
        for i, name in enumerate(column_names):
            # Create visibility array for all traces
            visible = []
            for j in range(num_subplots):
                # For each subplot, make only the trace for current fuel type visible
                for k in range(len(column_names)):
                    visible.append(True if k == i else False)

            button = dict(label=name, method="update", args=[{"visible": visible}])
            buttons.append(button)
        return buttons

    def make_alpha_gradient(hex_color, steps=10):
        rgba_colors = []

        for i in range(steps):
            alpha = 0 if i == 0 else min(1, (i / (steps / 1.25)))
            rgba_colors.append(f"rgba{pc.hex_to_rgb(hex_color) + (alpha,)}")

        return [[i / (steps - 1), rgba] for i, rgba in enumerate(rgba_colors)]

    figs = {}

    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():

        tz = get_tz_from_centroid(region_abbrev)

        # We need to ensure all jobs have the same fuel types for the buttons to work
        all_columns = set()
        for job in region_models:
            all_columns.update(job.fuel_mix.columns)

        # Create subplot titles with model dates
        subplot_titles = [j.model_date for j in region_models]

        # Create subplots
        fig = sp.make_subplots(
            rows=len(region_models),
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
        )

        all_columns_list = sorted(list(all_columns))  # Sort to ensure consistent order

        trace_idx = 0

        # Add traces for each job
        for job_idx, job in enumerate(region_models, start=1):
            for fuel_idx, fuel in enumerate(all_columns_list):
                fuel_mix = job.fuel_mix.tz_convert(tz)
                pivot = create_pivot_table(fuel_mix, fuel)

                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns,
                        y=pivot.index,
                        colorscale=make_alpha_gradient(fuel_cp[fuel], steps=10),
                        name=f"{fuel} ({job.model_date})",
                        visible=(fuel_idx == 0),
                        showscale=False,  # Only show colorscale for the first job
                        zmin=0,
                        zmax=1,
                    ),
                    row=job_idx,
                    col=1,
                )

                trace_idx += 1

        # Add buttons that control visibility across all subplots
        buttons = create_buttons(all_columns_list, len(region_models))

        # Month names mapping
        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }

        # Update layout
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0,
                    xanchor="left",
                    y=-0.15,
                    buttons=buttons,
                )
            ],
            height=(300 * len(region_models))
            + 50,  # Adjust height based on number of jobs
            xaxis_title="Hour of Day",
        )

        # Update all y-axes to show month names
        for model_ix in range(1, len(region_models) + 1):
            fig.update_yaxes(
                title="Month",
                tickvals=list(month_names.keys()),
                ticktext=list(month_names.values()),
                row=model_ix,
                col=1,
            )

            # Update all x-axes
            fig.update_xaxes(
                title=(
                    "Hour of Day" if model_ix == len(region_models) else None
                ),  # Only show title on bottom plot
                tickmode="linear",
                dtick=2,  # Show every 2 hours
                row=model_ix,
                col=1,
            )

        figs[region_abbrev] = fig

    return figs


def plot_bland_altman(
    factory: DataHandlerFactory, n_samples: int = 2500
) -> Dict[str, go.Figure]:
    """
    A Bland-Altman plot is useful to see the difference between two measurements (models).

    It includes a scatterplot of the deltas between the two measurements, along with a reference line.
    signifying the mean of the deltas, and confidence intervals for the mean.
    """
    figs = {}
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():

        if len(region_models) == 1:
            return None

        merged_moers = pd.concat(
            [dh.moers for dh in region_models], axis="columns"
        ).dropna()
        merged_moers.columns = [dh.model_date for dh in region_models]
        merged_moers = merged_moers.sort_index(axis="columns")  # put newest models last

        # TODO: consider more than two models
        if len(merged_moers.columns) < 2:
            merged_moers = merged_moers.iloc[:, :2]

        merged_moers = merged_moers.assign(
            delta=merged_moers.max(axis="columns") - merged_moers.min(axis="columns"),
            mean=merged_moers.mean(axis="columns"),
        )
        merged_moers = merged_moers.sample(min(n_samples, len(merged_moers)))

        fig = go.Figure()

        # scatter
        fig.add_trace(
            go.Scatter(
                x=merged_moers["mean"],
                y=merged_moers["delta"],
                mode="markers",
                marker=dict(opacity=0.5),
                name="Delta of Model Values",
            )
        )

        # reference line
        fig.add_trace(
            go.Scatter(
                x=[merged_moers["mean"].min(), merged_moers["mean"].max()],
                y=[merged_moers["delta"].mean(), merged_moers["delta"].mean()],
                mode="lines",
                line=dict(color="red"),
                name="Mean Delta",
            )
        )

        # ci
        fig.add_trace(
            go.Scatter(
                x=[merged_moers["mean"].min(), merged_moers["mean"].max()],
                y=[
                    merged_moers["delta"].mean() + merged_moers["delta"].std() * 1.96,
                    merged_moers["delta"].mean() + merged_moers["delta"].std() * 1.96,
                ],
                mode="lines",
                line=dict(color="red", dash="dash"),
                opacity=0.5,
                name="95% CI",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[merged_moers["mean"].min(), merged_moers["mean"].max()],
                y=[
                    merged_moers["delta"].mean() - merged_moers["delta"].std() * 1.96,
                    merged_moers["delta"].mean() - merged_moers["delta"].std() * 1.96,
                ],
                mode="lines",
                line=dict(color="red", dash="dash"),
                opacity=0.5,
                showlegend=False,
            )
        )

        fig.update_layout(
            height=500,
            xaxis_title="Mean",
            yaxis_title="Delta",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        figs[region_abbrev] = fig
    return figs


def calc_precision_recall(
    forecasts_v_moers: pd.DataFrame,
    horizon_minutes: int,
    impute_curtail_threshold: int,
    pred_col="predicted_value",
    truth_col="signal_value",
) -> Tuple[float, float]:
    """
    Calculate precision and recall for the all forecast values UP TO the specified horizon,
    based on the given threshold for curtailment.

    Parameters:
        forecasts_v_moers (pd.DataFrame): DataFrame containing forecasts and signal value.
        horizon_minutes (int): The horizon in minutes to consider for the calculation.
        impute_curtail_threshold (int): Threshold to convert continuous MOER to binary curtailment.

    Returns:
        Tuple[float, float]: Precision and recall values.
    """
    # Filter forecasts based on the horizon
    filtered_forecasts = forecasts_v_moers[
        forecasts_v_moers["horizon_mins"] <= horizon_minutes
    ]

    pred_bool = filtered_forecasts[pred_col] <= impute_curtail_threshold
    truth_bool = filtered_forecasts[truth_col] <= impute_curtail_threshold

    # TP / (TP + FP)
    precision = (
        (pred_bool & truth_bool).sum() / pred_bool.sum() if pred_bool.sum() > 0 else 0.0
    )

    # TP / (TP + FN)
    recall = (
        (pred_bool & truth_bool).sum() / truth_bool.sum()
        if truth_bool.sum() > 0
        else 0.0
    )

    return precision, recall


def plot_precision_recall(
    factory: DataHandlerFactory,
    horizons_hr=[1, 6, 12, 18, 24, 72],
    impute_curtail_threshold=100,
) -> Dict[str, go.Figure]:
    """
    Create a Plotly figure with subplots: each model in a separate row
    showing precision and recall across horizons.
    """

    figs = {}

    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        # One row per model
        fig = sp.make_subplots(
            rows=len(region_models),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"Model {m.model_date}" for m in region_models],
            vertical_spacing=0.15,
        )

        precision_y_max = recall_y_max = precision_y_min = recall_y_min = 0
        x_values = [f"{h}hr" for h in horizons_hr]

        for row_idx, model_job in enumerate(region_models, start=1):
            values = [
                calc_precision_recall(
                    model_job.forecasts_v_moers,
                    (h * 60) - 5,
                    impute_curtail_threshold=impute_curtail_threshold,
                )
                for h in horizons_hr
            ]

            # Convert from fractions to percentages
            precision_values, recall_values = zip(*values)
            precision_values = [p * 100 for p in precision_values]
            recall_values = [r * 100 for r in recall_values]

            precision_y_max = max(precision_values + [precision_y_max])
            recall_y_max = max(recall_values + [recall_y_max])

            # Precision trace
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=precision_values,
                    name="Precision",
                    marker_color="#636EFA",
                    text=[f"{y:.1f}%" for y in precision_values],
                    textposition="outside",
                    showlegend=(row_idx == 1),  # only show once
                ),
                row=row_idx,
                col=1,
            )

            # Recall trace
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=recall_values,
                    name="Recall",
                    marker_color="#EF553B",
                    text=[f"{y:.1f}%" for y in recall_values],
                    textposition="outside",
                    showlegend=(row_idx == 1),  # only show once
                ),
                row=row_idx,
                col=1,
            )

        # Set uniform y-axis range across subplots
        y_range = [
            min(precision_y_min, recall_y_min)
            - (0.25 * max(precision_y_max, recall_y_max)),
            max(precision_y_max, recall_y_max)
            + (0.25 * max(precision_y_max, recall_y_max)),
        ]

        for row_idx in range(1, len(region_models) + 1):
            fig.update_yaxes(range=y_range, row=row_idx, col=1)

        fig.update_layout(
            height=300 * len(region_models),
            xaxis_title="Horizon (Hours)",
            yaxis_title="Precision / Recall (%)",
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50),
            barmode="group",
        )

        figs[region_abbrev] = fig

    return figs


def plot_forecasts_vs_signal(
    factory: DataHandlerFactory,
    horizons=["15min", "1h", "8h"],
) -> Dict[str, go.Figure]:
    """
    Plot forecasts vs signal for different forecast pull frequencies.
    Assumes that forecast is pulled every increment of `horizons`. (e.g. once every 15 minutes, once every hour, once every 8 hours),
    and these forecasts are stringed together to form a continuous forecast time series.

    Only plots the longest contiguous block of forecasts to avoid gaps in the forecast time series.
    """

    horizons = sorted([pd.Timedelta(h) for h in horizons])

    figs = {}

    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():

        fig = sp.make_subplots(
            rows=len(region_models),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[j.model_date for j in region_models],
            vertical_spacing=0.1,  # Reduced spacing for better alignment
        )
        tz = get_tz_from_centroid(region_abbrev)

        for j, model_job in enumerate(region_models, start=1):

            fh = model_job.forecasts_v_moers
            if fh is None or fh.empty:
                # nothing to plot for this model_job
                continue

            # Before plotting, identify the longest contiguous generated_at block (5-min cadence)
            # to ensure we plot a continuous series even when forecast_sample_size < 1.0
            if j == 1:
                # Extract generated_at level (expected freq 5T) and sort unique values
                gen_at = fh.index.get_level_values("generated_at").unique()
                gen_at = pd.DatetimeIndex(gen_at).sort_values()

                # compute gaps between successive generated_at timestamps in minutes
                deltas = gen_at.to_series().diff().dt.total_seconds().div(60)

                # treat a gap as > 5 minutes (allow a small tolerance)
                gap_mask = deltas > 6  # >6 minutes considered a gap

                # find contiguous blocks by grouping on cumulative gaps
                block_id = gap_mask.cumsum()
                blocks = pd.DataFrame({"ts": gen_at, "block": block_id.values})

                # compute block lengths and pick the longest block
                block_lengths = blocks.groupby("block").size()
                longest_block = block_lengths.idxmax()
                block_times = blocks[blocks["block"] == longest_block]["ts"]

                if block_times.empty:
                    # fallback: use the full range
                    start_ts, end_ts = gen_at.min(), gen_at.max()
                else:
                    start_ts, end_ts = block_times.min(), block_times.max()

            # filter forecasts to only generated_at within the longest contiguous block
            fh_mask = (fh.index.get_level_values("generated_at") >= start_ts) & (
                fh.index.get_level_values("generated_at") <= end_ts
            )
            fh = fh[fh_mask].sort_index()

            # plot each resample_horizon as its own trace using the filtered forecasts

            for ix, resample_horizon in enumerate(horizons):

                # filter df down to only rows where 'generated_at' is the first one in each resample_horizon period
                gen_at_f = fh.index.get_level_values("generated_at")
                first_gen_at = pd.DatetimeIndex(
                    gen_at_f.to_series().resample(resample_horizon).first().dropna()
                )

                # include final forecast in first_gen_at to show horizon
                first_gen_at = first_gen_at.append(gen_at_f[-1:])
                df_filtered = fh[
                    fh.index.get_level_values("generated_at").isin(first_gen_at)
                ]

                # filter df to avoid overlapping forecasts, only keep rows where horizon_mins < resample_horizon in minutes
                df_filtered = df_filtered[
                    df_filtered["horizon_mins"] <= resample_horizon.total_seconds() / 60
                ]
                df_filtered = df_filtered.sort_index()
                df_filtered = (
                    df_filtered.reset_index()
                    .drop_duplicates(subset="point_time", keep="first")
                    .set_index(["generated_at", "point_time"])
                )

                if df_filtered.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=df_filtered.index.get_level_values("point_time").tz_convert(
                            tz
                        ),
                        y=df_filtered["predicted_value"],
                        mode="lines",
                        name=f"Forecast Pull Freq: {resample_horizon}",
                        opacity=0.9,
                        line=dict(
                            color=pc.qualitative.Plotly[
                                ix % len(pc.qualitative.Plotly)
                            ],
                            width=3,
                            dash="dash",
                        ),
                    ),
                    row=j,
                    col=1,
                )

            # get unique point_time values, and plot the signal_value as a solid line
            signal = (
                fh.reset_index()
                .drop_duplicates("point_time")
                .set_index("point_time")
                .sort_index()["signal_value"]
            )
            signal = signal.loc[start_ts : end_ts + max(horizons)].sort_index()
            fig.add_trace(
                go.Scatter(
                    x=signal.index.tz_convert(tz),
                    y=signal.values,
                    mode="lines",
                    name="Signal Value",
                    opacity=0.9,
                    line=dict(color="black", width=1),
                ),
                row=j,
                col=1,
            )

            # Update layout for the figure
            fig.update_layout(
                height=500,
                yaxis=dict(
                    title=f"{model_job.signal_type}",
                    fixedrange=True,  # Disable y-axis panning
                ),
                showlegend=True,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(
                    type="date",
                    tickformat="%Y-%m-%d %H:%M",
                    tickangle=45,
                    showgrid=True,
                    range=[
                        start_ts,
                        end_ts + max(horizons),
                    ],
                ),
            )

        figs[region_abbrev] = fig

    return figs


def parse_report_command_line_args(sys_args):
    parser = argparse.ArgumentParser(
        description="Parse command line arguments to report_moers script"
    )

    parser.add_argument(
        "-region", "--region_list", nargs="+", help="List of region abbrevs."
    )

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
        default=str(Path(__file__).parents[2] / "analysis"),
        help="Top level directory to save model report. Default is '<project_root>/research/moer_reports'.",
    )

    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=["signal", "fuel_mix", "forecast"],
        choices=["signal", "fuel_mix", "forecast"],
        help="Steps to run. Default is ['signal', 'fuel_mix', 'forecast'].",
    )

    parser.add_argument(
        "-fw",
        "--first_week_of_month_only",
        action="store_true",
        help="If set, only sample the first week of each month for fuel mix plots.",
    )

    args = parser.parse_args(sys_args)

    now = datetime.now(timezone.utc)
    if args.end is None:
        args.end = round_time(now - timedelta(days=7))
    if args.start is None:
        args.start = args.end - timedelta(days=7)

    return args


PLOTS = {
    "signal": [
        plot_sample_moers,
        plot_distribution_moers,
        plot_heatmaps,
        plot_bland_altman,
        plot_max_impact_potential,
    ],
    "fuel_mix": [plot_sample_fuelmix, plot_fuelmix_heatmap],
    "forecast": [
        plot_forecasts_vs_signal,
        plot_norm_mae,
        plot_rank_corr,
        plot_precision_recall,
        plot_impact_forecast_metrics,
    ],
}


def generate_report(
    region_list: Union[Tuple[str, List[str]], List[str]],
    model_date_list: List[str],
    signal_type: str,
    eval_start: datetime,
    eval_end: datetime,
    output_dir: Path,
    steps: Literal["signal", "fuel_mix", "forecast"] = [
        "signal",
        "forecast",
    ],  # no fuel_mix by default
    first_week_of_month_only: bool = False,
):

    if isinstance(region_list, str):
        region_title = region_list
        region_list = [region_list]
    elif isinstance(region_list, tuple):
        region_title = region_list[0]
        region_list = region_list[1]
    else:
        region_title = "&".join(region_list)

    if isinstance(model_date_list, str):
        model_date_list = [model_date_list]

    region_list = sorted(region_list)
    model_date_list = sorted(model_date_list)

    filename = f"{signal_type}_{region_title}_{'&'.join(model_date_list)}_model_stats"

    # run notebook
    output_path = output_dir / f"{filename}.html"
    input_template_path = Path(__file__).parent / r"report_card_template.html"

    factory = DataHandlerFactory(
        eval_start=eval_start,
        eval_end=eval_end,
        regions=region_list,
        model_dates=model_date_list,
        signal_types=signal_type,
        forecast_max_horizon=72 * 60,
    )

    kwargs = {
        "first_week_of_month_only": first_week_of_month_only,
    }

    plotly_html = {}
    for step in steps:
        for plot_func in PLOTS[step]:
            _func_params = inspect.signature(plot_func).parameters
            _kwargs = {k: v for k, v in kwargs.items() if k in _func_params}
            _plot = plot_func(factory, **_kwargs)

            if _plot is None:
                continue

            elif isinstance(_plot, dict):
                plotly_html[plot_func.__name__] = {}
                for region_abbrev, _p in _plot.items():
                    plotly_html[plot_func.__name__][region_abbrev] = _p.to_html(
                        full_html=False, include_plotlyjs=False, include_mathjax=False
                    )
            else:
                assert isinstance(_plot, go.Figure)

                plotly_html[plot_func.__name__] = _plot.to_html(
                    full_html=False, include_plotlyjs=False, include_mathjax=False
                )

    plotly_html["report_input_dict"] = json.dumps(
        dict(
            region_list=region_list,
            model_date_list=model_date_list,
            signal_type=signal_type,
            eval_start=str(eval_start),
            eval_end=str(eval_end),
            steps=steps,
        ),
        indent=4,
    )

    plotly_html["collected_model_meta"] = json.dumps(
        factory.collected_model_meta, indent=4
    )

    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(input_template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(
                j2_template.render(
                    plotly_html,
                    show_signal="signal" in steps,
                    show_fuel_mix="fuel_mix" in steps,
                    show_forecast="forecast" in steps,
                )
            )


if __name__ == "__main__":

    cli_args = parse_report_command_line_args(sys.argv[1:])

    generate_report(
        region_list=cli_args.region_list,
        model_date_list=cli_args.models,
        signal_type=cli_args.signal_type,
        eval_start=cli_args.start,
        eval_end=cli_args.end,
        output_dir=Path(cli_args.output_dir),
        steps=cli_args.steps,
        first_week_of_month_only=cli_args.first_week_of_month_only,
    )
