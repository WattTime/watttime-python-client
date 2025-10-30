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


def plot_distribution_moers(factory: DataHandlerFactory) -> Dict[str, go.Figure]:
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
    data_handler, 
    horizon_mins, 
    pred_col="predicted_value", 
    truth_col="signal_value",
    ci=False,
):
    """
    Calculate normalized MAE in lbs/Mwh.
    
    Parameters:
        data_handler: AnalysisDataHandler object with forecasts_v_moers DataFrame
        horizon_mins: Forecast horizon in minutes
        pred_col: Column name for predictions
        truth_col: Column name for ground truth
        ci: If True, returns dict with confidence intervals; if False, returns scalar (default False)
    
    Returns:
        float or dict: If ci=False, returns normalized MAE value.
                      If ci=True, returns {"norm_mae": value, "norm_mae_ci_lower": value, "norm_mae_ci_upper": value}
    """

    df = data_handler.forecasts_v_moers
    filtered_df = df[df["horizon_mins"] == horizon_mins].dropna()
    filtered_df["abs_error"] = (filtered_df[truth_col] - filtered_df[pred_col]).abs()
    norm_mae = (filtered_df["abs_error"].mean() / filtered_df[truth_col].mean()) * 100.0
    norm_mae = round(norm_mae, 1)
    
    if not ci:
        return norm_mae
    
    # Calculate per-forecast normalized MAE for confidence intervals
    df_with_date = filtered_df.reset_index()
    df_with_date["date"] = df_with_date["generated_at"].dt.normalize()
    
    # Group by date and calculate daily normalized MAE
    daily_mae = df_with_date.groupby("date").apply(
        lambda g: (g[truth_col] - g[pred_col]).abs().sum() / g[truth_col].sum() * 100.0 if g[truth_col].sum() > 0 else np.nan,
        include_groups=False
    )
    daily_mae = daily_mae.dropna()
    
    n = len(daily_mae)
    if n < 2:
        return {
            "norm_mae": norm_mae,
            "norm_mae_ci_lower": np.nan,
            "norm_mae_ci_upper": np.nan,
        }
    
    # Calculate 95% CI using t-distribution
    # Use the module-level `stats` import to avoid creating a local binding that
    # interferes with nested functions (see NameError closure issues).
    mae_se = daily_mae.sem()
    t_critical = stats.t.ppf(0.975, df=n-1)
    
    # Note: using daily_mae.mean() instead of norm_mae for consistency with CI calculation
    mae_mean = daily_mae.mean()
    mae_ci_lower = mae_mean - (t_critical * mae_se)
    mae_ci_upper = mae_mean + (t_critical * mae_se)
    
    return {
        "norm_mae": round(mae_mean, 1),
        "norm_mae_ci_lower": round(mae_ci_lower, 1),
        "norm_mae_ci_upper": round(mae_ci_upper, 1),
    }


def calc_rank_corr(
    data_handler, 
    horizon_mins, 
    pred_col="predicted_value", 
    truth_col="signal_value",
    ci=False,
):
    """
    Compute mean daily Spearman rank correlation up to a given forecast horizon.
    
    Parameters:
        data_handler: AnalysisDataHandler object with forecasts_v_moers DataFrame
        horizon_mins: Maximum forecast horizon in minutes
        pred_col: Column name for predictions
        truth_col: Column name for ground truth
        ci: If True, returns dict with confidence intervals; if False, returns scalar (default False)
    
    Returns:
        float or dict: If ci=False, returns rank correlation value.
                      If ci=True, returns {"rank_corr": value, "rank_corr_ci_lower": value, "rank_corr_ci_upper": value}
    """

    df = data_handler.forecasts_v_moers
    filtered_df = df[df["horizon_mins"] <= horizon_mins].dropna(
        subset=[pred_col, truth_col]
    )
    if filtered_df.empty:
        return np.nan

    def safe_spearmanr(group_df):
        if group_df[truth_col].nunique() < 2 or group_df[pred_col].nunique() < 2:
            return np.nan
        return stats.spearmanr(group_df[truth_col], group_df[pred_col]).statistic

    corr_by_forecast = filtered_df.groupby("generated_at").apply(
        safe_spearmanr, include_groups=False
    )
    
    rank_corr = float(np.nanmean(corr_by_forecast))
    if not np.isnan(rank_corr):
        rank_corr = round(rank_corr, 3)
    
    if not ci:
        return rank_corr
    
    # Calculate per-day correlation for confidence intervals
    df_reset = corr_by_forecast.reset_index()
    df_reset.columns = ["generated_at", "correlation"]
    df_reset = df_reset.dropna()
    df_reset["date"] = df_reset["generated_at"].dt.normalize()
    
    # Group by date and average correlations within each day
    daily_corr = df_reset.groupby("date")["correlation"].mean()
    
    n = len(daily_corr)
    if n < 2:
        return {
            "rank_corr": rank_corr,
            "rank_corr_ci_lower": np.nan,
            "rank_corr_ci_upper": np.nan,
        }
    
    # Calculate 95% CI using t-distribution
    # Use the module-level `stats` import to avoid creating a local binding
    corr_se = daily_corr.sem()
    t_critical = stats.t.ppf(0.975, df=n-1)
    
    # Note: using daily_corr.mean() instead of rank_corr for consistency with CI calculation
    corr_mean = daily_corr.mean()
    corr_ci_lower = corr_mean - (t_critical * corr_se)
    corr_ci_upper = corr_mean + (t_critical * corr_se)
    
    return {
        "rank_corr": round(corr_mean, 3),
        "rank_corr_ci_lower": round(corr_ci_lower, 3),
        "rank_corr_ci_upper": round(corr_ci_upper, 3),
    }


def bootstrap_confidence_interval(
    df_with_date: pd.DataFrame,
    metric_func,
    effective_forecast_samp_rate: float,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
):
    """
    Calculate bootstrap confidence intervals for a metric, accounting for forecast sampling.
    
    This function implements a day-level block bootstrap that appropriately handles
    uncertainty when forecasts are sampled on entire days (not all days have forecasts).
    
    Parameters:
        df_with_date: DataFrame with a 'date' column (normalized datetime) for grouping
        metric_func: Function that takes a DataFrame and returns a dict of metrics
                    Example: lambda df: {"metric": df["value"].sum()}
        effective_forecast_samp_rate: Fraction of days in date range that have forecasts (0-1)
        n_bootstrap: Number of bootstrap iterations (default 1000)
        confidence_level: Confidence level for intervals (default 0.95 = 95% CI)
        random_seed: Random seed for reproducibility (default 42)
    
    Returns:
        dict: Point estimates and CIs for each metric returned by metric_func
              Keys: {metric}_ci_lower, {metric}_ci_upper for each metric
    
    Algorithm:
        - If effective_forecast_samp_rate < 0.95: Use reduced effective sample size to inflate CIs
        - If effective_forecast_samp_rate >= 0.95: Use standard bootstrap (full sample size)
        - Both approaches use block resampling of entire days to preserve temporal structure
    
    """
    
    n_days = len(df_with_date)
    
    if n_days == 0:
        raise ValueError("DataFrame is empty, cannot calculate confidence intervals")
    
    # Determine bootstrap strategy based on sampling rate
    if effective_forecast_samp_rate < 0.95:
        # Sampled forecasts: use effective sample size to account for sampling uncertainty
        effective_n_days = max(2, int(n_days * effective_forecast_samp_rate))
    else:
        # Full or nearly-full data: standard bootstrap
        effective_n_days = n_days
    
    np.random.seed(random_seed)
    bootstrap_results = []
    
    for _ in range(n_bootstrap):
        # Block resample: sample entire days with replacement
        sampled_days = df_with_date.sample(n=effective_n_days, replace=True)
        metrics = metric_func(sampled_days)
        bootstrap_results.append(metrics)
    
    bootstrap_df = pd.DataFrame(bootstrap_results)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_dict = {}
    for metric_name in bootstrap_df.columns:
        ci_dict[f"{metric_name}_ci_lower"] = np.percentile(
            bootstrap_df[metric_name], lower_percentile
        )
        ci_dict[f"{metric_name}_ci_upper"] = np.percentile(
            bootstrap_df[metric_name], upper_percentile
        )
    
    return ci_dict


def pick_k_optimal_charge(in_df: pd.DataFrame, truth_col, charge_mins: int):
    """
    Select the k optimal charging periods (lowest truth_col values) within each window.
    
    This represents the theoretical best-case scenario where we have perfect knowledge
    of future values and can select the optimal charging times.
    
    Parameters:
        in_df: DataFrame with window_start column and point_time as index (or in MultiIndex)
        truth_col: Column name containing the actual signal values
        charge_mins: Total minutes of charging needed per window
    
    Returns:
        Series of boolean charge_status values indicating optimal charge periods
    """
    df = in_df.copy()

    charge_needed = charge_mins // 5
    df["charge_status"] = False
    df = df.sort_index(ascending=True)

    for window_start, w_df in df.reset_index().groupby("window_start"):
        # Pick the `charge_needed` lowest truth_col values in this window
        # If there are ties, 'first' ensures deterministic selection        
        selected_points = w_df.sort_values(truth_col, ascending=True).head(
            charge_needed
        )

        # Mark these point_times as charge periods in the original dataframe
        # Handle both simple index (point_time) and MultiIndex (generated_at, point_time)
        try:
            df.loc[selected_points["point_time"], "charge_status"] = True
        except KeyError:
            # MultiIndex case - use the full index from selected_points
            df.loc[selected_points.set_index(['generated_at', 'point_time']).index, "charge_status"] = True

    return df["charge_status"]


def simulate_charge(
    in_df: pd.DataFrame, 
    sort_col: str, 
    charge_mins: int,
    discount_factor: float = 0.0,
    discount_horizon_hours: float = 6.0
):
    """
    Simulate a greedy charging strategy that makes sequential decisions at each forecast time.
    
    At each decision point (generated_at), determines whether to charge based on:
    1. The forecast's predictions for future point_times
    2. How those predictions rank among all forecasted values
    3. Remaining opportunities to charge before window ends
    
    The algorithm processes forecasts chronologically and makes irrevocable charging decisions
    without knowledge of future forecasts (causally valid simulation).
    
    The function automatically detects the forecast frequency (interval between generated_at times)
    and determines how many 5-minute periods each forecast can schedule. For example:
    - 5-minute frequency: each forecast schedules 1 period
    - 30-minute frequency: each forecast schedules up to 6 periods
    - 60-minute frequency: each forecast schedules up to 12 periods
    
    Parameters:
        in_df: DataFrame with MultiIndex (generated_at, point_time)
        sort_col: Column name to use for ranking (e.g., 'predicted_value')
        charge_mins: Total minutes of charging needed in each window
        discount_factor: If > 0, applies exponential time-discounting to future predictions
                        to account for increasing uncertainty (default 0.0 = no discounting)
        discount_horizon_hours: Reference horizon for discounting (default 6h)
    
    Returns:
        Series of boolean charge_status values
    
    Algorithm:
        For each window and each generated_at time T:
        1. Determine how many periods this forecast can schedule (based on forecast frequency)
        2. From the available forecasted periods, select the best N periods
        3. Count how many forecasted values are better (with optional discounting)
        4. Decide whether to schedule charging for the selected periods
        5. Continue to next decision point
    """

    df = in_df.copy()

    assert sort_col in df.columns

    charge_needed = charge_mins // 5  # Total number of 5-min periods needed
    df = df.assign(charge_status=False)
    df = df.sort_index(ascending=True)
    
    for w in df.reset_index().groupby("window_start"):
        window_start, w_df = w
        w_df = w_df.sort_values(["generated_at", "point_time"], ascending=True)
        
        # Detect forecast frequency by looking at time between consecutive generated_at values
        unique_generated_at = sorted(w_df["generated_at"].unique())
        if len(unique_generated_at) > 1:
            # Calculate median interval to be robust to missing forecasts
            intervals = pd.Series(unique_generated_at).diff().dropna()
            forecast_freq_minutes = int(intervals.median().total_seconds() / 60)
        else:
            # Only one forecast, assume 5-minute frequency
            forecast_freq_minutes = 5
        
        # Calculate how many 5-minute periods each forecast can schedule
        periods_per_forecast = max(1, forecast_freq_minutes // 5)
        
        charge_periods = []
        scheduled_periods = set()  # Track which periods are already scheduled
        _charge_needed = charge_needed
        generated_at_groups = list(w_df.groupby("generated_at"))
        total_groups = len(generated_at_groups)

        for processed_count, (generated_at, g_df) in enumerate(generated_at_groups):
            if _charge_needed == 0:
                break
                
            # Filter to schedulable periods: not already scheduled, within [generated_at, generated_at + forecast_freq)
            # This represents the "decision window" - we can only schedule for the next forecast period
            g_df_reset = g_df.reset_index()
            next_generated_at = generated_at + pd.Timedelta(minutes=forecast_freq_minutes)
            
            schedulable_mask = (
                ~g_df_reset["point_time"].isin(scheduled_periods) &
                (g_df_reset["point_time"] >= generated_at) &
                (g_df_reset["point_time"] < next_generated_at)
            )
            schedulable_df = g_df_reset[schedulable_mask]
            
            if len(schedulable_df) == 0:
                continue
            
            # Determine how many periods to schedule in this decision
            # Can schedule up to periods_per_forecast, but limited by remaining need and available periods
            max_to_schedule = min(periods_per_forecast, _charge_needed, len(schedulable_df))
            
            # Apply discounting if specified
            if discount_factor > 0:
                hours_ahead = (schedulable_df["point_time"] - generated_at).dt.total_seconds() / 3600
                weights = np.exp(-discount_factor * hours_ahead / discount_horizon_hours)
                schedulable_df = schedulable_df.copy()
                schedulable_df["weighted_value"] = schedulable_df[sort_col] / weights
                sort_column = "weighted_value"
            else:
                sort_column = sort_col
            
            # Select the best max_to_schedule periods from schedulable periods
            best_periods = schedulable_df.nsmallest(max_to_schedule, sort_column)
            
            # Evaluate: Count how many periods in the ENTIRE forecast (all of g_df) are better than worst selected
            # This gives us the ranking quality without the scheduling constraint
            worst_selected_value = best_periods[sort_col].max()
            
            if discount_factor > 0:
                # For discounting, evaluate against all forecasted periods with weights
                all_future_mask = g_df_reset["point_time"] > best_periods["point_time"].max()
                all_future_df = g_df_reset[all_future_mask]
                if len(all_future_df) > 0:
                    hours_ahead_all = (all_future_df["point_time"] - generated_at).dt.total_seconds() / 3600
                    weights_all = np.exp(-discount_factor * hours_ahead_all / discount_horizon_hours)
                    better_mask = all_future_df[sort_col] < worst_selected_value
                    n_below_worst = (better_mask * weights_all).sum()
                else:
                    n_below_worst = 0
            else:
                # Simple count: how many periods in the entire forecast are better than worst selected
                all_future_mask = g_df_reset["point_time"] > best_periods["point_time"].max()
                n_below_worst = (g_df_reset[all_future_mask][sort_col] < worst_selected_value).sum()
            
            remaining_generated_at = total_groups - processed_count - 1
            remaining_capacity = remaining_generated_at * periods_per_forecast
            should_charge_now = (n_below_worst <= _charge_needed) or (
                remaining_capacity <= _charge_needed
            )

            if should_charge_now:
                # Schedule charging for the selected periods
                for _, row in best_periods.iterrows():
                    pt = row["point_time"]
                    charge_periods.append((generated_at, pt))
                    scheduled_periods.add(pt)
                    _charge_needed -= 1

        # Mark the scheduled periods in the dataframe
        df.loc[charge_periods, "charge_status"] = True

    # Validation: check that we achieved reasonable charging coverage
    charged_total = int(df["charge_status"].sum())
    
    # Calculate expected capacity based on forecast frequency
    window_group_counts = df.reset_index().groupby("window_start")["generated_at"].nunique()
    
    # For each window, calculate its forecast frequency and expected capacity
    expected_total_needed = 0
    for window_start, n_forecasts in window_group_counts.items():
        # Get forecasts for this window
        window_mask = df.reset_index()["window_start"] == window_start
        window_df = df.reset_index()[window_mask]
        unique_gen_at = sorted(window_df["generated_at"].unique())
        
        if len(unique_gen_at) > 1:
            intervals = pd.Series(unique_gen_at).diff().dropna()
            freq_mins = int(intervals.median().total_seconds() / 60)
        else:
            freq_mins = 5
        
        periods_per_fc = max(1, freq_mins // 5)
        window_capacity = n_forecasts * periods_per_fc
        expected_total_needed += min(charge_needed, window_capacity)

    # Upper bound: cannot exceed expected capacity (with small buffer for edge cases)
    assert (
        charged_total <= expected_total_needed + (len(window_group_counts) * 2)
    ), f"Charge status is too high ({charged_total} > {expected_total_needed}), check the logic for simulate_charge"

    # Lower bound: be within 10% of expected (relaxed to account for greedy scheduling constraints)
    # The greedy algorithm may not perfectly achieve theoretical capacity due to:
    # - Period conflicts from overlapping forecast coverage
    # - Suboptimal early decisions in the greedy algorithm
    # - Edge effects at window boundaries
    # - Timing misalignments between forecast pulls and optimal charge periods
    if expected_total_needed > 0:
        min_acceptable = int(np.floor(0.90 * expected_total_needed))
        assert (
            charged_total >= min_acceptable
        ), f"Charge status is too low ({charged_total} < {min_acceptable}, which is 90% of {expected_total_needed}), check the logic for simulate_charge"

    return df["charge_status"]


def assign_windows(
    timestamps: List[pd.Timestamp] | pd.DatetimeIndex,
    window_size: Union[int, pd.Timedelta],
    window_start_time: Optional[str] = None,
) -> List[pd.Timestamp]:
    """
    Assign a window_start timestamp for each input time based on:
    - A fixed window length (window_size)
    - An optional daily start time (window_start_time in "HH:MM").

    Behavior:
        - If window_start_time is provided: one window per day starting at that local time.
            For each timestamp t, compute daily_start = local midnight + window_start_time and
            daily_end = daily_start + window_size. Return daily_start only if daily_start <= t < daily_end;
            otherwise return NaT for that row (the row is outside the window).
    - If window_start_time is None: windows are tiled back-to-back every `window_size`
      from a fixed base (the first timestamp's local midnight), producing non-overlapping
      windows of length `window_size` across the entire timeline.

    All input timestamps are expected to be timezone-aware and already localized.
    Returns a list of tz-aware pandas Timestamps in the same timezone as inputs.
    """

    # Normalize inputs
    if isinstance(window_size, int):
        window_size = pd.Timedelta(minutes=window_size)

    if isinstance(timestamps, list):
        if len(timestamps) == 0:
            return []
        ts_index = pd.DatetimeIndex(timestamps)
    elif isinstance(timestamps, pd.DatetimeIndex):
        if len(timestamps) == 0:
            return []
        ts_index = timestamps
    else:
        # Fallback: try to construct a DatetimeIndex
        ts_index = pd.DatetimeIndex(timestamps)
        if len(ts_index) == 0:
            return []

    # Branch 1: daily-anchored windows at specific local time
    if window_start_time:
        try:
            hour_str, minute_str = window_start_time.split(":")
            start_hours, start_minutes = int(hour_str), int(minute_str)
        except Exception as e:
            raise ValueError(
                f"Invalid window_start_time '{window_start_time}'. Expected 'HH:MM'."
            ) from e

        # For each timestamp, find the most recent daily anchor whose window [start, start+window_size)
        # contains t. This supports windows that cross midnight and windows longer than 24h by
        # checking up to ceil(window_size/1day) days back.
        anchors = []
        start_offset = pd.Timedelta(hours=start_hours, minutes=start_minutes)
        one_day = pd.Timedelta(days=1)
        days_back_to_check = int(np.ceil(window_size / one_day))
        for t in ts_index:
            # Start with today's anchor; if t is before it, start from previous day's anchor
            base_anchor = t.normalize() + start_offset
            candidate = base_anchor if t >= base_anchor else base_anchor - one_day
            assigned = pd.NaT
            for k in range(days_back_to_check):
                a = candidate - (k * one_day)
                if (t >= a) and (t < a + window_size):
                    assigned = a
                    break
            anchors.append(assigned)

        return anchors

    # Branch 2: continuous tiling of windows across the timeline
    # Use a stable base so windows are non-overlapping and consistent across days
    base = ts_index.min().normalize()
    delta = ts_index - base  # TimedeltaIndex
    # Floor-divide to count how many full windows have elapsed since base
    steps = delta // window_size
    anchor_idx = base + (steps * window_size)
    return list(anchor_idx)


def calc_rank_compare_metrics(
    data_handler,
    charge_mins,
    window_mins,
    window_start_time=None,
    pred_col="predicted_value",
    truth_col="signal_value",
    load_kw=1000,
    forecast_pull_mins=30,
):
    """
    Calculate emissions reduction metrics comparing forecast-based charging to baseline.
    
    Parameters:
        data_handler: AnalysisDataHandler with forecasts_v_moers data
        charge_mins: Total minutes of charging needed per window
        window_mins: Window duration in minutes
        window_start_time: Optional time string (HH:MM) for daily window start
        pred_col: Column name for forecast predictions
        truth_col: Column name for ground truth values
        load_kw: Load in kilowatts
        forecast_pull_mins: Forecast pull frequency in minutes (default 30)
                           Controls how often forecasts are retrieved.
                           E.g., 30 means forecasts pulled every 30 minutes.
    
    Returns:
        dict: Reduction and potential metrics with confidence intervals
    """
    in_df = data_handler.forecasts_v_moers
    df = in_df.copy()

    if isinstance(window_mins, int):
        window_mins = pd.Timedelta(minutes=window_mins)
    
    forecast_pull_delta = pd.Timedelta(minutes=forecast_pull_mins)

    effective_forecast_samp_rate = data_handler.effective_forecast_sample_rate

    # reset index before filtering
    df = df.reset_index()

    # Filter generated_at times to match the forecast_pull frequency BEFORE assigning windows
    # This significantly reduces the volume of data that needs window assignment
    # Keep only forecasts where generated_at minute aligns with the forecast pull frequency
    # e.g., if forecast_pull_mins=30, keep forecasts at :00 and :30 minutes
    df = df[df["generated_at"].dt.minute % forecast_pull_mins == 0]

    window_starts = assign_windows(
        df["generated_at"], window_mins, window_start_time
    )

    df = df.assign(window_start=window_starts)
    
    window_ends = []
    for ws in window_starts:
        if pd.isna(ws):
            window_ends.append(pd.NaT)
        else:
            window_ends.append(ws + window_mins)
    
    df = df.assign(window_end=window_ends)

    # Filter out rows that do not fall within the valid window
    # Instead of using .loc to set values then dropna, directly filter the dataframe
    # This is much faster as it avoids the expensive .loc assignment operation
    valid_mask = (
        (df["point_time"] < df["window_end"])
        & (df["generated_at"] < df["window_end"])
        & (df["generated_at"] >= df["window_start"])
    )
    df = df[valid_mask]

    # Filter out windows where we don't have enough generated_ats to cover the full window duration
    # With the new forecast_pull frequency, we need fewer forecasts
    expected_gen_ats = int(np.ceil(window_mins / forecast_pull_delta))
    window_gen_at_counts = df.groupby("window_start")["generated_at"].nunique()
    # Allow some tolerance (90% of expected)
    min_required_gen_ats = max(1, int(np.floor(0.9 * expected_gen_ats)))
    valid_windows = window_gen_at_counts[window_gen_at_counts >= min_required_gen_ats].index
    df = df[df["window_start"].isin(valid_windows)]

    # Filter out rows where generated_at != first point_time in the window
    # e.g. we pulled an incomplete forecast (not sure why this would happen, but safety check)
    # Use direct boolean indexing instead of .loc for better performance
    first_point_times = df.groupby('generated_at')["point_time"].transform("first")
    df = df[df['generated_at'] == first_point_times]

    df = df.set_index(['generated_at', 'point_time']).sort_index(ascending=True)

    # For pick_k_optimal_charge, we need ALL nowcast values (every 5 minutes), not just those from filtered forecasts
    # Go back to the ORIGINAL unfiltered data to get complete nowcast coverage
    df_for_optimal = in_df.reset_index()
    df_for_optimal = df_for_optimal[df_for_optimal['generated_at'] == df_for_optimal['point_time']]
    
    # Apply window assignment
    window_starts_optimal = assign_windows(
        df_for_optimal["point_time"], window_mins, window_start_time
    )
    df_for_optimal = df_for_optimal.assign(window_start=window_starts_optimal)
    
    # Keep only windows that have forecasts (from valid_windows), but DON'T validate
    # based on nowcast count - we need ALL available nowcast data for baseline/optimal
    df_for_optimal = df_for_optimal[df_for_optimal["window_start"].isin(valid_windows)]
    
    # NOW validate that each window has enough nowcast data for baseline calculation
    # We need at least charge_mins // 5 periods per window
    min_periods_needed = charge_mins // 5
    nowcast_counts = df_for_optimal.groupby('window_start').size()
    windows_with_enough_data = nowcast_counts[nowcast_counts >= min_periods_needed].index
    df_for_optimal = df_for_optimal[df_for_optimal["window_start"].isin(windows_with_enough_data)]
    
    # Update valid_windows to only include windows that have both forecasts AND enough nowcast data
    valid_windows = windows_with_enough_data
    df = df.reset_index()
    df = df[df["window_start"].isin(valid_windows)]
    df = df.set_index(['generated_at', 'point_time']).sort_index(ascending=True)
    
    df_for_optimal = df_for_optimal[['point_time', truth_col, 'window_start']].copy()
    df_for_optimal = df_for_optimal.drop_duplicates(subset=['point_time'], keep='first').set_index('point_time').sort_index()

    # Calculate optimal charge status and emissions on the optimal dataset
    df_for_optimal['charge_status'] = pick_k_optimal_charge(df_for_optimal, truth_col, charge_mins)
    df_for_optimal['truth_charge_emissions'] = (
        df_for_optimal[truth_col]
        * df_for_optimal['charge_status']
        * (load_kw / 1000)
        * (5 / 60)
    )
    
    # Calculate baseline: immediate charging on the same optimal dataset
    # This ensures baseline uses the same complete set of point_times as optimal
    df_for_optimal = df_for_optimal.sort_index()
    df_for_optimal['sequential_rank'] = df_for_optimal.groupby('window_start').cumcount()
    df_for_optimal['baseline_charge_status'] = df_for_optimal['sequential_rank'] < charge_mins // 5
    df_for_optimal['baseline_charge_emissions'] = (
        df_for_optimal[truth_col]
        * df_for_optimal['baseline_charge_status']
        * (load_kw / 1000)
        * (5 / 60)
    )
    
    # Aggregate optimal and baseline results by window
    optimal_by_window = df_for_optimal.groupby('window_start').agg({
        'charge_status': 'sum',
        'truth_charge_emissions': 'sum',
        'baseline_charge_status': 'sum',
        'baseline_charge_emissions': 'sum'
    }).rename(columns={
        'charge_status': 'truth_charge_count',
        'baseline_charge_status': 'baseline_charge_count'
    })
    
    # Calculate pred_charge_status on the forecast dataframe
    df = df.assign(
        pred_charge_status=simulate_charge(df, pred_col, charge_mins),
    )

    df = df.assign(
        pred_charge_emissions=df[truth_col]
        * df["pred_charge_status"]
        * (load_kw / 1000)
        * (5 / 60),
    )

    # Aggregate forecast results by window and merge with optimal/baseline results
    by_window = df.groupby("window_start").agg({
        'pred_charge_status': 'sum',
        'pred_charge_emissions': 'sum',
    }).rename(columns={
        'pred_charge_status': 'pred_charge_count',
    })
    
    by_window = by_window.join(optimal_by_window, how='inner')
    
    # Validate that charge counts are correct for each window
    expected_charge_count = charge_mins // 5
    for window_start, row in by_window.iterrows():
        assert row['baseline_charge_count'] == expected_charge_count, \
            f"Window {window_start}: baseline_charge_count={row['baseline_charge_count']} != {expected_charge_count}"
        assert row['truth_charge_count'] == expected_charge_count, \
            f"Window {window_start}: truth_charge_count={row['truth_charge_count']} != {expected_charge_count}"
        # pred_charge_count may be less due to forecast limitations, but should be close
        assert row['pred_charge_count'] >= int(0.9 * expected_charge_count), \
            f"Window {window_start}: pred_charge_count={row['pred_charge_count']} < 90% of {expected_charge_count}"
        assert row['pred_charge_count'] <= expected_charge_count + 2, \
            f"Window {window_start}: pred_charge_count={row['pred_charge_count']} > {expected_charge_count}"
        

    # Validate sum of emissions follows order baseline >= pred >= truth
    summed_window = by_window[['baseline_charge_emissions', 'pred_charge_emissions', 'truth_charge_emissions']].sum()
    assert summed_window['baseline_charge_emissions'] >= summed_window['pred_charge_emissions'], \
        "Total baseline emissions should be >= total predicted emissions. Although this may be effected by forecast_pull_mins"
    assert summed_window['pred_charge_emissions'] >= summed_window['truth_charge_emissions'], \
        "Total predicted emissions should be >= total truth (optimal) emissions"
    
    by_window = by_window.reset_index()
    n = len(by_window)

    if n == 0:
        reduction = np.nan
        potential = np.nan
        reduction_ci_lower = np.nan
        reduction_ci_upper = np.nan
        potential_ci_lower = np.nan
        potential_ci_upper = np.nan
    else:
        # Calculate point estimates
        y_best_emissions = by_window["truth_charge_emissions"].sum()
        y_pred_emissions = by_window["pred_charge_emissions"].sum()
        y_base_emissions = by_window["baseline_charge_emissions"].sum()

        assert y_pred_emissions >= y_best_emissions
        assert y_base_emissions >= y_best_emissions

        reduction = (y_base_emissions - y_pred_emissions) / n
        potential = (y_base_emissions - y_best_emissions) / n

        # Calculate confidence intervals using standard error (95% CI)
        # Each window represents an independent sample
        by_window['reduction_per_window'] = (
            by_window['baseline_charge_emissions'] - by_window['pred_charge_emissions']
        )
        by_window['potential_per_window'] = (
            by_window['baseline_charge_emissions'] - by_window['truth_charge_emissions']
        )
        
        # Standard error of the mean
        reduction_se = by_window['reduction_per_window'].sem()
        potential_se = by_window['potential_per_window'].sem()
        
        # 95% CI using t-distribution (more conservative for small samples)
        # Use module-level `stats` to avoid creating a local binding
        t_critical = stats.t.ppf(0.975, df=n-1)  # 95% CI, two-tailed
        
        reduction_ci_lower = reduction - (t_critical * reduction_se)
        reduction_ci_upper = reduction + (t_critical * reduction_se)
        potential_ci_lower = potential - (t_critical * potential_se)
        potential_ci_upper = potential + (t_critical * potential_se)

    return {
        "reduction": round(reduction, 1),
        "potential": round(potential, 1),
        "reduction_ci_lower": round(reduction_ci_lower, 1),
        "reduction_ci_upper": round(reduction_ci_upper, 1),
        "potential_ci_lower": round(potential_ci_lower, 1),
        "potential_ci_upper": round(potential_ci_upper, 1),
    }


def plot_norm_mae(
    factory: DataHandlerFactory, horizons_hr=[1, 6, 12, 18, 24, 72], ci=True
) -> Dict[str, go.Figure]:
    """
    Create a Plotly bar chart for normalized MAE by horizon with one subplot per region (abbrev).
    
    Parameters:
        factory: DataHandlerFactory with data handlers
        horizons_hr: List of horizons in hours
        ci: If True, adds confidence interval error bars (default True)
    """

    y_min = y_max = 0
    figs = {}

    # Iterate through each region and create a bar plot
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        fig = go.Figure()
        x_values = [f"{h}hr" for h in horizons_hr]  # Add horizon labels

        for model_job in region_models:
            # Calculate metrics with or without CIs
            results = [
                calc_norm_mae(model_job, (h * 60) - 5, ci=ci)
                for h in horizons_hr
            ]
            
            if ci:
                # Extract values and CIs from dict results
                y_values = [r["norm_mae"] for r in results]
                ci_lower = [r["norm_mae_ci_lower"] for r in results]
                ci_upper = [r["norm_mae_ci_upper"] for r in results]
                
                # Calculate error bar arrays (distance from point estimate)
                error_minus = [y - lower if not np.isnan(lower) else 0 for y, lower in zip(y_values, ci_lower)]
                error_plus = [upper - y if not np.isnan(upper) else 0 for y, upper in zip(y_values, ci_upper)]
                
                # Update y_max to include CI upper bounds
                y_max = max([u for u in ci_upper if not np.isnan(u)] + [y_max])
            else:
                # Scalar results
                y_values = results
                error_minus = None
                error_plus = None
            
            y_max = max(y_values + [y_max])
            y_min = min(y_values + [y_min])

            error_y_config = dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                width=20,
                thickness=1,
                color="rgba(0, 0, 0, 0.3)"  # black with some transparency
            ) if ci else None

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=model_job.model_date,
                    text=[f"{y:.1f}%" for y in y_values],
                    textposition="outside",
                    error_y=error_y_config,
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
    factory: DataHandlerFactory, horizons_hr=[1, 6, 12, 18, 24, 72], ci=True
) -> Dict[str, go.Figure]:
    """
    Create a Plotly bar plot for rank correlation by horizon with one subplot per region (abbrev).

    Parameters:
        factory: DataHandlerFactory with data handlers
        horizons_hr: List of horizons in hours
        ci: If True, adds confidence interval error bars (default True)

    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure with subplots.
    """

    y_min = y_max = 0
    figs = {}

    # Iterate through each region and create a bar plot
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        fig = go.Figure()

        # Extract data for the bar plot
        x_values = [f"{h}hr" for h in horizons_hr]  # Add horizon labels

        for model_job in region_models:
            # Calculate metrics with or without CIs
            results = [
                calc_rank_corr(model_job, (h * 60) - 5, ci=ci)
                for h in horizons_hr
            ]
            
            if ci:
                # Extract values and CIs from dict results
                y_values = [r["rank_corr"] for r in results]
                ci_lower = [r["rank_corr_ci_lower"] for r in results]
                ci_upper = [r["rank_corr_ci_upper"] for r in results]
                
                # Calculate error bar arrays (distance from point estimate)
                error_minus = [y - lower if not np.isnan(lower) else 0 for y, lower in zip(y_values, ci_lower)]
                error_plus = [upper - y if not np.isnan(upper) else 0 for y, upper in zip(y_values, ci_upper)]
                
                # Update y_max/y_min to include CI bounds
                y_max = max([u for u in ci_upper if not np.isnan(u)] + [y_max])
                y_min = min([l for l in ci_lower if not np.isnan(l)] + [y_min])
            else:
                # Scalar results
                y_values = results
                error_minus = None
                error_plus = None
            
            y_max = max(y_values + [y_max])
            y_min = min(y_values + [y_min])

            error_y_config = dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                color="rgba(0, 0, 0, 0.3)",  # black with some transparency
                width=20,
                thickness=1,
            ) if ci else None

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=model_job.model_date,
                    text=[f"{y:.3f}" for y in y_values],
                    textposition="outside",
                    error_y=error_y_config,
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
            range=[y_min - (0.25 * abs(y_max - y_min)), y_max + (0.25 * abs(y_max - y_min))]
        )

    return figs


AER_SCENARIOS = {
    "EV-night": {
        "charge_mins": 3 * 60,
        "window_mins": 8 * 60,
        "window_start_time": "19:00",
        "load_kw": 19,
        "forecast_pull_mins": 30,
    },
    "EV-day": {
        "charge_mins": 3 * 60,
        "window_mins": 8 * 60,
        "window_start_time": "09:00",
        "load_kw": 19,
        "forecast_pull_mins": 30,
    },
    "Thermostat": {
        "charge_mins": 30,
        "window_mins": 60,
        "load_kw": 3,  # typical AC
        "forecast_pull_mins": 30,
    },
    "10 kW / 24hr / 25% Duty Cycle": {
        "charge_mins": int(24 * 60 * 0.25),
        "window_mins": 24 * 60,
        "load_kw": 10,
        "forecast_pull_mins": 30,
    },
    "10 kW / 72hr / 25% Duty Cycle": {
        "charge_mins": int(24 * 3 * 60 * 0.25),
        "window_mins": 24 * 3 * 60,
        "load_kw": 10,
        "forecast_pull_mins": 30,
    },
    "10 kW / 24hr / 50% Duty Cycle": {
        "charge_mins": int(24 * 60 * 0.5),
        "window_mins": 24 * 60,
        "load_kw": 10,
        "forecast_pull_mins": 30,
    },
    "10 kW / 72hr / 50% Duty Cycle": {
        "charge_mins": (int(24 * 3 * 60 * 0.5)),
        "window_mins": 24 * 3 * 60,
        "load_kw": 10,
        "forecast_pull_mins": 30,
    },
}


def plot_impact_forecast_metrics(
    factory: DataHandlerFactory,
    scenarios=[
        "10 kW / 24hr / 25% Duty Cycle",
        # "10 kW / 72hr / 25% Duty Cycle",
        "10 kW / 24hr / 50% Duty Cycle",
        # "10 kW / 72hr / 50% Duty Cycle",
        "EV-night",
        "EV-day",
        # "Thermostat",
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

        for model_ix, model_job in enumerate(region_models, start=1):

            _metrics = [
                {
                    **calc_rank_compare_metrics(
                        model_job, **AER_SCENARIOS[s],
                    ),
                    "scenario": s,
                }
                for s in scenarios
            ]

            _metrics = pd.DataFrame(_metrics)

            # Calculate error bar values for potential
            # CI values are absolute, so we need to compute distance from the mean
            potential_error = [
                [
                    _metrics["potential"].iloc[i] - _metrics["potential_ci_lower"].iloc[i],
                    _metrics["potential_ci_upper"].iloc[i] - _metrics["potential"].iloc[i],
                ]
                for i in range(len(_metrics))
            ]
            potential_error_symmetric = list(zip(*potential_error))

            fig.add_trace(
                go.Bar(
                    x=_metrics["scenario"],
                    y=_metrics["potential"],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=potential_error_symmetric[1],  # upper errors
                        arrayminus=potential_error_symmetric[0],  # lower errors
                        color="rgba(0, 0, 0, 0.25)",  # black
                        width=20,
                        thickness=1,
                    ),
                    name=f"CO2 Potential Savings",  # Legend only in the first subplot
                    marker=dict(
                        color="rgba(200, 200, 200, 0.8)"
                    ),  # Light gray for potential
                    hovertemplate="%{x}: %{y:.1f} lbs CO2<extra></extra>",
                    legendgroup="Potential Savings",  # Group legend for potential savings
                    showlegend=(
                        model_ix == 1
                    ),  # Show legend only for the first subplot
                    text=[f"{y:.1f} lbs" for y in _metrics["potential"]],
                    textposition="outside",
                ),
                row=model_ix,
                col=1,
            )

            # Calculate error bar values for reduction
            # CI values are absolute, so we need to compute distance from the mean
            reduction_error = [
                [
                    _metrics["reduction"].iloc[i] - _metrics["reduction_ci_lower"].iloc[i],
                    _metrics["reduction_ci_upper"].iloc[i] - _metrics["reduction"].iloc[i],
                ]
                for i in range(len(_metrics))
            ]
            reduction_error_symmetric = list(zip(*reduction_error))

            # Calculate text positions above error bars
            text_labels = [
                f"{(r / (p + 1e-6)) * 100:.1f}%"
                for r, p in zip(_metrics["reduction"], _metrics["potential"])
            ]
            
            fig.add_trace(
                go.Bar(
                    x=_metrics["scenario"],
                    y=_metrics["reduction"],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=reduction_error_symmetric[1],  # upper errors
                        arrayminus=reduction_error_symmetric[0],  # lower errors
                        color="rgba(0, 0, 0, 0.25)",  # black for reduction
                        width=20,
                        thickness=1,
                    ),
                    name="Forecast Achieved",
                    marker=dict(color="rgba(0, 128, 0, 0.8)"),  # Green for reduction
                    hovertemplate="%{x}: %{y:.1f} lbs CO2<extra></extra>",
                    legendgroup="Potential Savings",  # Group legend for potential savings
                    text=text_labels,
                    textposition="outside",
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

        # Track which fuel types have been added to legend
        fuels_in_legend = set()

        for model_ix, model_job in enumerate(region_models, start=1):

            stacked_values = model_job.fuel_mix.reindex(times)

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
    truth_col="signal_value",
    load_kw=1000,
    **kwargs
):
    """Predict the maximum potential CO2 Savings, without any forecast to estimate the upper limit of impact from a given signal value.
    This version is resilient to partial windows (e.g. where the data does not fully cover the expected number of 5-min intervals).
    """
    df = in_df.copy()

    needed_rows = charge_mins // 5
    load_factor = load_kw / 1000

    if isinstance(window_mins, int):
        window_mins = pd.Timedelta(minutes=window_mins)

    window_starts = assign_windows(
        df.index.get_level_values("point_time"), window_mins, window_start_time
    )

    df = df.assign(
        window_start=window_starts,
        window_end=[i + window_mins for i in window_starts],
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
    charge_emissions = truth_values * df["charge_status"] * load_factor * (5 / 60)

    df["sequential_rank"] = df.groupby("window_start").cumcount()
    baseline_charge_status = df["sequential_rank"] < needed_rows
    baseline_charge_emissions = truth_values * baseline_charge_status * load_factor * (5 / 60)

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
    scenarios: List[str] = [
        "10 kW / 24hr / 25% Duty Cycle",
        # "10 kW / 72hr / 25% Duty Cycle",
        "10 kW / 24hr / 50% Duty Cycle",
        # "10 kW / 72hr / 50% Duty Cycle",
        "EV-night",
        "EV-day",
        # "Thermostat",
    ],
):
    figs = {}
    y_max = 0
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        fig = go.Figure()

        for model_job in region_models:

            _metrics = [
                {
                    **calc_max_potential(model_job.moers, **AER_SCENARIOS[s]),
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
                pivot = create_pivot_table(job.fuel_mix, fuel)

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
    horizons=["30min", "2h", "8h"],
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

                # format resample_horizon for legend
                if resample_horizon >= pd.Timedelta("1h"):
                    rh_str = f"{int(resample_horizon.total_seconds() // 3600)}h"
                else:
                    rh_str = f"{int(resample_horizon.total_seconds() // 60)}min"

                fig.add_trace(
                    go.Scatter(
                        x=df_filtered.index.get_level_values("point_time"),
                        y=df_filtered["predicted_value"],
                        mode="lines",
                        name=f"Forecast Pull Freq: {rh_str}",
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
                .sort_index(ascending=True)["signal_value"]
            )
            signal = signal.loc[start_ts : end_ts + max(horizons)]
            fig.add_trace(
                go.Scatter(
                    x=signal.index,
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


MOER_PLOTS = {
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
        plot_impact_forecast_metrics,
    ],
}

OTHER_PLOTS = {
    "signal": [
        plot_sample_moers,
        plot_heatmaps,
    ],
    "forecast": [
        plot_forecasts_vs_signal,
        plot_norm_mae,
        plot_precision_recall,
    ],
}


PLOTS = {
    'co2_moer': MOER_PLOTS,
    'co2_health_damages': MOER_PLOTS,
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
        "fuel_mix",
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

    if signal_type in PLOTS:
        plot_dict = PLOTS[signal_type]
    elif len(signal_type) == 11 and 'tail' in signal_type:  # shh
        plot_dict = OTHER_PLOTS
    else:
        raise NotImplementedError(f"Plots not defined for signal_type {signal_type}")

    for step in steps:
        for plot_func in plot_dict[step]:
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
