import argparse
import calendar
import sys
import json
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse
from operator import attrgetter
from pathlib import Path
from typing import List, Optional, Union, Literal, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.colors as pc
import scipy.stats as stats

from jinja2 import Template
from watttime.evaluation.get_wt_api_forecast_evaluation_data import (
    AnalysisDataHandler,
    DataHandlerFactory,
)
from watttime.evaluation.fuels_cp import fuel_cp

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


def get_random_overlapping_period(dfs, max_period="30D", resample_freq="1H"):
    """
    Find a random overlapping time period between multiple DataFrames' datetime indices,
    maximizing the period up to `max_period`.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames with datetime indices.
        max_period (str): Maximum time period as a string (e.g., '7D' for 7 days).

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

    # Otherwise, return the first overlap range that satisfies the max_period
    else:
        overlap_range = pd.date_range(
            start=start_overlap, end=start_overlap + max_timedelta, freq=resample_freq
        )
    assert all(
        [all(t in df.index for t in overlap_range) for df in dfs]
    ), "Not all DataFrames contain the overlapping range."
    return overlap_range


def plot_sample_moers(
    factory: DataHandlerFactory, max_sample_period="30D"
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
        [j.moers for j in factory.data_handlers], max_sample_period
    )
    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():

        fig = go.Figure()

        for model_job in region_models:

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=model_job.moers.loc[times]["signal_value"],
                    mode="lines",
                    name=model_job.model_date,
                    line=dict(width=2),
                    showlegend=True,
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
    Plot the distribution of old and new MOER values for each region, creating a stacked subplot.

    Args:
        jobs (list): List of AnalysisDataHandler objects with a .moers atribute.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with stacked subplots of distributions.
    """
    # Create a figure with a subplot for each region, stacked vertically
    figs = {}

    for region_abbrev, region_models in factory.data_handlers_by_region_dict.items():
        fig = go.Figure()
        for model_job in region_models:

            # Add a histogram trace for the new MOER distribution
            fig.add_trace(
                go.Box(
                    x=model_job.moers["signal_value"].dropna(),
                    name=model_job.model_date,
                    opacity=0.6,
                )
            )

        # Update layout for the figure
        fig.update_layout(
            height=300,
            xaxis_title=f"{region_models[0].signal_type} Values",
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
                type="linear",
                tickangle=45,
                showgrid=True,
            ),
            yaxis_visible=False,
            yaxis_showticklabels=False,
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

        for j, job in enumerate(region_models, start=1):
            heat = job.moers.assign(
                month=job.moers.index.month, hour=job.moers.index.hour
            )
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
    """Returns mean daily Rank Correlation"""

    filtered_df = df[df["horizon_mins"] <= horizon_mins].dropna()
    corr = filtered_df.groupby("generated_at").apply(
        lambda group_df: stats.spearmanr(
            group_df[truth_col], group_df[pred_col]
        ).statistic
    )

    return np.nanmean(corr).round(3)

def simulate_charge(df: pd.DataFrame, sort_col: str, charge_mins: int):
    
    assert sort_col in df.columns
    include_generated_at = "generated_at" in df.index.names
    
    charge_needed = charge_mins // 5
    df = df.assign(charge_status=False)
    df = df.sort_index(ascending=True)
    for w in df.reset_index().groupby("window_start"):
        window_start, w_df = w
        
        if include_generated_at:
            charge_periods = []
            _charge_needed = charge_needed
            for g in w_df.groupby("generated_at"):
                generated_at, g_df = g
            
                # how many forecasted points are below the current value?
                n_below_now = (g_df[sort_col] <= g_df.iloc[0][sort_col]).sum()
                should_charge_now = n_below_now <= _charge_needed
                                
                if should_charge_now:
                    charge_periods.append((generated_at, generated_at))  # point_time and generated_at must be equal
                    _charge_needed -= 1
                
                if _charge_needed == 0:
                    break
        
        else:
            w_df['rank'] = w_df[sort_col].rank(ascending=True, method='first')
            charge_periods = w_df.loc[w_df['rank'] <= charge_needed, 'point_time']
        
        df.loc[charge_periods, 'charge_status'] = True
    
    assert df['charge_status'].sum() >= (len(df['window_start'].unique()) - 1) * charge_needed
    assert df['charge_status'].sum() <= (len(df['window_start'].unique())) * charge_needed
    return df['charge_status']


def calc_rank_compare_metrics(
    in_df,
    charge_mins,
    window_mins,
    window_starts=None,
    pred_col="predicted_value",
    truth_col="signal_value",
    load_kw=1000,
):
    df = in_df.copy()

    if window_starts:
        # Extract unique dates and create window ranges
        unique_dates = df.index.get_level_values("generated_at").unique()
        window_ranges = []
        for date in unique_dates:
            for start_time in window_starts:
                start = pd.Timestamp(f"{date} {start_time}")
                end = start + pd.Timedelta(minutes=window_mins)
                window_ranges.append((start, end))

        # Convert to DataFrame
        window_df = pd.DataFrame(window_ranges, columns=["window_start", "window_end"])
        window_df = window_df.sort_values("window_start")
        tz = df.index.get_level_values("generated_at")[0].tz
        window_df = window_df.assign(
            window_start=pd.to_datetime(
                window_df["window_start"], errors="coerce"
            ).dt.tz_convert(tz)
        )
        window_df = window_df.dropna(subset=["window_start"])

        # Use merge_asof to efficiently assign windows
        df = pd.merge_asof(
            df.reset_index().sort_values(["generated_at", "point_time"], ascending=True),
            window_df,
            left_on="generated_at",
            right_on="window_start",
            direction="backward",
        ).set_index(df.index.names)

    else:
        # Assign each row to a rolling window of `window_mins` based on `generated_at`
        df["window_start"] = df.index.get_level_values("generated_at").floor(
            f"{window_mins}min"
        )
        df["window_end"] = df["window_start"] + pd.Timedelta(f"{window_mins} min")

    # Filter out rows that do not fall within the valid window
    df.loc[
        (df.index.get_level_values("point_time") >= df["window_end"])
        | (df.index.get_level_values("generated_at") >= df["window_end"]) |
        (df.index.get_level_values("generated_at") <= df['window_start']),
        "window_start",
    ] = pd.NaT
    df = df.dropna(subset=["window_start"])

    df = df.assign(
        truth_charge_status=simulate_charge(df, truth_col, charge_mins),
        pred_charge_status=simulate_charge(df, pred_col, charge_mins),
    )

    df = df.assign(
        truth_charge_emissions=df[truth_col]
        * df["truth_charge_status"]
        * (load_kw / 1000),
        pred_charge_emissions=df[truth_col]
        * df["pred_charge_status"]
        * (load_kw / 1000),
    )

    # baseline: immediate charging rather than AER
    df = df.sort_index()
    df = df.assign(sequential_rank=df.groupby("window_start").cumcount())
    df = df.assign(baseline_charge_status=df["sequential_rank"] < charge_mins // 5)
    df = df.assign(
        baseline_charge_emissions=df[truth_col]
        * df["baseline_charge_status"]
        * (load_kw / 1000)
    )

    assert (
        df["baseline_charge_status"].sum()
        >= (len(df["window_start"].unique()) - 1) * charge_mins // 5
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

    reduction = (y_base_emissions - y_pred_emissions) / len(
        set(df.index.get_level_values("generated_at").date)
    )
    potential = (y_base_emissions - y_best_emissions) / len(
        set(df.index.get_level_values("generated_at").date)
    )

    return {
        "reduction": round(reduction, 1),
        "potential": round(potential, 1),
    }


def plot_norm_mae(
    factory: DataHandlerFactory, horizons_hr=[1, 6, 12, 18, 24]
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
    factory: DataHandlerFactory, horizons_hr=[1, 6, 12, 18, 24]
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


# TODO: translate to local time for start windows?
AER_SCENARIOS = {
    "EV-night": {
        "charge_mins": 3 * 60,
        "window_mins": 12 * 60,
        "window_starts": ["19:00"],
        "load_kw": 19,
    },
    "EV-day": {
        "charge_mins": 2 * 60,
        "window_mins": 8 * 60,
        "window_starts": ["09:00"],
        "load_kw": 19,
    },
    "Thermostat": {
        "charge_mins": 30,
        "window_mins": 60,
        "window_starts": None,
        "load_kw": 3,  # typical AC
    },
}


def plot_impact_forecast_metrics(
    factory: DataHandlerFactory, scenarios=["EV-night", "EV-day", "Thermostat"]
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
                        model_job.forecasts_v_moers, **AER_SCENARIOS[s]
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
                    hovertemplate="%{x}: %{y:.1f} lbs CO2/MWh<extra></extra>",
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
                    hovertemplate="%{x}: %{y:.1f} lbs CO2/MWh<extra></extra>",
                ),
                row=model_ix,
                col=1,
            )

            y_max = max(_metrics["reduction"].max(), _metrics["potential"].max(), y_max)

        # Update layout
        fig.update_layout(
            height=300 * len(region_models),
            xaxis_title="Scenario",
            yaxis_title="CO2 Savings (lbs/MWh)",
            barmode="group",  # Grouped bars (side by side)
            showlegend=True,  # Show legend
            margin=dict(l=50, r=50, t=50, b=50),
        )

        # Update axes for all subplots
        for model_ix in range(1, len(region_models) + 1):
            fig.update_xaxes(title_text="Scenario", row=model_ix, col=1)
            fig.update_yaxes(title_text="CO2 Savings (lbs/MWh)", row=model_ix, col=1)

        figs[region_abbrev] = fig

    # Set uniform y-axis range for all subplots
    for region, fig in figs.items():
        figs[region] = fig.update_yaxes(range=[0, y_max + (0.25 * y_max)])

    return figs


def plot_sample_fuelmix(
    factory: DataHandlerFactory, max_sample_period="30D"
) -> Dict[str, go.Figure]:

    figs = {}
    times = get_random_overlapping_period(
        [j.fuel_mix for j in factory.data_handlers], max_sample_period
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

        for model_ix, model_job in enumerate(region_models, start=1):

            stacked_values = model_job.fuel_mix.loc[times]

            # Create cumulative values for stacking
            for fuel_ix in range(1, len(stacked_values.columns)):
                stacked_values.iloc[:, fuel_ix] += stacked_values.iloc[:, fuel_ix - 1]

            # Add each fuel type as an area
            for fuel_ix, fuel in enumerate(model_job.fuel_mix.columns):
                fig.add_trace(
                    go.Scatter(
                        x=stacked_values.index,
                        y=stacked_values.iloc[:, fuel_ix],
                        fill="tonexty" if model_ix > 0 else "tozeroy",
                        mode="none",  # Hide lines to emphasize the filled area
                        name=fuel,
                        fillcolor=fuel_cp[fuel],
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
    window_starts=None,
    truth_col="signal_value",
    load_kw=1000,
):
    """Predict the maximum potential CO2 Savings, without any forecast to estimate the upper limit of impact from a given signal value."""

    df = in_df.copy()

    if window_starts:
        # Extract unique dates and create window ranges
        unique_dates = set(df.index.date)
        window_ranges = []
        for date in unique_dates:
            for start_time in window_starts:
                start = pd.Timestamp(f"{date} {start_time}")
                end = start + pd.Timedelta(minutes=window_mins)
                window_ranges.append((start, end))

        # Convert to DataFrame
        window_df = pd.DataFrame(window_ranges, columns=["window_start", "window_end"])
        window_df = window_df.sort_values("window_start")
        tz = df.index.get_level_values("point_time")[0].tz
        window_df = window_df.assign(
            window_start=pd.to_datetime(
                window_df["window_start"], errors="coerce"
            ).dt.tz_localize(tz),
            window_end=pd.to_datetime(
                window_df["window_end"], errors="coerce"
            ).dt.tz_localize(tz),
        )
        window_df = window_df.dropna(subset=["window_start"])

        # Ensure df is sorted by "point_time" before merging
        df = df.sort_index(level="point_time")

        # Use merge_asof to efficiently assign windows
        df = pd.merge_asof(
            df.reset_index().sort_values("point_time"),
            window_df,
            left_on="point_time",
            right_on="window_start",
            direction="backward",
        ).set_index(df.index.names)

    else:
        # Assign each row to a rolling window of `window_mins` based on `generated_at`
        df["window_start"] = df.index.get_level_values("point_time").floor(
            f"{window_mins}min", ambiguous=False
        )
        df["window_end"] = df["window_start"] + pd.Timedelta(f"{window_mins} min")

    # Filter out rows that do not fall within the valid window
    df = df.dropna(subset=["window_start"])
    df = df.loc[
        (df.index.get_level_values("point_time") < df["window_end"])
        & (df.index.get_level_values("point_time") >= df["window_start"])
    ]

    df = df.assign(charge_status=simulate_charge(df, truth_col, charge_mins))
    df = df.assign(
        charge_emissions=df[truth_col] * df["charge_status"] * (load_kw / 1000)
    )

    # baseline: immediate charging rather than AER
    df = df.assign(sequential_rank=df.groupby("window_start").cumcount())
    df = df.assign(baseline_charge_status=df["sequential_rank"] < charge_mins // 5)
    df = df.assign(
        baseline_charge_emissions=df[truth_col]
        * df["baseline_charge_status"]
        * (load_kw / 1000)
    )

    assert (
        df["baseline_charge_status"].sum()
        >= (len(df["window_start"].unique()) - 1) * charge_mins // 5
    )
    assert (
        df["baseline_charge_status"].sum()
        <= (len(df["window_start"].unique())) * charge_mins // 5
    )

    y_best_total = df["charge_emissions"].sum()
    y_baseline_total = df["baseline_charge_emissions"].sum()
    
    assert y_best_total <= y_baseline_total

    potential = (y_baseline_total - y_best_total) / len(set(df.index.date))

    return {"potential": round(potential, 1)}


def plot_max_impact_potential(
    factory: DataHandlerFactory,
    scenarios: List[str] = ["EV-night", "EV-day", "Thermostat"],
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
                    hovertemplate="%{x}: %{y:.1f} lbs CO2/MWh<extra></extra>",
                )
            )

            y_max = max(_metrics["potential"].max(), y_max)

            fig.update_layout(
                height=300 * len(region_models),
                yaxis=dict(
                    title="lbs CO2/MWh", fixedrange=True  # Disable y-axis panning
                ),
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
        merged_moers = merged_moers.sample(n_samples)

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
    "forecast": [plot_norm_mae, plot_rank_corr, plot_impact_forecast_metrics],
}


def generate_report(
    region_list: List[str],
    model_date_list: List[str],
    signal_type: str,
    eval_start: datetime,
    eval_end: datetime,
    output_dir: Path,
    steps: Literal["signal", "fuel_mix", "forecast"] = [
        "signal",
        "forecast",
    ],  # no fuel_mix by default
):
    filename = (
        f"{signal_type}_{'&'.join(region_list)}_{'&'.join(model_date_list)}_model_stats"
    )

    # run notebook
    output_path = output_dir / f"{filename}.html"
    input_template_path = Path(__file__).parent / r"report_card_template.html"

    factory = DataHandlerFactory(
        eval_start=eval_start,
        eval_end=eval_end,
        regions=region_list,
        model_dates=model_date_list,
        signal_types=signal_type,
    )

    plotly_html = {}
    for step in steps:
        for plot_func in PLOTS[step]:
            _plot = plot_func(factory)

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
    )
