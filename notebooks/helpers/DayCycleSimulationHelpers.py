from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from pandas.testing import assert_frame_equal

from day_cycle_simulation.generation import (
    build_scheduler_day_user_table,
    generate_synthetic_day_population,
)
from day_cycle_simulation.load_curve import (
    BITS_PER_GB,
    build_15_minute_target_load_table,
    load_hourly_load_curve,
)


def export_doc_figure(fig, filename: str, doc_img_dir: Path) -> Path:
    """Save one discussion figure into the repository image directory."""
    output_path = Path(doc_img_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    return output_path


def bin_index_to_clock(bin_index: int, *, bin_duration_min: int = 15) -> str:
    """Convert a bin index into a clock label such as 03:30."""
    total_minutes = int(bin_duration_min) * int(bin_index)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def build_catalog_table(config) -> pd.DataFrame:
    """Summarise the active preset catalog in reader-facing terms."""
    rows = []
    for value, weight in zip(config.distance_presets_m, config.distance_weights):
        rows.append(
            {
                "Preset family": "Distance",
                "Preset value": f"{float(value):.0f} m",
                "Sampling weight": float(weight),
                "Interpretation": "Link distance passed to the later radio layers",
            }
        )
    for value, weight in zip(config.total_data_presets_bits, config.total_data_weights):
        rows.append(
            {
                "Preset family": "Total session data",
                "Preset value": f"{float(value) / BITS_PER_GB:.2f} GB",
                "Sampling weight": float(weight),
                "Interpretation": "Whole-session payload before per-bin expansion",
            }
        )
    for value, weight in zip(
        config.nominal_duration_presets_bins,
        config.nominal_duration_weights,
    ):
        rows.append(
            {
                "Preset family": "Nominal session duration",
                "Preset value": f"{int(value)} bin ({15 * int(value)} min)",
                "Sampling weight": float(weight),
                "Interpretation": "Nominal occupancy before the per-bin user table is built",
            }
        )
    return pd.DataFrame(rows)


def build_session_table(sessions, *, bin_duration_s: float) -> pd.DataFrame:
    """Build a compact session table for discussion and plotting."""
    rows = []
    for session in sessions:
        start_bin = int(session.start_bin)
        duration_bins = int(session.nominal_duration_bins)
        stop_bin = start_bin + duration_bins
        total_data_bits = float(session.total_data_bits)
        required_rate_bps = total_data_bits / duration_bins / float(bin_duration_s)
        rows.append(
            {
                "session_id": int(session.session_id),
                "distance_m": float(session.distance_m),
                "total_data_gb": total_data_bits / BITS_PER_GB,
                "start_bin": start_bin,
                "stop_bin": stop_bin,
                "start_time": bin_index_to_clock(start_bin),
                "stop_time": bin_index_to_clock(stop_bin),
                "duration_bins": duration_bins,
                "duration_min": 15 * duration_bins,
                "required_rate_mbps": required_rate_bps / 1e6,
                "traffic_gb_per_bin": total_data_bits / duration_bins / BITS_PER_GB,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["start_bin", "stop_bin", "session_id"])
        .reset_index(drop=True)
    )


def expand_sessions_to_scheduler_rows(sessions, *, bin_duration_s: float) -> pd.DataFrame:
    """Expand each session into one scheduler row per occupied bin."""
    rows = []
    for session in sessions:
        required_rate_bps = (
            float(session.total_data_bits)
            / float(session.nominal_duration_bins)
            / float(bin_duration_s)
        )
        stop_bin = int(session.start_bin) + int(session.nominal_duration_bins)
        for bin_index in range(int(session.start_bin), stop_bin):
            rows.append(
                {
                    "bin_index": int(bin_index),
                    "user_id": int(session.session_id),
                    "distance_m": float(session.distance_m),
                    "required_rate_bps": float(required_rate_bps),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["bin_index", "user_id", "distance_m", "required_rate_bps"],
    )


def assign_session_lanes(session_table: pd.DataFrame) -> pd.DataFrame:
    """Pack overlapping sessions into the lowest free visual lane."""
    lane_stop_bins = []
    lane_indices = []

    for row in session_table.itertuples(index=False):
        assigned_lane = None
        for lane_index, occupied_until in enumerate(lane_stop_bins):
            if int(row.start_bin) >= int(occupied_until):
                assigned_lane = lane_index
                lane_stop_bins[lane_index] = int(row.stop_bin)
                break
        if assigned_lane is None:
            assigned_lane = len(lane_stop_bins)
            lane_stop_bins.append(int(row.stop_bin))
        lane_indices.append(int(assigned_lane))

    lane_table = session_table.copy()
    lane_table["lane_index"] = lane_indices
    return lane_table


def build_realized_mix_table(session_table: pd.DataFrame) -> pd.DataFrame:
    """Summarise the realised session mix after generation."""
    rows = []
    session_count = float(len(session_table))

    for distance_m, count in session_table["distance_m"].value_counts().sort_index().items():
        rows.append(
            {
                "Preset family": "Distance",
                "Realised value": f"{float(distance_m):.0f} m",
                "Session count": int(count),
                "Session share": float(count) / session_count,
            }
        )

    for total_data_gb, count in session_table["total_data_gb"].value_counts().sort_index().items():
        rows.append(
            {
                "Preset family": "Total session data",
                "Realised value": f"{float(total_data_gb):.2f} GB",
                "Session count": int(count),
                "Session share": float(count) / session_count,
            }
        )

    for duration_bins, count in session_table["duration_bins"].value_counts().sort_index().items():
        rows.append(
            {
                "Preset family": "Nominal session duration",
                "Realised value": f"{int(duration_bins)} bin ({15 * int(duration_bins)} min)",
                "Session count": int(count),
                "Session share": float(count) / session_count,
            }
        )

    return pd.DataFrame(rows)


def build_bin_validation_table(
    target_load_table: pd.DataFrame,
    session_table: pd.DataFrame,
    scheduler_day_user_table: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild per-bin load and user counts from the generated sessions."""
    rebuilt_bits = np.zeros(len(target_load_table), dtype=float)
    for row in session_table.itertuples(index=False):
        rebuilt_bits[int(row.start_bin) : int(row.stop_bin)] += (
            float(row.traffic_gb_per_bin) * BITS_PER_GB
        )

    active_users = (
        scheduler_day_user_table.groupby("bin_index")["user_id"]
        .nunique()
        .reindex(range(len(target_load_table)), fill_value=0)
        .to_numpy(dtype=int)
    )

    validation_table = target_load_table.copy()
    validation_table["target_load_gb_in_bin"] = validation_table["target_bits_in_bin"] / BITS_PER_GB
    validation_table["rebuilt_load_gb_in_bin"] = rebuilt_bits / BITS_PER_GB
    validation_table["residual_load_gb_in_bin"] = (
        validation_table["target_load_gb_in_bin"] - validation_table["rebuilt_load_gb_in_bin"]
    )
    validation_table["active_users"] = active_users
    return validation_table


def build_day_cycle_discussion_artifacts(load_curve_csv: Path, config) -> Dict[str, object]:
    """Build the full set of discussion tables used by the notebook."""
    hourly_load_table = load_hourly_load_curve(load_curve_csv)
    target_load_table = build_15_minute_target_load_table(hourly_load_table)
    sessions = generate_synthetic_day_population(config, target_load_table)
    session_table = build_session_table(sessions, bin_duration_s=float(config.bin_duration_s))
    lane_table = assign_session_lanes(session_table)

    manual_scheduler_day_user_table = (
        expand_sessions_to_scheduler_rows(sessions, bin_duration_s=float(config.bin_duration_s))
        .sort_values(["bin_index", "user_id"])
        .reset_index(drop=True)
    )
    scheduler_day_user_table = (
        build_scheduler_day_user_table(load_curve_csv, config)
        .sort_values(["bin_index", "user_id"])
        .reset_index(drop=True)
    )
    assert_frame_equal(scheduler_day_user_table, manual_scheduler_day_user_table)

    catalog_table = build_catalog_table(config)
    realized_mix_table = build_realized_mix_table(session_table)
    bin_validation_table = build_bin_validation_table(
        target_load_table,
        session_table,
        scheduler_day_user_table,
    )

    generator_summary = pd.DataFrame(
        [
            {
                "Load curve file": load_curve_csv.name,
                "Hourly samples": int(len(hourly_load_table)),
                "Quarter-hour bins": int(len(target_load_table)),
                "Synthetic sessions": int(len(session_table)),
                "Scheduler rows": int(len(scheduler_day_user_table)),
                "Max concurrent session lanes": int(lane_table["lane_index"].max() + 1),
            }
        ]
    )

    target_preview = (
        target_load_table.loc[:7, ["bin_index", "hour", "target_bits_in_bin"]]
        .assign(target_load_gb_in_bin=lambda df: df["target_bits_in_bin"] / BITS_PER_GB)
        .drop(columns="target_bits_in_bin")
        .rename(
            columns={
                "bin_index": "Bin index",
                "hour": "Hour",
                "target_load_gb_in_bin": "Target load in bin (GB)",
            }
        )
    )

    population_summary = pd.DataFrame(
        [
            {
                "Synthetic sessions": int(len(session_table)),
                "Mean active users per bin": float(bin_validation_table["active_users"].mean()),
                "Peak active users in one bin": int(bin_validation_table["active_users"].max()),
                "Peak per-user required rate (Mbps)": float(session_table["required_rate_mbps"].max()),
                "Target day traffic (TB)": float(target_load_table["target_bits_in_bin"].sum() / 8e12),
                "Max residual load in bin (GB)": float(bin_validation_table["residual_load_gb_in_bin"].max()),
            }
        ]
    )

    example_session_id = int(
        session_table.loc[session_table["duration_bins"].eq(3)]
        .sort_values(["start_bin", "session_id"])
        .iloc[0]["session_id"]
    )
    example_session_view = session_table.loc[
        session_table["session_id"].eq(example_session_id),
        [
            "session_id",
            "distance_m",
            "total_data_gb",
            "start_time",
            "stop_time",
            "duration_bins",
            "duration_min",
            "required_rate_mbps",
        ],
    ].reset_index(drop=True).rename(
        columns={
            "session_id": "Session ID",
            "distance_m": "Distance (m)",
            "total_data_gb": "Total data (GB)",
            "start_time": "Start time",
            "stop_time": "Stop time",
            "duration_bins": "Duration (bins)",
            "duration_min": "Duration (min)",
            "required_rate_mbps": "Required rate (Mbps)",
        }
    )

    example_scheduler_rows = (
        scheduler_day_user_table.loc[
            scheduler_day_user_table["user_id"].eq(example_session_id)
        ]
        .assign(required_rate_mbps=lambda df: df["required_rate_bps"] / 1e6)
        .drop(columns="required_rate_bps")
        .reset_index(drop=True)
        .rename(
            columns={
                "bin_index": "Bin index",
                "user_id": "User ID",
                "distance_m": "Distance (m)",
                "required_rate_mbps": "Required rate (Mbps)",
            }
        )
    )

    validation_summary = pd.DataFrame(
        [
            {
                "Target day traffic (TB)": float(target_load_table["target_bits_in_bin"].sum() / 8e12),
                "Generated day traffic (TB)": float(bin_validation_table["rebuilt_load_gb_in_bin"].sum() / 1e3),
                "Captured traffic share": float(
                    bin_validation_table["rebuilt_load_gb_in_bin"].sum()
                    / bin_validation_table["target_load_gb_in_bin"].sum()
                ),
                "Max residual load in bin (GB)": float(bin_validation_table["residual_load_gb_in_bin"].max()),
                "Peak active users": int(bin_validation_table["active_users"].max()),
            }
        ]
    )

    session_sample_table = (
        session_table.loc[
            :7,
            [
                "session_id",
                "distance_m",
                "total_data_gb",
                "start_time",
                "stop_time",
                "duration_bins",
                "required_rate_mbps",
            ],
        ]
        .rename(
            columns={
                "session_id": "Session ID",
                "distance_m": "Distance (m)",
                "total_data_gb": "Total data (GB)",
                "start_time": "Start time",
                "stop_time": "Stop time",
                "duration_bins": "Duration (bins)",
                "required_rate_mbps": "Required rate (Mbps)",
            }
        )
        .reset_index(drop=True)
    )

    return {
        "hourly_load_table": hourly_load_table,
        "target_load_table": target_load_table,
        "sessions": sessions,
        "session_table": session_table,
        "lane_table": lane_table,
        "scheduler_day_user_table": scheduler_day_user_table,
        "catalog_table": catalog_table,
        "realized_mix_table": realized_mix_table,
        "bin_validation_table": bin_validation_table,
        "generator_summary": generator_summary,
        "target_preview": target_preview,
        "population_summary": population_summary,
        "session_sample_table": session_sample_table,
        "example_session_view": example_session_view,
        "example_scheduler_rows": example_scheduler_rows,
        "validation_summary": validation_summary,
        "smallest_total_data_gb": float(min(config.total_data_presets_bits) / BITS_PER_GB),
    }


def _format_table_for_display(
    df: pd.DataFrame,
    formats: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return a copy with user-facing formatting applied safely column-wise."""
    display_df = df.copy()
    if not formats:
        return display_df

    def _format_value(value, formatter):
        if pd.isna(value):
            return ""
        if callable(formatter):
            try:
                return formatter(value)
            except Exception:
                return str(value)
        try:
            return formatter.format(value)
        except Exception:
            try:
                return formatter.format(float(value))
            except Exception:
                return str(value)

    for column, formatter in formats.items():
        if column not in display_df.columns:
            continue
        display_df[column] = display_df[column].map(
            lambda value, fmt=formatter: _format_value(value, fmt)
        )

    return display_df


def style_dataframe(
    df: pd.DataFrame,
    *,
    formats: Mapping[str, object] | None = None,
    caption: str | None = None,
):
    """Render a clean captioned HTML table without pandas Styler."""
    from IPython.display import HTML

    display_df = _format_table_for_display(df, formats=formats)
    header_cells = "".join(
        f'<th style="text-align:left; padding:6px 10px; border-bottom:1px solid #999;">{escape(str(column))}</th>'
        for column in display_df.columns
    )
    body_rows = []
    for _, row in display_df.iterrows():
        cells = "".join(
            f'<td style="text-align:left; padding:6px 10px; vertical-align:top;">{escape(str(value))}</td>'
            for value in row.tolist()
        )
        body_rows.append(f"<tr>{cells}</tr>")

    caption_html = (
        f'<caption style="caption-side:top; text-align:left; font-weight:bold; padding-bottom:6px;">{escape(caption)}</caption>'
        if caption
        else ""
    )
    html = (
        '<table style="border-collapse:collapse; width:auto;">'
        f'{caption_html}'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>'
    )
    return HTML(html)


def style_summary_table(df: pd.DataFrame, *, caption: str | None = None):
    formats = {
        column: "{:.3f}" for column in df.columns if "share" in column.lower()
    }
    for column in df.columns:
        lower = column.lower()
        if "weight" in lower:
            formats[column] = "{:.3f}"
        elif "rate" in lower:
            formats[column] = "{:.2f}"
        elif "traffic" in lower or "load" in lower:
            formats[column] = "{:.3f}"
        elif "mean" in lower:
            formats[column] = "{:.2f}"
    return style_dataframe(df, formats=formats, caption=caption)


def style_catalog_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={"Sampling weight": "{:.3f}"},
        caption=caption,
    )


def style_mix_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={"Session share": "{:.3f}"},
        caption=caption,
    )


def style_target_preview_table(df: pd.DataFrame, *, caption: str | None = None):
    return style_dataframe(
        df,
        formats={"Target load in bin (GB)": "{:.3f}"},
        caption=caption,
    )


def style_session_overview_table(df: pd.DataFrame, *, caption: str | None = None):
    formats = {
        "Distance (m)": "{:.0f}",
        "Total data (GB)": "{:.2f}",
        "Required rate (Mbps)": "{:.2f}",
    }
    return style_dataframe(df, formats=formats, caption=caption)


def style_scheduler_rows_table(df: pd.DataFrame, *, caption: str | None = None):
    formats = {
        "Distance (m)": "{:.0f}",
        "Required rate (Mbps)": "{:.2f}",
    }
    return style_dataframe(df, formats=formats, caption=caption)


def _set_bin_boundaries(ax, *, day_bin_count: int):
    for boundary in range(0, int(day_bin_count) + 1, 4):
        ax.axvline(boundary - 0.5, color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.set_xlim(-0.5, int(day_bin_count) - 0.5)
    ax.set_xticks(range(0, int(day_bin_count), 8))


def plot_hourly_offered_load(hourly_load_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        hourly_load_table["hour"],
        hourly_load_table["total_load_gbph"],
        color="#1f4e79",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    ax.fill_between(
        hourly_load_table["hour"],
        hourly_load_table["total_load_gbph"],
        color="#9ecae1",
        alpha=0.35,
    )
    ax.set_xlim(1, 24)
    ax.set_xticks(range(1, 25, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Offered load (GB/h)")
    ax.set_title("Hourly offered load used by the day-cycle generator")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_target_bins(target_load_table: pd.DataFrame, bin_validation_table: pd.DataFrame, *, day_bin_count: int):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(
        target_load_table["bin_index"],
        bin_validation_table["target_load_gb_in_bin"],
        width=0.9,
        color="#4c78a8",
        alpha=0.35,
        label="Target load in each quarter-hour bin",
    )
    ax.step(
        target_load_table["bin_index"],
        bin_validation_table["target_load_gb_in_bin"],
        where="mid",
        color="#1f4e79",
        linewidth=2.0,
        label="Piecewise-constant hourly expansion",
    )
    _set_bin_boundaries(ax, day_bin_count=day_bin_count)
    ax.set_xlabel("Quarter-hour bin index")
    ax.set_ylabel("Target load in bin (GB)")
    ax.set_title("Quarter-hour target load after hourly expansion")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def _build_session_volume_color_map(session_table: pd.DataFrame) -> Dict[float, tuple]:
    volumes = sorted(float(value) for value in session_table["total_data_gb"].unique())
    cmap_positions = np.linspace(0.35, 0.80, max(len(volumes), 2))[: len(volumes)]
    colors = plt.cm.Blues(cmap_positions)
    return {volume: color for volume, color in zip(volumes, colors)}


def plot_session_blocks(lane_table: pd.DataFrame, *, day_bin_count: int):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    color_map = _build_session_volume_color_map(lane_table)

    for row in lane_table.itertuples(index=False):
        rectangle = patches.Rectangle(
            (float(row.start_bin) - 0.5, float(row.lane_index)),
            float(row.duration_bins),
            0.9,
            facecolor=color_map[float(row.total_data_gb)],
            edgecolor="#0f3057",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.add_patch(rectangle)

    _set_bin_boundaries(ax, day_bin_count=day_bin_count)

    legend_handles = [
        patches.Patch(
            facecolor=color_map[volume],
            edgecolor="#0f3057",
            label=f"{volume:.2f} GB",
        )
        for volume in sorted(color_map)
    ]
    ax.legend(handles=legend_handles, title="Total session data", ncol=min(5, len(legend_handles)), loc="upper center")
    ax.set_ylim(-0.2, float(lane_table["lane_index"].max()) + 1.2)
    ax.set_xlabel("Quarter-hour bin index")
    ax.set_ylabel("Visual session lane")
    ax.set_title("Generated session placement across the simulated day")
    ax.grid(False)
    plt.tight_layout()
    return fig, ax


def plot_target_rebuild_and_activity(bin_validation_table: pd.DataFrame, *, day_bin_count: int):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    axes[0].bar(
        bin_validation_table["bin_index"],
        bin_validation_table["target_load_gb_in_bin"],
        width=0.9,
        color="#4c78a8",
        alpha=0.35,
        label="Target load from the quarter-hour bins",
    )
    axes[0].plot(
        bin_validation_table["bin_index"],
        bin_validation_table["rebuilt_load_gb_in_bin"],
        color="#111111",
        linewidth=1.9,
        label="Load rebuilt from generated sessions",
    )
    axes[0].set_ylabel("Load in bin (GB)")
    axes[0].set_title("Generated sessions reproduce the quarter-hour target load")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        bin_validation_table["bin_index"],
        bin_validation_table["active_users"],
        width=0.9,
        color="#dd8452",
        alpha=0.85,
    )
    axes[1].set_xlabel("Quarter-hour bin index")
    axes[1].set_ylabel("Active users")
    axes[1].set_title("Scheduler-facing active users in each quarter-hour bin")
    axes[1].grid(True, axis="y", alpha=0.3)

    for ax in axes:
        _set_bin_boundaries(ax, day_bin_count=day_bin_count)

    plt.tight_layout()
    return fig, axes
