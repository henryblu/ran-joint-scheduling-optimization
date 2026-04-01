"""Helper functions for the TDMA scheduling walkthrough notebook.

The notebook uses these helpers to keep the teaching flow focused on the
scheduler logic rather than on plotting and table formatting details.
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, patches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from configs import USER_REQUIREMENT_COLUMNS, build_pa_characteristics_table
from models import PASwitchPolicy, UserRequest


def build_user_requirements_table(users: Iterable[UserRequest]) -> pd.DataFrame:
    """Materialize the scheduler-ready user table used throughout the notebook."""

    rows = [
        {
            "user_id": int(user.user_id),
            "distance_m": float(user.distance_m),
            "required_rate_bps": float(user.required_rate_bps),
        }
        for user in users
    ]
    return pd.DataFrame(rows, columns=USER_REQUIREMENT_COLUMNS)


def to_mbps(value_bps: float) -> float:
    return float(value_bps) / 1e6



def watts_to_dbm(power_w: Any) -> np.ndarray:
    power_w = np.asarray(power_w, dtype=float)
    return 10.0 * np.log10(np.clip(power_w, 1e-12, None) * 1e3)



def build_request_group_map(user_table: pd.DataFrame) -> dict[int, int]:
    request_group_map: dict[tuple[float, float], int] = {}
    next_group = 1
    user_group_lookup: dict[int, int] = {}

    for row in user_table.itertuples(index=False):
        key = (float(row.distance_m), float(row.required_rate_bps))
        if key not in request_group_map:
            request_group_map[key] = next_group
            next_group += 1
        user_group_lookup[int(row.user_id)] = int(request_group_map[key])

    return user_group_lookup



def build_pa_maps(pa_catalog_or_problem: Any) -> tuple[pd.DataFrame, dict[int, str], dict[int, str]]:
    pa_table = build_pa_characteristics_table(pa_catalog_or_problem).copy()
    pa_label_map = {
        int(row.pa_id): str(row.scenario_label)
        for row in pa_table.itertuples(index=False)
    }
    base_colors = ["#c75d2c", "#0b7a75", "#4e79a7", "#8e6c8a"]
    pa_color_map = {
        pa_id: base_colors[idx % len(base_colors)]
        for idx, pa_id in enumerate(sorted(pa_label_map))
    }
    return pa_table, pa_label_map, pa_color_map



def summarize_batch_user_spaces(
    batch_space: Any,
    user_table: pd.DataFrame,
    pa_label_map: dict[int, str],
    request_group_lookup: dict[int, int],
) -> pd.DataFrame:
    rows = []
    for user_row in user_table.itertuples(index=False):
        user_id = int(user_row.user_id)
        candidate_table = batch_space.user_parameter_spaces[user_id].copy()
        best_row = candidate_table.nsmallest(1, "p_dc_active_w").iloc[0]
        rows.append(
            {
                "user_id": user_id,
                "request_group": int(request_group_lookup[user_id]),
                "candidate_count": int(len(candidate_table)),
                "min_rate_mbps": to_mbps(candidate_table["rate_active_bps"].min()),
                "max_rate_mbps": to_mbps(candidate_table["rate_active_bps"].max()),
                "min_active_power_w": float(candidate_table["p_dc_active_w"].min()),
                "max_active_power_w": float(candidate_table["p_dc_active_w"].max()),
                "lowest_power_pa": str(pa_label_map[int(best_row["pa_id"])]),
            }
        )
    return pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)



def summarize_prepared_problem(
    problem: Any,
    user_table: pd.DataFrame,
    pa_label_map: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for user_row in user_table.itertuples(index=False):
        user_id = int(user_row.user_id)
        candidate_table = problem.user_candidate_spaces[user_id].copy()
        best_row = candidate_table.nsmallest(1, "p_dc_avg_frame_w").iloc[0]
        rows.append(
            {
                "user_id": user_id,
                "candidate_count": int(len(candidate_table)),
                "min_slots": int(candidate_table["n_slots"].min()),
                "max_slots": int(candidate_table["n_slots"].max()),
                "lowest_power_slots": int(best_row["n_slots"]),
                "min_avg_power_w": float(candidate_table["p_dc_avg_frame_w"].min()),
                "max_avg_rate_mbps": to_mbps(candidate_table["rate_avg_frame_bps"].max()),
                "lowest_power_pa": str(pa_label_map[int(best_row["pa_id"])]),
            }
        )
    return pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)



def summarize_policy_result(
    policy_name: str,
    result: Any,
    user_table: pd.DataFrame,
    pa_label_map: dict[int, str],
) -> dict[str, Any]:
    schedule = result.best_schedule
    row_table = pd.DataFrame(schedule["rows"]).copy()
    active_power_w = float(row_table["p_dc_avg_frame_w"].sum())
    inactive_power_w = float(schedule["schedule_p_dc_total_avg_frame_w"] - active_power_w)
    used_pa_labels = ", ".join(
        sorted(pa_label_map[int(pa_id)] for pa_id in row_table["pa_id"].unique())
    )
    return {
        "switch_policy": str(policy_name),
        "solve_time_s": float(result.search_stats.get("solve_time_s", np.nan)),
        "total_power_w": float(schedule["schedule_p_dc_total_avg_frame_w"]),
        "active_power_w": active_power_w,
        "inactive_power_w": inactive_power_w,
        "slot_total": int(schedule["slot_total"]),
        "unused_slots": int(schedule["unused_slots"]),
        "total_rate_mbps": to_mbps(schedule["total_rate_bps"]),
        "requested_rate_mbps": to_mbps(user_table["required_rate_bps"].sum()),
        "used_pas": used_pa_labels,
    }



def build_policy_user_table(
    policy_name: str,
    result: Any,
    user_table: pd.DataFrame,
    pa_label_map: dict[int, str],
) -> pd.DataFrame:
    required_rate_lookup = {
        int(row.user_id): float(row.required_rate_bps)
        for row in user_table.itertuples(index=False)
    }
    rows = []
    for row in sorted(result.best_schedule["rows"], key=lambda item: int(item["user_id"])):
        user_id = int(row["user_id"])
        rows.append(
            {
                "switch_policy": str(policy_name),
                "user_id": user_id,
                "pa_label": str(pa_label_map[int(row["pa_id"])]),
                "n_slots": int(row["n_slots"]),
                "n_prb": int(row["n_prb"]),
                "layers": int(row["layers"]),
                "mcs": int(row["mcs"]),
                "required_rate_mbps": to_mbps(required_rate_lookup[user_id]),
                "achieved_rate_mbps": to_mbps(row["rate_avg_frame_bps"]),
                "avg_frame_power_w": float(row["p_dc_avg_frame_w"]),
            }
        )
    return pd.DataFrame(rows)



def build_schedule_blocks(
    schedule_result: dict[str, Any],
    problem: Any,
    pa_label_map: dict[int, str],
) -> dict[str, Any]:
    schedule_rows = sorted(schedule_result["rows"], key=lambda item: int(item["user_id"]))
    user_ids = [int(row["user_id"]) for row in schedule_rows]
    color_levels = np.linspace(0.25, 0.85, max(len(user_ids), 1))
    user_color_map = {
        user_id: colors.to_hex(plt.cm.cividis(level))
        for user_id, level in zip(user_ids, color_levels[::-1], strict=False)
    }

    blocks = []
    slot_cursor = 0
    for row in schedule_rows:
        block = {
            "user_id": int(row["user_id"]),
            "pa_label": str(pa_label_map[int(row["pa_id"])]),
            "n_prb": int(row["n_prb"]),
            "n_slots": int(row["n_slots"]),
            "layers": int(row["layers"]),
            "mcs": int(row["mcs"]),
            "p_dc_avg_frame_w": float(row["p_dc_avg_frame_w"]),
            "slot_start": int(slot_cursor),
            "slot_end": int(slot_cursor + int(row["n_slots"])),
            "color": str(user_color_map[int(row["user_id"])]),
        }
        blocks.append(block)
        slot_cursor = block["slot_end"]

    total_prbs = int(max(block["n_prb"] for block in blocks))
    total_slots = int(problem.window_n_slots)
    frame_slots = int(problem.window_n_slots // problem.window_n_frames)

    unused_blocks = []
    for block in blocks:
        unused_prbs = total_prbs - int(block["n_prb"])
        if unused_prbs > 0:
            unused_blocks.append(
                {
                    "x": int(block["n_prb"]),
                    "y": int(block["slot_start"]),
                    "width": int(unused_prbs),
                    "height": int(block["n_slots"]),
                }
            )
    if slot_cursor < total_slots:
        unused_blocks.append(
            {
                "x": 0,
                "y": int(slot_cursor),
                "width": int(total_prbs),
                "height": int(total_slots - slot_cursor),
            }
        )

    return {
        "blocks": blocks,
        "total_prbs": total_prbs,
        "total_slots": total_slots,
        "frame_slots": frame_slots,
        "window_boundaries": list(range(frame_slots, total_slots, frame_slots)),
        "unused_blocks": unused_blocks,
    }



def build_power_grid(allocation_view: dict[str, Any]) -> np.ndarray:
    grid = np.full((allocation_view["total_slots"], allocation_view["total_prbs"]), np.nan)
    for block in allocation_view["blocks"]:
        grid[block["slot_start"]:block["slot_end"], :block["n_prb"]] = block["p_dc_avg_frame_w"]
    return grid



def build_mcs_grid(allocation_view: dict[str, Any]) -> np.ndarray:
    grid = np.full((allocation_view["total_slots"], allocation_view["total_prbs"]), np.nan)
    for block in allocation_view["blocks"]:
        grid[block["slot_start"]:block["slot_end"], :block["n_prb"]] = block["mcs"]
    return grid



def plot_user_batch_overview(
    user_table: pd.DataFrame,
    request_group_lookup: dict[int, int],
) -> None:
    batch_view = user_table.copy()
    batch_view["request_group"] = batch_view["user_id"].map(request_group_lookup)
    batch_view["required_rate_mbps"] = batch_view["required_rate_bps"] / 1e6
    group_counts = batch_view.groupby("request_group")["user_id"].count().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    group_palette = plt.cm.Set2(np.linspace(0.15, 0.85, max(len(group_counts), 1)))
    group_color_map = {
        int(group_id): group_palette[idx]
        for idx, group_id in enumerate(group_counts.index.tolist())
    }
    bar_colors = [group_color_map[int(group)] for group in batch_view["request_group"]]

    axes[0].bar(batch_view["user_id"].astype(int), batch_view["required_rate_mbps"], color=bar_colors)
    for row in batch_view.itertuples(index=False):
        axes[0].text(
            int(row.user_id),
            float(row.required_rate_mbps) + 0.45,
            f"{float(row.distance_m):.0f} m\nGroup {int(row.request_group)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_xlabel("User id")
    axes[0].set_ylabel("Requested average rate (Mbps)")
    axes[0].set_title("Scheduler-ready demand for the tutorial batch")
    axes[0].set_xticks(batch_view["user_id"].astype(int).tolist())
    axes[0].set_ylim(0.0, float(batch_view["required_rate_mbps"].max()) * 1.25)

    axes[1].bar(group_counts.index.astype(int), group_counts.values, color=[group_color_map[int(i)] for i in group_counts.index])
    axes[1].set_xlabel("Request group id")
    axes[1].set_ylabel("Users sharing the same request")
    axes[1].set_title("Repeated request groups can reuse one single-user solve")
    axes[1].set_xticks(group_counts.index.astype(int).tolist())
    axes[1].set_ylim(0.0, max(group_counts.max() + 0.75, 1.5))

    fig.tight_layout()
    plt.show()



def plot_full_frame_candidate_spaces(
    batch_space: Any,
    user_table: pd.DataFrame,
    request_group_lookup: dict[int, int],
    pa_color_map: dict[int, str],
    pa_label_map: dict[int, str],
) -> None:
    batch_view = user_table.copy()
    batch_view["request_group"] = batch_view["user_id"].map(request_group_lookup).astype(int)
    unique_group_rows = batch_view.drop_duplicates(subset="request_group").reset_index(drop=True)

    fig, axes = plt.subplots(
        1,
        len(unique_group_rows),
        figsize=(5.5 * len(unique_group_rows), 4.5),
        sharey=True,
    )
    if len(unique_group_rows) == 1:
        axes = [axes]

    max_active_power_w = max(
        (
            float(batch_space.user_parameter_spaces[int(user_row.user_id)]["p_dc_active_w"].max())
            for user_row in unique_group_rows.itertuples(index=False)
            if not batch_space.user_parameter_spaces[int(user_row.user_id)].empty
        ),
        default=0.0,
    )
    y_axis_upper_w = max(max_active_power_w * 1.08, 1.0)

    for ax, user_row in zip(axes, unique_group_rows.itertuples(index=False), strict=False):
        user_id = int(user_row.user_id)
        candidate_table = batch_space.user_parameter_spaces[user_id].copy()
        required_rate_mbps = to_mbps(user_row.required_rate_bps)

        for pa_id, pa_rows in candidate_table.groupby("pa_id", sort=True):
            pa_rows = pa_rows.copy()
            ax.scatter(
                pa_rows["rate_active_bps"] / 1e6,
                pa_rows["p_dc_active_w"],
                s=8,
                alpha=0.10,
                color=pa_color_map[int(pa_id)],
            )
            best_feasible_row = (
                pa_rows[pa_rows["rate_active_bps"] >= float(user_row.required_rate_bps)]
                .nsmallest(1, "p_dc_active_w")
            )
            if not best_feasible_row.empty:
                ax.scatter(
                    best_feasible_row["rate_active_bps"] / 1e6,
                    best_feasible_row["p_dc_active_w"],
                    s=150,
                    marker="*",
                    color=pa_color_map[int(pa_id)],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )

        ax.axvline(required_rate_mbps, color="black", linestyle="--", linewidth=1.1)
        ax.set_title(
            f"User {user_id} (group {request_group_lookup[user_id]})\n"
            f"{float(user_row.distance_m):.0f} m, {required_rate_mbps:.0f} Mbps"
        )
        ax.set_xlabel("Full-frame active rate (Mbps)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, y_axis_upper_w)

    axes[0].set_ylabel("Full-frame active PA DC power (W)")
    fig.suptitle("Full-frame operating points that enter the TDMA layer", y=1.04)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=pa_color_map[pa_id],
            markersize=7,
            label=pa_label_map[pa_id],
        )
        for pa_id in sorted(pa_label_map)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="Required average rate",
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#bbbbbb",
            markeredgecolor="black",
            markersize=12,
            label="Cheapest feasible row inside one PA family",
        )
    )
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=True)
    fig.tight_layout()
    plt.show()



def plot_space_reduction(
    batch_user_space_summary: pd.DataFrame,
    prepared_problem_summary: pd.DataFrame,
) -> None:
    merged = batch_user_space_summary.merge(
        prepared_problem_summary,
        on="user_id",
        suffixes=("_full_frame", "_tdma"),
    )
    user_labels = [f"User {int(user_id)}" for user_id in merged["user_id"]]
    x = np.arange(len(merged))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    axes[0].bar(x - width / 2, merged["candidate_count_full_frame"], width=width, label="Full-frame points")
    axes[0].bar(x + width / 2, merged["candidate_count_tdma"], width=width, label="TDMA candidate rows")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(user_labels)
    axes[0].set_ylabel("Candidate rows")
    axes[0].set_title("Window resolution and pruning shrink each user space")
    axes[0].legend(frameon=True)

    y_positions = np.arange(len(merged), 0, -1)
    for y, row in zip(y_positions, merged.itertuples(index=False), strict=False):
        axes[1].hlines(y, row.min_slots, row.max_slots, linewidth=5)
        axes[1].plot(row.lowest_power_slots, y, marker="o", markersize=9, color="black")
        axes[1].text(row.max_slots + 0.08, y, f"best row: {int(row.lowest_power_slots)} slots", va="center")
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(user_labels)
    axes[1].set_xlabel("Allocated slot count in the resolved window")
    axes[1].set_ylabel("User")
    axes[1].set_title("Each user is reduced to a finite slot range")
    axes[1].set_xlim(left=0.0)

    fig.tight_layout()
    plt.show()



def plot_policy_comparison(
    policy_summary_view: pd.DataFrame,
    schedule_choice_view: pd.DataFrame,
    policy_order: list[str],
) -> None:
    policy_plot_summary = policy_summary_view.set_index("switch_policy").loc[policy_order]
    slot_plot = (
        schedule_choice_view.pivot(index="user_id", columns="switch_policy", values="n_slots")
        .sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].bar(
        policy_order,
        policy_plot_summary["active_power_w"],
        color="#0b7a75",
        label="Active contribution",
    )
    axes[0].bar(
        policy_order,
        policy_plot_summary["inactive_power_w"],
        bottom=policy_plot_summary["active_power_w"],
        color="#c75d2c",
        label="Inactive contribution",
    )
    for idx, policy_name in enumerate(policy_order):
        total_power_w = float(policy_plot_summary.loc[policy_name, "total_power_w"])
        axes[0].text(idx, total_power_w + 0.05, f"{total_power_w:.2f} W", ha="center", va="bottom")
    axes[0].set_ylabel("Average-frame PA DC power (W)")
    axes[0].set_title("The policy changes the objective, not the demand")
    axes[0].legend(frameon=True)

    x = np.arange(len(slot_plot.index))
    width = 0.34
    axes[1].bar(
        x - width / 2,
        slot_plot[PASwitchPolicy.STANDBY.value],
        width=width,
        color="#4e79a7",
        label=PASwitchPolicy.STANDBY.value,
    )
    axes[1].bar(
        x + width / 2,
        slot_plot[PASwitchPolicy.HARD_OFF.value],
        width=width,
        color="#f28e2b",
        label=PASwitchPolicy.HARD_OFF.value,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"User {int(user_id)}" for user_id in slot_plot.index])
    axes[1].set_ylabel("Allocated slot count")
    axes[1].set_title("Hard-off tends to use a more bursty slot pattern")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    plt.show()



def plot_selected_tdma_rows(
    problem: Any,
    user_table: pd.DataFrame,
    policy_results: dict[str, Any],
    schedule_choice_view: pd.DataFrame,
    pa_color_map: dict[int, str],
    pa_label_map: dict[int, str],
) -> None:
    user_ids = user_table["user_id"].astype(int).tolist()
    fig, axes = plt.subplots(1, len(user_ids), figsize=(5.5 * len(user_ids), 4.5), sharey=True)
    if len(user_ids) == 1:
        axes = [axes]

    selection_markers = {
        PASwitchPolicy.STANDBY.value: "o",
        PASwitchPolicy.HARD_OFF.value: "s",
    }

    for ax, user_row in zip(axes, user_table.itertuples(index=False), strict=False):
        user_id = int(user_row.user_id)
        candidate_table = problem.user_candidate_spaces[user_id].copy()
        for pa_id, pa_rows in candidate_table.groupby("pa_id", sort=True):
            ax.scatter(
                pa_rows["n_slots"],
                pa_rows["p_dc_avg_frame_w"],
                s=30,
                alpha=0.55,
                color=pa_color_map[int(pa_id)],
            )

        for policy_name, marker in selection_markers.items():
            selected_row = schedule_choice_view[
                (schedule_choice_view["switch_policy"] == policy_name)
                & (schedule_choice_view["user_id"] == user_id)
            ].iloc[0]
            pa_id = int(
                pd.DataFrame(policy_results[policy_name].best_schedule["rows"])
                .set_index("user_id")
                .loc[user_id, "pa_id"]
            )
            ax.scatter(
                selected_row["n_slots"],
                selected_row["avg_frame_power_w"],
                s=150,
                marker=marker,
                color=pa_color_map[pa_id],
                edgecolor="black",
                linewidth=1.2,
                zorder=3,
            )

        ax.set_title(
            f"User {user_id}\n{float(user_row.distance_m):.0f} m, {to_mbps(user_row.required_rate_bps):.0f} Mbps"
        )
        ax.set_xlabel("Allocated slot count in the resolved window")
        ax.set_xticks(sorted(candidate_table["n_slots"].astype(int).unique()))
        ax.set_ylim(bottom=0.0)

    axes[0].set_ylabel("Average-frame PA DC power of one row (W)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=pa_color_map[pa_id],
            markersize=8,
            label=pa_label_map[pa_id],
        )
        for pa_id in sorted(pa_label_map)
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker=selection_markers[PASwitchPolicy.STANDBY.value],
                color="black",
                markerfacecolor="white",
                linestyle="None",
                markersize=9,
                label="standby winner",
            ),
            Line2D(
                [0],
                [0],
                marker=selection_markers[PASwitchPolicy.HARD_OFF.value],
                color="black",
                markerfacecolor="white",
                linestyle="None",
                markersize=9,
                label="hard_off winner",
            ),
        ]
    )
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=True)
    fig.tight_layout()
    plt.show()



def add_time_frequency_overlays(ax: Any, allocation_view: dict[str, Any], *, show_legend: bool = False) -> None:
    for boundary in allocation_view["window_boundaries"]:
        ax.axhline(boundary, color="#9e9e9e", linewidth=1.2, linestyle=":")

    for block in allocation_view["unused_blocks"]:
        unused_rect = patches.Rectangle(
            (block["x"], block["y"]),
            block["width"],
            block["height"],
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            linestyle="--",
        )
        ax.add_patch(unused_rect)

    for block in sorted(allocation_view["blocks"], key=lambda item: int(item["user_id"]), reverse=True):
        rect = patches.Rectangle(
            (0, block["slot_start"]),
            block["n_prb"],
            block["n_slots"],
            fill=False,
            edgecolor=block["color"],
            linewidth=2.6,
        )
        ax.add_patch(rect)

    if show_legend:
        legend_handles = [
            patches.Patch(
                facecolor="none",
                edgecolor=block["color"],
                linewidth=2.6,
                label=f"User {block['user_id']} allocation",
            )
            for block in allocation_view["blocks"]
        ]
        legend_handles.append(
            patches.Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=2.0,
                linestyle="--",
                label="Unused resources",
            )
        )
        if allocation_view["window_boundaries"]:
            legend_handles.append(
                patches.Patch(
                    facecolor="none",
                    edgecolor="#9e9e9e",
                    linewidth=1.2,
                    linestyle=":",
                    label="Frame boundary",
                )
            )
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)



def cuboid(ax: Any, x: int, y: int, z: int, dx: int, dy: int, dz: int, color: str) -> None:
    vertices = np.array(
        [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ]
    )
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]
    poly = Poly3DCollection(faces, facecolors=color, edgecolor="black", alpha=0.45)
    ax.add_collection3d(poly)



def plot_3d_schedule(allocation_view: dict[str, Any], problem: Any) -> None:
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    for block in allocation_view["blocks"]:
        cuboid(
            ax,
            block["slot_start"],
            0,
            0,
            block["n_slots"],
            block["n_prb"],
            block["layers"],
            block["color"],
        )

    for boundary in allocation_view["window_boundaries"]:
        ax.plot(
            [boundary, boundary],
            [0, allocation_view["total_prbs"]],
            [0, 0],
            color="#9e9e9e",
            linestyle=":",
            linewidth=1.2,
        )

    ax.set_xlabel("Slot index within the resolved window", labelpad=12)
    ax.set_ylabel("PRB index within the allocated band", labelpad=14)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Active spatial layers", rotation=90, labelpad=18)
    ax.set_xlim(0, allocation_view["total_slots"])
    ax.set_ylim(allocation_view["total_prbs"], 0)
    ax.set_zlim(0, problem.n_tx_chains)
    ax.set_xticks(np.arange(0, allocation_view["total_slots"] + 1, 1))
    ax.set_yticks(np.arange(0, allocation_view["total_prbs"] + 1, 50))
    ax.set_zticks(np.arange(0, problem.n_tx_chains + 1, 1))
    ax.set_title("One valid packing of the standby schedule on the resource grid")
    ax.view_init(elev=22, azim=-62)
    plt.tight_layout()
    plt.show()



def plot_power_heatmaps(
    allocation_views: dict[str, dict[str, Any]],
    policy_order: list[str],
    policy_summary_view: pd.DataFrame,
) -> None:
    power_grids_dbm = {}
    active_power_dbm_values = []
    for policy_name, allocation_view in allocation_views.items():
        power_grid_w = build_power_grid(allocation_view)
        power_grid_dbm = np.where(
            np.isfinite(power_grid_w),
            watts_to_dbm(power_grid_w),
            np.nan,
        )
        power_grids_dbm[policy_name] = power_grid_dbm
        active_power_dbm_values.append(power_grid_dbm[np.isfinite(power_grid_dbm)])

    finite_values = np.concatenate(active_power_dbm_values)
    vmin = float(np.floor(finite_values.min()) - 1.0)
    vmax = float(np.ceil(finite_values.max()) + 1.0)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    power_cmap = plt.cm.inferno.copy()
    power_cmap.set_bad(color="#efefef")

    fig, axes = plt.subplots(1, len(policy_order), figsize=(13, 5), sharey=True)
    if len(policy_order) == 1:
        axes = [axes]

    policy_totals = policy_summary_view.set_index("switch_policy")
    for ax, policy_name in zip(axes, policy_order, strict=False):
        allocation_view = allocation_views[policy_name]
        image = ax.imshow(
            power_grids_dbm[policy_name],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=power_cmap,
            extent=(0, allocation_view["total_prbs"], 0, allocation_view["total_slots"]),
            vmin=vmin,
            vmax=vmax,
        )
        add_time_frequency_overlays(
            ax,
            allocation_view,
            show_legend=(policy_name == policy_order[-1]),
        )
        ax.set_title(
            f"{policy_name}\n{policy_totals.loc[policy_name, 'total_power_w']:.2f} W total"
        )
        ax.set_xlabel("PRB index within the allocated band")
        ax.set_xticks(np.arange(0, allocation_view["total_prbs"] + 1, 50))
        ax.set_xticks(np.arange(0, allocation_view["total_prbs"] + 1, 5), minor=True)
        ax.set_yticks(np.arange(0, allocation_view["total_slots"] + 1, 1))
        ax.grid(which="minor", axis="x", linewidth=0.35, color="white", alpha=0.35)
        ax.grid(which="major", axis="y", linewidth=0.55, color="white", alpha=0.55)

    axes[0].set_ylabel("Slot index within the resolved window")
    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("Selected row average-frame PA DC power [dBm]")
    fig.suptitle("The chosen rows projected onto the time-frequency grid", y=1.02)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    plt.show()



def plot_mcs_heatmaps(
    allocation_views: dict[str, dict[str, Any]],
    policy_order: list[str],
    policy_summary_view: pd.DataFrame,
) -> None:
    mcs_cmap = colors.ListedColormap(plt.cm.plasma(np.linspace(0.05, 0.95, 29)))
    mcs_cmap.set_bad(color="#efefef")
    mcs_norm = colors.BoundaryNorm(np.arange(-0.5, 29.5, 1.0), mcs_cmap.N)

    fig, axes = plt.subplots(1, len(policy_order), figsize=(13, 5), sharey=True)
    if len(policy_order) == 1:
        axes = [axes]

    policy_totals = policy_summary_view.set_index("switch_policy")
    for ax, policy_name in zip(axes, policy_order, strict=False):
        allocation_view = allocation_views[policy_name]
        mcs_grid = build_mcs_grid(allocation_view)
        image = ax.imshow(
            mcs_grid,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=mcs_cmap,
            norm=mcs_norm,
            extent=(0, allocation_view["total_prbs"], 0, allocation_view["total_slots"]),
        )
        add_time_frequency_overlays(
            ax,
            allocation_view,
            show_legend=(policy_name == policy_order[-1]),
        )
        ax.set_title(
            f"{policy_name}\n{policy_totals.loc[policy_name, 'slot_total']} used slots"
        )
        ax.set_xlabel("PRB index within the allocated band")
        ax.set_xticks(np.arange(0, allocation_view["total_prbs"] + 1, 50))
        ax.set_xticks(np.arange(0, allocation_view["total_prbs"] + 1, 5), minor=True)
        ax.set_yticks(np.arange(0, allocation_view["total_slots"] + 1, 1))
        ax.grid(which="minor", axis="x", linewidth=0.35, color="white", alpha=0.35)
        ax.grid(which="major", axis="y", linewidth=0.55, color="white", alpha=0.55)

    axes[0].set_ylabel("Slot index within the resolved window")
    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("MCS index")
    cbar.set_ticks(np.arange(0, 29, 2))
    fig.suptitle("The same schedule geometry, colored by the chosen MCS", y=1.02)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    plt.show()
