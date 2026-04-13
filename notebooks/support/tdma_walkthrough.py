from __future__ import annotations

"""Lean TDMA walkthrough support layered on top of the production scheduler."""

from collections import Counter
from itertools import product
from math import prod
from types import SimpleNamespace
from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from multi_user_tdma_scheduler.api import prepare_joint_schedule_problem
from multi_user_tdma_scheduler.joint_search import ExactJointScheduleSearch
from multi_user_tdma_scheduler.models import USER_CANDIDATE_COLUMNS
from multi_user_tdma_scheduler.tdma_space import (
    slot_lower_bound,
    validate_single_frame_schedule_feasibility,
)


def _frame_avg_power_series(table: pd.DataFrame, *, frame_n_slots: int) -> pd.Series:
    """Return the frame-averaged PA DC power for each quantized row."""

    return (
        table["n_slots"].astype(float)
        * table["p_dc_active_w"].astype(float)
        / float(frame_n_slots)
    )


def _frame_avg_rate_series(table: pd.DataFrame, *, frame_n_slots: int) -> pd.Series:
    """Return the frame-averaged delivered rate for each quantized row."""

    return (
        table["n_slots"].astype(float)
        * table["rate_active_bps"].astype(float)
        / float(frame_n_slots)
    )


def build_tdma_preparation_artifacts(batch_space: Any) -> SimpleNamespace:
    """Build the notebook TDMA preparation view from the production batch artifact."""

    full_frame_user_spaces = {
        int(user_row.user_id): (
            batch_space.user_parameter_spaces[int(user_row.user_id)][BATCH_USER_PARAMETER_SPACE_COLUMNS]
            .copy()
            .reset_index(drop=True)
        )
        for user_row in batch_space.user_requirements.itertuples(index=False)
    }
    frame_n_slots = validate_single_frame_schedule_feasibility(
        batch_space,
        full_frame_user_spaces,
    )
    quantized_user_spaces = {
        int(user_row.user_id): _quantize_user_rows(
            user_id=int(user_row.user_id),
            required_rate_bps=float(user_row.required_rate_bps),
            active_table=full_frame_user_spaces[int(user_row.user_id)],
            frame_n_slots=int(frame_n_slots),
        )
        for user_row in batch_space.user_requirements.itertuples(index=False)
    }
    prepared_problem = prepare_joint_schedule_problem(batch_space)
    annotated_user_spaces = {
        int(user_id): _annotate_quantized_rows(
            quantized_table,
            prepared_problem.user_candidate_spaces[int(user_id)],
            frame_n_slots=int(frame_n_slots),
        )
        for user_id, quantized_table in quantized_user_spaces.items()
    }
    exact_frame_share_lower_bound = _frame_share_lower_bound(
        batch_space,
        full_frame_user_spaces=full_frame_user_spaces,
    )

    return SimpleNamespace(
        frame_n_slots=int(frame_n_slots),
        exact_frame_share_lower_bound=float(exact_frame_share_lower_bound),
        exact_slot_lower_bound=int(
            slot_lower_bound(
                batch_space,
                full_frame_user_spaces,
                int(frame_n_slots),
            )
        ),
        full_frame_user_spaces=full_frame_user_spaces,
        quantized_user_spaces=quantized_user_spaces,
        annotated_user_spaces=annotated_user_spaces,
        prepared_problem=prepared_problem,
    )


def plot_scheduler_input_spaces(
    batch_space: Any,
    user_table: pd.DataFrame,
    *,
    user_color_map: dict[int, str],
    pa_color_map: dict[int, str],
    pa_label_map: dict[int, str],
    full_frontiers_by_user: dict[int, pd.DataFrame] | None = None,
):
    """Plot the lookup-stage full-frame candidate menus passed into TDMA prep."""

    user_ids = user_table["user_id"].astype(int).tolist()
    n_cols = 2 if len(user_ids) > 2 else max(len(user_ids), 1)
    n_rows = int(np.ceil(len(user_ids) / max(n_cols, 1)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.3 * n_rows),
        sharey=True,
    )
    flat_axes = np.atleast_1d(axes).ravel()
    pa_ids = sorted(pa_label_map)
    marker_map = _build_pa_marker_map(pa_ids)
    max_power_w = max(
        float(
            _plot_source_table_for_user(
                int(user_id),
                batch_space=batch_space,
                full_frontiers_by_user=full_frontiers_by_user,
            )["p_dc_active_w"].max()
        )
        for user_id in user_ids
    )

    for ax, user_row in zip(flat_axes, user_table.itertuples(index=False), strict=False):
        user_id = int(user_row.user_id)
        candidate_table = _sort_full_frame_menu_rows(batch_space.user_parameter_spaces[user_id].copy())
        if full_frontiers_by_user is not None:
            _scatter_pa_rows(
                ax,
                _sort_full_frame_menu_rows(full_frontiers_by_user[user_id].copy()),
                x_column="rate_active_bps",
                y_column="p_dc_active_w",
                x_scale=1e6,
                pa_color_map=pa_color_map,
                marker_map=marker_map,
                size=24,
                alpha=0.18,
            )
        _scatter_pa_rows(
            ax,
            candidate_table,
            x_column="rate_active_bps",
            y_column="p_dc_active_w",
            x_scale=1e6,
            pa_color_map=pa_color_map,
            marker_map=marker_map,
            size=40,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.axvline(
            float(user_row.required_rate_bps) / 1e6,
            color=str(user_color_map[user_id]),
            linestyle="--",
            linewidth=1.6,
        )
        ax.set_title(
            f"User {user_id}\n{float(user_row.distance_m):.0f} m, {float(user_row.required_rate_bps) / 1e6:.1f} Mbps",
            color=str(user_color_map[user_id]),
            fontsize=11,
        )
        ax.set_xlabel("Full-frame active rate (Mbps)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, max_power_w * 1.12)
        ax.grid(True, alpha=0.22)
        for spine in ax.spines.values():
            spine.set_edgecolor(str(user_color_map[user_id]))
            spine.set_linewidth(1.3)

    for ax in flat_axes[len(user_ids):]:
        ax.set_visible(False)

    flat_axes[0].set_ylabel("Active PA DC input power (W)")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=marker_map[int(pa_id)],
                color="w",
                markerfacecolor=pa_color_map[int(pa_id)],
                markeredgecolor="white",
                markersize=8,
                label=pa_label_map[int(pa_id)],
            )
            for pa_id in pa_ids
        ]
        + [
            Line2D([0], [0], color="#4b5563", linestyle="--", linewidth=1.6, label="Required rate")
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(pa_ids) + 1, 3),
        frameon=True,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig, axes


def build_joint_search_trace(
    annotated_user_spaces: dict[int, pd.DataFrame],
    problem: Any,
) -> SimpleNamespace:
    """Build notebook-only branch-and-bound trace artifacts for the exact search."""

    search = _InstrumentedExactJointScheduleSearch(problem)
    result = search.run()
    user_ids = sorted(int(user_id) for user_id in annotated_user_spaces)
    quantized_counts = {
        int(user_id): int(len(annotated_user_spaces[int(user_id)]))
        for user_id in user_ids
    }
    prepared_counts = {
        int(user_id): int(len(problem.user_candidate_spaces[int(user_id)]))
        for user_id in user_ids
    }
    ranked_counts = {
        int(user_id): int(len(search.ranked_user_rows[int(user_id)]))
        for user_id in search.user_order
    }
    depth_rows = [
        _build_depth_row(
            depth=int(depth),
            user_id=int(user_id),
            search=search,
        )
        for depth, user_id in enumerate(search.user_order)
    ]

    return SimpleNamespace(
        result=result,
        user_ids=user_ids,
        user_order=[int(user_id) for user_id in search.user_order],
        quantized_counts=quantized_counts,
        prepared_counts=prepared_counts,
        ranked_counts=ranked_counts,
        quantized_joint_cases=int(prod(int(count) for count in quantized_counts.values())),
        prepared_joint_cases=int(prod(int(count) for count in prepared_counts.values())),
        ranked_joint_cases=int(prod(int(count) for count in ranked_counts.values())),
        search_stats=dict(search.search_stats),
        depth_table=pd.DataFrame(depth_rows),
    )


def build_joint_allocation_examples(problem: Any, schedule_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the alternate-feasible schedule examples shown next to the optimum."""

    optimal_summary = {
        "rows": sorted(schedule_result["rows"], key=lambda row: int(row["user_id"])),
        "slot_total": int(schedule_result["slot_total"]),
        "unused_slots": int(schedule_result["unused_slots"]),
        "total_rate_bps": float(schedule_result["total_rate_bps"]),
        "schedule_p_dc_total_avg_frame_w": float(schedule_result["schedule_p_dc_total_avg_frame_w"]),
        "feasible": True,
    }
    lowest_power_summary = _summarize_schedule_rows(
        [
            candidate_table.assign(
                schedule_cost=_frame_avg_power_series(
                    candidate_table,
                    frame_n_slots=int(problem.frame_n_slots),
                )
            )
            .sort_values(["schedule_cost", "n_slots", "mcs", "n_prb"])
            .iloc[0]
            .to_dict()
            for candidate_table in problem.user_candidate_spaces.values()
        ],
        frame_n_slots=int(problem.frame_n_slots),
    )
    infeasible_summary = lowest_power_summary
    infeasible_label = "User-local low-power choices"
    if infeasible_summary["feasible"]:
        infeasible_summary = _summarize_schedule_rows(
            [
                candidate_table.sort_values(
                    ["n_slots", "p_dc_active_w", "mcs", "n_prb"],
                    ascending=[False, True, True, True],
                )
                .iloc[0]
                .to_dict()
                for candidate_table in problem.user_candidate_spaces.values()
            ],
            frame_n_slots=int(problem.frame_n_slots),
        )
        infeasible_label = "A joint choice that overfills the frame"

    comparison_schedule = _minimum_slot_schedule(problem)
    comparison_label = "Minimum-slot feasible allocation"
    if comparison_schedule is not None and _same_schedule_rows(
        comparison_schedule["rows"],
        optimal_summary["rows"],
    ):
        comparison_schedule = None
    if comparison_schedule is None:
        comparison_schedule = _best_distinct_feasible_schedule(problem, optimal_summary)
        if comparison_schedule is not None:
            comparison_label = _comparison_schedule_label(comparison_schedule, optimal_summary)
    if comparison_schedule is None:
        comparison_schedule = _best_single_swap_schedule(problem, optimal_summary)
        if comparison_schedule is not None:
            comparison_label = _comparison_schedule_label(comparison_schedule, optimal_summary)

    cases: list[dict[str, Any]] = []
    if not infeasible_summary["feasible"]:
        cases.append({"label": infeasible_label, "summary": infeasible_summary})
    if comparison_schedule is not None and not _same_schedule_rows(
        comparison_schedule["rows"],
        optimal_summary["rows"],
    ):
        cases.append({"label": comparison_label, "summary": comparison_schedule})
    cases.append({"label": "Optimal allocation", "summary": optimal_summary})
    return cases


def _frame_share_lower_bound(batch_space: Any, *, full_frame_user_spaces: dict[int, pd.DataFrame]) -> float:
    return float(
        sum(
            float(user_row.required_rate_bps)
            / float(full_frame_user_spaces[int(user_row.user_id)]["rate_active_bps"].max())
            for user_row in batch_space.user_requirements.itertuples(index=False)
        )
    )


def _quantize_user_rows(
    *,
    user_id: int,
    required_rate_bps: float,
    active_table: pd.DataFrame,
    frame_n_slots: int,
) -> pd.DataFrame:
    if active_table.empty:
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    rate_active_bps = active_table["rate_active_bps"].astype(float).to_numpy()
    required_slots = np.ceil(
        float(frame_n_slots) * float(required_rate_bps) / rate_active_bps - 1e-12
    ).astype(int)
    feasible_mask = (
        (rate_active_bps > 0.0)
        & (required_slots >= 1)
        & (required_slots <= int(frame_n_slots))
    )
    if not np.any(feasible_mask):
        return pd.DataFrame(columns=USER_CANDIDATE_COLUMNS)

    quantized_table = active_table.loc[
        feasible_mask,
        BATCH_USER_PARAMETER_SPACE_COLUMNS,
    ].copy().reset_index(drop=True)
    quantized_table["user_id"] = int(user_id)
    quantized_table["n_slots"] = required_slots[feasible_mask]
    return quantized_table[USER_CANDIDATE_COLUMNS].copy()


def _annotate_quantized_rows(
    quantized_table: pd.DataFrame,
    kept_table: pd.DataFrame,
    *,
    frame_n_slots: int,
) -> pd.DataFrame:
    if quantized_table.empty:
        return quantized_table.assign(
            pruning_role=pd.Series(dtype=str),
            delivered_rate_bps=pd.Series(dtype=float),
            p_dc_avg_frame_w=pd.Series(dtype=float),
        )

    compare_columns = list(quantized_table.columns)
    quantized_view = quantized_table.copy().reset_index(drop=True)
    quantized_view["_dup_rank"] = quantized_view.groupby(compare_columns, dropna=False).cumcount()
    kept_view = kept_table.copy().reset_index(drop=True)
    if kept_view.empty:
        kept_lookup = pd.DataFrame(columns=compare_columns + ["_dup_rank", "pruning_role"])
    else:
        kept_view["_dup_rank"] = kept_view.groupby(compare_columns, dropna=False).cumcount()
        kept_lookup = kept_view[compare_columns + ["_dup_rank"]].copy()
        kept_lookup["pruning_role"] = "kept"

    annotated = quantized_view.merge(
        kept_lookup,
        on=compare_columns + ["_dup_rank"],
        how="left",
    )
    annotated["pruning_role"] = annotated["pruning_role"].fillna("dominated")
    annotated["delivered_rate_bps"] = _frame_avg_rate_series(
        annotated,
        frame_n_slots=int(frame_n_slots),
    )
    annotated["p_dc_avg_frame_w"] = _frame_avg_power_series(
        annotated,
        frame_n_slots=int(frame_n_slots),
    )
    return annotated.drop(columns=["_dup_rank"]).reset_index(drop=True)


def _build_depth_row(*, depth: int, user_id: int, search: "_InstrumentedExactJointScheduleSearch") -> dict[str, int]:
    depth_stats = dict(search.depth_stats[int(depth)])
    suffix_combo_count = (
        int(
            prod(
                len(search.ranked_user_rows[int(remaining_user_id)])
                for remaining_user_id in search.user_order[int(depth) + 1 :]
            )
        )
        if int(depth) + 1 < len(search.user_order)
        else 1
    )
    pruned_row_total = int(depth_stats["rows_considered"] - depth_stats["rows_recurse"])
    return {
        "depth": int(depth),
        "user_id": int(user_id),
        "rows_considered": int(depth_stats["rows_considered"]),
        "rows_recurse": int(depth_stats["rows_recurse"]),
        "pruned_time_direct": int(depth_stats["pruned_time_direct"]),
        "pruned_power_direct": int(depth_stats["pruned_power_direct"]),
        "pruned_time_bound": int(depth_stats["pruned_time_bound"]),
        "pruned_power_bound": int(depth_stats["pruned_power_bound"]),
        "pruned_rank_bound": int(depth_stats["pruned_rank_bound"]),
        "pruned_row_total": int(pruned_row_total),
        "suffix_combo_count": int(suffix_combo_count),
        "pruned_suffix_completions": int(pruned_row_total * suffix_combo_count),
    }


def _build_pa_marker_map(pa_ids: list[int]) -> dict[int, str]:
    markers = ["o", "s", "^", "D", "P", "X"]
    return {int(pa_id): markers[idx % len(markers)] for idx, pa_id in enumerate(sorted(pa_ids))}


def _plot_source_table_for_user(
    user_id: int,
    *,
    batch_space: Any,
    full_frontiers_by_user: dict[int, pd.DataFrame] | None,
) -> pd.DataFrame:
    if full_frontiers_by_user is None:
        return batch_space.user_parameter_spaces[int(user_id)]
    return full_frontiers_by_user[int(user_id)]


def _sort_full_frame_menu_rows(candidate_table: pd.DataFrame) -> pd.DataFrame:
    return candidate_table.sort_values(["rate_active_bps", "p_dc_active_w", "pa_id", "mcs", "n_prb"])


def _scatter_pa_rows(
    ax,
    candidate_table: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    pa_color_map: dict[int, str],
    marker_map: dict[int, str],
    size: float,
    alpha: float,
    x_scale: float = 1.0,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    color_override: str | None = None,
) -> None:
    if candidate_table.empty:
        return

    for pa_id, pa_rows in candidate_table.groupby("pa_id", sort=True):
        resolved_color = (
            str(color_override)
            if color_override is not None
            else str(pa_color_map[int(pa_id)])
        )
        ax.scatter(
            pa_rows[x_column].astype(float) / float(x_scale),
            pa_rows[y_column].astype(float),
            s=size,
            alpha=alpha,
            color=resolved_color,
            marker=marker_map[int(pa_id)],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )


def _summarize_schedule_rows(rows: list[dict[str, Any]], *, frame_n_slots: int) -> dict[str, Any]:
    sorted_rows = [dict(row) for row in sorted(rows, key=lambda row: int(row["user_id"]))]
    slot_total = int(sum(int(row["n_slots"]) for row in sorted_rows))
    total_rate_bps = float(sum(_row_frame_avg_rate(row, frame_n_slots) for row in sorted_rows))
    total_power_w = float(sum(_row_frame_avg_power(row, frame_n_slots) for row in sorted_rows))
    return {
        "rows": sorted_rows,
        "slot_total": int(slot_total),
        "unused_slots": int(frame_n_slots - slot_total),
        "total_rate_bps": float(total_rate_bps),
        "schedule_p_dc_total_avg_frame_w": float(total_power_w),
        "feasible": bool(slot_total <= int(frame_n_slots)),
    }


def _best_single_swap_schedule(problem: Any, optimal_summary: dict[str, Any]) -> dict[str, Any] | None:
    selection_lookup = {
        int(row["user_id"]): dict(row)
        for row in optimal_summary["rows"]
    }
    optimal_rank = (
        float(optimal_summary["schedule_p_dc_total_avg_frame_w"]),
        int(optimal_summary["slot_total"]),
        -float(optimal_summary["total_rate_bps"]),
    )
    best_summary = None
    best_rank = None

    for user_id, candidate_table in problem.user_candidate_spaces.items():
        base_rows = [dict(selection_lookup[int(selected_user_id)]) for selected_user_id in sorted(selection_lookup)]
        for candidate_row in candidate_table.to_dict("records"):
            if _same_schedule_rows([candidate_row], [selection_lookup[int(user_id)]]):
                continue

            swapped_rows = [
                dict(candidate_row) if int(row["user_id"]) == int(user_id) else dict(row)
                for row in base_rows
            ]
            summary = _summarize_schedule_rows(
                swapped_rows,
                frame_n_slots=int(problem.frame_n_slots),
            )
            if not summary["feasible"]:
                continue

            rank = (
                float(summary["schedule_p_dc_total_avg_frame_w"]),
                int(summary["slot_total"]),
                -float(summary["total_rate_bps"]),
            )
            if rank <= optimal_rank:
                continue
            if best_rank is None or rank < best_rank:
                best_summary = summary
                best_rank = rank

    return best_summary


def _minimum_slot_schedule(problem: Any) -> dict[str, Any] | None:
    """Return one feasible schedule that minimizes total allocated slots."""

    frame_n_slots = int(problem.frame_n_slots)
    user_ids = sorted(int(user_id) for user_id in problem.user_candidate_spaces)
    if not user_ids:
        return None

    schedule_by_slot_total: dict[int, tuple[float, list[dict[str, Any]]]] = {0: (0.0, [])}
    for user_id in user_ids:
        next_schedule_by_slot_total: dict[int, tuple[float, list[dict[str, Any]]]] = {}
        for slot_total, (total_power_w, selected_rows) in schedule_by_slot_total.items():
            for candidate_row in problem.user_candidate_spaces[int(user_id)].to_dict("records"):
                next_slot_total = int(slot_total + int(candidate_row["n_slots"]))
                if next_slot_total > frame_n_slots:
                    continue

                next_total_power_w = float(
                    total_power_w + _row_frame_avg_power(candidate_row, frame_n_slots)
                )
                current_best = next_schedule_by_slot_total.get(next_slot_total)
                if current_best is not None and float(current_best[0]) <= next_total_power_w + 1e-12:
                    continue

                next_schedule_by_slot_total[next_slot_total] = (
                    next_total_power_w,
                    [*selected_rows, dict(candidate_row)],
                )

        schedule_by_slot_total = next_schedule_by_slot_total
        if not schedule_by_slot_total:
            return None

    minimum_slot_total = min(schedule_by_slot_total)
    return _summarize_schedule_rows(
        schedule_by_slot_total[int(minimum_slot_total)][1],
        frame_n_slots=frame_n_slots,
    )


def _best_distinct_feasible_schedule(problem: Any, optimal_summary: dict[str, Any]) -> dict[str, Any] | None:
    user_ids = sorted(int(user_id) for user_id in problem.user_candidate_spaces)
    if not user_ids:
        return None

    candidate_pools = {
        int(user_id): _candidate_pool_for_visual_comparison(
            problem.user_candidate_spaces[int(user_id)],
            frame_n_slots=int(problem.frame_n_slots),
            max_slot_options=6,
        )
        for user_id in user_ids
    }
    if any(len(pool) == 0 for pool in candidate_pools.values()):
        return None

    min_changed_users = min(max(len(user_ids) - 1, 2), len(user_ids))
    feasible_summaries: list[tuple[tuple[float, int, float, int], dict[str, Any]]] = []
    fallback_summaries: list[tuple[tuple[float, int, float, int], dict[str, Any]]] = []

    for selected_rows in product(*(candidate_pools[user_id] for user_id in user_ids)):
        summary = _summarize_schedule_rows(
            [dict(row) for row in selected_rows],
            frame_n_slots=int(problem.frame_n_slots),
        )
        if not summary["feasible"]:
            continue
        if float(summary["schedule_p_dc_total_avg_frame_w"]) <= float(optimal_summary["schedule_p_dc_total_avg_frame_w"]) + 1e-12:
            continue

        changed_users = sum(
            _row_signature(row) != _row_signature(optimal_row)
            for row, optimal_row in zip(summary["rows"], optimal_summary["rows"], strict=False)
        )
        if changed_users == 0:
            continue

        slot_diff = sum(
            abs(int(row["n_slots"]) - int(optimal_row["n_slots"]))
            for row, optimal_row in zip(summary["rows"], optimal_summary["rows"], strict=False)
        )
        rank = (
            float(summary["schedule_p_dc_total_avg_frame_w"]),
            -int(changed_users),
            -int(slot_diff),
            int(summary["slot_total"]),
        )
        if changed_users >= min_changed_users and slot_diff >= 2:
            feasible_summaries.append((rank, summary))
            continue
        if slot_diff >= 1:
            fallback_summaries.append((rank, summary))

    if feasible_summaries:
        feasible_summaries.sort(key=lambda item: item[0])
        return feasible_summaries[0][1]
    if fallback_summaries:
        fallback_summaries.sort(key=lambda item: item[0])
        return fallback_summaries[0][1]
    return None


def _candidate_pool_for_visual_comparison(
    candidate_table: pd.DataFrame,
    *,
    frame_n_slots: int,
    max_slot_options: int,
) -> list[dict[str, Any]]:
    if candidate_table.empty:
        return []

    working_table = candidate_table.copy()
    working_table["frame_avg_power_w"] = _frame_avg_power_series(
        working_table,
        frame_n_slots=int(frame_n_slots),
    )
    best_rows = [
        slot_rows.sort_values(["frame_avg_power_w", "mcs", "n_prb", "layers"])
        .iloc[0]
        .drop(labels=["frame_avg_power_w"])
        .to_dict()
        for _, slot_rows in working_table.groupby("n_slots", sort=True)
    ]
    best_rows.sort(
        key=lambda row: (
            int(row["n_slots"]),
            float(int(row["n_slots"]) * float(row["p_dc_active_w"]) / float(frame_n_slots)),
            int(row["mcs"]),
            int(row["n_prb"]),
        )
    )
    return best_rows[: int(max_slot_options)]


def _comparison_schedule_label(
    comparison_summary: dict[str, Any],
    optimal_summary: dict[str, Any],
) -> str:
    if int(comparison_summary["slot_total"]) < int(optimal_summary["slot_total"]):
        return "Feasible but more bursty allocation"
    if int(comparison_summary["slot_total"]) > int(optimal_summary["slot_total"]):
        return "Feasible but less bursty allocation"
    return "Feasible but higher-power allocation"


def _row_signature(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row["pa_id"]),
        int(row["n_prb"]),
        int(row["layers"]),
        int(row["mcs"]),
        int(row["n_slots"]),
    )


def _same_schedule_rows(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> bool:
    return Counter(tuple(sorted(row.items())) for row in left_rows) == Counter(
        tuple(sorted(row.items())) for row in right_rows
    )


def _row_frame_avg_power(row: pd.Series | dict[str, Any], frame_n_slots: int) -> float:
    return float(int(row["n_slots"]) * float(row["p_dc_active_w"]) / float(frame_n_slots))


def _row_frame_avg_rate(row: pd.Series | dict[str, Any], frame_n_slots: int) -> float:
    return float(int(row["n_slots"]) * float(row["rate_active_bps"]) / float(frame_n_slots))


class _InstrumentedExactJointScheduleSearch(ExactJointScheduleSearch):
    """Notebook-only wrapper that records a compact exact-search trace."""

    def __init__(self, problem: Any):
        super().__init__(problem)
        self.depth_stats = {
            depth: {
                "rows_considered": 0,
                "rows_recurse": 0,
                "pruned_time_direct": 0,
                "pruned_power_direct": 0,
                "pruned_time_bound": 0,
                "pruned_power_bound": 0,
                "pruned_rank_bound": 0,
            }
            for depth in range(len(self.user_order))
        }

    def _on_depth_row_considered(self, *, depth: int) -> None:
        self.depth_stats[int(depth)]["rows_considered"] += 1

    def _on_depth_prune(self, *, depth: int, reason: str) -> None:
        self.depth_stats[int(depth)][str(reason)] += 1

    def _on_depth_recurse(self, *, depth: int) -> None:
        self.depth_stats[int(depth)]["rows_recurse"] += 1


__all__ = [
    "build_joint_allocation_examples",
    "build_joint_search_trace",
    "build_tdma_preparation_artifacts",
    "plot_scheduler_input_spaces",
]
