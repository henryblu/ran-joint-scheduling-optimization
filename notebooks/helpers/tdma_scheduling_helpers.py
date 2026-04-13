from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string
import sys
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors, patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from multi_user_tdma_scheduler.api import prepare_joint_schedule_problem, run_joint_schedule_search

from .DayCycleSimulationHelpers import build_day_cycle_discussion_artifacts
from .candidate_space_helpers import export_doc_figure
from .table_lookup_helpers import (
    build_cached_batch_user_parameter_space,
    build_table_lookup_artifacts,
    load_cached_distance_binned_table,
    pick_example_scheduler_bin,
)
from .visual_identity import (
    NotebookTheme,
    apply_3d_axis_style,
    build_color_cycle,
    create_themed_figure,
    get_notebook_theme,
    render_html_table,
    style_colorbar,
    style_legend,
)


@dataclass(frozen=True)
class TdmaBinView:
    """Compact cross-bin schedule view used by the final notebook section."""

    label: str
    user_count: int
    requested_rate_mbps: float
    schedule_power_w: float
    slot_total: int
    allocation_view: dict[str, Any]
    problem: Any


@dataclass(frozen=True)
class TdmaSchedulingArtifacts:
    """Lean notebook payload for the TDMA scheduling walkthrough."""

    example_bin_index: int
    user_table: pd.DataFrame
    pa_label_map: dict[int, str]
    pa_color_map: dict[int, str]
    user_color_map: dict[int, str]
    annotated_user_spaces: dict[int, pd.DataFrame]
    problem: Any
    trace_artifacts: SimpleNamespace
    optimal_result: Any
    comparison_cases: list[dict[str, Any]]
    selected_allocation_view: dict[str, Any]
    bin_views: tuple[TdmaBinView, ...]


class TdmaSchedulingHelpers:
    """Theme-aware presentation helpers for Notebook 4."""

    def __init__(self, *, theme: str | NotebookTheme = "aalto_elec"):
        self.theme = get_notebook_theme(theme)

    def build_artifacts(
        self,
        *,
        load_curve_csv: Path,
        day_cycle_config,
        target_user_count: int = 4,
    ) -> TdmaSchedulingArtifacts:
        """Build the compact TDMA scheduling views used in Notebook 4.

        Steps:
        1. Reuse one active scheduler bin from the day-level demand artifact.
        2. Convert the cached full-frame lookup rows into the prepared TDMA search problem.
        3. Solve the worked bin and assemble the lighter-versus-heavier bin comparison used at the end.
        """

        distance_binned_table = load_cached_distance_binned_table()
        day_artifacts = build_day_cycle_discussion_artifacts(
            Path(load_curve_csv),
            day_cycle_config,
        )
        scheduler_day_user_table = day_artifacts["scheduler_day_user_table"].copy()

        example_context = self._build_example_bin_context(
            scheduler_day_user_table=scheduler_day_user_table,
            distance_binned_table=distance_binned_table,
            target_user_count=int(target_user_count),
        )
        pa_color_map = self._build_pa_color_map(example_context["pa_label_map"])
        user_color_map = self._build_user_color_map(
            example_context["user_table"]["user_id"].astype(int).tolist()
        )
        selected_allocation_view = build_schedule_blocks(
            example_context["optimal_result"].best_schedule,
            example_context["problem"],
            example_context["pa_label_map"],
            user_color_map=user_color_map,
        )

        return TdmaSchedulingArtifacts(
            example_bin_index=int(example_context["example_bin_index"]),
            user_table=example_context["user_table"].copy(),
            pa_label_map=dict(example_context["pa_label_map"]),
            pa_color_map=pa_color_map,
            user_color_map=user_color_map,
            annotated_user_spaces={
                int(user_id): user_space.copy()
                for user_id, user_space in example_context["annotated_user_spaces"].items()
            },
            problem=example_context["problem"],
            trace_artifacts=example_context["trace_artifacts"],
            optimal_result=example_context["optimal_result"],
            comparison_cases=list(example_context["comparison_cases"]),
            selected_allocation_view=selected_allocation_view,
            bin_views=self._build_bin_views(
                scheduler_day_user_table=scheduler_day_user_table,
                distance_binned_table=distance_binned_table,
                example_bin_index=int(example_context["example_bin_index"]),
            ),
        )

    def render_table(
        self,
        df: pd.DataFrame,
        *,
        formats: dict[str, object] | None = None,
        caption: str | None = None,
    ):
        """Render one notebook table with the active visual identity."""

        return render_html_table(
            df,
            theme=self.theme,
            formats=formats,
            caption=caption,
        )

    def plot_workflow(
        self,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the notebook-level flow from cached lookup rows to the exact TDMA solve."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(11.8, 2.8),
        )
        ax.set_axis_off()

        boxes = [
            (
                0.03,
                "Batch artifact",
                self.theme.neutral_light,
                self.theme.neutral_dark,
            ),
            (
                0.29,
                "Quantize rows\nonto one frame",
                self._with_alpha(self.theme.primary, 0.12),
                self.theme.primary,
            ),
            (
                0.55,
                "Keep cheapest row\nper slots and PA",
                self._with_alpha(self.theme.highlight, 0.22),
                self.theme.accent,
            ),
            (
                0.81,
                "Exact joint\nsearch",
                self._with_alpha(self.theme.secondary, 0.12),
                self.theme.secondary,
            ),
        ]

        for x0, label, facecolor, edgecolor in boxes:
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x0, 0.22),
                    0.16,
                    0.54,
                    boxstyle="round,pad=0.02,rounding_size=0.04",
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.3,
                )
            )
            ax.text(
                x0 + 0.08,
                0.49,
                label,
                ha="center",
                va="center",
                fontsize=10.8,
                color=self.theme.text,
            )

        for start, end in ((0.19, 0.29), (0.45, 0.55), (0.71, 0.81)):
            ax.annotate(
                "",
                xy=(end - 0.01, 0.49),
                xytext=(start + 0.01, 0.49),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.6,
                    "color": self.theme.neutral_dark,
                },
            )

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_search_space_summary(
        self,
        artifacts: TdmaSchedulingArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Show how local TDMA pruning reduces the per-user search tables."""

        trace_artifacts = artifacts.trace_artifacts
        user_ids = [
            int(user_id)
            for user_id in artifacts.user_table["user_id"].astype(int).tolist()
        ]
        stage_specs = [
            (
                "(a)",
                trace_artifacts.quantized_counts,
                trace_artifacts.quantized_joint_cases,
            ),
            (
                "(b)",
                trace_artifacts.prepared_counts,
                trace_artifacts.prepared_joint_cases,
            ),
        ]

        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=1,
            ncols=2,
            figsize=(10.8, 4.6),
            sharey=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()
        max_count = max(
            max((int(stage_counts[int(user_id)]) for user_id in user_ids), default=0)
            for _, stage_counts, _ in stage_specs
        )
        y_upper = max(max_count * 1.18, 1.0)

        for ax, (panel_title, stage_counts, _) in zip(
            flat_axes,
            stage_specs,
            strict=False,
        ):
            x_positions = np.arange(len(user_ids))
            bar_values = [int(stage_counts[int(user_id)]) for user_id in user_ids]
            ax.bar(
                x_positions,
                bar_values,
                color=[artifacts.user_color_map[int(user_id)] for user_id in user_ids],
                edgecolor=self.theme.background,
                linewidth=0.9,
                alpha=0.92,
            )
            for idx, value in enumerate(bar_values):
                ax.text(
                    x_positions[idx],
                    float(value) + y_upper * 0.025,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=9.6,
                    color=self.theme.text,
                )

            ax.set_xticks(x_positions, [f"U{int(user_id)}" for user_id in user_ids])
            ax.set_ylim(0.0, y_upper)
            ax.set_xlabel("Active user")
            ax.text(
                0.5,
                -0.18,
                panel_title,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                color=self.theme.text,
            )

        flat_axes[0].set_ylabel("Rows per user")
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        self._export_figure(fig, export_path)
        return fig, flat_axes

    def plot_quantized_user_spaces(
        self,
        artifacts: TdmaSchedulingArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot the per-user TDMA menus after local pruning."""

        user_ids = artifacts.user_table["user_id"].astype(int).tolist()
        n_cols = 2 if len(user_ids) > 2 else max(len(user_ids), 1)
        n_rows = int(np.ceil(len(user_ids) / max(n_cols, 1)))
        frame_n_slots = int(artifacts.problem.frame_n_slots)
        max_power_w = max(
            float(user_space["p_dc_active_w"].max())
            for user_space in artifacts.annotated_user_spaces.values()
            if not user_space.empty
        )
        marker_cycle = ["o", "s", "^", "D", "P", "X"]
        marker_map = {
            int(pa_id): marker_cycle[idx % len(marker_cycle)]
            for idx, pa_id in enumerate(sorted(artifacts.pa_label_map))
        }

        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=n_rows,
            ncols=n_cols,
            figsize=(5.3 * n_cols, 4.2 * n_rows),
            sharey=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for panel_index, (ax, user_row) in enumerate(
            zip(
                flat_axes,
                artifacts.user_table.itertuples(index=False),
                strict=False,
            )
        ):
            row_index = panel_index // n_cols
            col_index = panel_index % n_cols
            user_id = int(user_row.user_id)
            user_color = str(artifacts.user_color_map[user_id])
            user_space = (
                artifacts.annotated_user_spaces[user_id]
                .copy()
                .sort_values(["n_slots", "p_dc_active_w", "pa_id", "mcs", "n_prb"])
            )
            pruned_rows = user_space.loc[user_space["pruning_role"].ne("kept")].copy()
            kept_rows = user_space.loc[user_space["pruning_role"].eq("kept")].copy()

            if not pruned_rows.empty:
                self._scatter_pa_rows(
                    ax,
                    pruned_rows,
                    x_column="n_slots",
                    y_column="p_dc_active_w",
                    pa_color_map=artifacts.pa_color_map,
                    marker_map=marker_map,
                    size=38,
                    alpha=0.95,
                    color_override=self.theme.neutral_light,
                    edgecolor=self.theme.grid,
                    linewidth=0.7,
                )
            self._scatter_pa_rows(
                ax,
                kept_rows,
                x_column="n_slots",
                y_column="p_dc_active_w",
                pa_color_map=artifacts.pa_color_map,
                marker_map=marker_map,
                size=56,
                alpha=0.92,
                edgecolor=self.theme.background,
                linewidth=0.7,
            )

            ax.text(
                0.03,
                0.97,
                self._panel_tag(panel_index),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.8,
                color=user_color,
                weight="bold",
            )
            ax.set_xlabel("Allocated slots" if row_index == n_rows - 1 else "")
            ax.set_ylabel("Active PA DC power (W)" if col_index == 0 else "")
            ax.set_xlim(-0.25, frame_n_slots + 0.25)
            ax.set_xticks(np.arange(0, frame_n_slots + 1, 1))
            ax.set_ylim(0.0, max(max_power_w * 1.12, 1.0))
            for spine_name in ("left", "bottom"):
                ax.spines[spine_name].set_color(user_color)
                ax.spines[spine_name].set_linewidth(1.2)

        for ax in flat_axes[len(user_ids):]:
            ax.set_visible(False)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_map[int(pa_id)],
                color="w",
                markerfacecolor=artifacts.pa_color_map[int(pa_id)],
                markeredgecolor=self.theme.background,
                markersize=8,
                label=artifacts.pa_label_map[int(pa_id)],
            )
            for pa_id in sorted(artifacts.pa_label_map)
        ]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.theme.neutral_light,
                markeredgecolor=self.theme.grid,
                markersize=8,
                label="Pruned row",
            )
        )
        legend = fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=min(len(legend_handles), 4),
            frameon=True,
        )
        style_legend(legend, theme=self.theme)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        self._export_figure(fig, export_path)
        return fig, flat_axes

    def plot_branch_and_bound_trace(
        self,
        artifacts: TdmaSchedulingArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Show the user-by-user search order of the exact solver."""

        depth_table = (
            artifacts.trace_artifacts.depth_table
            .copy()
            .sort_values("depth")
            .reset_index(drop=True)
        )
        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.6, 3.6),
        )
        ax.set_axis_off()

        depth_count = max(len(depth_table), 1)
        box_width = min(0.18, 0.78 / depth_count)
        x_positions = np.linspace(0.05, 0.95 - box_width, depth_count)
        for idx, depth_row in enumerate(depth_table.itertuples(index=False)):
            user_id = int(depth_row.user_id)
            search_rows = int(
                artifacts.trace_artifacts.ranked_counts.get(user_id, depth_row.rows_considered)
            )
            box_spec = (x_positions[idx], 0.26, box_width, 0.5)
            x0, y0, width, height = box_spec
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x0, y0),
                    width,
                    height,
                    boxstyle="round,pad=0.015,rounding_size=0.02",
                    facecolor=self._with_alpha(artifacts.user_color_map[user_id], 0.14),
                    edgecolor=artifacts.user_color_map[user_id],
                    linewidth=1.5,
                )
            )
            ax.text(
                x0 + width / 2.0,
                y0 + height * 0.73,
                f"Depth {int(depth_row.depth)}",
                ha="center",
                va="center",
                fontsize=10.1,
                weight="bold",
                color=self.theme.text,
            )
            ax.text(
                x0 + width / 2.0,
                y0 + height * 0.34,
                (
                    f"User {user_id}\n"
                    f"{search_rows} candidate rows\n"
                    f"{int(depth_row.depth) + 1} user"
                    f"{'' if int(depth_row.depth) == 0 else 's'} fixed"
                ),
                ha="center",
                va="center",
                fontsize=9.2,
                color=self.theme.text,
            )
            if idx == 0:
                continue
            ax.annotate(
                "",
                xy=(x_positions[idx] - 0.012, 0.51),
                xytext=(x_positions[idx] - 0.055, 0.51),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.5,
                    "color": self.theme.neutral_dark,
                },
            )

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_joint_search_case_metrics(
        self,
        artifacts: TdmaSchedulingArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Compare the worked joint schedules by slot consumption and power."""

        frame_n_slots = int(artifacts.problem.frame_n_slots)
        case_rows = []
        case_order = []
        for case in artifacts.comparison_cases:
            case_label = str(case["label"])
            case_order.append(case_label)
            for row in sorted(case["summary"]["rows"], key=lambda item: int(item["user_id"])):
                case_rows.append(
                    {
                        "case_label": case_label,
                        "user_id": int(row["user_id"]),
                        "frame_avg_power_w": float(
                            int(row["n_slots"]) * float(row["p_dc_active_w"]) / float(frame_n_slots)
                        ),
                        "n_slots": int(row["n_slots"]),
                    }
                )
        case_table = pd.DataFrame(case_rows)
        user_ids = sorted(case_table["user_id"].astype(int).unique().tolist())

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(9.3, 5.5),
        )
        slot_table = (
            case_table.pivot(index="case_label", columns="user_id", values="n_slots")
            .reindex(index=case_order, columns=user_ids)
            .fillna(0)
        )
        running_bottom = np.zeros(len(case_order), dtype=float)
        x_positions = np.arange(len(case_order))
        for user_id in user_ids:
            user_values = slot_table[int(user_id)].to_numpy(dtype=float)
            ax.bar(
                x_positions,
                user_values,
                bottom=running_bottom,
                color=artifacts.user_color_map[int(user_id)],
                edgecolor=self.theme.background,
                linewidth=0.8,
                label=f"User {int(user_id)}",
            )
            running_bottom = running_bottom + user_values

        total_power_by_case = (
            case_table.groupby("case_label")["frame_avg_power_w"]
            .sum()
            .reindex(case_order)
            .fillna(0.0)
        )
        total_slots_by_case = slot_table.sum(axis=1)
        y_upper = max(
            float(max(total_slots_by_case.max(), frame_n_slots) * 1.16),
            1.0,
        )
        for idx, case_label in enumerate(case_order):
            ax.text(
                x_positions[idx],
                float(total_slots_by_case.loc[case_label]) + y_upper * 0.02,
                (
                    f"{float(total_power_by_case.loc[case_label]):.3f} W\n"
                    f"{int(total_slots_by_case.loc[case_label])} slots"
                ),
                ha="center",
                va="bottom",
                fontsize=9.2,
                color=self.theme.text,
            )

        ax.axhline(
            float(frame_n_slots),
            color=self.theme.neutral_dark,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.set_xticks(
            x_positions,
            [
                str(label)
                .replace(" feasible allocation", "\nfeasible allocation")
                .replace("Optimal allocation", "Optimal\nallocation")
                for label in case_order
            ],
        )
        ax.set_ylabel("Consumed slots")
        ax.set_ylim(0.0, y_upper)
        ax.set_yticks(
            np.arange(
                0,
                int(np.ceil(y_upper)) + 1,
                _slot_tick_step(int(np.ceil(y_upper))),
            )
        )
        legend = ax.legend(loc="upper right", frameon=True)
        style_legend(legend, theme=self.theme)

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_selected_schedule(
        self,
        artifacts: TdmaSchedulingArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, Any]:
        """Plot the selected joint allocation on the 3D time-frequency-layer grid."""

        cmap = self._build_power_colormap()
        color_norm = _power_norm_from_blocks(artifacts.selected_allocation_view["blocks"])
        fig = plt.figure(figsize=(10.4, 6.6))
        fig.patch.set_facecolor(self.theme.background)
        ax = fig.add_subplot(111, projection="3d")
        _plot_schedule_3d_on_axis(
            ax,
            artifacts.selected_allocation_view,
            problem=artifacts.problem,
            color_norm=color_norm,
            cmap=cmap,
        )
        apply_3d_axis_style(ax, theme=self.theme)

        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=color_norm, cmap=cmap),
            ax=ax,
            fraction=0.04,
            pad=0.08,
        )
        colorbar.set_label("Selected-row active power (W)")
        style_colorbar(colorbar, theme=self.theme)
        fig.subplots_adjust(left=0.04, right=0.88, bottom=0.06, top=0.94)

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_bin_transition(
        self,
        artifacts: TdmaSchedulingArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, list[Any]]:
        """Show how the optimal 3D allocation changes between two day bins."""

        if not artifacts.bin_views:
            raise ValueError("artifacts.bin_views must contain at least one schedule view.")

        cmap = self._build_power_colormap()
        all_blocks = [
            block
            for bin_view in artifacts.bin_views
            for block in bin_view.allocation_view["blocks"]
        ]
        color_norm = _power_norm_from_blocks(all_blocks)

        fig = plt.figure(figsize=(6.2 * len(artifacts.bin_views), 5.8))
        fig.patch.set_facecolor(self.theme.background)
        axes = []
        for idx, bin_view in enumerate(artifacts.bin_views, start=1):
            ax = fig.add_subplot(1, len(artifacts.bin_views), idx, projection="3d")
            _plot_schedule_3d_on_axis(
                ax,
                bin_view.allocation_view,
                problem=bin_view.problem,
                color_norm=color_norm,
                cmap=cmap,
            )
            apply_3d_axis_style(ax, theme=self.theme)
            ax.text2D(
                0.5,
                -0.08,
                self._panel_tag(idx - 1),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                color=self.theme.text,
            )
            axes.append(ax)

        colorbar_ax = fig.add_axes([0.95, 0.18, 0.018, 0.62])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=color_norm, cmap=cmap),
            cax=colorbar_ax,
        )
        colorbar.set_label("Selected-row active power (W)")
        style_colorbar(colorbar, theme=self.theme)
        fig.subplots_adjust(left=0.04, right=0.87, bottom=0.10, top=0.92, wspace=0.20)

        self._export_figure(fig, export_path)
        return fig, axes

    def _build_example_bin_context(
        self,
        *,
        scheduler_day_user_table: pd.DataFrame,
        distance_binned_table,
        target_user_count: int,
    ) -> dict[str, object]:
        """Resolve the worked scheduler bin into the prepared TDMA problem."""

        example_bin_index = pick_example_scheduler_bin(
            scheduler_day_user_table,
            target_user_count=int(target_user_count),
        )
        user_table = self._select_bin_user_table(
            scheduler_day_user_table,
            bin_index=int(example_bin_index),
        )
        lookup_artifacts = build_table_lookup_artifacts(
            user_table,
            distance_binned_table=distance_binned_table,
        )
        batch_space = build_cached_batch_user_parameter_space(
            user_table,
            lookup_artifacts=lookup_artifacts,
        )
        from .tdma_walkthrough_helpers import (
            build_joint_allocation_examples,
            build_joint_search_trace,
            build_tdma_preparation_artifacts,
        )

        prep_artifacts = build_tdma_preparation_artifacts(batch_space)
        problem = prep_artifacts.prepared_problem
        trace_artifacts = build_joint_search_trace(
            prep_artifacts.annotated_user_spaces,
            problem,
        )
        optimal_result = run_joint_schedule_search(problem)
        comparison_cases = [
            case
            for case in build_joint_allocation_examples(
                problem,
                optimal_result.best_schedule,
            )
            if bool(case["summary"].get("feasible", True))
        ]

        return {
            "example_bin_index": int(example_bin_index),
            "user_table": user_table,
            "pa_label_map": {
                int(pa_id): str(label)
                for pa_id, label in lookup_artifacts.pa_label_map.items()
            },
            "annotated_user_spaces": {
                int(user_id): user_space.copy()
                for user_id, user_space in prep_artifacts.annotated_user_spaces.items()
            },
            "problem": problem,
            "trace_artifacts": trace_artifacts,
            "optimal_result": optimal_result,
            "comparison_cases": comparison_cases,
        }

    def _build_bin_views(
        self,
        *,
        scheduler_day_user_table: pd.DataFrame,
        distance_binned_table,
        example_bin_index: int,
    ) -> tuple[TdmaBinView, ...]:
        """Build the lighter-versus-heavier bin comparison used at the end."""

        bin_summary = (
            scheduler_day_user_table.groupby("bin_index")
            .agg(
                user_count=("user_id", "nunique"),
                total_rate_bps=("required_rate_bps", "sum"),
            )
            .loc[lambda table: table["user_count"].gt(0)]
            .reset_index()
        )
        stressed_candidates = bin_summary.sort_values(
            ["user_count", "total_rate_bps", "bin_index"],
            ascending=[False, False, True],
        )
        stressed_bin_index = int(stressed_candidates.iloc[0]["bin_index"])
        if stressed_bin_index == int(example_bin_index) and len(stressed_candidates) > 1:
            stressed_bin_index = int(stressed_candidates.iloc[1]["bin_index"])

        bin_views: list[TdmaBinView] = []
        for label, bin_index in (
            ("Reference bin", int(example_bin_index)),
            ("Higher-load bin", int(stressed_bin_index)),
        ):
            bin_user_table = self._select_bin_user_table(
                scheduler_day_user_table,
                bin_index=int(bin_index),
            )
            lookup_artifacts = build_table_lookup_artifacts(
                bin_user_table,
                distance_binned_table=distance_binned_table,
            )
            batch_space = build_cached_batch_user_parameter_space(
                bin_user_table,
                lookup_artifacts=lookup_artifacts,
            )
            problem = prepare_joint_schedule_problem(batch_space)
            result = run_joint_schedule_search(problem)
            user_color_map = self._build_user_color_map(
                bin_user_table["user_id"].astype(int).tolist()
            )
            allocation_view = build_schedule_blocks(
                result.best_schedule,
                problem,
                lookup_artifacts.pa_label_map,
                user_color_map=user_color_map,
            )
            bin_views.append(
                TdmaBinView(
                    label=label,
                    user_count=int(bin_user_table["user_id"].nunique()),
                    requested_rate_mbps=float(bin_user_table["required_rate_bps"].sum()) / 1e6,
                    schedule_power_w=float(result.best_schedule["schedule_p_dc_total_avg_frame_w"]),
                    slot_total=int(result.best_schedule["slot_total"]),
                    allocation_view=allocation_view,
                    problem=problem,
                )
            )

        return tuple(bin_views)

    def _build_pa_color_map(self, pa_label_map: dict[int, str]) -> dict[int, str]:
        """Return one theme-aware categorical color map for the PA families."""

        base_colors = build_color_cycle(self.theme, include_highlight=True)
        return {
            int(pa_id): base_colors[idx % len(base_colors)]
            for idx, pa_id in enumerate(sorted(int(pa_id) for pa_id in pa_label_map))
        }

    def _build_user_color_map(self, user_ids: list[int]) -> dict[int, str]:
        """Return one theme-aware categorical color map for the active users."""

        resolved_user_ids = sorted(int(user_id) for user_id in user_ids)
        if not resolved_user_ids:
            return {}

        user_cmap = colors.LinearSegmentedColormap.from_list(
            f"{self.theme.name}_tdma_users",
            [self.theme.secondary, self.theme.primary, self.theme.accent],
        )
        color_levels = np.linspace(0.10, 0.85, len(resolved_user_ids))
        return {
            int(user_id): colors.to_hex(user_cmap(level))
            for user_id, level in zip(resolved_user_ids, color_levels, strict=False)
        }

    def _build_power_colormap(self):
        """Return the theme-aware sequential colormap used for schedule power."""

        return colors.LinearSegmentedColormap.from_list(
            f"{self.theme.name}_tdma_power",
            [
                self.theme.neutral_light,
                self.theme.highlight,
                self.theme.primary,
                self.theme.secondary,
            ],
        )

    @staticmethod
    def _select_bin_user_table(
        scheduler_day_user_table: pd.DataFrame,
        *,
        bin_index: int,
    ) -> pd.DataFrame:
        """Return the scheduler-facing user table for one active bin."""

        return (
            scheduler_day_user_table.loc[
                lambda table: table["bin_index"].eq(int(bin_index)),
                ["user_id", "distance_m", "required_rate_bps"],
            ]
            .sort_values("user_id")
            .reset_index(drop=True)
        )

    def _export_figure(self, fig: plt.Figure, export_path: Path | None) -> None:
        """Save one figure when the notebook requests a document export."""

        if export_path is None:
            return

        resolved_path = Path(export_path)
        export_doc_figure(
            fig,
            resolved_path.name,
            doc_img_dir=resolved_path.parent,
        )

    @staticmethod
    def _panel_tag(index: int) -> str:
        panel_index = int(index) % len(string.ascii_lowercase)
        return f"({string.ascii_lowercase[panel_index]})"

    @staticmethod
    def _with_alpha(color_value: str, alpha: float) -> tuple[float, float, float, float]:
        red, green, blue = colors.to_rgb(color_value)
        return (red, green, blue, float(alpha))

    @staticmethod
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
                pa_rows[x_column].astype(float),
                pa_rows[y_column].astype(float),
                s=size,
                alpha=alpha,
                color=resolved_color,
                marker=marker_map[int(pa_id)],
                edgecolor=edgecolor,
                linewidth=linewidth,
            )

def build_schedule_blocks(
    schedule_result: dict[str, Any],
    problem: Any,
    pa_label_map: dict[int, str],
    user_color_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Build a rectangular schedule view from one selected joint allocation."""

    schedule_rows = sorted(schedule_result["rows"], key=lambda item: int(item["user_id"]))
    user_ids = [int(row["user_id"]) for row in schedule_rows]
    if user_color_map is None:
        color_levels = np.linspace(0.25, 0.85, max(len(user_ids), 1))
        resolved_user_color_map = {
            user_id: colors.to_hex(plt.cm.cividis(level))
            for user_id, level in zip(user_ids, color_levels[::-1], strict=False)
        }
    else:
        fallback_levels = np.linspace(0.25, 0.85, max(len(user_ids), 1))
        fallback_color_map = {
            user_id: colors.to_hex(plt.cm.cividis(level))
            for user_id, level in zip(user_ids, fallback_levels[::-1], strict=False)
        }
        resolved_user_color_map = {
            int(user_id): str(user_color_map.get(int(user_id), fallback_color_map[int(user_id)]))
            for user_id in user_ids
        }

    blocks = []
    slot_cursor = 0
    for row in schedule_rows:
        block = {
            "user_id": int(row["user_id"]),
            "pa_id": int(row["pa_id"]),
            "pa_label": str(pa_label_map[int(row["pa_id"])]),
            "n_prb": int(row["n_prb"]),
            "n_slots": int(row["n_slots"]),
            "layers": int(row["layers"]),
            "mcs": int(row["mcs"]),
            "p_dc_active_w": float(row["p_dc_active_w"]),
            "p_dc_avg_frame_w": float(
                int(row["n_slots"]) * float(row["p_dc_active_w"]) / float(problem.frame_n_slots)
            ),
            "slot_start": int(slot_cursor),
            "slot_end": int(slot_cursor + int(row["n_slots"])),
            "color": str(resolved_user_color_map[int(row["user_id"])]),
        }
        blocks.append(block)
        slot_cursor = block["slot_end"]

    total_prbs = int(max(block["n_prb"] for block in blocks))
    total_slots = int(problem.frame_n_slots)

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
        "frame_slots": total_slots,
        "frame_boundaries": [],
        "unused_blocks": unused_blocks,
    }
def _power_norm_from_blocks(blocks: list[dict[str, Any]]) -> colors.Normalize:
    power_values = [
        float(block.get("p_dc_active_w", block.get("p_dc_avg_frame_w", 0.0)))
        for block in blocks
    ]
    vmin = float(min(power_values))
    vmax = float(max(power_values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _plot_schedule_3d_on_axis(
    ax,
    allocation_view: dict[str, Any],
    *,
    problem: Any,
    color_norm: colors.Normalize,
    cmap,
    title: str | None = None,
) -> None:
    sorted_blocks = sorted(
        allocation_view["blocks"],
        key=lambda block: (int(block["slot_start"]), int(block["slot_end"]), int(block["n_prb"])),
    )
    for block in sorted_blocks:
        active_power_w = float(block.get("p_dc_active_w", block.get("p_dc_avg_frame_w", 0.0)))
        facecolor = cmap(color_norm(active_power_w))
        _cuboid(
            ax,
            int(block["slot_start"]),
            0,
            0,
            int(block["n_slots"]),
            int(block["n_prb"]),
            int(block["layers"]),
            facecolor,
        )

    frame_slots = int(allocation_view["total_slots"])
    total_prbs = int(allocation_view["total_prbs"])
    if title:
        ax.set_title(title, pad=12, fontsize=11)
    ax.set_xlabel("Slot index", labelpad=10)
    ax.set_ylabel("PRB index", labelpad=12)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Spatial layers", rotation=90, labelpad=12)
    ax.set_xlim(0, frame_slots)
    ax.set_ylim(total_prbs, 0)
    ax.set_zlim(0, int(problem.n_tx_chains))
    ax.set_xticks(np.arange(0, frame_slots + 1, _slot_tick_step(frame_slots)))
    ax.set_yticks(np.arange(0, total_prbs + 1, 50 if total_prbs > 100 else 25))
    ax.set_zticks(np.arange(0, int(problem.n_tx_chains) + 1, 1))
    ax.view_init(elev=24, azim=-60)


def _cuboid(
    ax,
    x: int,
    y: int,
    z: int,
    dx: int,
    dy: int,
    dz: int,
    facecolor,
) -> None:
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
    poly = Poly3DCollection(
        faces,
        facecolors=facecolor,
        edgecolor=(0.1, 0.1, 0.1, 0.45),
        alpha=0.58,
        linewidth=0.65,
        zsort="average",
    )
    ax.add_collection3d(poly)


def _slot_tick_step(slot_count: int) -> int:
    if int(slot_count) <= 12:
        return 1
    if int(slot_count) <= 24:
        return 2
    return 5
__all__ = [
    "PROJECT_ROOT",
    "TdmaBinView",
    "TdmaSchedulingArtifacts",
    "TdmaSchedulingHelpers",
    "build_schedule_blocks",
]
