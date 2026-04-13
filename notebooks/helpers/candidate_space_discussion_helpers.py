from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from configs import pa_dc_power
from candidate_table_generation.pruning import prune_candidate_frontier
from day_cycle_simulation.models import SyntheticSessionGenerationConfig
from .candidate_space_helpers import (
    _draw_candidate_allocation_axis,
    annotate_same_pa_dominance,
    export_doc_figure,
    prepare_feasible_plot_table,
    select_dominated_example_pair,
    select_slice_rows,
)
from .candidate_table_generation_helpers import (
    _build_full_frame_candidate_table,
    _resolve_candidate_table_engine_state,
    _select_distance_bin,
)
from .DayCycleSimulationHelpers import bin_index_to_clock, build_day_cycle_discussion_artifacts
from .single_user_study_helpers import (
    build_single_user_scenario,
    run_single_user_scenario,
    summarize_single_user_scenario,
)
from .table_lookup_helpers import (
    build_cached_batch_user_parameter_space,
    build_table_lookup_artifacts,
    load_cached_distance_binned_table,
    pick_example_scheduler_bin,
)
from .tdma_walkthrough_helpers import plot_scheduler_input_spaces as plot_lookup_scheduler_input_spaces
from .visual_identity import (
    NotebookTheme,
    apply_axis_style,
    build_color_cycle,
    create_themed_figure,
    get_notebook_theme,
    render_html_table,
    style_legend,
)


@dataclass(frozen=True)
class CandidateSpaceArtifacts:
    """Lean notebook payload for the candidate-space walkthrough."""

    distance_m: float
    required_rate_bps: float
    required_rate_mbps: float
    frame_slot_count: int
    total_prbs: int
    max_layers: int
    pa_label_by_id: dict[int, str]
    pa_color_map: dict[int, str]
    single_user_summary: pd.DataFrame
    best_feasible_row: pd.Series
    best_feasible_label: str
    slice_comparison: pd.DataFrame
    slice_best_row: pd.Series
    slice_comparison_row: pd.Series
    lower_power_role_label: str
    comparison_role_label: str
    throughput_band: pd.DataFrame
    pruning_pair_table: pd.DataFrame
    band_dominance_table: pd.DataFrame
    pruning_summary: pd.DataFrame
    pruned_frontier_table: pd.DataFrame
    bin_validation_table: pd.DataFrame
    day_bin_count: int
    example_bin_index: int
    example_user_table: pd.DataFrame
    batch_space: Any
    full_frontiers_by_user: dict[int, pd.DataFrame]
    user_color_map: dict[int, str]


class CandidateSpaceHelpers:
    """Theme-aware presentation helpers for Notebook 3."""

    def __init__(self, *, theme: str | NotebookTheme = "aalto_elec"):
        self.theme = get_notebook_theme(theme)

    def build_artifacts(
        self,
        *,
        load_curve_csv: Path,
        day_cycle_config: SyntheticSessionGenerationConfig,
        distance_m: float = 300.0,
        required_rate_bps: float = 200e6,
        throughput_slice_tolerance_mbps: float = 0.05,
        throughput_band_mbps: tuple[float, float] = (200.0, 205.0),
        target_user_count: int = 4,
    ) -> CandidateSpaceArtifacts:
        """Build the compact candidate-space views used in Notebook 3.

        Steps:
        1. Resolve one fixed single-user case and derive the feasible rows used to explain the tradeoffs.
        2. Build the pruning and stored-frontier views for the same distance bin.
        3. Reuse one day-level active bin to assemble the lookup-stage handoff into per-user candidate menus.
        """

        single_user_context = self._build_single_user_candidate_context(
            distance_m=float(distance_m),
            required_rate_bps=float(required_rate_bps),
            throughput_slice_tolerance_mbps=float(throughput_slice_tolerance_mbps),
            throughput_band_mbps=throughput_band_mbps,
        )
        lookup_context = self._build_lookup_context(
            load_curve_csv=Path(load_curve_csv),
            day_cycle_config=day_cycle_config,
            target_user_count=int(target_user_count),
        )

        return CandidateSpaceArtifacts(
            distance_m=float(distance_m),
            required_rate_bps=float(required_rate_bps),
            required_rate_mbps=float(required_rate_bps) / 1e6,
            frame_slot_count=int(single_user_context["frame_slot_count"]),
            total_prbs=int(single_user_context["total_prbs"]),
            max_layers=int(single_user_context["max_layers"]),
            pa_label_by_id=dict(single_user_context["pa_label_by_id"]),
            pa_color_map=dict(lookup_context["pa_color_map"]),
            single_user_summary=single_user_context["single_user_summary"].copy(),
            best_feasible_row=single_user_context["best_feasible_row"].copy(),
            best_feasible_label=str(single_user_context["best_feasible_label"]),
            slice_comparison=single_user_context["slice_comparison"].copy(),
            slice_best_row=single_user_context["slice_best_row"].copy(),
            slice_comparison_row=single_user_context["slice_comparison_row"].copy(),
            lower_power_role_label=str(single_user_context["lower_power_role_label"]),
            comparison_role_label=str(single_user_context["comparison_role_label"]),
            throughput_band=single_user_context["throughput_band"].copy(),
            pruning_pair_table=single_user_context["pruning_pair_table"].copy(),
            band_dominance_table=single_user_context["band_dominance_table"].copy(),
            pruning_summary=single_user_context["pruning_summary"].copy(),
            pruned_frontier_table=single_user_context["pruned_frontier_table"].copy(),
            bin_validation_table=lookup_context["bin_validation_table"].copy(),
            day_bin_count=int(lookup_context["day_bin_count"]),
            example_bin_index=int(lookup_context["example_bin_index"]),
            example_user_table=lookup_context["example_user_table"].copy(),
            batch_space=lookup_context["batch_space"],
            full_frontiers_by_user={
                int(user_id): frontier.copy()
                for user_id, frontier in lookup_context["full_frontiers_by_user"].items()
            },
            user_color_map=dict(lookup_context["user_color_map"]),
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

    def plot_candidate_allocation(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, Any]:
        """Plot one feasible row as a themed time-frequency-layer allocation block."""

        fig = plt.figure(figsize=(12.0, 8.0))
        fig.patch.set_facecolor(self.theme.background)
        ax = fig.add_subplot(111, projection="3d")
        _draw_candidate_allocation_axis(
            ax=ax,
            total_slots=int(artifacts.frame_slot_count),
            total_prbs=int(artifacts.total_prbs),
            max_layers=int(artifacts.max_layers),
            allocation_row=artifacts.best_feasible_row,
            allocation_color=self.theme.primary,
            allocation_edgecolor=self.theme.secondary,
            envelope_color=self.theme.neutral_light,
            envelope_edgecolor=self.theme.neutral_dark,
            label_color=self.theme.text,
        )
        self._apply_3d_axis_theme(ax)
        fig.subplots_adjust(left=0.06, right=0.88, bottom=0.10, top=0.95)

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_same_throughput_tradeoff(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, tuple[Any, Any]]:
        """Plot two same-throughput candidate rows side by side."""

        fig = plt.figure(figsize=(16.0, 7.2))
        fig.patch.set_facecolor(self.theme.background)
        left_ax = fig.add_subplot(121, projection="3d")
        right_ax = fig.add_subplot(122, projection="3d")

        _draw_candidate_allocation_axis(
            ax=left_ax,
            total_slots=int(artifacts.frame_slot_count),
            total_prbs=int(artifacts.total_prbs),
            max_layers=int(artifacts.max_layers),
            allocation_row=artifacts.slice_best_row,
            allocation_color=self.theme.primary,
            allocation_edgecolor=self.theme.secondary,
            z_label_x=1.08,
            envelope_color=self.theme.neutral_light,
            envelope_edgecolor=self.theme.neutral_dark,
            label_color=self.theme.text,
        )
        _draw_candidate_allocation_axis(
            ax=right_ax,
            total_slots=int(artifacts.frame_slot_count),
            total_prbs=int(artifacts.total_prbs),
            max_layers=int(artifacts.max_layers),
            allocation_row=artifacts.slice_comparison_row,
            allocation_color=self.theme.highlight,
            allocation_edgecolor=self.theme.accent,
            z_label_x=1.08,
            envelope_color=self.theme.neutral_light,
            envelope_edgecolor=self.theme.neutral_dark,
            label_color=self.theme.text,
        )
        self._apply_3d_axis_theme(left_ax)
        self._apply_3d_axis_theme(right_ax)
        left_ax.text2D(
            0.5,
            -0.045,
            "(a)",
            transform=left_ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color=self.theme.text,
        )
        right_ax.text2D(
            0.5,
            -0.045,
            "(b)",
            transform=right_ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color=self.theme.text,
        )
        fig.subplots_adjust(left=0.04, right=0.95, bottom=0.10, top=0.94, wspace=0.16)

        self._export_figure(fig, export_path)
        return fig, (left_ax, right_ax)

    def plot_throughput_band(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the feasible candidates across the worked throughput band."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(8.6, 5.2),
        )
        throughput_band = artifacts.throughput_band
        cloud_artist = ax.scatter(
            throughput_band["rate_mbps"].astype(float),
            throughput_band["active_pa_power_w"].astype(float),
            c=throughput_band["total_prb_slots"].astype(float),
            cmap=self._build_prb_colormap(),
            s=68,
            alpha=0.88,
            edgecolors=self.theme.background,
            linewidths=0.35,
        )
        ax.scatter(
            [float(artifacts.slice_best_row["rate_mbps"])],
            [float(artifacts.slice_best_row["active_pa_power_w"])],
            facecolors="none",
            edgecolors=self.theme.secondary,
            linewidths=2.0,
            s=190,
            marker="o",
            label=f"{artifacts.lower_power_role_label} ({artifacts.pa_label_by_id[int(artifacts.slice_best_row['pa_id'])]})",
            zorder=4,
        )
        ax.scatter(
            [float(artifacts.slice_comparison_row["rate_mbps"])],
            [float(artifacts.slice_comparison_row["active_pa_power_w"])],
            facecolors="none",
            edgecolors=self.theme.accent,
            linewidths=2.1,
            s=220,
            marker="D",
            label=f"{artifacts.comparison_role_label} ({artifacts.pa_label_by_id[int(artifacts.slice_comparison_row['pa_id'])]})",
            zorder=5,
        )
        ax.set_xlim(
            float(throughput_band["rate_mbps"].min()-0.5),
            float(throughput_band["rate_mbps"].max()+0.5),
        )
        ax.set_xlabel("Average throughput (Mbps)")
        ax.set_ylabel("Active PA DC power (W)")
        style_legend(ax, theme=self.theme)

        colorbar = fig.colorbar(cloud_artist, ax=ax, label="Allocated PRBs per frame")
        self._style_colorbar(colorbar)
        fig.tight_layout()

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_pruning_band(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot kept and dominated rows across the worked throughput band."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(9.4, 5.4),
        )
        pruning_slice_table = artifacts.band_dominance_table
        pa_ids = sorted(pruning_slice_table["pa_id"].astype(int).unique())
        marker_by_pa = {
            int(pa_id): marker
            for pa_id, marker in zip(pa_ids, ["o", "s", "^", "D"], strict=False)
        }
        color_norm = colors.Normalize(
            vmin=float(pruning_slice_table["total_prb_slots"].min()),
            vmax=float(pruning_slice_table["total_prb_slots"].max()),
        )
        color_map = self._build_prb_colormap()

        for pa_id in pa_ids:
            pa_rows = pruning_slice_table.loc[pruning_slice_table["pa_id"].eq(int(pa_id))].copy()
            marker = marker_by_pa[int(pa_id)]
            kept_rows = pa_rows.loc[pa_rows["pruning_role"].eq("kept")]
            dominated_rows = pa_rows.loc[pa_rows["pruning_role"].eq("dominated")]
            pa_label = artifacts.pa_label_by_id[int(pa_id)]

            if not kept_rows.empty:
                ax.scatter(
                    kept_rows["rate_mbps"].astype(float),
                    kept_rows["active_pa_power_w"].astype(float),
                    c=kept_rows["total_prb_slots"].astype(float),
                    cmap=color_map,
                    norm=color_norm,
                    s=72,
                    alpha=0.9,
                    marker=marker,
                    edgecolors=self.theme.neutral_dark,
                    linewidths=0.55,
                    label=f"{pa_label} kept",
                )
            if dominated_rows.empty:
                continue

            edge_colors = color_map(color_norm(dominated_rows["total_prb_slots"].astype(float)))
            ax.scatter(
                dominated_rows["rate_mbps"].astype(float),
                dominated_rows["active_pa_power_w"].astype(float),
                facecolors="none",
                edgecolors=edge_colors,
                s=95,
                marker=marker,
                linewidths=1.5,
                label=f"{pa_label} dominated",
            )

        ax.axvline(
            float(artifacts.required_rate_mbps),
            color=self.theme.neutral_dark,
            linestyle="--",
            linewidth=1.2,
        )
        ax.set_xlim(
            float(pruning_slice_table["rate_mbps"].min()-0.5),
            float(pruning_slice_table["rate_mbps"].max()+0.5),
        )
        ax.set_xlabel("Average throughput (Mbps)")
        ax.set_ylabel("Active PA DC power (W)")
        legend = ax.legend(frameon=True, ncol=2)
        style_legend(legend, theme=self.theme)

        scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax, label="Allocated PRBs per frame")
        self._style_colorbar(colorbar)
        fig.subplots_adjust(left=0.09, right=0.88, bottom=0.13, top=0.88)

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_frontier_compaction(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the row-count reduction introduced by same-PA pruning."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(8.0, 4.8),
        )
        x_positions = np.arange(len(artifacts.pruning_summary))
        width = 0.35

        ax.bar(
            x_positions - width / 2,
            artifacts.pruning_summary["rows_before_pruning"].astype(int),
            width=width,
            color=self.theme.primary,
            alpha=0.82,
            label="Full-frame rows",
        )
        ax.bar(
            x_positions + width / 2,
            artifacts.pruning_summary["rows_after_pruning"].astype(int),
            width=width,
            color=self.theme.secondary,
            alpha=0.82,
            label="Stored frontier rows",
        )
        ax.set_xticks(x_positions.tolist())
        ax.set_xticklabels(artifacts.pruning_summary["pa_label"].tolist())
        ax.set_ylabel("Row count")
        legend = ax.legend(loc="upper right", frameon=True)
        style_legend(legend, theme=self.theme)
        fig.tight_layout()

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_pruned_frontier(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the stored full-frame frontier for the worked distance bin."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(8.2, 5.2),
        )
        for pa_id, pa_rows in artifacts.pruned_frontier_table.groupby("pa_id", sort=True):
            ax.scatter(
                pa_rows["rate_active_bps"].astype(float) / 1e6,
                pa_rows["p_dc_active_w"].astype(float),
                s=36,
                alpha=0.86,
                color=artifacts.pa_color_map[int(pa_id)],
                edgecolors=self.theme.background,
                linewidths=0.45,
                label=artifacts.pa_label_by_id[int(pa_id)],
            )

        ax.set_xlabel("Active rate for the full frame (Mbps)")
        ax.set_ylabel("Active PA DC power (W)")
        legend = ax.legend(loc="upper right", frameon=True)
        style_legend(legend, theme=self.theme)
        fig.tight_layout()

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_lookup_bin_context(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot the selected scheduler bin inside the day-level demand trace."""

        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=2,
            ncols=1,
            figsize=(12.0, 7.5),
            sharex=True,
            squeeze=False,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )
        load_ax, active_ax = axes.ravel()
        validation_table = artifacts.bin_validation_table

        load_ax.bar(
            validation_table["bin_index"].astype(int),
            validation_table["target_load_gb_in_bin"].astype(float),
            width=0.9,
            color=self.theme.primary,
            alpha=0.22,
            label="Target load from the quarter-hour bins",
        )
        load_ax.plot(
            validation_table["bin_index"].astype(int),
            validation_table["rebuilt_load_gb_in_bin"].astype(float),
            color=self.theme.accent,
            linewidth=2.0,
            label="Load rebuilt from generated sessions",
        )
        load_ax.set_ylabel("Load in bin (GB)")
        style_legend(load_ax, theme=self.theme)

        active_ax.bar(
            validation_table["bin_index"].astype(int),
            validation_table["active_users"].astype(int),
            width=0.9,
            color=self.theme.secondary,
            alpha=0.85,
        )
        active_ax.set_xlabel("Quarter-hour bin index")
        active_ax.set_ylabel("Active users")

        for ax in (load_ax, active_ax):
            self._set_day_bin_axis(ax, day_bin_count=int(artifacts.day_bin_count))
            self._highlight_selected_bin(
                ax,
                bin_index=int(artifacts.example_bin_index),
                label=f"Chosen bin ({bin_index_to_clock(int(artifacts.example_bin_index))})",
            )

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, (load_ax, active_ax)

    def plot_active_bin_snapshot(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the active users in the selected scheduler-facing bin."""

        plot_table = artifacts.example_user_table.sort_values("user_id").copy()
        plot_table["required_rate_mbps"] = plot_table["required_rate_bps"].astype(float) / 1e6
        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(8.4, 4.8),
        )

        for row in plot_table.itertuples(index=False):
            user_id = int(row.user_id)
            user_color = artifacts.user_color_map[user_id]
            ax.scatter(
                [float(row.distance_m)],
                [float(row.required_rate_mbps)],
                s=94,
                color=user_color,
                edgecolor=self.theme.background,
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                f"User {user_id}",
                (float(row.distance_m), float(row.required_rate_mbps)),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=user_color,
            )

        ax.set_xlabel("User distance (m)")
        ax.set_ylabel("Required throughput (Mbps)")
        fig.tight_layout()

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_scheduler_input_spaces(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ):
        """Plot the per-user full-frame menus that leave the lookup stage."""

        fig, axes = plot_lookup_scheduler_input_spaces(
            artifacts.batch_space,
            artifacts.example_user_table,
            user_color_map=artifacts.user_color_map,
            pa_color_map=artifacts.pa_color_map,
            pa_label_map=artifacts.pa_label_by_id,
            full_frontiers_by_user=artifacts.full_frontiers_by_user,
        )
        fig.patch.set_facecolor(self.theme.background)
        for ax, user_row in zip(
            np.atleast_1d(axes).ravel(),
            artifacts.example_user_table.sort_values("user_id").itertuples(index=False),
            strict=False,
        ):
            if not ax.get_visible():
                continue
            apply_axis_style(ax, theme=self.theme)
            user_color = artifacts.user_color_map[int(user_row.user_id)]
            for spine in ax.spines.values():
                if not spine.get_visible():
                    continue
                spine.set_edgecolor(user_color)
                spine.set_linewidth(1.25)
        for legend in fig.legends:
            style_legend(legend, theme=self.theme)

        self._export_figure(fig, export_path)
        return fig, axes

    def _build_single_user_candidate_context(
        self,
        *,
        distance_m: float,
        required_rate_bps: float,
        throughput_slice_tolerance_mbps: float,
        throughput_band_mbps: tuple[float, float],
    ) -> dict[str, object]:
        """Resolve the single-user candidate and pruning views for Notebook 3."""

        scenario = build_single_user_scenario(
            distance_m=float(distance_m),
            required_rate_bps=float(required_rate_bps),
        )
        scenario_views = summarize_single_user_scenario(scenario)
        example_candidate_view = scenario_views["example_candidate_view"].copy()
        feasible_table = run_single_user_scenario(scenario).copy().reset_index(drop=True)
        pa_label_by_id = {
            int(pa_id): str(pa.scenario_label)
            for pa_id, pa in enumerate(scenario.context.pa_catalog)
        }
        best_feasible_row = (
            feasible_table.sort_values(
                [
                    "p_dc_avg_total_w",
                    "bandwidth_hz",
                    "n_prb",
                    "n_slots_on",
                    "layers",
                    "mcs",
                    "pa_id",
                    "bwp_idx",
                ]
            )
            .reset_index(drop=True)
            .iloc[0]
        )
        feasible_plot_table = self._attach_active_pa_power(
            feasible_table,
            scenario=scenario,
        )
        required_rate_mbps = float(required_rate_bps) / 1e6
        throughput_slice = (
            feasible_plot_table.loc[
                feasible_plot_table["rate_mbps"].between(
                    required_rate_mbps - float(throughput_slice_tolerance_mbps),
                    required_rate_mbps + float(throughput_slice_tolerance_mbps),
                    inclusive="both",
                )
            ]
            .sort_values(
                [
                    "active_pa_power_w",
                    "total_prb_slots",
                    "mcs",
                    "n_prb",
                    "n_slots_on",
                    "layers",
                    "pa_id",
                    "rate_mbps",
                ]
            )
            .reset_index(drop=True)
        )
        min_rate_mbps, max_rate_mbps = throughput_band_mbps
        throughput_band = (
            feasible_plot_table.loc[
                feasible_plot_table["rate_mbps"].between(
                    float(min_rate_mbps),
                    float(max_rate_mbps),
                    inclusive="both",
                )
            ]
            .sort_values(
                [
                    "rate_mbps",
                    "active_pa_power_w",
                    "total_prb_slots",
                    "mcs",
                    "n_prb",
                    "n_slots_on",
                    "layers",
                    "pa_id",
                ]
            )
            .reset_index(drop=True)
        )
        if throughput_band.empty:
            raise ValueError("No feasible candidates were found inside the requested throughput band.")

        slice_best_row, slice_comparison_row = select_slice_rows(
            throughput_slice,
            power_column="active_pa_power_w",
        )
        lower_power_role_label = "Lower-power row"
        comparison_role_label = (
            "Lower-resource higher-MCS row"
            if int(slice_comparison_row["total_prb_slots"]) < int(slice_best_row["total_prb_slots"])
            else "Higher-MCS comparison"
        )
        annotated_feasible_table = annotate_same_pa_dominance(
            feasible_plot_table,
            resource_column="total_prb_slots",
            power_column="active_pa_power_w",
            rate_column="rate_mbps",
        )
        band_dominance_table = (
            annotated_feasible_table.loc[
                annotated_feasible_table["rate_mbps"].between(
                    float(min_rate_mbps),
                    float(max_rate_mbps),
                    inclusive="both",
                )
            ]
            .sort_values(
                [
                    "rate_mbps",
                    "active_pa_power_w",
                    "total_prb_slots",
                    "pa_id",
                    "pruning_role",
                    "mcs",
                    "n_prb",
                    "n_slots_on",
                    "layers",
                ]
            )
            .reset_index(drop=True)
        )
        dominated_row, dominating_row = select_dominated_example_pair(
            band_dominance_table,
            rate_column="rate_mbps",
            power_column="active_pa_power_w",
            resource_column="total_prb_slots",
            min_rate_mbps=float(min_rate_mbps),
            max_rate_mbps=float(max_rate_mbps),
            lookup_table=annotated_feasible_table,
            require_dominator_in_table=True,
        )
        selected_distance_m = _select_distance_bin(int(distance_m))
        full_frame_candidate_table = _build_full_frame_candidate_table(
            int(selected_distance_m),
            engine_state=_resolve_candidate_table_engine_state(),
        )
        pruned_frontier_table = prune_candidate_frontier(full_frame_candidate_table)

        return {
            "frame_slot_count": int(example_candidate_view.loc[0, "available_slots"]),
            "total_prbs": int(example_candidate_view.loc[0, "available_prbs"]),
            "max_layers": int(example_candidate_view.loc[0, "available_layers"]),
            "pa_label_by_id": pa_label_by_id,
            "best_feasible_row": best_feasible_row,
            "best_feasible_label": pa_label_by_id[int(best_feasible_row["pa_id"])],
            "single_user_summary": self._build_single_user_summary_table(
                distance_m=float(distance_m),
                required_rate_mbps=required_rate_mbps,
                best_feasible_row=best_feasible_row,
                best_feasible_label=pa_label_by_id[int(best_feasible_row["pa_id"])],
            ),
            "slice_comparison": self._build_slice_comparison_table(
                pa_label_by_id=pa_label_by_id,
                slice_best_row=slice_best_row,
                slice_comparison_row=slice_comparison_row,
                lower_power_role_label=lower_power_role_label,
                comparison_role_label=comparison_role_label,
            ),
            "slice_best_row": slice_best_row,
            "slice_comparison_row": slice_comparison_row,
            "lower_power_role_label": lower_power_role_label,
            "comparison_role_label": comparison_role_label,
            "throughput_band": throughput_band,
            "pruning_pair_table": self._build_pruning_pair_table(
                pa_label_by_id=pa_label_by_id,
                dominating_row=dominating_row,
                dominated_row=dominated_row,
            ),
            "band_dominance_table": band_dominance_table,
            "pruning_summary": self._build_pruning_summary(
                full_frame_candidate_table,
                pruned_frontier_table,
                pa_label_by_id=pa_label_by_id,
            ),
            "pruned_frontier_table": pruned_frontier_table.copy(),
        }

    def _build_lookup_context(
        self,
        *,
        load_curve_csv: Path,
        day_cycle_config: SyntheticSessionGenerationConfig,
        target_user_count: int,
    ) -> dict[str, object]:
        """Resolve the lookup-stage handoff for one selected scheduler bin."""

        distance_binned_table = load_cached_distance_binned_table()
        day_artifacts = build_day_cycle_discussion_artifacts(load_curve_csv, day_cycle_config)
        example_bin_index = pick_example_scheduler_bin(
            day_artifacts["scheduler_day_user_table"],
            target_user_count=int(target_user_count),
        )
        example_user_table = (
            day_artifacts["scheduler_day_user_table"].loc[
                lambda table: table["bin_index"].eq(int(example_bin_index)),
                ["user_id", "distance_m", "required_rate_bps"],
            ]
            .sort_values("user_id")
            .reset_index(drop=True)
        )
        lookup_artifacts = build_table_lookup_artifacts(
            example_user_table,
            distance_binned_table=distance_binned_table,
        )

        return {
            "bin_validation_table": day_artifacts["bin_validation_table"].copy(),
            "day_bin_count": int(day_cycle_config.day_bin_count),
            "example_bin_index": int(example_bin_index),
            "example_user_table": example_user_table,
            "batch_space": build_cached_batch_user_parameter_space(
                example_user_table,
                lookup_artifacts=lookup_artifacts,
            ),
            "full_frontiers_by_user": lookup_artifacts.full_frontiers_by_user,
            "pa_color_map": self._build_pa_color_map(lookup_artifacts.pa_label_map),
            "user_color_map": self._build_user_color_map(
                example_user_table["user_id"].astype(int).tolist()
            ),
        }

    def _attach_active_pa_power(
        self,
        feasible_table: pd.DataFrame,
        *,
        scenario,
    ) -> pd.DataFrame:
        """Attach active PA power to the plot-ready feasible candidate table."""

        plot_table = prepare_feasible_plot_table(feasible_table)
        n_tx_chains = int(scenario.context.deployment.n_tx_chains)
        plot_table["active_pa_power_w"] = [
            float(n_tx_chains)
            * float(
                pa_dc_power(
                    scenario.context.pa_catalog[int(row.pa_id)],
                    float(row.p_out_total_w) / float(n_tx_chains),
                )
            )
            for row in plot_table.itertuples(index=False)
        ]
        return plot_table

    def _build_single_user_summary_table(
        self,
        *,
        distance_m: float,
        required_rate_mbps: float,
        best_feasible_row: pd.Series,
        best_feasible_label: str,
    ) -> pd.DataFrame:
        """Return the one-row single-user summary shown at the start of Notebook 3."""

        return pd.DataFrame(
            [
                {
                    "Distance (m)": round(float(distance_m), 1),
                    "Required throughput (Mbps)": round(float(required_rate_mbps), 1),
                    "PA": best_feasible_label,
                    "Slots": int(best_feasible_row["n_slots_on"]),
                    "n_PRB per slot": int(best_feasible_row["n_prb"]),
                    "Allocated PRBs per frame": int(best_feasible_row["n_prb"])
                    * int(best_feasible_row["n_slots_on"]),
                    "Layers": int(best_feasible_row["layers"]),
                    "MCS index": int(best_feasible_row["mcs"]),
                    "Achieved throughput (Mbps)": round(
                        float(best_feasible_row["rate_ach_bps"]) / 1e6,
                        3,
                    ),
                    "Average power (W)": round(float(best_feasible_row["p_dc_avg_total_w"]), 3),
                }
            ]
        )

    def _build_slice_comparison_table(
        self,
        *,
        pa_label_by_id: dict[int, str],
        slice_best_row: pd.Series,
        slice_comparison_row: pd.Series,
        lower_power_role_label: str,
        comparison_role_label: str,
    ) -> pd.DataFrame:
        """Return the small comparison table used in the throughput-band section."""

        return pd.DataFrame(
            [
                {
                    "Role": lower_power_role_label,
                    "PA": pa_label_by_id[int(slice_best_row["pa_id"])],
                    "Slots": int(slice_best_row["n_slots_on"]),
                    "n_PRB per slot": int(slice_best_row["n_prb"]),
                    "Allocated PRBs per frame": int(slice_best_row["total_prb_slots"]),
                    "Layers": int(slice_best_row["layers"]),
                    "MCS index": int(slice_best_row["mcs"]),
                    "Average throughput (Mbps)": round(float(slice_best_row["rate_mbps"]), 3),
                    "Active PA DC power (W)": round(float(slice_best_row["active_pa_power_w"]), 3),
                },
                {
                    "Role": comparison_role_label,
                    "PA": pa_label_by_id[int(slice_comparison_row["pa_id"])],
                    "Slots": int(slice_comparison_row["n_slots_on"]),
                    "n_PRB per slot": int(slice_comparison_row["n_prb"]),
                    "Allocated PRBs per frame": int(slice_comparison_row["total_prb_slots"]),
                    "Layers": int(slice_comparison_row["layers"]),
                    "MCS index": int(slice_comparison_row["mcs"]),
                    "Average throughput (Mbps)": round(float(slice_comparison_row["rate_mbps"]), 3),
                    "Active PA DC power (W)": round(float(slice_comparison_row["active_pa_power_w"]), 3),
                },
            ]
        )

    def _build_pruning_pair_table(
        self,
        *,
        pa_label_by_id: dict[int, str],
        dominating_row: pd.Series,
        dominated_row: pd.Series,
    ) -> pd.DataFrame:
        """Return the kept-versus-dominated comparison table shown before pruning."""

        return pd.DataFrame(
            [
                {
                    "Role": "Dominating row",
                    "PA": pa_label_by_id[int(dominating_row["pa_id"])],
                    "Slots": int(dominating_row["n_slots_on"]),
                    "n_PRB per slot": int(dominating_row["n_prb"]),
                    "Allocated PRBs per frame": int(dominating_row["total_prb_slots"]),
                    "Layers": int(dominating_row["layers"]),
                    "MCS index": int(dominating_row["mcs"]),
                    "Average throughput (Mbps)": round(float(dominating_row["rate_mbps"]), 3),
                    "Active PA DC power (W)": round(float(dominating_row["active_pa_power_w"]), 3),
                },
                {
                    "Role": "Dominated row",
                    "PA": pa_label_by_id[int(dominated_row["pa_id"])],
                    "Slots": int(dominated_row["n_slots_on"]),
                    "n_PRB per slot": int(dominated_row["n_prb"]),
                    "Allocated PRBs per frame": int(dominated_row["total_prb_slots"]),
                    "Layers": int(dominated_row["layers"]),
                    "MCS index": int(dominated_row["mcs"]),
                    "Average throughput (Mbps)": round(float(dominated_row["rate_mbps"]), 3),
                    "Active PA DC power (W)": round(float(dominated_row["active_pa_power_w"]), 3),
                },
            ]
        )

    def _build_pruning_summary(
        self,
        full_frame_candidate_table: pd.DataFrame,
        pruned_frontier_table: pd.DataFrame,
        *,
        pa_label_by_id: dict[int, str],
    ) -> pd.DataFrame:
        """Return the per-PA row-count reduction introduced by strict pruning."""

        rows = []
        pa_ids = sorted(
            {
                *full_frame_candidate_table["pa_id"].dropna().astype(int).tolist(),
                *pruned_frontier_table["pa_id"].dropna().astype(int).tolist(),
            }
        )
        for pa_id in pa_ids:
            rows.append(
                {
                    "pa_id": int(pa_id),
                    "pa_label": pa_label_by_id[int(pa_id)],
                    "rows_before_pruning": int(
                        full_frame_candidate_table["pa_id"].eq(int(pa_id)).sum()
                    ),
                    "rows_after_pruning": int(
                        pruned_frontier_table["pa_id"].eq(int(pa_id)).sum()
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _build_prb_colormap(self):
        """Return the restrained colormap used for PRB-slot emphasis in Notebook 3."""

        return colors.LinearSegmentedColormap.from_list(
            f"{self.theme.name}_candidate_prbs",
            [self.theme.neutral_light, self.theme.primary, self.theme.secondary],
        )

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
            f"{self.theme.name}_lookup_users",
            [self.theme.secondary, self.theme.primary, self.theme.accent],
        )
        color_levels = np.linspace(0.1, 0.85, len(resolved_user_ids))
        return {
            int(user_id): colors.to_hex(user_cmap(level))
            for user_id, level in zip(resolved_user_ids, color_levels, strict=False)
        }

    def _set_day_bin_axis(self, ax: plt.Axes, *, day_bin_count: int) -> None:
        """Apply the shared day-bin framing used in the lookup-context plot."""

        for boundary in range(0, int(day_bin_count) + 1, 4):
            ax.axvline(boundary - 0.5, color=self.theme.grid, linewidth=0.8, zorder=0)
        ax.set_xlim(-0.5, int(day_bin_count) - 0.5)
        ax.set_xticks(list(range(0, int(day_bin_count), 8)))

    def _highlight_selected_bin(self, ax: plt.Axes, *, bin_index: int, label: str) -> None:
        """Highlight the selected teaching bin on one day-level axis."""

        left = float(bin_index) - 0.45
        right = float(bin_index) + 0.45
        ax.axvspan(left, right, color=self.theme.highlight, alpha=0.12, zorder=0.2)
        ax.axvline(
            float(bin_index),
            color=self.theme.highlight,
            linestyle="--",
            linewidth=1.3,
        )
        y_top = ax.get_ylim()[1]
        ax.text(
            float(bin_index),
            y_top * 0.97,
            label,
            ha="center",
            va="top",
            fontsize=9,
            color=self.theme.text,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": self.theme.background,
                "edgecolor": self.theme.highlight,
                "alpha": 0.95,
            },
        )

    def _style_colorbar(self, colorbar) -> None:
        """Apply the shared notebook theme to a colorbar."""

        colorbar.outline.set_edgecolor(self.theme.neutral_dark)
        colorbar.outline.set_linewidth(0.9)
        colorbar.ax.tick_params(colors=self.theme.neutral_dark, labelcolor=self.theme.neutral_dark)
        colorbar.ax.yaxis.label.set_color(self.theme.text)

    def _apply_3d_axis_theme(self, ax) -> None:
        """Apply the shared notebook colors to one 3D axis."""

        background_rgba = colors.to_rgba(self.theme.background, 1.0)
        grid_rgba = colors.to_rgba(self.theme.grid, 1.0)

        ax.set_facecolor(self.theme.background)
        ax.tick_params(colors=self.theme.neutral_dark, labelcolor=self.theme.neutral_dark)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(background_rgba)
            axis._axinfo["grid"]["color"] = grid_rgba
            axis._axinfo["grid"]["linewidth"] = 0.8
            axis.line.set_color(self.theme.neutral_dark)
        ax.xaxis.label.set_color(self.theme.text)
        ax.yaxis.label.set_color(self.theme.text)

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


__all__ = [
    "CandidateSpaceArtifacts",
    "CandidateSpaceHelpers",
]
