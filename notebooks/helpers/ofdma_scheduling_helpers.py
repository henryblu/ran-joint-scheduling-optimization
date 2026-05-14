from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import pandas as pd


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

NOTEBOOK_PATH = (PROJECT_ROOT / "notebooks").resolve()
if str(NOTEBOOK_PATH) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_PATH))

from support.day_cycle import build_day_cycle_discussion_artifacts
from support.candidate_space import export_doc_figure
from support.ofdma_walkthrough import build_ofdma_walkthrough_artifacts
from support.table_lookup import (
    build_cached_batch_user_parameter_space,
    build_table_lookup_artifacts,
    load_cached_distance_binned_table,
    pick_example_scheduler_bin,
)
from support.theme import (
    NotebookTheme,
    apply_3d_axis_style,
    build_color_cycle,
    get_notebook_theme,
    style_colorbar,
    style_legend,
)

from helpers.tdma_scheduling_helpers import _plot_schedule_3d_on_axis


OFDMA_POWER_NORM = colors.Normalize(vmin=2.0, vmax=18.0, clip=True)


@dataclass(frozen=True)
class OfdmaStageView:
    """One plot-ready OFDMA scheduler stage."""

    key: str
    label: str
    packed_frame: Any
    allocation_view: dict[str, Any]
    average_frame_dc_power_w: float
    active_slot_count: int
    allocation_count: int


@dataclass(frozen=True)
class OfdmaSchedulingArtifacts:
    """Lean notebook payload for the OFDMA scheduling walkthrough."""

    example_bin_index: int
    user_table: pd.DataFrame
    pa_label_map: dict[int, str]
    pa_color_map: dict[int, str]
    user_color_map: dict[int, str]
    problem: Any
    raw_candidate_counts: dict[int, int]
    pruned_candidate_counts: dict[int, int]
    stage_views: tuple[OfdmaStageView, ...]


class OfdmaSchedulingHelpers:
    """Theme-aware presentation helpers for the OFDMA scheduling notebook."""

    def __init__(self, *, theme: str | NotebookTheme = "aalto_elec"):
        self.theme = get_notebook_theme(theme)

    def build_artifacts(
        self,
        *,
        load_curve_csv: Path,
        day_cycle_config,
        target_user_count: int = 4,
        example_bin_index: int | None = None,
    ) -> OfdmaSchedulingArtifacts:
        """Build compact OFDMA views using aggregate slot RF output for PA DC."""

        distance_binned_table = load_cached_distance_binned_table()
        day_artifacts = build_day_cycle_discussion_artifacts(
            Path(load_curve_csv),
            day_cycle_config,
        )
        scheduler_day_user_table = day_artifacts["scheduler_day_user_table"].copy()
        resolved_bin_index = (
            pick_example_scheduler_bin(
                scheduler_day_user_table,
                target_user_count=int(target_user_count),
            )
            if example_bin_index is None
            else int(example_bin_index)
        )
        user_table = self._select_bin_user_table(
            scheduler_day_user_table,
            bin_index=int(resolved_bin_index),
        )
        lookup_artifacts = build_table_lookup_artifacts(
            user_table,
            distance_binned_table=distance_binned_table,
        )
        batch_space = build_cached_batch_user_parameter_space(
            user_table,
            lookup_artifacts=lookup_artifacts,
        )
        walkthrough = build_ofdma_walkthrough_artifacts(batch_space)
        pa_label_map = {
            int(pa_id): str(label)
            for pa_id, label in lookup_artifacts.pa_label_map.items()
        }
        pa_color_map = self._build_pa_color_map(pa_label_map)
        user_color_map = self._build_user_color_map(
            user_table["user_id"].astype(int).tolist()
        )

        return OfdmaSchedulingArtifacts(
            example_bin_index=int(resolved_bin_index),
            user_table=user_table.copy(),
            pa_label_map=pa_label_map,
            pa_color_map=pa_color_map,
            user_color_map=user_color_map,
            problem=walkthrough.problem,
            raw_candidate_counts={
                int(user_id): int(len(candidates))
                for user_id, candidates in walkthrough.raw_user_candidates.items()
            },
            pruned_candidate_counts={
                int(user_id): int(len(candidates))
                for user_id, candidates in walkthrough.pruned_user_candidates.items()
            },
            stage_views=self._build_stage_views(
                walkthrough,
                pa_label_map=pa_label_map,
                user_color_map=user_color_map,
            ),
        )

    def plot_stage_frame(
        self,
        artifacts: OfdmaSchedulingArtifacts,
        stage_index: int,
        *,
        color_by: str = "power",
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, Any]:
        """Plot one OFDMA scheduler stage on the shared 3D frame."""

        stage_view = artifacts.stage_views[int(stage_index)]
        resolved_color_by = str(color_by).strip().lower()
        if resolved_color_by not in {"power", "user"}:
            raise ValueError("color_by must be either 'power' or 'user'.")

        cmap = self._build_power_colormap()
        fig = plt.figure(figsize=(10.4, 6.6))
        fig.patch.set_facecolor(self.theme.background)
        ax = fig.add_subplot(111, projection="3d")
        _plot_schedule_3d_on_axis(
            ax,
            stage_view.allocation_view,
            problem=artifacts.problem,
            color_norm=OFDMA_POWER_NORM,
            cmap=cmap,
            block_alpha=1.0,
            manual_draw_order=True,
            use_block_colors=bool(resolved_color_by == "user"),
        )
        apply_3d_axis_style(ax, theme=self.theme)

        if resolved_color_by == "power":
            colorbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=OFDMA_POWER_NORM, cmap=cmap),
                ax=ax,
                fraction=0.04,
                pad=0.08,
            )
            colorbar.set_label("Slot PA DC input power (W)")
            style_colorbar(colorbar, theme=self.theme)
            fig.subplots_adjust(left=0.04, right=0.88, bottom=0.06, top=0.94)
        else:
            legend_handles = self._build_stage_user_legend_handles(
                stage_view,
                user_color_map=artifacts.user_color_map,
            )
            if legend_handles:
                legend = ax.legend(
                    handles=legend_handles,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=True,
                )
                style_legend(legend, theme=self.theme)
            fig.subplots_adjust(left=0.04, right=0.80, bottom=0.06, top=0.94)

        self._export_figure(fig, export_path)
        return fig, ax

    def _build_stage_views(
        self,
        walkthrough,
        *,
        pa_label_map: dict[int, str],
        user_color_map: dict[int, str],
    ) -> tuple[OfdmaStageView, ...]:
        stage_specs = (
            ("baseline", "Build a feasible frame", walkthrough.baseline_frame),
            ("pa_switch", "Switch to lower-power PA rows", walkthrough.pa_switched_frame),
            ("slack", "Spend remaining slack", walkthrough.slack_refined_frame),
            ("compact", "Compact the final frame", walkthrough.final_frame),
        )
        return tuple(
            OfdmaStageView(
                key=str(key),
                label=str(label),
                packed_frame=packed_frame,
                allocation_view=build_ofdma_frame_blocks(
                    packed_frame,
                    walkthrough.problem,
                    pa_label_map,
                    user_color_map=user_color_map,
                ),
                average_frame_dc_power_w=float(packed_frame.average_frame_dc_power_w),
                active_slot_count=int(sum(slot.active for slot in packed_frame.slot_schedules)),
                allocation_count=int(
                    sum(len(slot.allocations) for slot in packed_frame.slot_schedules)
                ),
            )
            for key, label, packed_frame in stage_specs
        )

    def _select_bin_user_table(
        self,
        scheduler_day_user_table: pd.DataFrame,
        *,
        bin_index: int,
    ) -> pd.DataFrame:
        user_table = (
            scheduler_day_user_table.loc[
                scheduler_day_user_table["bin_index"].astype(int).eq(int(bin_index))
            ]
            .sort_values("user_id")
            .reset_index(drop=True)
        )
        if user_table.empty:
            raise ValueError(f"Scheduler bin {int(bin_index)} does not contain active users.")
        return user_table

    def _build_pa_color_map(self, pa_label_map: dict[int, str]) -> dict[int, str]:
        color_cycle = build_color_cycle(self.theme, include_highlight=True)
        return {
            int(pa_id): color_cycle[idx % len(color_cycle)]
            for idx, pa_id in enumerate(sorted(pa_label_map))
        }

    def _build_user_color_map(self, user_ids: list[int]) -> dict[int, str]:
        palette = _build_distinct_user_palette()
        return {
            int(user_id): palette[idx % len(palette)]
            for idx, user_id in enumerate(sorted(user_ids))
        }

    def _build_power_colormap(self):
        return colors.LinearSegmentedColormap.from_list(
            f"{self.theme.name}_ofdma_power",
            [
                self.theme.neutral_light,
                self.theme.highlight,
                self.theme.primary,
                self.theme.secondary,
            ],
        )

    def _build_stage_user_legend_handles(
        self,
        stage_view: OfdmaStageView,
        *,
        user_color_map: dict[int, str],
    ) -> list[Line2D]:
        active_user_ids = sorted(
            {
                int(block["user_id"])
                for block in stage_view.allocation_view["blocks"]
            }
        )
        return [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=str(user_color_map[int(user_id)]),
                markeredgecolor=self.theme.background,
                markersize=8,
                label=f"User {int(user_id)}",
            )
            for user_id in active_user_ids
        ]

    def _export_figure(self, fig: plt.Figure, export_path: Path | None) -> Path | None:
        if export_path is None:
            return None
        return export_doc_figure(fig, Path(export_path).name, Path(export_path).parent)


def build_ofdma_frame_blocks(
    packed_frame,
    problem,
    pa_label_map: dict[int, str],
    *,
    user_color_map: dict[int, str],
) -> dict[str, Any]:
    """Return the 3D block view for one OFDMA slot schedule."""

    blocks = []
    active_slots = [
        slot
        for slot in packed_frame.slot_schedules
        if slot.active
    ]
    for slot in active_slots:
        prb_cursor = 0
        display_allocations = sorted(
            slot.allocations,
            key=lambda allocation: (
                int(allocation.user_id),
                int(allocation.pa_id),
                int(allocation.n_prb),
                int(allocation.mcs),
            ),
        )
        for allocation in display_allocations:
            block = {
                "user_id": int(allocation.user_id),
                "pa_id": int(allocation.pa_id),
                "pa_label": str(pa_label_map[int(allocation.pa_id)]),
                "n_prb": int(allocation.n_prb),
                "n_slots": 1,
                "layers": int(allocation.layers),
                "mcs": int(allocation.mcs),
                "p_dc_active_w": float(slot.dc_power_w),
                "p_dc_avg_frame_w": float(slot.dc_power_w) / float(problem.frame_n_slots),
                "slot_start": int(slot.slot_index),
                "slot_end": int(slot.slot_index) + 1,
                "source_slot_index": int(slot.slot_index),
                "prb_start": int(prb_cursor),
                "color": str(user_color_map[int(allocation.user_id)]),
            }
            blocks.append(block)
            prb_cursor += int(allocation.n_prb)

    return {
        "blocks": blocks,
        "total_prbs": int(problem.prb_max),
        "total_slots": int(problem.frame_n_slots),
        "frame_slots": int(problem.frame_n_slots),
        "frame_boundaries": [],
        "unused_blocks": [],
    }


def _build_distinct_user_palette() -> tuple[str, ...]:
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        ordered_indices = list(range(0, int(cmap.N), 2)) + list(range(1, int(cmap.N), 2))
        palette.extend(
            colors.to_hex(cmap(color_index))
            for color_index in ordered_indices
        )
    return tuple(palette)


__all__ = [
    "OfdmaSchedulingArtifacts",
    "OfdmaSchedulingHelpers",
    "OfdmaStageView",
    "OFDMA_POWER_NORM",
    "PROJECT_ROOT",
    "build_ofdma_frame_blocks",
]
