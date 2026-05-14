from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from candidate_table_generation import DISTANCE_BIN_GRID_M
from candidate_table_generation.pruning import prune_candidate_frontier
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from support.candidate_space import annotate_same_pa_dominance, export_doc_figure
from support.candidate_table_generation import (
    _build_slot_normalized_candidate_table,
    _resolve_candidate_table_engine_state,
    _select_distance_bin,
)
from support.theme import (
    NotebookTheme,
    apply_axis_style,
    create_themed_figure,
    get_notebook_theme,
    render_html_table,
    style_legend,
)


@dataclass(frozen=True)
class CandidateSpaceArtifacts:
    """Compact notebook payload for the candidate-space frontier walkthrough."""

    worked_distance_m: int
    comparison_distance_m: int
    supported_distance_bins: tuple[int, ...]
    pa_label_by_id: dict[int, str]
    pa_color_map: dict[int, str]
    row_contract: pd.Series
    full_space_table: pd.DataFrame
    annotated_full_space_table: pd.DataFrame
    zoom_table: pd.DataFrame
    zoom_x_limits_bits: tuple[float, float]
    zoom_y_limits_w: tuple[float, float]
    dominated_row: pd.Series
    dominating_row: pd.Series
    worked_frontier_table: pd.DataFrame
    comparison_frontier_table: pd.DataFrame


class CandidateSpaceHelpers:
    """Theme-aware presentation helpers for the candidate-space notebook."""

    def __init__(self, *, theme: str | NotebookTheme = "aalto_elec"):
        self.theme = get_notebook_theme(theme)

    def build_artifacts(
        self,
        *,
        worked_distance_m: int = 300,
        comparison_distance_m: int = 50,
    ) -> CandidateSpaceArtifacts:
        """Build the one-slot feasible-space and frontier views used in the notebook."""

        engine_state = _resolve_candidate_table_engine_state()
        worked_distance_bin = _select_distance_bin(int(worked_distance_m))
        comparison_distance_bin = _select_distance_bin(int(comparison_distance_m))
        pa_label_by_id = {
            int(pa_id): str(pa.scenario_label)
            for pa_id, pa in enumerate(engine_state.pa_catalog)
        }
        pa_color_map = self._build_pa_color_map(pa_label_by_id)

        worked_context = self._build_worked_space_context(
            distance_m=int(worked_distance_bin),
            pa_label_by_id=pa_label_by_id,
        )
        worked_frontier_table = self._prepare_candidate_table(
            prune_candidate_frontier(worked_context["full_space_table"])
        )
        comparison_frontier_table = self._prepare_candidate_table(
            prune_candidate_frontier(
                _build_slot_normalized_candidate_table(
                    int(comparison_distance_bin),
                    engine_state=engine_state,
                )
            )
        )

        return CandidateSpaceArtifacts(
            worked_distance_m=int(worked_distance_bin),
            comparison_distance_m=int(comparison_distance_bin),
            supported_distance_bins=tuple(int(value) for value in DISTANCE_BIN_GRID_M),
            pa_label_by_id=pa_label_by_id,
            pa_color_map=pa_color_map,
            row_contract=worked_context["row_contract"].copy(),
            full_space_table=worked_context["full_space_table"].copy(),
            annotated_full_space_table=worked_context["annotated_full_space_table"].copy(),
            zoom_table=worked_context["zoom_table"].copy(),
            zoom_x_limits_bits=tuple(worked_context["zoom_x_limits_bits"]),
            zoom_y_limits_w=tuple(worked_context["zoom_y_limits_w"]),
            dominated_row=worked_context["dominated_row"].copy(),
            dominating_row=worked_context["dominating_row"].copy(),
            worked_frontier_table=worked_frontier_table.copy(),
            comparison_frontier_table=comparison_frontier_table.copy(),
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

    def plot_inherited_row_contract(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot one compact card for the inherited evaluated single-slot row contract."""

        row = artifacts.row_contract
        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(11.8, 4.6),
        )
        ax.set_axis_off()
        self._add_card_group(
            ax,
            x=0.04,
            y=0.16,
            width=0.92,
            height=0.68,
            header="Evaluated single-slot row",
            rows=[
                ("pa_id", f"{int(row['pa_id'])} ({artifacts.pa_label_by_id[int(row['pa_id'])]})"),
                ("n_prb", str(int(row["n_prb"]))),
                ("layers", str(int(row["layers"]))),
                ("mcs", str(int(row["mcs"]))),
                ("bits_per_slot", self._format_bits_per_slot(float(row["bits_per_slot"]))),
                ("p_out_total_w", self._format_power_w(float(row["p_out_total_w"]))),
                ("p_dc_active_w", self._format_power_w(float(row["p_dc_active_w"]))),
                ("feasible", "True"),
            ],
            accent_color=self.theme.primary,
            n_columns=4,
        )
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_single_slot_space(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the full feasible one-slot space for the worked distance bin."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(9.4, 5.4),
        )
        self._plot_candidate_cloud(
            ax=ax,
            table=artifacts.full_space_table,
            pa_color_map=artifacts.pa_color_map,
            pa_label_by_id=artifacts.pa_label_by_id,
            alpha=0.62,
            size=16,
            legend=True,
        )
        ax.set_xlabel("Payload per active slot (kbit)")
        ax.set_ylabel("Active PA DC power (W)")
        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_single_slot_space_zoom(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot the full one-slot space and one local zoom region."""

        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=1,
            ncols=2,
            figsize=(13.2, 5.6),
            squeeze=False,
        )
        full_ax, zoom_ax = axes.ravel()
        self._plot_candidate_cloud(
            ax=full_ax,
            table=artifacts.full_space_table,
            pa_color_map=artifacts.pa_color_map,
            pa_label_by_id=artifacts.pa_label_by_id,
            alpha=0.34,
            size=10,
            legend=True,
        )
        self._plot_zoom_box(
            full_ax,
            x_limits_bits=artifacts.zoom_x_limits_bits,
            y_limits_w=artifacts.zoom_y_limits_w,
        )
        full_ax.set_xlabel("Payload per active slot (kbit)")
        full_ax.set_ylabel("Active PA DC power (W)")

        self._plot_candidate_cloud(
            ax=zoom_ax,
            table=artifacts.zoom_table,
            pa_color_map=artifacts.pa_color_map,
            pa_label_by_id=artifacts.pa_label_by_id,
            alpha=0.72,
            size=28,
            legend=False,
        )
        zoom_ax.set_xlim(
            float(artifacts.zoom_x_limits_bits[0]) / 1e3,
            float(artifacts.zoom_x_limits_bits[1]) / 1e3,
        )
        zoom_ax.set_ylim(*artifacts.zoom_y_limits_w)
        zoom_ax.set_xlabel("Payload per active slot (kbit)")
        zoom_ax.set_ylabel("Active PA DC power (W)")
        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, (full_ax, zoom_ax)

    def plot_pruning_rule_diagram(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the exact same-PA strict pruning rule used by the stored frontier."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.4, 4.8),
        )
        ax.set_axis_off()
        self._add_diagram_box(
            ax,
            x=0.05,
            y=0.20,
            width=0.24,
            height=0.58,
            header="Kept row",
            lines=[
                "same pa_id",
                "n_prb_kept <= n_prb_drop",
                "p_dc_active_w_kept <= p_dc_active_w_drop",
                "bits_per_slot_kept >= bits_per_slot_drop",
            ],
            facecolor=colors.to_hex(colors.to_rgba(self.theme.primary, 0.08)),
            edgecolor=self.theme.primary,
        )
        self._add_diagram_box(
            ax,
            x=0.38,
            y=0.28,
            width=0.24,
            height=0.42,
            header="Strict improvement",
            lines=[
                "at least one axis improves strictly",
                "n_prb_kept < n_prb_drop",
                "or",
                "p_dc_active_w_kept < p_dc_active_w_drop",
                "or",
                "bits_per_slot_kept > bits_per_slot_drop",
            ],
            facecolor=self.theme.background,
            edgecolor=self.theme.grid,
        )
        self._add_diagram_box(
            ax,
            x=0.71,
            y=0.20,
            width=0.24,
            height=0.58,
            header="Dominated row",
            lines=[
                "same pa_id",
                "more PRBs is allowed here",
                "more active power is allowed here",
                "less payload per slot is allowed here",
            ],
            facecolor=colors.to_hex(colors.to_rgba(self.theme.highlight, 0.20)),
            edgecolor=self.theme.accent,
        )
        self._add_diagram_arrow(ax, (0.29, 0.49), (0.38, 0.49))
        self._add_diagram_arrow(ax, (0.62, 0.49), (0.71, 0.49))
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_dominated_vs_kept_example(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot one explicit dominated-row versus kept-row pruning example."""

        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=1,
            ncols=2,
            figsize=(13.4, 5.6),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.6, 1.0]},
        )
        scatter_ax, card_ax = axes.ravel()
        card_ax.set_axis_off()

        self._plot_candidate_cloud(
            ax=scatter_ax,
            table=artifacts.zoom_table,
            pa_color_map=artifacts.pa_color_map,
            pa_label_by_id=artifacts.pa_label_by_id,
            alpha=0.45,
            size=22,
            legend=False,
        )
        self._highlight_example_pair(scatter_ax, artifacts)
        scatter_ax.set_xlim(
            float(artifacts.zoom_x_limits_bits[0]) / 1e3,
            float(artifacts.zoom_x_limits_bits[1]) / 1e3,
        )
        scatter_ax.set_ylim(*artifacts.zoom_y_limits_w)
        scatter_ax.set_xlabel("Payload per active slot (kbit)")
        scatter_ax.set_ylabel("Active PA DC power (W)")

        dominated = artifacts.dominated_row
        dominating = artifacts.dominating_row
        self._add_card_group(
            card_ax,
            x=0.05,
            y=0.54,
            width=0.90,
            height=0.34,
            header="Kept row",
            rows=self._build_pair_card_rows(dominating),
            accent_color=self.theme.primary,
            n_columns=2,
        )
        self._add_card_group(
            card_ax,
            x=0.05,
            y=0.12,
            width=0.90,
            height=0.34,
            header="Pruned row",
            rows=self._build_pair_card_rows(dominated),
            accent_color=self.theme.highlight,
            n_columns=2,
        )
        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, (scatter_ax, card_ax)

    def plot_space_vs_frontier(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot the worked full feasible space beside the stored frontier."""

        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=1,
            ncols=2,
            figsize=(13.0, 5.4),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        full_ax, frontier_ax = axes.ravel()
        self._plot_candidate_cloud(
            ax=full_ax,
            table=artifacts.full_space_table,
            pa_color_map=artifacts.pa_color_map,
            pa_label_by_id=artifacts.pa_label_by_id,
            alpha=0.40,
            size=10,
            legend=True,
        )
        self._plot_candidate_cloud(
            ax=frontier_ax,
            table=artifacts.worked_frontier_table,
            pa_color_map=artifacts.pa_color_map,
            pa_label_by_id=artifacts.pa_label_by_id,
            alpha=0.86,
            size=18,
            legend=False,
        )
        for axis in (full_ax, frontier_ax):
            axis.set_xlabel("Payload per active slot (kbit)")
            axis.set_ylabel("Active PA DC power (W)")
        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, (full_ax, frontier_ax)

    def plot_frontier_distance_comparison(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot stored frontiers at two distances on the same visual scale."""

        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=1,
            ncols=2,
            figsize=(13.0, 5.4),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        short_ax, long_ax = axes.ravel()
        combined_bits_max = max(
            float(artifacts.comparison_frontier_table["bits_per_slot"].max()),
            float(artifacts.worked_frontier_table["bits_per_slot"].max()),
        )
        combined_power_max = max(
            float(artifacts.comparison_frontier_table["p_dc_active_w"].max()),
            float(artifacts.worked_frontier_table["p_dc_active_w"].max()),
        )

        self._plot_candidate_cloud(
            ax=short_ax,
            table=artifacts.comparison_frontier_table,
            pa_color_map=artifacts.pa_color_map,
            pa_label_by_id=artifacts.pa_label_by_id,
            alpha=0.88,
            size=22,
            legend=True,
        )
        self._plot_candidate_cloud(
            ax=long_ax,
            table=artifacts.worked_frontier_table,
            pa_color_map=artifacts.pa_color_map,
            pa_label_by_id=artifacts.pa_label_by_id,
            alpha=0.88,
            size=22,
            legend=False,
        )
        for axis in (short_ax, long_ax):
            axis.set_xlim(0.0, combined_bits_max / 1e3 * 1.04)
            axis.set_ylim(0.0, combined_power_max * 1.04)
            axis.set_xlabel("Payload per active slot (kbit)")
            axis.set_ylabel("Active PA DC power (W)")
        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, (short_ax, long_ax)

    def plot_distance_bin_storage_strip(
        self,
        artifacts: CandidateSpaceArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the supported distance-bin strip used for stored frontier construction."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.0, 2.6),
        )
        y_values = np.zeros(len(artifacts.supported_distance_bins), dtype=float)
        ax.scatter(
            list(artifacts.supported_distance_bins),
            y_values,
            s=46,
            color=self.theme.neutral_light,
            edgecolors=self.theme.neutral_dark,
            linewidths=0.45,
            zorder=2,
        )
        for distance_m, color in (
            (artifacts.comparison_distance_m, self.theme.highlight),
            (artifacts.worked_distance_m, self.theme.primary),
        ):
            ax.scatter(
                [distance_m],
                [0.0],
                s=140,
                color=color,
                edgecolors=self.theme.accent,
                linewidths=0.8,
                zorder=3,
            )
            ax.text(
                float(distance_m),
                0.12,
                f"{int(distance_m)} m",
                ha="center",
                va="bottom",
                fontsize=9,
                color=self.theme.text,
            )
        ax.set_yticks([])
        ax.set_xlabel("Supported static distance bin (m)")
        ax.set_xlim(min(artifacts.supported_distance_bins) - 10, max(artifacts.supported_distance_bins) + 10)
        ax.set_ylim(-0.2, 0.35)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_handoff_strip(
        self,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the notebook handoff from stored frontier construction to lookup."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(13.0, 2.8),
        )
        ax.set_axis_off()
        boxes = [
            ("Stored frontier", 0.05, self.theme.primary),
            ("Distance-bin snap", 0.31, self.theme.neutral_dark),
            ("Trusted frontier retrieval", 0.53, self.theme.neutral_dark),
            ("Later rate expansion", 0.75, self.theme.highlight),
        ]
        for label, x_pos, accent in boxes:
            self._add_diagram_box(
                ax,
                x=x_pos,
                y=0.28,
                width=0.18,
                height=0.40,
                header=label,
                lines=[],
                facecolor=colors.to_hex(colors.to_rgba(accent, 0.12)),
                edgecolor=accent,
            )
        self._add_diagram_arrow(ax, (0.23, 0.48), (0.31, 0.48))
        self._add_diagram_arrow(ax, (0.49, 0.48), (0.53, 0.48))
        self._add_diagram_arrow(ax, (0.71, 0.48), (0.75, 0.48))
        self._export_figure(fig, export_path)
        return fig, ax

    def _build_worked_space_context(
        self,
        *,
        distance_m: int,
        pa_label_by_id: dict[int, str],
    ) -> dict[str, object]:
        """Resolve the worked one-slot space, one example pair, and one zoom slice."""

        full_space_table = self._prepare_candidate_table(
            _build_slot_normalized_candidate_table(int(distance_m))
        )
        annotated_full_space_table = self._prepare_candidate_table(
            annotate_same_pa_dominance(
                full_space_table,
                resource_column="n_prb",
                power_column="p_dc_active_w",
                rate_column="bits_per_slot",
            )
        )
        dominated_row, dominating_row = self._select_worked_pair(annotated_full_space_table)
        zoom_table, zoom_x_limits_bits, zoom_y_limits_w = self._build_zoom_slice(
            annotated_full_space_table,
            dominated_row=dominated_row,
            dominating_row=dominating_row,
        )
        return {
            "row_contract": dominating_row.reindex(
                [*BATCH_USER_PARAMETER_SPACE_COLUMNS, "bits_per_slot_kbit"]
            ),
            "full_space_table": full_space_table,
            "annotated_full_space_table": annotated_full_space_table,
            "zoom_table": zoom_table,
            "zoom_x_limits_bits": zoom_x_limits_bits,
            "zoom_y_limits_w": zoom_y_limits_w,
            "dominated_row": dominated_row,
            "dominating_row": dominating_row,
        }

    def _prepare_candidate_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Attach display-oriented columns without changing the stored contract."""

        plot_table = table.copy()
        if "bits_per_slot" in plot_table.columns:
            plot_table["bits_per_slot_kbit"] = plot_table["bits_per_slot"].astype(float) / 1e3
        return plot_table.reset_index(drop=True)

    def _select_worked_pair(
        self,
        annotated_table: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """Choose one explicit dominated-versus-kept example for the notebook."""

        dominated_rows = annotated_table.loc[
            annotated_table["pruning_role"].eq("dominated")
        ].copy()
        if dominated_rows.empty:
            raise ValueError("The worked table does not contain any dominated one-slot rows.")

        bits_mid_low = float(dominated_rows["bits_per_slot"].quantile(0.25))
        bits_mid_high = float(dominated_rows["bits_per_slot"].quantile(0.75))
        bits_midpoint = float(dominated_rows["bits_per_slot"].median())
        candidates = []
        for row in dominated_rows.itertuples(index=False):
            dominator_matches = annotated_table.loc[
                annotated_table["row_id"].eq(int(row.dominator_row_id))
            ]
            if dominator_matches.empty:
                continue
            dominator = dominator_matches.iloc[0]
            bits_ratio = float(dominator["bits_per_slot"]) / max(float(row.bits_per_slot), 1e-12)
            power_gap = float(row.p_dc_active_w) - float(dominator["p_dc_active_w"])
            if bits_ratio > 1.06:
                continue
            if power_gap < 0.02:
                continue

            candidates.append(
                {
                    "dominated_row": row,
                    "dominating_row": dominator,
                    "same_n_prb": int(dominator["n_prb"]) == int(row.n_prb),
                    "mid_band": bits_mid_low <= float(row.bits_per_slot) <= bits_mid_high,
                    "bits_ratio_gap": abs(bits_ratio - 1.0),
                    "power_gap": power_gap,
                    "bits_distance_to_mid": abs(float(row.bits_per_slot) - bits_midpoint),
                }
            )

        if not candidates:
            raise ValueError("Unable to select one dominated one-slot example for the notebook.")

        best_pair = sorted(
            candidates,
            key=lambda item: (
                not bool(item["same_n_prb"]),
                not bool(item["mid_band"]),
                float(item["bits_ratio_gap"]),
                -float(item["power_gap"]),
                float(item["bits_distance_to_mid"]),
            ),
        )[0]
        dominated_row = annotated_table.loc[
            annotated_table["row_id"].eq(int(best_pair["dominated_row"].row_id))
        ].iloc[0]
        dominating_row = best_pair["dominating_row"]
        return dominated_row, dominating_row

    def _build_zoom_slice(
        self,
        annotated_table: pd.DataFrame,
        *,
        dominated_row: pd.Series,
        dominating_row: pd.Series,
    ) -> tuple[pd.DataFrame, tuple[float, float], tuple[float, float]]:
        """Build one local zoom around the worked dominated-versus-kept pair."""

        bits_values = [float(dominated_row["bits_per_slot"]), float(dominating_row["bits_per_slot"])]
        power_values = [float(dominated_row["p_dc_active_w"]), float(dominating_row["p_dc_active_w"])]
        x_span = max(abs(max(bits_values) - min(bits_values)), 3000.0)
        y_span = max(abs(max(power_values) - min(power_values)), 0.8)
        x_limits_bits = (
            max(0.0, min(bits_values) - 0.45 * x_span),
            max(bits_values) + 0.45 * x_span,
        )
        y_limits_w = (
            max(0.0, min(power_values) - 0.65 * y_span),
            max(power_values) + 0.65 * y_span,
        )
        zoom_table = annotated_table.loc[
            annotated_table["bits_per_slot"].between(
                float(x_limits_bits[0]),
                float(x_limits_bits[1]),
                inclusive="both",
            )
            & annotated_table["p_dc_active_w"].between(
                float(y_limits_w[0]),
                float(y_limits_w[1]),
                inclusive="both",
            )
        ].copy()
        return zoom_table, x_limits_bits, y_limits_w

    def _plot_candidate_cloud(
        self,
        *,
        ax: plt.Axes,
        table: pd.DataFrame,
        pa_color_map: dict[int, str],
        pa_label_by_id: dict[int, str],
        alpha: float,
        size: float,
        legend: bool,
    ) -> None:
        """Plot one candidate cloud using the slot-normalized axes."""

        for pa_id, pa_rows in table.groupby("pa_id", sort=True):
            ax.scatter(
                pa_rows["bits_per_slot_kbit"].astype(float),
                pa_rows["p_dc_active_w"].astype(float),
                s=size,
                alpha=alpha,
                color=pa_color_map[int(pa_id)],
                edgecolors=self.theme.background,
                linewidths=0.35,
                label=pa_label_by_id[int(pa_id)],
            )
        if legend:
            style_legend(ax.legend(frameon=True, loc="upper left"), theme=self.theme)

    def _plot_zoom_box(
        self,
        ax: plt.Axes,
        *,
        x_limits_bits: tuple[float, float],
        y_limits_w: tuple[float, float],
    ) -> None:
        """Draw one zoom box on the full-space plot."""

        ax.add_patch(
            Rectangle(
                (float(x_limits_bits[0]) / 1e3, float(y_limits_w[0])),
                (float(x_limits_bits[1]) - float(x_limits_bits[0])) / 1e3,
                float(y_limits_w[1]) - float(y_limits_w[0]),
                facecolor="none",
                edgecolor=self.theme.accent,
                linewidth=1.2,
                linestyle="--",
            )
        )

    def _highlight_example_pair(
        self,
        ax: plt.Axes,
        artifacts: CandidateSpaceArtifacts,
    ) -> None:
        """Highlight the kept and pruned rows inside the local zoom."""

        dominated = artifacts.dominated_row
        dominating = artifacts.dominating_row
        ax.scatter(
            [float(dominating["bits_per_slot"]) / 1e3],
            [float(dominating["p_dc_active_w"])],
            s=170,
            color=self.theme.primary,
            edgecolors=self.theme.accent,
            linewidths=1.0,
            zorder=4,
        )
        ax.scatter(
            [float(dominated["bits_per_slot"]) / 1e3],
            [float(dominated["p_dc_active_w"])],
            s=170,
            facecolors="none",
            edgecolors=self.theme.highlight,
            linewidths=2.0,
            zorder=5,
        )
        ax.text(
            float(dominating["bits_per_slot"]) / 1e3,
            float(dominating["p_dc_active_w"]) + 0.14,
            "Kept",
            ha="center",
            va="bottom",
            fontsize=9,
            color=self.theme.text,
        )
        ax.text(
            float(dominated["bits_per_slot"]) / 1e3,
            float(dominated["p_dc_active_w"]) + 0.14,
            "Pruned",
            ha="center",
            va="bottom",
            fontsize=9,
            color=self.theme.text,
        )

    def _build_pair_card_rows(self, row: pd.Series) -> list[tuple[str, str]]:
        """Return the compact comparison rows used in the worked example card."""

        return [
            ("pa_id", str(int(row["pa_id"]))),
            ("n_prb", str(int(row["n_prb"]))),
            ("layers", str(int(row["layers"]))),
            ("mcs", str(int(row["mcs"]))),
            ("bits_per_slot", self._format_bits_per_slot(float(row["bits_per_slot"]))),
            ("p_dc_active_w", self._format_power_w(float(row["p_dc_active_w"]))),
        ]

    def _build_pa_color_map(self, pa_label_by_id: dict[int, str]) -> dict[int, str]:
        """Return one stable color per PA family for the notebook sequence."""

        color_map: dict[int, str] = {}
        fallback_palette = [self.theme.secondary, self.theme.neutral_dark]
        fallback_index = 0
        for pa_id in sorted(int(value) for value in pa_label_by_id):
            label = str(pa_label_by_id[int(pa_id)])
            if label == "4W PA":
                color_map[int(pa_id)] = self.theme.primary
                continue
            if label == "8W PA":
                color_map[int(pa_id)] = self.theme.highlight
                continue
            color_map[int(pa_id)] = fallback_palette[fallback_index % len(fallback_palette)]
            fallback_index += 1
        return color_map

    def _add_card_group(
        self,
        ax: plt.Axes,
        *,
        x: float,
        y: float,
        width: float,
        height: float,
        header: str,
        rows: list[tuple[str, str]],
        accent_color: str,
        n_columns: int = 1,
    ) -> None:
        """Draw one grouped notebook card."""

        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=self.theme.background,
            edgecolor=self.theme.grid,
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        header_patch = Rectangle(
            (x, y + height - 0.09),
            width,
            0.09,
            transform=ax.transAxes,
            facecolor=colors.to_hex(colors.to_rgba(accent_color, 0.18)),
            edgecolor="none",
        )
        ax.add_patch(header_patch)
        ax.text(
            x + 0.02,
            y + height - 0.045,
            header,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=self.theme.text,
        )

        resolved_columns = max(int(n_columns), 1)
        column_width = width / float(resolved_columns)
        rows_per_column = int(np.ceil(len(rows) / float(resolved_columns)))
        for index, (label, value) in enumerate(rows):
            column = index // rows_per_column
            row_index = index % rows_per_column
            anchor_x = x + column * column_width + 0.02
            anchor_y = y + height - 0.13 - row_index * 0.11
            ax.text(
                anchor_x,
                anchor_y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                color=self.theme.neutral_dark,
            )
            ax.text(
                anchor_x,
                anchor_y - 0.035,
                value,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color=self.theme.text,
            )

    def _add_diagram_box(
        self,
        ax: plt.Axes,
        *,
        x: float,
        y: float,
        width: float,
        height: float,
        header: str,
        lines: list[str],
        facecolor: str,
        edgecolor: str,
    ) -> None:
        """Draw one rounded diagram box."""

        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.02,
            y + height - 0.05,
            header,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=self.theme.text,
        )
        if not lines:
            return

        line_step = (height - 0.11) / max(len(lines), 1)
        for index, line in enumerate(lines):
            ax.text(
                x + 0.02,
                y + height - 0.09 - index * line_step,
                line,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color=self.theme.text,
                family="monospace" if any(symbol in line for symbol in ("<", ">", "=")) else None,
            )

    def _add_diagram_arrow(
        self,
        ax: plt.Axes,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        """Draw one diagram arrow in axes coordinates."""

        arrow = FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color=self.theme.neutral_dark,
        )
        ax.add_patch(arrow)

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
    def _format_bits_per_slot(bits_per_slot: float) -> str:
        """Format one slot payload value for notebook display."""

        return f"{float(bits_per_slot) / 1e3:.2f} kbit"

    @staticmethod
    def _format_power_w(power_w: float) -> str:
        """Format one power value with a stable engineering unit."""

        resolved_power = float(power_w)
        if resolved_power >= 10.0:
            return f"{resolved_power:.2f} W"
        return f"{resolved_power:.3f} W"


__all__ = [
    "CandidateSpaceArtifacts",
    "CandidateSpaceHelpers",
]
