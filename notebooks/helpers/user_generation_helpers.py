from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from day_cycle_simulation.models import SyntheticSessionGenerationConfig
from support.day_cycle import (
    BITS_PER_GB,
    bin_index_to_clock,
    build_day_cycle_discussion_artifacts,
    export_doc_figure,
)
from support.theme import (
    NotebookTheme,
    apply_axis_style,
    create_themed_figure,
    get_notebook_theme,
    render_html_table,
    style_legend,
)


_SESSION_TABLE_FORMATS = {
    "Distance (m)": "{:.0f}",
    "Total data (GB)": "{:.2f}",
    "Required rate (Mbps)": "{:.2f}",
}

_SCHEDULER_ROW_FORMATS = {
    "Distance (m)": "{:.0f}",
    "Required rate (Mbps)": "{:.2f}",
}


@dataclass(frozen=True)
class UserGenerationArtifacts:
    """Lean notebook payload for the user-generation walkthrough."""

    hourly_load_table: pd.DataFrame
    target_load_table: pd.DataFrame
    lane_table: pd.DataFrame
    bin_validation_table: pd.DataFrame
    example_session_view: pd.DataFrame
    example_scheduler_rows: pd.DataFrame
    day_bin_count: int


@dataclass(frozen=True)
class _SessionLayoutSpec:
    """Theme-aware rendering settings for one session-layout view."""

    column: str
    title: str
    colorbar_label: str
    cmap_colors: tuple[str, ...]


class UserGenerationHelpers:
    """Theme-aware presentation helpers for Notebook 2."""

    def __init__(self, *, theme: str | NotebookTheme = "aalto_elec"):
        self.theme = get_notebook_theme(theme)

    def build_artifacts(
        self,
        *,
        load_curve_csv: Path,
        config: SyntheticSessionGenerationConfig,
    ) -> UserGenerationArtifacts:
        """Build the compact demand-generation views used in Notebook 2.

        Steps:
        1. Load one hourly demand profile and expand it onto the 15-minute day bins.
        2. Generate the synthetic sessions and the scheduler-facing per-bin user rows.
        3. Validate the rebuilt demand residuals and keep only the notebook views that explain the workflow.
        """

        artifact_map = build_day_cycle_discussion_artifacts(
            Path(load_curve_csv),
            config,
        )
        self._validate_residual_bounds(
            artifact_map["bin_validation_table"],
            smallest_total_data_bits=float(min(config.total_data_presets_bits)),
        )

        return UserGenerationArtifacts(
            hourly_load_table=artifact_map["hourly_load_table"].copy(),
            target_load_table=artifact_map["target_load_table"].copy(),
            lane_table=artifact_map["lane_table"].copy(),
            bin_validation_table=artifact_map["bin_validation_table"].copy(),
            example_session_view=artifact_map["example_session_view"].copy(),
            example_scheduler_rows=artifact_map["example_scheduler_rows"].copy(),
            day_bin_count=int(config.day_bin_count),
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

    def display_session_expansion(self, artifacts: UserGenerationArtifacts) -> None:
        """Show one example session and the scheduler rows it expands into."""

        display(
            self.render_table(
                artifacts.example_session_view,
                formats=_SESSION_TABLE_FORMATS,
                caption="One example generated session",
            )
        )
        display(
            self.render_table(
                artifacts.example_scheduler_rows,
                formats=_SCHEDULER_ROW_FORMATS,
                caption="Scheduler rows produced by that session",
            )
        )

    def plot_hourly_offered_load(
        self,
        artifacts: UserGenerationArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the hourly demand curve that seeds the day-level generator."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.0, 4.5),
        )
        hourly_load_table = artifacts.hourly_load_table

        ax.plot(
            hourly_load_table["hour"],
            hourly_load_table["total_load_gbph"],
            color=self.theme.primary,
            linewidth=2.4,
            marker="o",
            markersize=4.2,
            markerfacecolor=self.theme.background,
            markeredgecolor=self.theme.primary,
            markeredgewidth=1.1,
        )
        ax.fill_between(
            hourly_load_table["hour"],
            hourly_load_table["total_load_gbph"],
            color=self.theme.primary,
            alpha=0.14,
        )
        ax.set_xlim(1.0, 24.0)
        ax.set_xticks(list(range(1, 25, 2)))
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Offered load (GB/h)")
        ax.set_title("Hourly offered load used by the day-cycle generator")
        fig.tight_layout()

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_target_bins(
        self,
        artifacts: UserGenerationArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the quarter-hour demand after expanding the hourly load curve."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.0, 4.5),
        )
        validation_table = artifacts.bin_validation_table

        ax.bar(
            artifacts.target_load_table["bin_index"],
            validation_table["target_load_gb_in_bin"],
            width=0.9,
            color=self.theme.primary,
            alpha=0.22,
            label="Target load in each quarter-hour bin",
        )
        ax.step(
            artifacts.target_load_table["bin_index"],
            validation_table["target_load_gb_in_bin"],
            where="mid",
            color=self.theme.secondary,
            linewidth=2.1,
            label="Piecewise-constant hourly expansion",
        )
        self._set_day_bin_axis(ax, day_bin_count=artifacts.day_bin_count)
        ax.set_xlabel("Quarter-hour bin index")
        ax.set_ylabel("Target load in bin (GB)")
        ax.set_title("Quarter-hour target load after hourly expansion")
        style_legend(ax, theme=self.theme)
        fig.tight_layout()

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_session_layout(
        self,
        artifacts: UserGenerationArtifacts,
        *,
        color_by: str = "throughput",
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the generated sessions across the day using one themed color encoding."""

        plot_spec = self._resolve_session_layout_spec(color_by)
        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(14.0, 6.5),
        )
        apply_axis_style(ax, theme=self.theme, grid_axis="none")

        lane_table = artifacts.lane_table
        color_values = lane_table[plot_spec.column].astype(float)
        color_norm = colors.Normalize(
            vmin=float(color_values.min()),
            vmax=float(color_values.max()),
        )
        color_map = colors.LinearSegmentedColormap.from_list(
            f"{self.theme.name}_{plot_spec.column}",
            list(plot_spec.cmap_colors),
        )

        for row in lane_table.itertuples(index=False):
            color_value = float(getattr(row, plot_spec.column))
            rectangle = patches.Rectangle(
                (float(row.start_bin) - 0.5, float(row.lane_index)),
                float(row.duration_bins),
                0.9,
                facecolor=color_map(color_norm(color_value)),
                edgecolor=self.theme.neutral_dark,
                linewidth=0.55,
                alpha=0.9,
            )
            ax.add_patch(rectangle)

        self._set_day_bin_axis(ax, day_bin_count=artifacts.day_bin_count)
        ax.set_ylim(-0.2, float(lane_table["lane_index"].max()) + 1.2)
        ax.set_xlabel("Quarter-hour bin index")
        ax.set_ylabel("Visual session lane")
        ax.set_title(plot_spec.title)

        scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(
            scalar_mappable,
            ax=ax,
            label=plot_spec.colorbar_label,
            fraction=0.035,
            pad=0.03,
        )
        colorbar.outline.set_edgecolor(self.theme.neutral_dark)
        colorbar.outline.set_linewidth(0.9)
        colorbar.ax.tick_params(colors=self.theme.neutral_dark, labelcolor=self.theme.neutral_dark)
        colorbar.ax.yaxis.label.set_color(self.theme.text)
        fig.subplots_adjust(left=0.07, right=0.90, bottom=0.11, top=0.90)

        self._export_figure(fig, export_path)
        return fig, ax

    def plot_day_validation(
        self,
        artifacts: UserGenerationArtifacts,
        *,
        highlight_bin_index: int | None = None,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot the rebuilt day load and the active-user count per bin."""

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
            validation_table["bin_index"],
            validation_table["target_load_gb_in_bin"],
            width=0.9,
            color=self.theme.primary,
            alpha=0.22,
            label="Target load from the quarter-hour bins",
        )
        load_ax.plot(
            validation_table["bin_index"],
            validation_table["rebuilt_load_gb_in_bin"],
            color=self.theme.accent,
            linewidth=2.0,
            label="Load rebuilt from generated sessions",
        )
        load_ax.set_ylabel("Load in bin (GB)")
        load_ax.set_title("Generated sessions reproduce the quarter-hour target load")
        style_legend(load_ax, theme=self.theme)

        active_ax.bar(
            validation_table["bin_index"],
            validation_table["active_users"],
            width=0.9,
            color=self.theme.secondary,
            alpha=0.85,
        )
        active_ax.set_xlabel("Quarter-hour bin index")
        active_ax.set_ylabel("Active users")
        active_ax.set_title("Scheduler-facing active users in each quarter-hour bin")

        for ax in (load_ax, active_ax):
            self._set_day_bin_axis(ax, day_bin_count=artifacts.day_bin_count)

        if highlight_bin_index is not None:
            highlight_label = f"Chosen bin ({bin_index_to_clock(int(highlight_bin_index))})"
            for ax in (load_ax, active_ax):
                left = float(highlight_bin_index) - 0.45
                right = float(highlight_bin_index) + 0.45
                ax.axvspan(left, right, color=self.theme.highlight, alpha=0.12, zorder=0.2)
                ax.axvline(
                    float(highlight_bin_index),
                    color=self.theme.highlight,
                    linestyle="--",
                    linewidth=1.3,
                )
                y_top = ax.get_ylim()[1]
                ax.text(
                    float(highlight_bin_index),
                    y_top * 0.97,
                    highlight_label,
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

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, (load_ax, active_ax)

    def _validate_residual_bounds(
        self,
        bin_validation_table: pd.DataFrame,
        *,
        smallest_total_data_bits: float,
    ) -> None:
        """Check that the session rebuild stays within the expected residual tolerance."""

        residual_bits = (
            bin_validation_table["residual_load_gb_in_bin"].to_numpy(dtype=float) * BITS_PER_GB
        )
        if not np.all(residual_bits >= -1.0):
            raise AssertionError("Generated day rebuild produced a negative residual load below tolerance.")
        if not np.all(residual_bits < float(smallest_total_data_bits) + 1.0):
            raise AssertionError("Generated day rebuild exceeded the smallest session-size residual bound.")

    def _resolve_session_layout_spec(self, color_by: str) -> _SessionLayoutSpec:
        """Resolve one notebook-facing session-color view."""

        normalized = str(color_by).strip().lower()
        if normalized == "throughput":
            return _SessionLayoutSpec(
                column="required_rate_mbps",
                title="Generated session placement across the simulated day",
                colorbar_label="Required throughput (Mbps)",
                cmap_colors=(
                    self.theme.neutral_light,
                    self.theme.highlight,
                    self.theme.primary,
                    self.theme.secondary,
                ),
            )
        if normalized == "distance":
            return _SessionLayoutSpec(
                column="distance_m",
                title="Generated session placement across the simulated day (colored by distance)",
                colorbar_label="Distance (m)",
                cmap_colors=(
                    self.theme.neutral_light,
                    self.theme.primary,
                    self.theme.accent,
                ),
            )

        raise ValueError("color_by must be either 'throughput' or 'distance'.")

    def _set_day_bin_axis(self, ax: plt.Axes, *, day_bin_count: int) -> None:
        """Apply the shared day-bin framing used across Notebook 2 figures."""

        for boundary in range(0, int(day_bin_count) + 1, 4):
            ax.axvline(boundary - 0.5, color=self.theme.grid, linewidth=0.8, zorder=0)
        ax.set_xlim(-0.5, int(day_bin_count) - 0.5)
        ax.set_xticks(list(range(0, int(day_bin_count), 8)))

    def _export_figure(self, fig: plt.Figure, export_path: Path | None) -> None:
        """Save one figure when the notebook requests a document export."""

        if export_path is None:
            return

        resolved_path = Path(export_path)
        export_doc_figure(
            fig,
            resolved_path.name,
            resolved_path.parent,
        )


__all__ = [
    "UserGenerationArtifacts",
    "UserGenerationHelpers",
]
