from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from support.day_cycle import export_doc_figure
from support.day_results import (
    build_day_results_artifacts,
    build_scenario_pa_choice_table,
    filter_pa_choice_table_to_500m_user_bins,
)
from support.theme import (
    NotebookTheme,
    apply_axis_style,
    create_themed_figure,
    get_notebook_theme,
    render_html_table,
    style_legend,
)


_SCENARIO_MODE_ORDER = (
    "4W only",
    "Mixed 4W/8W",
    "8W only",
)

_SCENARIO_MODE_THEME_ROLES = {
    "4W only": "primary",
    "Mixed 4W/8W": "secondary",
    "8W only": "neutral_dark",
    "Other": "neutral_light",
}

_PA_CHOICE_THEME_ROLES = {
    "4W PA": "primary",
    "Mixed PA use": "secondary",
    "8W PA": "neutral_dark",
    "Infeasible": "highlight",
}

_PA_MODE_LABELS = {
    ("4W PA",): "4W only",
    ("8W PA",): "8W only",
    ("4W PA", "8W PA"): "Mixed 4W/8W",
}


@dataclass(frozen=True)
class ScenarioResultsArtifacts:
    """Lean notebook payload for the day-results comparison notebook."""

    scenario_order: tuple[str, ...]
    display_labels: dict[str, str]
    scenario_story_table: pd.DataFrame
    scenario_mode_summary: pd.DataFrame
    bin_table_all: pd.DataFrame
    hard_off_500m_table: pd.DataFrame
    bin_duration_s: float
    burden_scenario_key: str


@dataclass(frozen=True)
class _ScenarioSpec:
    """One notebook-facing scenario definition entry."""

    key: str
    path: Path
    label: str
    role: str
    within_bin_rule: str
    why_it_matters: str


class ScenarioResultsHelpers:
    """Theme-aware presentation helpers for Notebook 5."""

    def __init__(self, *, theme: str | NotebookTheme = "aalto_elec"):
        self.theme = get_notebook_theme(theme)

    def build_artifacts(
        self,
        *,
        scenario_specs: Sequence[Mapping[str, object]],
        burden_scenario_key: str = "optimized",
    ) -> ScenarioResultsArtifacts:
        """Build the compact scenario-comparison views used in Notebook 5.

        Steps:
        1. Load the compared day-run exports and flatten them into bin-level result tables.
        2. Assemble the scenario-definition table and the solved-bin PA-mode summary.
        3. Keep only the cross-scenario traces and the focused 500 m burden slice used by the discussion figures.
        """

        normalized_specs = self._normalize_scenario_specs(scenario_specs)
        scenario_order = tuple(spec.key for spec in normalized_specs)
        display_labels = {spec.key: spec.label for spec in normalized_specs}
        if burden_scenario_key not in display_labels:
            supported = ", ".join(scenario_order)
            raise ValueError(
                f"Unknown burden_scenario_key '{burden_scenario_key}'. Supported scenarios: {supported}."
            )

        artifact_map = build_day_results_artifacts(
            {spec.key: spec.path for spec in normalized_specs}
        )
        bin_table_all = artifact_map["bin_table_all"].copy()
        allocation_table_all = artifact_map["allocation_table_all"].copy()
        hard_off_pa_choice_table = build_scenario_pa_choice_table(
            allocation_table_all,
            bin_table_all,
            scenario_label=burden_scenario_key,
            pa_label_map=artifact_map["scenario_runs"][burden_scenario_key]["pa_label_map"],
        )

        return ScenarioResultsArtifacts(
            scenario_order=scenario_order,
            display_labels=display_labels,
            scenario_story_table=self._build_scenario_story_table(
                normalized_specs,
                run_overview_table=artifact_map["run_overview_table"].copy(),
            ),
            scenario_mode_summary=self._build_scenario_mode_summary(
                bin_table_all=bin_table_all,
                allocation_table_all=allocation_table_all,
                scenario_order=scenario_order,
            ),
            bin_table_all=bin_table_all,
            hard_off_500m_table=filter_pa_choice_table_to_500m_user_bins(
                hard_off_pa_choice_table
            ),
            bin_duration_s=float(artifact_map["bin_duration_s"]),
            burden_scenario_key=str(burden_scenario_key),
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

    def plot_scenario_pa_modes(
        self,
        artifacts: ScenarioResultsArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot how each scenario uses the PA families across solved bins."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(9.6, 4.0),
        )
        apply_axis_style(ax, theme=self.theme, grid_axis="x")

        summary = (
            artifacts.scenario_mode_summary.reindex(artifacts.scenario_order)
            .fillna(0.0)
            .copy()
        )
        active_modes = [
            mode for mode in summary.columns if float(summary[mode].sum()) > 0.0
        ]
        scenario_labels = [
            artifacts.display_labels[scenario_label]
            for scenario_label in summary.index.tolist()
        ]
        mode_colors = {
            mode: self.theme.color(_SCENARIO_MODE_THEME_ROLES.get(mode, "neutral_light"))
            for mode in active_modes
        }
        left = np.zeros(len(summary), dtype=float)

        for mode in active_modes:
            values = summary[mode].to_numpy(dtype=float)
            ax.barh(
                scenario_labels,
                values,
                left=left,
                height=0.65,
                color=mode_colors[mode],
                label=mode,
            )
            label_color = self._contrasting_text_color(mode_colors[mode])
            for row_index, value in enumerate(values):
                if value <= 0.0:
                    continue
                ax.text(
                    left[row_index] + value / 2.0,
                    row_index,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color=label_color,
                    fontsize=10,
                    fontweight="bold",
                )
            left += values

        ax.invert_yaxis()
        ax.set_xlim(0.0, float(summary.sum(axis=1).max()) + 2.0)
        ax.set_xlabel("Solved quarter-hour bins")
        ax.legend(
            frameon=True,
            ncol=min(len(active_modes), 3),
            bbox_to_anchor=(0.5, 1.14),
            loc="upper center",
        )
        style_legend(ax, theme=self.theme)

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_day_power_trajectories(
        self,
        artifacts: ScenarioResultsArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the day-level total PA DC power trace for each scenario."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.0, 4.8),
        )
        color_map = self._build_scenario_color_map(artifacts.scenario_order)
        day_bin_count = int(artifacts.bin_table_all["bin_index"].max()) + 1

        for scenario_label in artifacts.scenario_order:
            scenario_bins = self._scenario_bins(
                artifacts.bin_table_all,
                scenario_label=scenario_label,
            )
            color = color_map[scenario_label]
            ax.plot(
                scenario_bins["bin_index"],
                scenario_bins["dc_total_w"],
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                markerfacecolor=self.theme.background,
                markeredgecolor=color,
                markeredgewidth=0.9,
                color=color,
                label=artifacts.display_labels[scenario_label],
            )
            finite_values = scenario_bins["dc_total_w"].to_numpy(dtype=float)
            if np.isfinite(finite_values).any():
                self._plot_infeasible_markers(
                    ax,
                    scenario_bins,
                    y_value=float(np.nanmax(finite_values)) * 1.02,
                )

        self._set_day_bin_axis(ax, day_bin_count=day_bin_count)
        ax.set_xlabel("Quarter-hour bin index")
        ax.set_ylabel("Total PA DC power (W)")
        ax.legend(frameon=True)
        style_legend(ax, theme=self.theme)

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_cumulative_day_energy(
        self,
        artifacts: ScenarioResultsArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the cumulative day energy for each scenario."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.0, 4.8),
        )
        color_map = self._build_scenario_color_map(artifacts.scenario_order)
        day_bin_count = int(artifacts.bin_table_all["bin_index"].max()) + 1

        for scenario_label in artifacts.scenario_order:
            scenario_bins = self._scenario_bins(
                artifacts.bin_table_all,
                scenario_label=scenario_label,
            )
            cumulative_energy_wh = np.nancumsum(
                scenario_bins["dc_total_w"].to_numpy(dtype=float)
                * float(artifacts.bin_duration_s)
                / 3600.0
            )
            ax.plot(
                scenario_bins["bin_index"],
                cumulative_energy_wh,
                linewidth=2.1,
                color=color_map[scenario_label],
                label=artifacts.display_labels[scenario_label],
            )

        self._set_day_bin_axis(ax, day_bin_count=day_bin_count)
        ax.set_xlabel("Quarter-hour bin index")
        ax.set_ylabel("Cumulative energy (Wh)")
        ax.legend(frameon=True)
        style_legend(ax, theme=self.theme)

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_day_load_power_scatter(
        self,
        artifacts: ScenarioResultsArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot total requested load against total PA DC power."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(8.8, 6.1),
        )
        color_map = self._build_scenario_color_map(artifacts.scenario_order)

        for scenario_label in artifacts.scenario_order:
            scenario_bins = self._scenario_bins(
                artifacts.bin_table_all,
                scenario_label=scenario_label,
            )
            scenario_bins = scenario_bins.loc[
                scenario_bins["status"].eq("solved")
            ].copy()
            if scenario_bins.empty:
                continue
            ax.scatter(
                scenario_bins["requested_rate_mbps"],
                scenario_bins["dc_total_w"],
                s=52,
                alpha=0.85,
                color=color_map[scenario_label],
                edgecolor=self.theme.neutral_dark,
                linewidth=0.45,
                label=artifacts.display_labels[scenario_label],
            )

        ax.set_xlabel("Total requested rate in bin (Mbps)")
        ax.set_ylabel("Total PA DC power (W)")
        ax.legend(frameon=True)
        style_legend(ax, theme=self.theme)

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_filtered_500m_total_burden_scatter(
        self,
        artifacts: ScenarioResultsArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the far-user burden split for the focused hard-off scenario."""

        burden_table = artifacts.hard_off_500m_table
        if burden_table.empty:
            raise ValueError(
                "The filtered PA-choice table does not contain any bins with a 500 m user."
            )

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(8.0, 5.8),
        )
        choice_labels = list(dict.fromkeys(burden_table["pa_choice_label"].astype(str)))
        pa_choice_colors = {
            label: self.theme.color(_PA_CHOICE_THEME_ROLES.get(label, "neutral_light"))
            for label in choice_labels
        }

        for choice_label in choice_labels:
            choice_rows = burden_table.loc[
                burden_table["pa_choice_label"].eq(choice_label)
            ].copy()
            marker = "x" if choice_label == "Infeasible" else "o"
            scatter_kwargs = {
                "s": 58,
                "marker": marker,
                "color": pa_choice_colors[choice_label],
                "label": choice_label,
                "zorder": 3,
            }
            if choice_label == "Infeasible":
                scatter_kwargs["linewidth"] = 1.2
            else:
                scatter_kwargs["edgecolor"] = self.theme.neutral_dark
                scatter_kwargs["linewidth"] = 0.5

            ax.scatter(
                choice_rows["requested_rate_500m_plus_mbps"],
                choice_rows["requested_rate_300m_to_499m_mbps"],
                **scatter_kwargs,
            )

            if choice_label not in {"8W PA", "Infeasible"}:
                continue
            for row in choice_rows.itertuples(index=False):
                ax.annotate(
                    f"{int(row.bin_index)}",
                    (
                        float(row.requested_rate_500m_plus_mbps),
                        float(row.requested_rate_300m_to_499m_mbps),
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=self.theme.neutral_dark,
                )

        ax.set_xlabel("Requested rate from users at >= 500 m (Mbps)")
        ax.set_ylabel("Requested rate from users at 300-499 m (Mbps)")
        ax.legend(frameon=True)
        style_legend(ax, theme=self.theme)

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def plot_day_used_slots(
        self,
        artifacts: ScenarioResultsArtifacts,
        *,
        export_path: Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot slot usage over the day for each scenario."""

        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.0, 4.8),
        )
        color_map = self._build_scenario_color_map(artifacts.scenario_order)
        day_bin_count = int(artifacts.bin_table_all["bin_index"].max()) + 1

        for scenario_label in artifacts.scenario_order:
            scenario_bins = self._scenario_bins(
                artifacts.bin_table_all,
                scenario_label=scenario_label,
            )
            color = color_map[scenario_label]
            ax.plot(
                scenario_bins["bin_index"],
                scenario_bins["used_slots"],
                linewidth=1.9,
                marker="o",
                markersize=2.8,
                markerfacecolor=self.theme.background,
                markeredgecolor=color,
                markeredgewidth=0.85,
                color=color,
                label=artifacts.display_labels[scenario_label],
            )

        self._set_day_bin_axis(ax, day_bin_count=day_bin_count)
        ax.set_xlabel("Quarter-hour bin index")
        ax.set_ylabel("Used slots")
        ax.legend(
            frameon=True,
            ncol=min(len(artifacts.scenario_order), 3),
        )
        style_legend(ax, theme=self.theme)

        fig.tight_layout()
        self._export_figure(fig, export_path)
        return fig, ax

    def _build_scenario_story_table(
        self,
        specs: Sequence[_ScenarioSpec],
        *,
        run_overview_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine notebook scenario notes with the flattened run summary."""

        overview_lookup = (
            run_overview_table.rename(columns={"Scenario": "scenario_label"})
            .assign(scenario_label=lambda table: table["scenario_label"].astype(str))
            .set_index("scenario_label")
        )
        rows = []

        for spec in specs:
            overview_row = overview_lookup.loc[spec.key]
            rows.append(
                {
                    "Scenario": spec.label,
                    "Role": spec.role,
                    "Within-bin rule": spec.within_bin_rule,
                    "Why it matters": spec.why_it_matters,
                    "Mean total power (W)": float(overview_row["Mean total power (W)"]),
                    "Day energy (Wh)": float(overview_row["Day energy (Wh)"]),
                }
            )

        return pd.DataFrame(rows)

    def _build_scenario_mode_summary(
        self,
        *,
        bin_table_all: pd.DataFrame,
        allocation_table_all: pd.DataFrame,
        scenario_order: Sequence[str],
    ) -> pd.DataFrame:
        """Summarize the PA family usage pattern for each solved bin."""

        solved_bins = bin_table_all.loc[
            bin_table_all["status"].eq("solved"),
            ["scenario_label", "bin_index"],
        ].copy()
        pa_labels_per_bin = (
            allocation_table_all.groupby(
                ["scenario_label", "bin_index"],
                dropna=False,
            )["pa_label"]
            .agg(lambda values: tuple(sorted(pd.unique(values))))
            .reset_index(name="pa_labels")
        )
        scenario_mode_table = solved_bins.merge(
            pa_labels_per_bin,
            on=["scenario_label", "bin_index"],
            how="left",
        )
        scenario_mode_table["pa_mode"] = (
            scenario_mode_table["pa_labels"].map(_PA_MODE_LABELS).fillna("Other")
        )
        summary = (
            scenario_mode_table.groupby(["scenario_label", "pa_mode"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reindex(index=list(scenario_order), fill_value=0)
        )

        ordered_columns = [
            *[column for column in _SCENARIO_MODE_ORDER if column in summary.columns],
            *[
                column
                for column in summary.columns
                if column not in _SCENARIO_MODE_ORDER
            ],
        ]
        return summary.reindex(columns=ordered_columns, fill_value=0)

    def _normalize_scenario_specs(
        self,
        scenario_specs: Sequence[Mapping[str, object]],
    ) -> tuple[_ScenarioSpec, ...]:
        """Validate and normalize the notebook-level scenario definitions."""

        if not scenario_specs:
            raise ValueError("scenario_specs must contain at least one scenario definition.")

        normalized_specs: list[_ScenarioSpec] = []

        for raw_spec in scenario_specs:
            spec = _ScenarioSpec(
                key=str(raw_spec["key"]),
                path=Path(raw_spec["path"]),
                label=str(raw_spec["label"]),
                role=str(raw_spec["role"]),
                within_bin_rule=str(raw_spec["within_bin_rule"]),
                why_it_matters=str(raw_spec["why_it_matters"]),
            )
            if not spec.path.exists():
                raise FileNotFoundError(f"Missing day-run export for {spec.key}: {spec.path}")
            normalized_specs.append(spec)

        return tuple(normalized_specs)

    def _build_scenario_color_map(
        self,
        scenario_order: Sequence[str],
    ) -> dict[str, str]:
        palette = (
            self.theme.neutral_dark,
            self.theme.secondary,
            self.theme.primary,
            self.theme.highlight,
        )
        return {
            scenario_label: palette[index % len(palette)]
            for index, scenario_label in enumerate(scenario_order)
        }

    def _scenario_bins(
        self,
        bin_table_all: pd.DataFrame,
        *,
        scenario_label: str,
    ) -> pd.DataFrame:
        return (
            bin_table_all.loc[bin_table_all["scenario_label"].eq(scenario_label)]
            .sort_values("bin_index")
            .reset_index(drop=True)
        )

    def _set_day_bin_axis(self, ax: plt.Axes, *, day_bin_count: int) -> None:
        """Apply the shared day-bin framing used across Notebook 5 figures."""

        for boundary in range(0, int(day_bin_count) + 1, 4):
            ax.axvline(
                boundary - 0.5,
                color=self.theme.grid,
                linewidth=0.8,
                zorder=0,
            )
        ax.set_xlim(-0.5, int(day_bin_count) - 0.5)
        ax.set_xticks(list(range(0, int(day_bin_count), 8)))

    def _plot_infeasible_markers(
        self,
        ax: plt.Axes,
        scenario_bins: pd.DataFrame,
        *,
        y_value: float,
    ) -> None:
        """Mark infeasible bins just above the solved power trace."""

        infeasible_bins = scenario_bins.loc[scenario_bins["status"].ne("solved")]
        if infeasible_bins.empty:
            return

        ax.scatter(
            infeasible_bins["bin_index"],
            np.full(len(infeasible_bins), y_value, dtype=float),
            marker="x",
            s=44,
            color=self.theme.highlight,
            linewidth=1.4,
            zorder=4,
        )

    def _contrasting_text_color(self, color: str) -> str:
        rgb = mcolors.to_rgb(color)
        luminance = (
            0.299 * rgb[0]
            + 0.587 * rgb[1]
            + 0.114 * rgb[2]
        )
        if luminance < 0.55:
            return self.theme.background
        return self.theme.text

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
    "ScenarioResultsArtifacts",
    "ScenarioResultsHelpers",
]
