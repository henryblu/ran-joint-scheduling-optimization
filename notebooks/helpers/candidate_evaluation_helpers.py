from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from downlink_candidate_evaluation import CandidatePowerModel, CandidateRateModel
from downlink_candidate_evaluation.mcs_requirements import McsRequirementModel
from single_user_solver.candidate_space import resolve_candidate_context
from single_user_solver.models import Candidate
from support.single_user_study import build_single_user_pa_curve_table, build_single_user_scenario
from support.theme import (
    NotebookTheme,
    apply_3d_axis_style,
    create_themed_figure,
    get_notebook_theme,
    render_html_table,
)


@dataclass(frozen=True)
class EvaluatedCandidateContext:
    """One fully resolved single-slot candidate at one fixed link distance."""

    distance_m: float
    path_loss_db: float
    candidate: Candidate
    pa_label: str
    pa_name: str
    deployment: object
    rrc: object
    pa: object
    mcs_row: dict[str, float]
    mcs_requirement: dict[str, float]
    re_counts: dict[str, float]
    rate_result: Any
    power_result: Any
    sinr_terms: dict[str, float]
    ps_solution: dict[str, float]
    p_out_ant_w: float
    p_dc_active_ant_w: float


@dataclass(frozen=True)
class CandidateEvaluationArtifacts:
    """Compact notebook payload for the candidate-evaluation walkthrough."""

    scenario_context: pd.DataFrame
    mcs_requirement_table: pd.DataFrame
    pa_curve_table: pd.DataFrame
    worked_candidate: EvaluatedCandidateContext
    comparison_candidate: EvaluatedCandidateContext


class CandidateEvaluationHelpers:
    """Theme-aware presentation helpers for the candidate-evaluation notebook."""

    def __init__(self, *, theme: str | NotebookTheme = "aalto_elec"):
        self.theme = get_notebook_theme(theme)

    def build_artifacts(
        self,
        *,
        worked_distance_m: float = 200.0,
        comparison_distance_m: float = 500.0,
        reference_required_rate_bps: float = 50e6,
        candidate: Candidate | None = None,
    ) -> CandidateEvaluationArtifacts:
        """Resolve one worked candidate and one distance comparison from production code."""

        resolved_candidate = candidate or Candidate(
            pa_id=0,
            n_prb=96,
            n_slots_on=1,
            layers=2,
            mcs=14,
        )
        if int(resolved_candidate.n_slots_on) != 1:
            raise ValueError("This notebook helper is restricted to one active slot.")

        worked_context = self._build_evaluated_candidate_context(
            distance_m=float(worked_distance_m),
            reference_required_rate_bps=float(reference_required_rate_bps),
            candidate=resolved_candidate,
        )
        comparison_context = self._build_evaluated_candidate_context(
            distance_m=float(comparison_distance_m),
            reference_required_rate_bps=float(reference_required_rate_bps),
            candidate=resolved_candidate,
        )
        pa_curve_table = build_single_user_pa_curve_table(
            build_single_user_scenario(
                distance_m=float(worked_distance_m),
                required_rate_bps=float(reference_required_rate_bps),
            )
        )
        return CandidateEvaluationArtifacts(
            scenario_context=self._build_scenario_context_table(
                worked_candidate=worked_context,
                comparison_candidate=comparison_context,
            ),
            mcs_requirement_table=self._build_mcs_requirement_table(worked_context),
            pa_curve_table=pa_curve_table.copy(),
            worked_candidate=worked_context,
            comparison_candidate=comparison_context,
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

    def plot_candidate_geometry(
        self,
        artifacts: CandidateEvaluationArtifacts,
    ) -> tuple[plt.Figure, Any]:
        """Plot the worked single-slot candidate inside the fixed NR resource envelope."""

        worked = artifacts.worked_candidate
        fig = plt.figure(figsize=(10.4, 6.6))
        fig.patch.set_facecolor(self.theme.background)
        ax = fig.add_subplot(111, projection="3d")
        apply_3d_axis_style(ax, theme=self.theme)

        self._draw_cuboid(
            ax,
            x=0.0,
            y=0.0,
            z=0.0,
            dx=float(worked.deployment.frame_n_slots),
            dy=float(worked.rrc.prb_max),
            dz=float(worked.deployment.n_tx_chains),
            facecolor=self.theme.highlight,
            alpha=0.08,
            edgecolor=self.theme.neutral_dark,
            linewidth=1.0,
        )
        self._draw_cuboid(
            ax,
            x=0.0,
            y=0.0,
            z=0.0,
            dx=float(worked.candidate.n_slots_on),
            dy=float(worked.candidate.n_prb),
            dz=float(worked.candidate.layers),
            facecolor=self.theme.primary,
            alpha=0.55,
            edgecolor=self.theme.secondary,
            linewidth=1.3,
        )

        ax.set_xlabel("Slot number", labelpad=10)
        ax.set_ylabel("PRB number", labelpad=12)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("Rank number", rotation=90, labelpad=12)
        ax.zaxis.label.set_clip_on(False)
        ax.set_xlim(0, int(worked.deployment.frame_n_slots))
        ax.set_ylim(int(worked.rrc.prb_max), 0)
        ax.set_zlim(0, int(worked.deployment.n_tx_chains))
        ax.set_xticks(np.arange(0, int(worked.deployment.frame_n_slots) + 1, 2))
        ax.set_yticks(np.arange(0, int(worked.rrc.prb_max) + 1, 50))
        ax.set_zticks(np.arange(0, int(worked.deployment.n_tx_chains) + 1, 1))
        ax.view_init(elev=24, azim=-58)
        fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.98)
        return fig, ax

    def plot_slot_payload_accounting(
        self,
        artifacts: CandidateEvaluationArtifacts,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot the worked slot symbol mix and the resulting RE accounting."""

        worked = artifacts.worked_candidate
        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=1,
            ncols=2,
            figsize=(11.8, 4.8),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.3, 1.0]},
        )
        symbol_ax, re_ax = axes.ravel()

        payload_symbols = int(worked.deployment.n_sym_data - worked.deployment.n_dmrs_sym)
        dmrs_symbols = int(worked.deployment.n_dmrs_sym)
        guard_symbols = int(worked.deployment.n_guard_sym)
        ul_symbols = int(worked.deployment.n_ul_sym)
        symbol_colors = (
            [self.theme.primary] * payload_symbols
            + [self.theme.highlight] * dmrs_symbols
            + [self.theme.neutral_light] * guard_symbols
            + [colors.to_hex(colors.to_rgba(self.theme.neutral_dark, 0.18))] * ul_symbols
        )
        symbol_labels = (
            ["Payload"] * payload_symbols
            + ["DMRS"] * dmrs_symbols
            + ["Guard"] * guard_symbols
            + ["UL"] * ul_symbols
        )

        for symbol_index, (fill_color, label) in enumerate(
            zip(symbol_colors, symbol_labels, strict=False)
        ):
            symbol_ax.add_patch(
                Rectangle(
                    (float(symbol_index), 0.0),
                    1.0,
                    1.0,
                    facecolor=fill_color,
                    edgecolor=self.theme.background,
                    linewidth=1.1,
                )
            )
            if label != "Payload":
                symbol_ax.text(
                    symbol_index + 0.5,
                    0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=self.theme.text,
                )
        symbol_ax.text(
            payload_symbols / 2.0,
            0.5,
            "Payload",
            ha="center",
            va="center",
            fontsize=10,
            color=self.theme.background,
            fontweight="bold",
        )
        symbol_ax.set_xlim(0.0, float(len(symbol_colors)))
        symbol_ax.set_ylim(0.0, 1.0)
        symbol_ax.set_xticks(np.arange(0.5, len(symbol_colors), 1.0))
        symbol_ax.set_xticklabels([str(index) for index in range(len(symbol_colors))])
        symbol_ax.set_xlabel("OFDM symbol number (schematic)")
        symbol_ax.set_yticks([])
        symbol_ax.grid(False)
        symbol_ax.spines["left"].set_visible(False)
        symbol_ax.spines["right"].set_visible(False)
        symbol_ax.spines["top"].set_visible(False)

        n_re_raw = float(worked.re_counts["n_re_raw"])
        n_pilot = float(worked.re_counts["n_pilot"])
        n_re_data = float(worked.re_counts["n_re_data"])
        x_positions = np.array([0.0, 1.1, 2.2])
        re_ax.bar([x_positions[0]], [n_re_raw], width=0.62, color=self.theme.secondary, alpha=0.85)
        re_ax.bar(
            [x_positions[1]],
            [-n_pilot],
            bottom=[n_re_raw],
            width=0.62,
            color=self.theme.highlight,
            alpha=0.92,
        )
        re_ax.bar([x_positions[2]], [n_re_data], width=0.62, color=self.theme.primary, alpha=0.88)
        re_ax.plot([x_positions[0], x_positions[1]], [n_re_raw, n_re_raw], color=self.theme.neutral_dark, linewidth=1.0)
        re_ax.plot([x_positions[1], x_positions[2]], [n_re_data, n_re_data], color=self.theme.neutral_dark, linewidth=1.0)
        re_ax.text(x_positions[0], n_re_raw + 250.0, f"{int(n_re_raw):,}", ha="center", va="bottom", fontsize=9, color=self.theme.text)
        re_ax.text(x_positions[1], n_re_raw - 0.5 * n_pilot, f"-{int(n_pilot):,}", ha="center", va="center", fontsize=9, color=self.theme.text)
        re_ax.text(x_positions[2], n_re_data + 250.0, f"{int(n_re_data):,}", ha="center", va="bottom", fontsize=9, color=self.theme.text)
        re_ax.set_xticks(x_positions.tolist())
        re_ax.set_xticklabels(["Scheduled RE", "DMRS RE", "Payload RE"])
        re_ax.set_ylabel("Resource elements in one active slot")
        re_ax.set_xlim(-0.55, 2.75)
        fig.tight_layout()
        return fig, (symbol_ax, re_ax)

    def plot_mcs_requirement(
        self,
        artifacts: CandidateEvaluationArtifacts,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the active MCS requirement table with the worked point highlighted."""

        worked = artifacts.worked_candidate
        table = artifacts.mcs_requirement_table
        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(9.4, 4.8),
        )

        color_by_qm = {
            2: self.theme.neutral_light,
            4: self.theme.primary,
            6: self.theme.secondary,
        }
        point_colors = [color_by_qm[int(value)] for value in table["qm"].astype(int)]
        ax.plot(table["mcs"].astype(int), table["rho_req_db"].astype(float), color=self.theme.neutral_dark, linewidth=1.3, alpha=0.8)
        ax.scatter(
            table["mcs"].astype(int),
            table["rho_req_db"].astype(float),
            c=point_colors,
            s=48,
            edgecolors=self.theme.background,
            linewidths=0.45,
            zorder=3,
        )
        ax.axvline(int(worked.candidate.mcs), color=self.theme.grid, linestyle="--", linewidth=1.0)
        ax.axhline(float(worked.mcs_requirement["rho_req_db"]), color=self.theme.grid, linestyle="--", linewidth=1.0)
        ax.scatter(
            [int(worked.candidate.mcs)],
            [float(worked.mcs_requirement["rho_req_db"])],
            s=165,
            color=self.theme.highlight,
            edgecolors=self.theme.accent,
            linewidths=1.2,
            zorder=4,
        )
        ax.text(
            int(worked.candidate.mcs) + 0.45,
            float(worked.mcs_requirement["rho_req_db"]) + 0.5,
            "\n".join(
                [
                    f"mcs = {int(worked.candidate.mcs)}",
                    f"qm = {int(worked.mcs_row['qm'])}",
                    f"eta = {float(worked.mcs_row['eta']):.4f}",
                    f"rho_req = {float(worked.mcs_requirement['rho_req_db']):.2f} dB",
                ]
            ),
            ha="left",
            va="bottom",
            fontsize=9,
            color=self.theme.text,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": self.theme.background,
                "edgecolor": self.theme.grid,
                "alpha": 0.96,
            },
        )
        ax.text(3.6, 1.35, "QPSK", fontsize=9, color=self.theme.neutral_dark)
        ax.text(11.0, 5.2, "16-QAM", fontsize=9, color=self.theme.primary)
        ax.text(22.0, 16.8, "64-QAM", fontsize=9, color=self.theme.secondary)
        ax.set_xlabel("MCS index")
        ax.set_ylabel("Required effective SINR (dB)")
        ax.set_xlim(-0.5, float(table["mcs"].max()) + 0.5)
        fig.tight_layout()
        return fig, ax

    def plot_link_budget_chain(
        self,
        artifacts: CandidateEvaluationArtifacts,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the worked link-budget and effective-SINR solve chain."""

        worked = artifacts.worked_candidate
        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(13.0, 5.8),
        )
        ax.set_axis_off()

        self._add_diagram_box(
            ax,
            x=0.04,
            y=0.56,
            width=0.18,
            height=0.30,
            header="Candidate",
            lines=[
                f"pa_id = {int(worked.candidate.pa_id)} ({worked.pa_label})",
                f"n_prb = {int(worked.candidate.n_prb)}",
                f"n_slots_on = {int(worked.candidate.n_slots_on)}",
                f"layers = {int(worked.candidate.layers)}",
                f"mcs = {int(worked.candidate.mcs)}",
            ],
            facecolor=self.theme.background,
            edgecolor=self.theme.grid,
        )
        self._add_diagram_box(
            ax,
            x=0.04,
            y=0.16,
            width=0.18,
            height=0.26,
            header="Distance context",
            lines=[
                f"distance = {float(worked.distance_m):.0f} m",
                f"path_loss_db = {float(worked.path_loss_db):.2f}",
                f"n_tx_chains = {int(worked.deployment.n_tx_chains)}",
                f"delta_f = {float(worked.rrc.delta_f_hz) / 1e3:.0f} kHz",
            ],
            facecolor=self.theme.background,
            edgecolor=self.theme.grid,
        )
        self._add_diagram_box(
            ax,
            x=0.29,
            y=0.56,
            width=0.18,
            height=0.30,
            header="Requirement",
            lines=[
                f"qm = {int(worked.mcs_row['qm'])}",
                f"eta = {float(worked.mcs_row['eta']):.4f}",
                f"rho_req_lin = {float(worked.mcs_requirement['rho_req_linear']):.4f}",
                f"rho_req_db = {float(worked.mcs_requirement['rho_req_db']):.2f}",
            ],
            facecolor=colors.to_hex(colors.to_rgba(self.theme.primary, 0.08)),
            edgecolor=self.theme.primary,
        )
        self._add_diagram_box(
            ax,
            x=0.52,
            y=0.12,
            width=0.24,
            height=0.74,
            header="Resolved terms",
            lines=[
                "B_occ = n_prb * 12 * delta_f",
                f"  = {self._format_bandwidth_hz(worked.sinr_terms['b_occ'])}",
                "K_active = n_prb * 12",
                f"  = {int(worked.sinr_terms['k_active_re'])}",
                f"N_pilot = {int(worked.sinr_terms['n_pilot'])}",
                f"g_l = {float(worked.sinr_terms['g_l']):.3e}",
                f"g_bf = {float(worked.sinr_terms['g_bf_linear']):.2f}",
                f"c_noise = {float(worked.sinr_terms['c_noise']):.3e}",
                "sigma_e2 = 1 / (rho_raw * N_pilot)",
                "rho_eff = rho_raw / (1 + rho_raw * sigma_e2)",
            ],
            facecolor=self.theme.background,
            edgecolor=self.theme.grid,
        )
        self._add_diagram_box(
            ax,
            x=0.80,
            y=0.30,
            width=0.16,
            height=0.38,
            header="Solved state",
            lines=[
                f"P_s,total = {self._format_power_w(worked.ps_solution['ps_min_w'])}",
                f"rho_raw = {float(worked.ps_solution['rho_ach_raw_linear']):.4f}",
                f"sigma_e2 = {float(worked.ps_solution['sigma_e2']):.3e}",
                f"rho_eff = {float(worked.ps_solution['rho_achieved_linear']):.4f}",
            ],
            facecolor=colors.to_hex(colors.to_rgba(self.theme.highlight, 0.18)),
            edgecolor=self.theme.accent,
        )

        self._add_diagram_arrow(ax, (0.22, 0.71), (0.29, 0.71))
        self._add_diagram_arrow(ax, (0.22, 0.29), (0.52, 0.29))
        self._add_diagram_arrow(ax, (0.47, 0.71), (0.52, 0.71))
        self._add_diagram_arrow(ax, (0.76, 0.49), (0.80, 0.49))
        return fig, ax

    def plot_rf_operating_point(
        self,
        artifacts: CandidateEvaluationArtifacts,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot the solved PA operating point on the measured curve."""

        worked = artifacts.worked_candidate
        fig, axes = create_themed_figure(
            theme=self.theme,
            nrows=1,
            ncols=2,
            figsize=(11.8, 4.9),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.55, 1.0]},
        )
        curve_ax, metrics_ax = axes.ravel()
        metrics_ax.set_axis_off()

        pa_rows = (
            artifacts.pa_curve_table.loc[
                artifacts.pa_curve_table["scenario_label"].eq(str(worked.pa_label))
                & artifacts.pa_curve_table["operating_state"].eq("active")
            ]
            .sort_values("pout_w")
            .reset_index(drop=True)
        )
        curve_color = self._pa_color(str(worked.pa_label))
        curve_ax.plot(
            pa_rows["pout_w"].astype(float),
            pa_rows["pdc_w"].astype(float),
            color=curve_color,
            linewidth=2.4,
        )
        curve_ax.scatter(
            [float(worked.p_out_ant_w)],
            [float(worked.p_dc_active_ant_w)],
            s=160,
            color=self.theme.highlight,
            edgecolors=self.theme.accent,
            linewidths=1.0,
            zorder=4,
        )
        curve_ax.axvline(float(worked.p_out_ant_w), color=self.theme.grid, linestyle="--", linewidth=1.0)
        curve_ax.axhline(float(worked.p_dc_active_ant_w), color=self.theme.grid, linestyle="--", linewidth=1.0)
        curve_ax.text(
            float(worked.p_out_ant_w),
            float(worked.p_dc_active_ant_w) + 0.18,
            "\n".join(
                [
                    f"P_out,ant = {self._format_power_w(worked.p_out_ant_w)}",
                    f"P_dc,ant = {self._format_power_w(worked.p_dc_active_ant_w)}",
                ]
            ),
            ha="left",
            va="bottom",
            fontsize=9,
            color=self.theme.text,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": self.theme.background,
                "edgecolor": self.theme.grid,
                "alpha": 0.96,
            },
        )
        curve_ax.set_xlabel("Per-chain RF output power (W)")
        curve_ax.set_ylabel("Per-chain PA DC power (W)")

        self._add_metric_box(
            metrics_ax,
            x=0.08,
            y=0.74,
            width=0.82,
            height=0.17,
            label="ps_total_w",
            value=self._format_power_w(worked.ps_solution["ps_min_w"]),
            facecolor=colors.to_hex(colors.to_rgba(self.theme.secondary, 0.08)),
        )
        self._add_metric_box(
            metrics_ax,
            x=0.08,
            y=0.52,
            width=0.82,
            height=0.17,
            label="p_out_total_w",
            value=self._format_power_w(worked.power_result.p_out_total_w),
            facecolor=colors.to_hex(colors.to_rgba(self.theme.primary, 0.08)),
        )
        self._add_metric_box(
            metrics_ax,
            x=0.08,
            y=0.30,
            width=0.82,
            height=0.17,
            label="p_out_ant_w",
            value=self._format_power_w(worked.p_out_ant_w),
            facecolor=colors.to_hex(colors.to_rgba(self.theme.highlight, 0.18)),
        )
        self._add_metric_box(
            metrics_ax,
            x=0.08,
            y=0.08,
            width=0.82,
            height=0.17,
            label="p_dc_active_total_w",
            value=self._format_power_w(worked.power_result.p_dc_active_total_w),
            facecolor=colors.to_hex(colors.to_rgba(self.theme.highlight, 0.28)),
        )
        fig.tight_layout()
        return fig, (curve_ax, metrics_ax)

    def plot_evaluated_candidate_record(
        self,
        artifacts: CandidateEvaluationArtifacts,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot one clean card for the evaluated single-slot operating row."""

        worked = artifacts.worked_candidate
        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.6, 6.2),
        )
        ax.set_axis_off()

        self._add_card_group(
            ax,
            x=0.04,
            y=0.55,
            width=0.28,
            height=0.34,
            header="Geometry",
            rows=[
                ("pa_id", f"{int(worked.candidate.pa_id)} ({worked.pa_label})"),
                ("n_prb", str(int(worked.candidate.n_prb))),
                ("n_slots_on", str(int(worked.candidate.n_slots_on))),
                ("layers", str(int(worked.candidate.layers))),
                ("mcs", str(int(worked.candidate.mcs))),
            ],
            accent_color=self.theme.primary,
        )
        self._add_card_group(
            ax,
            x=0.36,
            y=0.55,
            width=0.28,
            height=0.34,
            header="Payload",
            rows=[
                ("n_re_raw", f"{int(worked.re_counts['n_re_raw']):,}"),
                ("n_pilot", f"{int(worked.re_counts['n_pilot']):,}"),
                ("n_re_data", f"{int(worked.re_counts['n_re_data']):,}"),
                ("bits_per_slot", self._format_bits_per_slot(worked.rate_result.bits_per_slot)),
                ("rate_ach_bps", self._format_rate_bps(worked.rate_result.rate_ach_bps)),
            ],
            accent_color=self.theme.secondary,
        )
        self._add_card_group(
            ax,
            x=0.68,
            y=0.55,
            width=0.28,
            height=0.34,
            header="SINR",
            rows=[
                ("rho_req_lin", f"{float(worked.power_result.gamma_req_lin):.4f}"),
                ("rho_req_db", f"{float(worked.mcs_requirement['rho_req_db']):.2f}"),
                ("rho_eff", f"{float(worked.power_result.gamma_achieved):.4f}"),
                ("sigma_e2", f"{float(worked.ps_solution['sigma_e2']):.3e}"),
                ("is_feasible", str(bool(worked.power_result.is_feasible))),
            ],
            accent_color=self.theme.highlight,
        )
        self._add_card_group(
            ax,
            x=0.04,
            y=0.10,
            width=0.92,
            height=0.32,
            header="Power",
            rows=[
                ("ps_total_w", self._format_power_w(worked.ps_solution["ps_min_w"])),
                ("p_out_total_w", self._format_power_w(worked.power_result.p_out_total_w)),
                ("p_out_ant_w", self._format_power_w(worked.p_out_ant_w)),
                ("p_dc_active_total_w", self._format_power_w(worked.power_result.p_dc_active_total_w)),
                ("p_dc_avg_total_w", self._format_power_w(worked.power_result.p_dc_avg_total_w)),
                ("infeasibility_reason", str(worked.power_result.infeasibility_reason)),
            ],
            accent_color=self.theme.accent,
            n_columns=3,
        )
        return fig, ax

    def plot_distance_comparison(
        self,
        artifacts: CandidateEvaluationArtifacts,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the same candidate class resolved at two different distances."""

        worked = artifacts.worked_candidate
        comparison = artifacts.comparison_candidate
        fig, ax = create_themed_figure(
            theme=self.theme,
            figsize=(12.4, 5.9),
        )
        ax.set_axis_off()

        self._add_card_group(
            ax,
            x=0.05,
            y=0.76,
            width=0.90,
            height=0.16,
            header="Shared candidate class",
            rows=[
                ("pa_id", str(int(worked.candidate.pa_id))),
                ("n_prb", str(int(worked.candidate.n_prb))),
                ("n_slots_on", str(int(worked.candidate.n_slots_on))),
                ("layers", str(int(worked.candidate.layers))),
                ("mcs", str(int(worked.candidate.mcs))),
                ("bits_per_slot", self._format_bits_per_slot(worked.rate_result.bits_per_slot)),
            ],
            accent_color=self.theme.primary,
            n_columns=3,
        )
        self._add_distance_card(
            ax,
            x=0.05,
            y=0.14,
            width=0.40,
            height=0.52,
            header=f"{float(worked.distance_m):.0f} m",
            context=worked,
            edgecolor=self.theme.primary,
            facecolor=colors.to_hex(colors.to_rgba(self.theme.primary, 0.06)),
        )
        self._add_distance_card(
            ax,
            x=0.55,
            y=0.14,
            width=0.40,
            height=0.52,
            header=f"{float(comparison.distance_m):.0f} m",
            context=comparison,
            edgecolor=self.theme.highlight,
            facecolor=colors.to_hex(colors.to_rgba(self.theme.highlight, 0.16)),
        )
        self._add_diagram_arrow(ax, (0.45, 0.40), (0.55, 0.40))
        return fig, ax

    def _build_evaluated_candidate_context(
        self,
        *,
        distance_m: float,
        reference_required_rate_bps: float,
        candidate: Candidate,
    ) -> EvaluatedCandidateContext:
        """Resolve one direct single-slot candidate evaluation from production code."""

        scenario = build_single_user_scenario(
            distance_m=float(distance_m),
            required_rate_bps=float(reference_required_rate_bps),
        )
        context = scenario.context
        rrc, pa = resolve_candidate_context(context.search_catalog, candidate)
        rate_model = CandidateRateModel(context.mcs_table)
        power_model = CandidatePowerModel(context.mcs_table)
        mcs_row = dict(context.mcs_table[int(candidate.mcs)])
        mcs_requirement = dict(
            McsRequirementModel(context.mcs_table).get_required_sinr_table(context.deployment)[
                int(candidate.mcs)
            ]
        )
        rate_result = rate_model.compute_candidate_rate(context.deployment, rrc, candidate)
        power_result = power_model.solve_candidate_power(
            context.deployment,
            rrc,
            candidate,
            pa,
            gamma_req_lin=float(mcs_requirement["rho_req_linear"]),
        )
        if not power_result.is_feasible:
            raise ValueError(
                "The worked candidate is infeasible under the current model contract: "
                f"{power_result.infeasibility_reason}."
            )

        sinr_terms = dict(
            power_model.sinr_model.build_sinr_terms(
                context.deployment,
                rrc,
                candidate,
                pa,
            )
        )
        ps_solution = dict(
            power_model.sinr_model.solve_required_source_power_for_target(
                float(mcs_requirement["rho_req_linear"]),
                context.deployment,
                rrc,
                candidate,
                pa,
            )
        )
        n_tx_chains = max(int(context.deployment.n_tx_chains), 1)
        return EvaluatedCandidateContext(
            distance_m=float(distance_m),
            path_loss_db=float(context.deployment.path_loss_db),
            candidate=candidate,
            pa_label=str(pa.scenario_label),
            pa_name=str(pa.pa_name),
            deployment=context.deployment,
            rrc=rrc,
            pa=pa,
            mcs_row=mcs_row,
            mcs_requirement=mcs_requirement,
            re_counts={
                "n_re_raw": float(rate_result.n_re_raw),
                "n_pilot": float(rate_result.n_pilot),
                "n_re_data": float(rate_result.n_re_data),
            },
            rate_result=rate_result,
            power_result=power_result,
            sinr_terms=sinr_terms,
            ps_solution=ps_solution,
            p_out_ant_w=float(power_result.p_out_total_w) / float(n_tx_chains),
            p_dc_active_ant_w=float(power_result.p_dc_active_total_w) / float(n_tx_chains),
        )

    def _build_scenario_context_table(
        self,
        *,
        worked_candidate: EvaluatedCandidateContext,
        comparison_candidate: EvaluatedCandidateContext,
    ) -> pd.DataFrame:
        """Return the compact scenario recap shown at the start of the notebook."""

        resource_envelope = (
            f"{int(worked_candidate.deployment.frame_n_slots)} slots x "
            f"{int(worked_candidate.rrc.prb_max)} PRBs x "
            f"{int(worked_candidate.deployment.n_tx_chains)} ranks"
        )
        candidate_signature = ", ".join(
            [
                f"pa_id = {int(worked_candidate.candidate.pa_id)}",
                f"n_prb = {int(worked_candidate.candidate.n_prb)}",
                f"n_slots_on = {int(worked_candidate.candidate.n_slots_on)}",
                f"layers = {int(worked_candidate.candidate.layers)}",
                f"mcs = {int(worked_candidate.candidate.mcs)}",
            ]
        )
        return pd.DataFrame(
            [
                {
                    "item": "Downlink channel",
                    "value": "3GPP NR PDSCH over the fixed micro-cell preset",
                },
                {
                    "item": "Carrier preset",
                    "value": (
                        f"{float(worked_candidate.deployment.channel_bw_hz) / 1e6:.0f} MHz at "
                        f"{float(worked_candidate.deployment.fc_hz) / 1e9:.1f} GHz, "
                        f"{float(worked_candidate.rrc.delta_f_hz) / 1e3:.0f} kHz SCS"
                    ),
                },
                {
                    "item": "Resource envelope",
                    "value": resource_envelope,
                },
                {
                    "item": "Worked distance",
                    "value": f"{float(worked_candidate.distance_m):.0f} m",
                },
                {
                    "item": "Comparison distance",
                    "value": f"{float(comparison_candidate.distance_m):.0f} m",
                },
                {
                    "item": "Worked candidate class",
                    "value": candidate_signature,
                },
            ]
        )

    def _build_mcs_requirement_table(
        self,
        worked_candidate: EvaluatedCandidateContext,
    ) -> pd.DataFrame:
        """Return the active MCS requirement table for plotting."""

        current_mcs_table = dict(build_single_user_scenario(
            distance_m=float(worked_candidate.distance_m),
            required_rate_bps=50e6,
        ).context.mcs_table)
        requirement_rows = McsRequirementModel(current_mcs_table).current_required_sinr_table(
            worked_candidate.deployment
        )
        rows = []
        for row in requirement_rows:
            mcs = int(row["mcs"])
            rows.append(
                {
                    "mcs": mcs,
                    "qm": int(current_mcs_table[mcs]["qm"]),
                    "eta": float(current_mcs_table[mcs]["eta"]),
                    "rho_req_linear": float(row["rho_req_linear"]),
                    "rho_req_db": float(row["rho_req_db"]),
                }
            )
        return pd.DataFrame(rows).sort_values("mcs").reset_index(drop=True)

    def _pa_color(self, pa_label: str) -> str:
        """Return the consistent PA-family color used across notebook plots."""

        if str(pa_label) == "4W PA":
            return self.theme.primary
        if str(pa_label) == "8W PA":
            return self.theme.highlight
        return self.theme.secondary

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
        """Draw one rounded notebook diagram box."""

        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.1,
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
        line_step = (height - 0.10) / max(len(lines), 1)
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
                family="monospace" if "=" in line else None,
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

    def _add_metric_box(
        self,
        ax: plt.Axes,
        *,
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        value: str,
        facecolor: str,
    ) -> None:
        """Draw one compact metric card."""

        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=facecolor,
            edgecolor=self.theme.grid,
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.04,
            y + height - 0.05,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            color=self.theme.text,
        )
        ax.text(
            x + 0.04,
            y + 0.05,
            value,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            color=self.theme.text,
            fontweight="bold",
        )

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
        """Draw one grouped record card with code-facing field names."""

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
            (x, y + height - 0.08),
            width,
            0.08,
            transform=ax.transAxes,
            facecolor=colors.to_hex(colors.to_rgba(accent_color, 0.18)),
            edgecolor="none",
        )
        ax.add_patch(header_patch)
        ax.text(
            x + 0.02,
            y + height - 0.04,
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
            row = index % rows_per_column
            anchor_x = x + column * column_width + 0.02
            anchor_y = y + height - 0.12 - row * 0.055
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
                anchor_y - 0.026,
                value,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color=self.theme.text,
            )

    def _add_distance_card(
        self,
        ax: plt.Axes,
        *,
        x: float,
        y: float,
        width: float,
        height: float,
        header: str,
        context: EvaluatedCandidateContext,
        edgecolor: str,
        facecolor: str,
    ) -> None:
        """Draw one distance-conditioned operating-point card."""

        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.014,rounding_size=0.02",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.1,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.03,
            y + height - 0.06,
            header,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=self.theme.text,
        )
        rows = [
            ("path_loss_db", f"{float(context.path_loss_db):.2f}"),
            ("rho_req_lin", f"{float(context.power_result.gamma_req_lin):.4f}"),
            ("ps_total_w", self._format_power_w(context.ps_solution["ps_min_w"])),
            ("p_out_total_w", self._format_power_w(context.power_result.p_out_total_w)),
            ("p_dc_active_total_w", self._format_power_w(context.power_result.p_dc_active_total_w)),
            ("rate_ach_bps", self._format_rate_bps(context.rate_result.rate_ach_bps)),
        ]
        for index, (label, value) in enumerate(rows):
            anchor_y = y + height - 0.13 - index * 0.065
            ax.text(
                x + 0.03,
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
                x + 0.22,
                anchor_y,
                value,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color=self.theme.text,
            )

    @staticmethod
    def _draw_cuboid(
        ax,
        *,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        dz: float,
        facecolor: str,
        alpha: float,
        edgecolor: str,
        linewidth: float,
    ) -> None:
        """Draw one translucent 3D cuboid."""

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
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            zsort="average",
        )
        ax.add_collection3d(poly)

    @staticmethod
    def _format_power_w(value_w: float) -> str:
        """Format one power value with a stable engineering unit."""

        resolved_value = float(value_w)
        if resolved_value >= 1.0:
            return f"{resolved_value:.2f} W"
        if resolved_value >= 1e-3:
            return f"{resolved_value * 1e3:.2f} mW"
        if resolved_value >= 1e-6:
            return f"{resolved_value * 1e6:.2f} uW"
        return f"{resolved_value:.3e} W"

    @staticmethod
    def _format_rate_bps(value_bps: float) -> str:
        """Format one rate value for notebook display."""

        resolved_value = float(value_bps)
        if resolved_value >= 1e6:
            return f"{resolved_value / 1e6:.2f} Mbps"
        if resolved_value >= 1e3:
            return f"{resolved_value / 1e3:.2f} kbps"
        return f"{resolved_value:.2f} bps"

    @staticmethod
    def _format_bits_per_slot(value_bits: float) -> str:
        """Format one slot payload value for notebook display."""

        resolved_value = float(value_bits)
        if resolved_value >= 1e6:
            return f"{resolved_value / 1e6:.3f} Mbit/slot"
        if resolved_value >= 1e3:
            return f"{resolved_value / 1e3:.2f} kbit/slot"
        return f"{resolved_value:.2f} bit/slot"

    @staticmethod
    def _format_bandwidth_hz(value_hz: float) -> str:
        """Format one occupied bandwidth value with a clean unit."""

        resolved_value = float(value_hz)
        if resolved_value >= 1e6:
            return f"{resolved_value / 1e6:.2f} MHz"
        if resolved_value >= 1e3:
            return f"{resolved_value / 1e3:.2f} kHz"
        return f"{resolved_value:.2f} Hz"


__all__ = [
    "CandidateEvaluationArtifacts",
    "CandidateEvaluationHelpers",
    "EvaluatedCandidateContext",
]
