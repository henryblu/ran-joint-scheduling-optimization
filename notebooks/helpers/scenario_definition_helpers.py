from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from uuid import uuid4

from IPython import get_ipython
from IPython.display import HTML, Javascript, display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from support.single_user_study import (
    build_single_user_pa_curve_table,
    build_single_user_scenario,
    summarize_single_user_scenario,
)
from support.theme import (
    NotebookTheme,
    apply_axis_style,
    create_themed_figure,
    get_notebook_theme,
    render_html_table,
    style_legend,
)


SCHEDULER_FACING_COLUMNS = tuple(BATCH_USER_PARAMETER_SPACE_COLUMNS)


@dataclass(frozen=True)
class ScenarioDefinitionArtifacts:
    """Lean notebook payload for the scenario-definition walkthrough."""

    scenario: object
    radio_assumptions: pd.DataFrame
    problem_statement: pd.DataFrame
    phy_resource_space: pd.DataFrame
    illustrative_user: pd.DataFrame
    pa_characteristics: pd.DataFrame
    pa_curve_table: pd.DataFrame
    candidate_space_view: pd.DataFrame


class ScenarioDefinitionHelpers:
    """Theme-aware presentation helpers for Notebook 1."""

    def __init__(self, *, theme: str | NotebookTheme = "aalto_elec"):
        self.theme = get_notebook_theme(theme)

    def build_artifacts(
        self,
        *,
        distance_m: float = 200.0,
        required_rate_bps: float = 120e6,
    ) -> ScenarioDefinitionArtifacts:
        """Build the compact scenario-first views used in Notebook 1.

        Steps:
        1. Resolve one illustrative single-user scenario under the canonical radio preset.
        2. Extract the deployment, PA, and scheduler-facing summaries from that scenario.
        3. Return the small tables that frame the study before candidate-table generation begins.
        """

        scenario = build_single_user_scenario(
            distance_m=float(distance_m),
            required_rate_bps=float(required_rate_bps),
        )
        summary_views = summarize_single_user_scenario(scenario)
        pa_characteristics = summary_views["pa_characteristics"].copy()
        deployment = scenario.context.deployment

        return ScenarioDefinitionArtifacts(
            scenario=scenario,
            radio_assumptions=self._build_radio_assumption_table(
                scenario.context.model_inputs,
                pa_characteristics=pa_characteristics,
            ),
            problem_statement=self._build_problem_statement_table(),
            phy_resource_space=self._build_phy_resource_space_table(scenario),
            illustrative_user=self._build_illustrative_user_table(
                distance_m=float(distance_m),
                required_rate_bps=float(required_rate_bps),
                resolved_path_loss_db=float(deployment.path_loss_db),
            ),
            pa_characteristics=pa_characteristics,
            pa_curve_table=build_single_user_pa_curve_table(scenario),
            candidate_space_view=summary_views["candidate_space_view"].copy(),
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

    def display_contract_flowchart(self) -> None:
        """Render the scenario-to-contract Mermaid diagram inside Jupyter."""

        container_id = f"scenario-definition-flowchart-{uuid4().hex}"
        diagram = "\n".join(
            [
                "flowchart LR",
                '    A["Fixed n78 microcell preset"]',
                '    B["Deployment context for one user condition"]',
                '    C["Measured PA set"]',
                '    D["Fixed 100 MHz resource space"]',
                '    E["Lean full-frame candidate-table contract"]',
                '    F["Candidate generation and later TDMA use"]',
                "",
                "    A --> B",
                "    B --> C",
                "    C --> D",
                "    D --> E",
                "    E --> F",
            ]
        )
        theme_variables = {
            "primaryColor": self.theme.neutral_light,
            "primaryBorderColor": self.theme.neutral_dark,
            "primaryTextColor": self.theme.text,
            "lineColor": self.theme.primary,
            "secondaryColor": self.theme.background,
            "secondaryBorderColor": self.theme.grid,
            "secondaryTextColor": self.theme.text,
            "tertiaryColor": self.theme.background,
            "tertiaryBorderColor": self.theme.grid,
            "tertiaryTextColor": self.theme.text,
            "mainBkg": self.theme.background,
            "textColor": self.theme.text,
            "nodeBorder": self.theme.neutral_dark,
            "clusterBkg": self.theme.background,
            "clusterBorder": self.theme.grid,
            "edgeLabelBackground": self.theme.background,
            "fontFamily": "Helvetica, Arial, sans-serif",
        }

        display(
            HTML(
                f'<div id="{container_id}" '
                f'style="padding:6px 0 14px 0; background:{self.theme.background};"></div>'
            )
        )
        display(
            Javascript(
                f"""
(async function() {{
  const container = document.getElementById({json.dumps(container_id)});
  if (!container) {{
    return;
  }}

  if (!window.__thesisMermaidImport) {{
    window.__thesisMermaidImport = import("https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs");
  }}

  const mermaidModule = await window.__thesisMermaidImport;
  const mermaid = mermaidModule.default;
  mermaid.initialize({{
    startOnLoad: false,
    securityLevel: "loose",
    theme: "base",
    themeVariables: {json.dumps(theme_variables)},
  }});

  container.className = "mermaid";
  container.removeAttribute("data-processed");
  container.textContent = {json.dumps(diagram)};
  await mermaid.run({{ nodes: [container] }});
}})();
"""
            )
        )

    def plot_pa_gain_and_pae(
        self,
        pa_curve_table: pd.DataFrame,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot gain and PAE against average PA output power on one shared figure."""

        fig, gain_ax = create_themed_figure(
            theme=self.theme,
            figsize=(9.4, 5.4),
        )
        pae_ax = gain_ax.twinx()
        apply_axis_style(
            pae_ax,
            theme=self.theme,
            grid_axis="none",
            use_theme_cycle=False,
            hide_spines=("top", "left", "bottom"),
        )
        pae_ax.patch.set_alpha(0.0)

        color_by_label = self._build_pa_color_map(pa_curve_table)
        for scenario_label, pa_curve_rows in pa_curve_table.groupby("scenario_label", sort=True):
            active_rows = pa_curve_rows.loc[pa_curve_rows["pout_w"].fillna(0.0) > 0.0].copy()
            if active_rows.empty:
                continue

            active_rows = active_rows.sort_values("pout_w")
            pout_w = active_rows["pout_w"].to_numpy(dtype=float)
            pin_w = active_rows["pin_w"].to_numpy(dtype=float)
            pdc_w = active_rows["pdc_w"].to_numpy(dtype=float)
            pout_dbm = self._to_dbm(pout_w)
            pin_dbm = self._to_dbm(pin_w)
            gain_db = pout_dbm - pin_dbm
            pae_percent = np.where(
                pdc_w > 0.0,
                100.0 * (pout_w - pin_w) / pdc_w,
                np.nan,
            )
            valid = np.isfinite(pout_dbm) & np.isfinite(gain_db) & np.isfinite(pae_percent)
            if not np.any(valid):
                continue

            color = color_by_label[str(scenario_label)]
            gain_ax.plot(
                pout_dbm[valid],
                gain_db[valid],
                color=color,
                linewidth=2.4,
            )
            pae_ax.plot(
                pout_dbm[valid],
                pae_percent[valid],
                color=color,
                linewidth=2.4,
                linestyle="--",
            )

        gain_ax.set_xlim(10.0, 40.0)
        gain_ax.set_ylim(25.0, 35.0)
        pae_ax.set_ylim(0.0, 50.0)
        gain_ax.set_xticks(list(range(10, 41, 5)))
        gain_ax.set_yticks(list(range(25, 36)))
        pae_ax.set_yticks(list(range(0, 51, 10)))
        gain_ax.set_xlabel("Average PA output power (dBm)")
        gain_ax.set_ylabel("Gain (dB)")
        pae_ax.set_ylabel("PAE (%)")

        pa_handles = [
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.4,
                label=label,
            )
            for label, color in color_by_label.items()
        ]
        metric_handles = [
            Line2D(
                [0],
                [0],
                color=self.theme.neutral_dark,
                linewidth=2.4,
                label="Gain",
            ),
            Line2D(
                [0],
                [0],
                color=self.theme.neutral_dark,
                linewidth=2.4,
                linestyle="--",
                label="PAE",
            ),
        ]
        pa_legend = gain_ax.legend(
            handles=pa_handles,
            loc="upper left",
            title="PA option",
        )
        style_legend(pa_legend, theme=self.theme)
        metric_legend = gain_ax.legend(
            handles=metric_handles,
            loc="upper right",
            title="Quantity",
        )
        style_legend(metric_legend, theme=self.theme)
        gain_ax.add_artist(pa_legend)
        fig.tight_layout()
        self._display_plot_caption(
            "Measured PA gain and efficiency over the active operating range.",
        )
        return fig, (gain_ax, pae_ax)

    def _build_radio_assumption_table(
        self,
        config,
        *,
        pa_characteristics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the fixed deployment, loss, and waveform assumptions of the study."""

        pa_labels = ", ".join(pa_characteristics["scenario_label"].astype(str).tolist())
        rows = [
            {
                "group": "Deployment",
                "assumption": "Scenario",
                "value": "3GPP-compliant micro cell in band n78",
            },
            {
                "group": "Deployment",
                "assumption": "Carrier frequency",
                "value": f"{float(config.fc_hz) / 1e9:.1f} GHz",
            },
            {
                "group": "Deployment",
                "assumption": "Channel bandwidth",
                "value": f"{float(config.channel_bw_hz) / 1e6:.0f} MHz",
            },
            {
                "group": "Propagation",
                "assumption": "Path-loss model",
                "value": str(config.pl_model),
            },
            {
                "group": "Propagation",
                "assumption": "BS / UE heights",
                "value": f"{float(config.h_bs_m):.1f} m / {float(config.h_ut_m):.1f} m",
            },
            {
                "group": "Propagation",
                "assumption": "Antenna gains",
                "value": f"{float(config.g_tx_db):.1f} dB TX, {float(config.g_rx_db):.1f} dB RX",
            },
            {
                "group": "Link budget",
                "assumption": "Thermal noise density",
                "value": f"{float(config.n0_dbm_per_hz):.0f} dBm/Hz",
            },
            {
                "group": "Link budget",
                "assumption": "LNA noise figure",
                "value": f"{float(config.lna_noise_figure_db):.1f} dB",
            },
            {
                "group": "Link budget",
                "assumption": "Implementation loss",
                "value": f"{float(config.l_impl_db):.1f} dB",
            },
            {
                "group": "Link budget",
                "assumption": "Shadow margin",
                "value": f"{float(config.shadow_margin_db):.1f} dB",
            },
            {
                "group": "Waveform",
                "assumption": "Subcarrier spacing",
                "value": f"{float(config.delta_f_hz) / 1e3:.0f} kHz",
            },
            {
                "group": "Waveform",
                "assumption": "Slot duration",
                "value": f"{float(config.t_slot_s) * 1e3:.1f} ms",
            },
            {
                "group": "Waveform",
                "assumption": "Reference frame",
                "value": (
                    f"{int(config.frame_n_slots)} slots = "
                    f"{float(config.frame_n_slots * config.t_slot_s) * 1e3:.1f} ms"
                ),
            },
            {
                "group": "Waveform",
                "assumption": "Slot symbol mix",
                "value": (
                    f"{int(config.n_sym_data)} data, {int(config.n_dmrs_sym)} DMRS, "
                    f"{int(config.n_guard_sym)} guard, {int(config.n_ul_sym)} UL"
                ),
            },
            {
                "group": "Waveform",
                "assumption": "Total OFDM symbols",
                "value": f"{int(config.n_sym_total)}",
            },
            {
                "group": "Waveform",
                "assumption": "DFT size",
                "value": f"{int(config.dft_size_N)}",
            },
            {
                "group": "Impairment model",
                "assumption": "PAPR assumption",
                "value": f"{float(config.papr_db):.1f} dB",
            },
            {
                "group": "Impairment model",
                "assumption": "Phase-noise gain factor",
                "value": f"{float(config.g_phi):.2f}",
            },
            {
                "group": "Impairment model",
                "assumption": "Phase-noise variance",
                "value": f"{float(config.sigma_phi2):.2f}",
            },
            {
                "group": "Impairment model",
                "assumption": "Quantization-noise variance",
                "value": f"{float(config.sigma_q2):.2f}",
            },
            {
                "group": "Impairment model",
                "assumption": "Mutual-information samples",
                "value": f"{int(config.mi_n_samples)}",
            },
            {
                "group": "RF chain",
                "assumption": "Transmit chains",
                "value": f"{int(config.n_tx_chains)} active chains in one 4-MIMO sub-array",
            },
            {
                "group": "RF chain",
                "assumption": "Per-chain PA options",
                "value": pa_labels,
            },
            {
                "group": "RF chain",
                "assumption": "PSD constraint active",
                "value": "Yes" if bool(config.use_psd_constraint) else "No",
            },
            {
                "group": "RF chain",
                "assumption": "PSD constraint",
                "value": f"{float(config.psd_max_w_per_hz):.1e} W/Hz",
            },
        ]
        return pd.DataFrame(rows)

    def _build_problem_statement_table(self) -> pd.DataFrame:
        """Return the compact thesis question stated at the scheduler-facing level."""

        return pd.DataFrame(
            [
                {
                    "question": "What is fixed?",
                    "answer": "The scenario, PA catalog, and compliant PHY search space are fixed by the shared radio preset.",
                },
                {
                    "question": "What is allocated?",
                    "answer": "Each active user receives one full-frame scheduler row built from PA choice, PRBs, layers, and MCS.",
                },
                {
                    "question": "What is optimized?",
                    "answer": "The active PA DC power is minimized while the full-frame row still meets the requested service rate.",
                },
            ]
        )

    def _build_phy_resource_space_table(self, scenario) -> pd.DataFrame:
        """Return the compliant PHY dimensions that survive into the stored table contract."""

        context = scenario.context
        max_prbs = max(int(rrc.prb_max) for rrc in context.rrc_catalog)
        return pd.DataFrame(
            [
                {
                    "dimension": "channel_bw_hz",
                    "admitted_values": f"{float(context.deployment.channel_bw_hz) / 1e6:.0f} MHz",
                    "note": "The candidate table is built for one fixed carrier bandwidth.",
                },
                {
                    "dimension": "n_prb",
                    "admitted_values": f"1 to {max_prbs} PRBs in steps of {int(context.search_shape.prb_step)}",
                    "note": "Frequency allocation varies only through the PRB count inside the fixed carrier.",
                },
                {
                    "dimension": "layers",
                    "admitted_values": ", ".join(str(int(value)) for value in context.search_shape.layers_space),
                    "note": "The active layer count is bounded by the 4-chain sub-array.",
                },
                {
                    "dimension": "mcs",
                    "admitted_values": f"{min(context.search_shape.mcs_space)} to {max(context.search_shape.mcs_space)}",
                    "note": "The NR MCS reference table fixes the admissible coding points.",
                },
                {
                    "dimension": "Stored table columns",
                    "admitted_values": ", ".join(SCHEDULER_FACING_COLUMNS),
                    "note": "This lean full-frame contract is reused by user lookup before TDMA slot quantization.",
                },
            ]
        )

    def _build_illustrative_user_table(
        self,
        *,
        distance_m: float,
        required_rate_bps: float,
        resolved_path_loss_db: float,
    ) -> pd.DataFrame:
        """Return the single illustrative user case used to make the scenario concrete."""

        return pd.DataFrame(
            [
                {
                    "distance_m": float(distance_m),
                    "required_rate_mbps": float(required_rate_bps) / 1e6,
                    "resolved_path_loss_db": float(resolved_path_loss_db),
                    "role": "Illustrative user condition carried into the later notebooks",
                }
            ]
        )

    def _build_pa_color_map(self, pa_curve_table: pd.DataFrame) -> dict[str, str]:
        preferred_colors = {
            "4W PA": self.theme.primary,
            "8W PA": self.theme.secondary,
        }
        scenario_labels = sorted(
            pa_curve_table["scenario_label"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        fallback_palette = [
            self.theme.accent,
            self.theme.neutral_dark,
        ]
        color_map: dict[str, str] = {}
        fallback_index = 0

        for label in scenario_labels:
            preferred_color = preferred_colors.get(label)
            if preferred_color is not None:
                color_map[label] = preferred_color
                continue
            color_map[label] = fallback_palette[fallback_index % len(fallback_palette)]
            fallback_index += 1

        return color_map

    def _display_plot_caption(self, caption: str) -> None:
        if get_ipython() is None:
            return

        display(
            HTML(
                '<div style="'
                f'margin-top:6px; color:{self.theme.neutral_dark}; font-size:0.95rem;'
                '">'
                f"{caption}"
                "</div>"
            )
        )

    @staticmethod
    def _to_dbm(power_w) -> np.ndarray:
        """Convert power in watts to dBm, returning NaN for non-positive values."""

        power_w = np.asarray(power_w, dtype=float)
        power_dbm = np.full(power_w.shape, np.nan, dtype=float)
        valid = power_w > 0.0
        power_dbm[valid] = 10.0 * np.log10(power_w[valid] * 1000.0)
        return power_dbm

__all__ = [
    "PROJECT_ROOT",
    "SCHEDULER_FACING_COLUMNS",
    "ScenarioDefinitionArtifacts",
    "ScenarioDefinitionHelpers",
]
