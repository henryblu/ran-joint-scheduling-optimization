from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from .single_user_study_helpers import (
    build_single_user_pa_curve_table,
    build_single_user_scenario,
    summarize_single_user_scenario,
)


SCHEDULER_FACING_COLUMNS = tuple(BATCH_USER_PARAMETER_SPACE_COLUMNS)


def build_scenario_definition_artifacts(
    *,
    distance_m: float = 200.0,
    required_rate_bps: float = 120e6,
) -> SimpleNamespace:
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
    config = scenario.context.model_inputs
    deployment = scenario.context.deployment

    return SimpleNamespace(
        scenario=scenario,
        radio_assumptions=_build_radio_assumption_table(config),
        problem_statement=_build_problem_statement_table(),
        phy_resource_space=_build_phy_resource_space_table(scenario),
        illustrative_user=_build_illustrative_user_table(
            distance_m=float(distance_m),
            required_rate_bps=float(required_rate_bps),
            resolved_path_loss_db=float(deployment.path_loss_db),
        ),
        pa_characteristics=summary_views["pa_characteristics"].copy(),
        pa_curve_table=build_single_user_pa_curve_table(scenario),
        candidate_space_view=summary_views["candidate_space_view"].copy(),
    )


def _build_radio_assumption_table(config) -> pd.DataFrame:
    """Return the fixed radio, frame, and hardware assumptions of the study."""

    return pd.DataFrame(
        [
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
                "group": "Noise",
                "assumption": "Noise density and NF",
                "value": (
                    f"{float(config.n0_dbm_per_hz):.0f} dBm/Hz, "
                    f"{float(config.lna_noise_figure_db):.1f} dB NF"
                ),
            },
            {
                "group": "Noise",
                "assumption": "Implementation and shadow margins",
                "value": f"{float(config.l_impl_db):.1f} dB impl., {float(config.shadow_margin_db):.1f} dB shadow",
            },
            {
                "group": "Frame",
                "assumption": "Subcarrier spacing",
                "value": f"{float(config.delta_f_hz) / 1e3:.0f} kHz",
            },
            {
                "group": "Frame",
                "assumption": "Slot duration",
                "value": f"{float(config.t_slot_s) * 1e3:.1f} ms",
            },
            {
                "group": "Frame",
                "assumption": "Reference window",
                "value": f"{int(config.n_slots_win)} slots = {float(config.n_slots_win * config.t_slot_s) * 1e3:.1f} ms",
            },
            {
                "group": "MIMO",
                "assumption": "Transmit chains",
                "value": f"{int(config.n_tx_chains)} active chains in one 4-MIMO sub-array",
            },
            {
                "group": "MIMO",
                "assumption": "Layer space",
                "value": ", ".join(str(int(value)) for value in config.layers_space),
            },
            {
                "group": "PA model",
                "assumption": "Per-chain PA options",
                "value": "QPA9942 and Bae et al. NR",
            },
            {
                "group": "PA model",
                "assumption": "PSD constraint",
                "value": f"{float(config.psd_max_w_per_hz):.1e} W/Hz",
            },
        ]
    )


def _build_problem_statement_table() -> pd.DataFrame:
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


def _build_phy_resource_space_table(scenario) -> pd.DataFrame:
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


__all__ = [
    "PROJECT_ROOT",
    "SCHEDULER_FACING_COLUMNS",
    "build_scenario_definition_artifacts",
]
