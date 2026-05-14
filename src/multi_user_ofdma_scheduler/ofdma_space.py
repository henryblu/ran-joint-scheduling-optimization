from __future__ import annotations

import math

from configs import SINGLE_USER_SEARCH_CONFIG
from configs.pa import pa_slot_dc_power
from models import BatchUserParameterSpace
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS

from .models import PreparedJointOfdmaProblem


def prepare_joint_ofdma_problem(
    batch_space: BatchUserParameterSpace,
) -> PreparedJointOfdmaProblem:
    """Prepare the slot-level OFDMA scheduler input from one trusted batch artifact."""

    user_requirements = (
        batch_space.user_requirements[["user_id", "required_rate_bps"]]
        .copy()
        .assign(
            user_id=lambda table: table["user_id"].astype(int),
            required_rate_bps=lambda table: table["required_rate_bps"].astype(float),
        )
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    user_slot_spaces = {}
    for user_row in user_requirements.itertuples(index=False):
        user_id = int(user_row.user_id)
        raw_user_slot_space = batch_space.user_parameter_spaces[user_id]
        if raw_user_slot_space.empty:
            user_slot_spaces[user_id] = (
                raw_user_slot_space.reindex(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)
                .copy()
                .reset_index(drop=True)
            )
            continue

        user_slot_space = _apply_active_pa_dc_contract(
            raw_user_slot_space[BATCH_USER_PARAMETER_SPACE_COLUMNS].copy(),
            n_tx_chains=int(batch_space.n_tx_chains),
            pa_catalog=tuple(batch_space.pa_catalog),
        )
        user_slot_spaces[user_id] = (
            user_slot_space
            .sort_values(
                [
                    "pa_id",
                    "n_prb",
                    "mcs",
                    "layers",
                    "bits_per_slot",
                    "p_dc_active_w",
                    "p_out_total_w",
                ],
                ascending=[True, True, True, True, True, True, True],
            )
            .reset_index(drop=True)
        )

    return PreparedJointOfdmaProblem(
        frame_n_slots=int(batch_space.frame_n_slots),
        t_slot_s=float(SINGLE_USER_SEARCH_CONFIG.t_slot_s),
        prb_max=int(
            math.floor(
                float(SINGLE_USER_SEARCH_CONFIG.channel_bw_hz)
                / (12.0 * float(SINGLE_USER_SEARCH_CONFIG.delta_f_hz))
            )
        ),
        n_tx_chains=int(batch_space.n_tx_chains),
        pa_catalog=tuple(batch_space.pa_catalog),
        user_requirements=user_requirements,
        user_slot_spaces=user_slot_spaces,
        )


def _apply_active_pa_dc_contract(
    candidate_table,
    *,
    n_tx_chains: int,
    pa_catalog: tuple,
):
    """Attach full active-slot PA DC power to OFDMA scheduler rows."""

    if candidate_table.empty:
        return candidate_table.copy().reindex(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    corrected_table = candidate_table.copy()
    corrected_table["p_dc_active_w"] = [
        pa_slot_dc_power(
            pa_catalog[int(row.pa_id)],
            p_out_total_w=float(row.p_out_total_w),
            n_tx_chains=int(n_tx_chains),
        )
        for row in corrected_table.itertuples(index=False)
    ]
    return corrected_table[BATCH_USER_PARAMETER_SPACE_COLUMNS].reset_index(drop=True)


__all__ = [
    "prepare_joint_ofdma_problem",
]
