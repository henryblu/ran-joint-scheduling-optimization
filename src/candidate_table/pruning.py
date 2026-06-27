from __future__ import annotations

import pandas as pd

from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS


_DOMINANCE_TOL = 1e-12


def prune_candidate_frontier(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Strict-prune dominated scheduler-facing rows within each PA family.

    A stored row is removable only when another row on the same PA family:
    1. uses no more PRBs,
    2. draws no more active DC power,
    3. uses no more aggregate RF output,
    4. delivers no less payload bits per active slot,
    5. and improves at least one of those axes strictly.
    """

    if candidate_table.empty:
        return candidate_table.copy().reindex(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)

    ranked_rows = candidate_table.sort_values(
        [
            "pa_id",
            "n_prb",
            "p_dc_active_w",
            "p_out_total_w",
            "bits_per_slot",
            "mcs",
            "layers",
        ],
        ascending=[True, True, True, True, False, True, True],
    ).to_dict("records")

    kept_rows = []
    kept_rows_by_pa = {}
    for row in ranked_rows:
        pa_id = int(row["pa_id"])
        kept_rows_for_pa = kept_rows_by_pa.setdefault(pa_id, [])
        if any(
            int(kept_row["n_prb"]) <= int(row["n_prb"])
            and float(kept_row["p_dc_active_w"]) <= float(row["p_dc_active_w"]) + _DOMINANCE_TOL
            and float(kept_row["p_out_total_w"]) <= float(row["p_out_total_w"]) + _DOMINANCE_TOL
            and float(kept_row["bits_per_slot"]) >= float(row["bits_per_slot"]) - _DOMINANCE_TOL
            and (
                int(kept_row["n_prb"]) < int(row["n_prb"])
                or float(kept_row["p_dc_active_w"]) < float(row["p_dc_active_w"]) - _DOMINANCE_TOL
                or float(kept_row["p_out_total_w"]) < float(row["p_out_total_w"]) - _DOMINANCE_TOL
                or float(kept_row["bits_per_slot"]) > float(row["bits_per_slot"]) + _DOMINANCE_TOL
            )
            for kept_row in kept_rows_for_pa
        ):
            continue

        kept_rows_for_pa.append(row)
        kept_rows.append(row)

    return pd.DataFrame(kept_rows, columns=BATCH_USER_PARAMETER_SPACE_COLUMNS).reset_index(drop=True)


__all__ = [
    "prune_candidate_frontier",
]
