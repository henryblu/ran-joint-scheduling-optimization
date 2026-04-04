from __future__ import annotations

import logging
from functools import lru_cache

from candidate_table_generation import build_distance_binned_candidate_table
from configs import SINGLE_USER_SEARCH_CONFIG, USER_REQUIREMENT_COLUMNS, build_pa_catalog
from models import BatchUserParameterSpace, UserRequest
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from run_reporting import build_console_message


LOGGER = logging.getLogger("day_run")
PA_CATALOG = tuple(build_pa_catalog(SINGLE_USER_SEARCH_CONFIG.pa_data_csv))


def lookup_user_parameter_space(request):
    """Return one user's feasible full-frame rows from the precomputed lookup table."""

    frontiers_by_distance_m = _distance_binned_candidate_table().frontiers_by_distance_m
    distance_bin_m = min(
        frontiers_by_distance_m,
        key=lambda value: abs(float(value) - float(request.distance_m)),
    )
    user_space = (
        frontiers_by_distance_m[int(distance_bin_m)]
        .loc[
            lambda table: table["rate_active_bps"].astype(float) >= float(request.required_rate_bps),
            BATCH_USER_PARAMETER_SPACE_COLUMNS,
        ]
        .copy()
        .reset_index(drop=True)
    )
    LOGGER.debug(
        build_console_message(
            level_tag="DEBUG",
            scope="SULK",
            stage="lookup",
            event="user",
            fields=[
                ("user", str(getattr(request, "user_id", "na"))),
                ("dist_m", f"{float(request.distance_m):.1f}"),
                ("bin_m", str(int(distance_bin_m))),
                ("rate_mbps", f"{float(request.required_rate_bps) / 1e6:.1f}"),
                ("bin_rows", str(int(len(frontiers_by_distance_m[int(distance_bin_m)])))),
                ("kept_rows", str(int(len(user_space)))),
            ],
        )
    )
    return user_space


def build_batch_user_parameter_space(user_table) -> BatchUserParameterSpace:
    """Build the trusted batch artifact by looking up one feasible space per user."""

    users = user_table[USER_REQUIREMENT_COLUMNS].copy()
    users["user_id"] = users["user_id"].astype(int)
    users["distance_m"] = users["distance_m"].astype(float)
    users["required_rate_bps"] = users["required_rate_bps"].astype(float)
    return BatchUserParameterSpace(
        user_requirements=users[["user_id", "required_rate_bps"]].copy(),
        user_parameter_spaces={
            int(user_row.user_id): lookup_user_parameter_space(
                UserRequest(
                    user_id=int(user_row.user_id),
                    distance_m=float(user_row.distance_m),
                    required_rate_bps=float(user_row.required_rate_bps),
                )
            )
            for user_row in users.itertuples(index=False)
        },
        frame_n_slots=int(SINGLE_USER_SEARCH_CONFIG.n_slots_win),
        n_tx_chains=int(SINGLE_USER_SEARCH_CONFIG.n_tx_chains),
        pa_catalog=PA_CATALOG,
    )


@lru_cache(maxsize=1)
def _distance_binned_candidate_table():
    """Resolve the precomputed distance-binned candidate table once per process."""

    return build_distance_binned_candidate_table()


__all__ = [
    "build_batch_user_parameter_space",
    "lookup_user_parameter_space",
]
