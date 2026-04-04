from __future__ import annotations

import logging
from functools import lru_cache

from candidate_table_generation import load_or_build_distance_binned_candidate_table
from .console_logging import emit_lookup_console_log
from configs import SINGLE_USER_SEARCH_CONFIG, USER_REQUIREMENT_COLUMNS, build_pa_catalog
from models import BatchUserParameterSpace, UserRequest
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS


PA_CATALOG = tuple(build_pa_catalog(SINGLE_USER_SEARCH_CONFIG.pa_data_csv))


def lookup_user_parameter_space(request):
    """Return one user's feasible full-frame rows from the precomputed lookup table."""

    frontiers_by_distance_m = _distance_binned_candidate_table().frontiers_by_distance_m
    distance_m = float(request.distance_m)
    max_distance_bin_m = max(frontiers_by_distance_m)
    if distance_m > float(max_distance_bin_m):
        emit_lookup_console_log(
            level=logging.WARNING,
            stage="lookup",
            event="range",
            fields=[
                ("user", str(getattr(request, "user_id", "na"))),
                ("dist_m", f"{distance_m:.1f}"),
                ("max_bin_m", str(int(max_distance_bin_m))),
            ],
        )
        return (
            frontiers_by_distance_m[int(min(frontiers_by_distance_m))]
            .iloc[0:0]
            .copy()
            .reset_index(drop=True)
        )

    distance_bin_m = next(
        distance_bin_m
        for distance_bin_m in sorted(frontiers_by_distance_m)
        if float(distance_bin_m) >= distance_m
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
    emit_lookup_console_log(
        level=logging.DEBUG,
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
    """Resolve the precomputed distance-binned candidate table once per process.

    The candidate-table package owns disk load/build/save. This thin wrapper only
    exists to memoize that artifact in memory so repeated user lookups inside one
    worker process do not keep reloading the JSON-backed table.
    """

    return load_or_build_distance_binned_candidate_table()


__all__ = [
    "build_batch_user_parameter_space",
    "lookup_user_parameter_space",
]
