"""Compatibility re-export for older internal imports.

The multi-user TDMA study logic now lives in `study.py` and `user_space.py`.
This module remains only to avoid breaking older imports while the notebooks are updated.
"""

from .user_space import (
    ACTIVE_OPERATING_COLUMNS,
    USER_CANDIDATE_COLUMNS,
    build_active_candidate_summary_df,
    build_user_candidate_space,
    build_user_candidate_spaces,
    enumerate_user_active_operating_tables,
    prune_exactly_dominated_user_space,
    resolve_repeated_frame_requirement,
)

__all__ = [
    "ACTIVE_OPERATING_COLUMNS",
    "USER_CANDIDATE_COLUMNS",
    "build_active_candidate_summary_df",
    "build_user_candidate_space",
    "build_user_candidate_spaces",
    "enumerate_user_active_operating_tables",
    "prune_exactly_dominated_user_space",
    "resolve_repeated_frame_requirement",
]
