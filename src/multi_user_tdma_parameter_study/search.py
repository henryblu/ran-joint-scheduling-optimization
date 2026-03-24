"""Compatibility re-export for older internal imports."""

from .api import enumerate_user_active_operating_tables
from .user_space import (
    ACTIVE_OPERATING_COLUMNS,
    USER_CANDIDATE_COLUMNS,
    build_active_candidate_summary_df,
    build_user_candidate_space,
    build_user_candidate_spaces,
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
