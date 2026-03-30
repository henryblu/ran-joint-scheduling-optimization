"""Shared user-request schema builders and trusted presets."""

from typing import Iterable

import pandas as pd

from models import UserRequest


USER_REQUIREMENT_COLUMNS = [
    "user_id",
    "distance_m",
    "required_rate_bps",
]


def build_user_requirements_table(users: Iterable[UserRequest]) -> pd.DataFrame:
    """Materialize a trusted user-request table from immutable preset requests."""

    rows = [
        {
            "user_id": int(user.user_id),
            "distance_m": float(user.distance_m),
            "required_rate_bps": float(user.required_rate_bps),
        }
        for user in users
    ]
    return pd.DataFrame(rows, columns=USER_REQUIREMENT_COLUMNS)


SIMPLE_TWO_USER_REQUESTS = (
    UserRequest(user_id=1, distance_m=100.0, required_rate_bps=5e6),
    UserRequest(user_id=2, distance_m=150.0, required_rate_bps=4e6),
)


TWO_FRAME_WINDOW_REQUESTS = (
    UserRequest(user_id=1, distance_m=100.0, required_rate_bps=10.1e6),
    UserRequest(user_id=2, distance_m=150.0, required_rate_bps=9.1e6),
)


INFEASIBLE_TWO_USER_REQUESTS = (
    UserRequest(user_id=1, distance_m=100.0, required_rate_bps=18e6),
    UserRequest(user_id=2, distance_m=150.0, required_rate_bps=18e6),
)


def get_user_preset(name: str):
    """Return one trusted user-group preset by name."""

    if str(name) == "simple_two_user":
        return SIMPLE_TWO_USER_REQUESTS
    if str(name) == "two_frame_window":
        return TWO_FRAME_WINDOW_REQUESTS
    if str(name) == "infeasible_two_user":
        return INFEASIBLE_TWO_USER_REQUESTS
    raise KeyError(f"Unknown user preset: {name}")


__all__ = [
    "INFEASIBLE_TWO_USER_REQUESTS",
    "SIMPLE_TWO_USER_REQUESTS",
    "TWO_FRAME_WINDOW_REQUESTS",
    "USER_REQUIREMENT_COLUMNS",
    "build_user_requirements_table",
    "get_user_preset",
]
