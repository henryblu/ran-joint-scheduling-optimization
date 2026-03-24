"""Shared immutable radio config aliases owned by the static config layer."""

from .defaults import COMMON_RADIO_CONFIG


SINGLE_USER_SEARCH_CONFIG = COMMON_RADIO_CONFIG
MULTI_USER_TDMA_CONFIG = COMMON_RADIO_CONFIG


def get_scenario_config(name: str):
    """Return the shared static radio config for a known scenario alias."""

    if str(name) in {"single_user_search", "multi_user_tdma"}:
        return COMMON_RADIO_CONFIG
    raise KeyError(f"Unknown radio scenario config: {name}")


__all__ = [
    "MULTI_USER_TDMA_CONFIG",
    "SINGLE_USER_SEARCH_CONFIG",
    "get_scenario_config",
]
