"""Static radio configuration inputs and defaults."""

from .defaults import COMMON_RADIO_CONFIG, DEFAULT_PA_DATA_CSV
from .mcs_tables import DEFAULT_NR_MCS_TABLE
from .presets import MULTI_USER_TDMA_CONFIG, SINGLE_USER_SEARCH_CONFIG, get_scenario_config
from .types import FrozenMcsTable, RadioConfig, freeze_mcs_table

__all__ = [
    "COMMON_RADIO_CONFIG",
    "DEFAULT_NR_MCS_TABLE",
    "DEFAULT_PA_DATA_CSV",
    "FrozenMcsTable",
    "MULTI_USER_TDMA_CONFIG",
    "RadioConfig",
    "SINGLE_USER_SEARCH_CONFIG",
    "freeze_mcs_table",
    "get_scenario_config",
]
