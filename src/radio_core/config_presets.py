from dataclasses import dataclass, replace

from .config_values import (
    COMMON_LINK_CONSTANTS,
    COMMON_PHY_CONSTANTS,
    COMMON_SCHEDULER_SPACE,
    DEFAULT_PA_DATA_CSV,
)
from .mcs_tables import DEFAULT_NR_MCS_TABLE
from .pa_models import PASwitchPolicy


@dataclass(frozen=True)
class ModelPreset:
    link: object
    phy: object
    scheduler: object
    mcs_table: dict
    pa_data_csv: str


@dataclass(frozen=True)
class TddPatternConfig:
    tdd_pattern_slots: int
    ul_slots: int


@dataclass(frozen=True)
class MultiUserRuntimeConfig:
    switch_policy: PASwitchPolicy
    max_configs_per_user: int
    max_schedule_windows: int


@dataclass(frozen=True)
class MultiUserPreset:
    model: ModelPreset
    tdd: TddPatternConfig
    runtime: MultiUserRuntimeConfig


SINGLE_USER_RESOURCE_MODEL_PRESET = ModelPreset(
    link=COMMON_LINK_CONSTANTS,
    phy=replace(COMMON_PHY_CONSTANTS, n_slots_win=7),
    scheduler=COMMON_SCHEDULER_SPACE,
    mcs_table=DEFAULT_NR_MCS_TABLE,
    pa_data_csv=DEFAULT_PA_DATA_CSV,
)

SINGLE_USER_POWER_OPTIMIZATION_PRESET = ModelPreset(
    link=COMMON_LINK_CONSTANTS,
    phy=replace(COMMON_PHY_CONSTANTS, n_slots_win=20),
    scheduler=COMMON_SCHEDULER_SPACE,
    mcs_table=DEFAULT_NR_MCS_TABLE,
    pa_data_csv=DEFAULT_PA_DATA_CSV,
)

MULTI_USER_TDMA_PRESET = MultiUserPreset(
    model=ModelPreset(
        link=COMMON_LINK_CONSTANTS,
        phy=replace(COMMON_PHY_CONSTANTS, n_slots_win=20),
        scheduler=COMMON_SCHEDULER_SPACE,
        mcs_table=DEFAULT_NR_MCS_TABLE,
        pa_data_csv=DEFAULT_PA_DATA_CSV,
    ),
    tdd=TddPatternConfig(
        tdd_pattern_slots=10,
        ul_slots=3,
    ),
    runtime=MultiUserRuntimeConfig(
        switch_policy=PASwitchPolicy.STANDBY,
        max_configs_per_user=300,
        max_schedule_windows=32,
    ),
)

__all__ = [
    "ModelPreset",
    "MultiUserPreset",
    "MultiUserRuntimeConfig",
    "MULTI_USER_TDMA_PRESET",
    "SINGLE_USER_POWER_OPTIMIZATION_PRESET",
    "SINGLE_USER_RESOURCE_MODEL_PRESET",
    "TddPatternConfig",
]
