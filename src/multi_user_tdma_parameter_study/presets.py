"""TDMA-specific preset helpers owned by the TDMA study layer."""

from dataclasses import dataclass

from pa_models import PASwitchPolicy
from radio_configs import MULTI_USER_TDMA_CONFIG, RadioConfig


@dataclass(frozen=True)
class TddPatternConfig:
    n_dl_symbols: int
    n_guard_symbols: int
    n_ul_symbols: int


@dataclass(frozen=True)
class MultiUserRuntimeConfig:
    switch_policy: PASwitchPolicy
    max_configs_per_user: int
    max_schedule_windows: int


@dataclass(frozen=True)
class MultiUserPreset:
    """TDMA-owned preset wrapper around shared radio config plus TDMA extras."""

    scenario: RadioConfig
    tdd: TddPatternConfig
    runtime: MultiUserRuntimeConfig


LEGACY_MULTI_USER_TDD = TddPatternConfig(
    n_dl_symbols=10,
    n_guard_symbols=1,
    n_ul_symbols=3,
)

LEGACY_MULTI_USER_RUNTIME = MultiUserRuntimeConfig(
    switch_policy=PASwitchPolicy.STANDBY,
    max_configs_per_user=300,
    max_schedule_windows=32,
)

MULTI_USER_TDMA_PRESET = MultiUserPreset(
    scenario=MULTI_USER_TDMA_CONFIG,
    tdd=LEGACY_MULTI_USER_TDD,
    runtime=LEGACY_MULTI_USER_RUNTIME,
)


__all__ = [
    "LEGACY_MULTI_USER_RUNTIME",
    "LEGACY_MULTI_USER_TDD",
    "MULTI_USER_TDMA_PRESET",
    "MultiUserPreset",
    "MultiUserRuntimeConfig",
    "TddPatternConfig",
]
