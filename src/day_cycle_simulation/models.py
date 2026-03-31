from __future__ import annotations

"""Core data models for synthetic day-cycle session generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticSessionGenerationConfig:
    """Resolved inputs for one synthetic day of session generation."""

    # Keep the generator boundary narrow: distance stays global and
    # independent, data and duration come from small weighted preset
    # catalogs, and start-bin placement stays owned by the load curve.
    day_bin_count: int
    bin_duration_s: float
    distance_presets_m: tuple[float, ...]
    distance_weights: tuple[float, ...]
    total_data_presets_bits: tuple[float, ...]
    total_data_weights: tuple[float, ...]
    nominal_duration_presets_bins: tuple[int, ...]
    nominal_duration_weights: tuple[float, ...]
    rng_seed: int


@dataclass(frozen=True)
class SyntheticSession:
    """One synthetic session demand record for a simulated day."""

    session_id: int
    distance_m: float
    total_data_bits: float
    start_bin: int
    nominal_duration_bins: int


__all__ = [
    "SyntheticSession",
    "SyntheticSessionGenerationConfig",
]
