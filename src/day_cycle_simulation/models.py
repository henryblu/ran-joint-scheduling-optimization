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


DEFAULT_DISTANCE_PRESETS_M = (50.0, 125.0, 200.0, 300.0, 500.0)
DEFAULT_DISTANCE_WEIGHTS = (0.08, 0.22, 0.34, 0.24, 0.12)

# The first generator version should be able to match the bin target exactly,
# so the runtime may still emit a smaller residual cleanup session when the
# remaining bin budget falls below the smallest preset.
DEFAULT_TOTAL_DATA_PRESETS_BITS = (
    1.6e10,  # 2 GB
    3.2e10,  # 4 GB
    6.4e10,  # 8 GB
    9.6e10,  # 12 GB
    1.6e11,  # 20 GB
)
DEFAULT_TOTAL_DATA_WEIGHTS = (0.10, 0.24, 0.34, 0.22, 0.10)

DEFAULT_NOMINAL_DURATION_PRESETS_BINS = (1, 2, 3)
DEFAULT_NOMINAL_DURATION_WEIGHTS = (0.30, 0.40, 0.30)
DEFAULT_SYNTHETIC_SESSION_GENERATION_CONFIG = SyntheticSessionGenerationConfig(
    day_bin_count=96,
    bin_duration_s=900.0,
    distance_presets_m=DEFAULT_DISTANCE_PRESETS_M,
    distance_weights=DEFAULT_DISTANCE_WEIGHTS,
    total_data_presets_bits=DEFAULT_TOTAL_DATA_PRESETS_BITS,
    total_data_weights=DEFAULT_TOTAL_DATA_WEIGHTS,
    nominal_duration_presets_bins=DEFAULT_NOMINAL_DURATION_PRESETS_BINS,
    nominal_duration_weights=DEFAULT_NOMINAL_DURATION_WEIGHTS,
    rng_seed=0,
)


__all__ = [
    "DEFAULT_DISTANCE_PRESETS_M",
    "DEFAULT_DISTANCE_WEIGHTS",
    "DEFAULT_NOMINAL_DURATION_PRESETS_BINS",
    "DEFAULT_NOMINAL_DURATION_WEIGHTS",
    "DEFAULT_SYNTHETIC_SESSION_GENERATION_CONFIG",
    "DEFAULT_TOTAL_DATA_PRESETS_BITS",
    "DEFAULT_TOTAL_DATA_WEIGHTS",
    "SyntheticSession",
    "SyntheticSessionGenerationConfig",
]
