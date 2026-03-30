"""Shared default presets for synthetic day-cycle session generation."""

from day_cycle_simulation.models import SyntheticSessionGenerationConfig


DEFAULT_DISTANCE_PRESETS_M = (50.0, 125.0, 200.0, 300.0, 500.0)
DEFAULT_DISTANCE_WEIGHTS = (0.08, 0.22, 0.34, 0.24, 0.12)

# The generator stops once every bin residual falls below the 2 GB floor,
# so only preset-sized sessions are emitted into the scheduler-facing day table.
DEFAULT_TOTAL_DATA_PRESETS_BITS = (
    0.8e10,  # 1 GB
    1.6e10,  # 2 GB
    3.2e10,  # 4 GB
    6.4e10,  # 8 GB
    12.8e10,  # 16 GB
)
DEFAULT_TOTAL_DATA_WEIGHTS = (0.10, 0.14, 0.29, 0.32, 0.50)

DEFAULT_NOMINAL_DURATION_PRESETS_BINS = (1, 2, 3)
DEFAULT_NOMINAL_DURATION_WEIGHTS = (0.50, 0.30, 0.20)

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
    "SyntheticSessionGenerationConfig",
]
