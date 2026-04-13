"""Synthetic day-population generation from a 15-minute target load table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from configs import USER_REQUIREMENT_COLUMNS

from .load_curve import build_15_minute_target_load_table, load_hourly_load_curve
from .models import SyntheticSession, SyntheticSessionGenerationConfig


RESIDUAL_TOL = 1.0
SCHEDULER_DAY_USER_COLUMNS = ["bin_index", *USER_REQUIREMENT_COLUMNS]


def build_scheduler_day_user_table(
    load_curve_csv,
    config: SyntheticSessionGenerationConfig,
) -> pd.DataFrame:
    """Build the lean day-wide scheduler request table consumed by main orchestration.

    Steps:
    1. Load the hourly offered-load curve from CSV.
    2. Expand it onto the 15-minute target bins owned by the day-cycle layer.
    3. Generate the full synthetic session population for that day.
    4. Expand each session onto the bin/user scheduler contract used downstream.
    """

    hourly_load_table = load_hourly_load_curve(load_curve_csv)
    target_load_table = build_15_minute_target_load_table(hourly_load_table)
    sessions = generate_synthetic_day_population(
        config=config,
        target_load_table=target_load_table,
    )
    return _build_scheduler_day_user_table_from_sessions(
        sessions,
        bin_duration_s=float(config.bin_duration_s),
    )


def generate_synthetic_day_population(
    config: SyntheticSessionGenerationConfig,
    target_load_table: pd.DataFrame,
) -> tuple[SyntheticSession, ...]:
    """Generate one synthetic day that tracks the target table with preset-sized sessions.

    Steps:
    1. Copy the target bin budget into a mutable remaining-demand vector.
    2. Repeatedly build one feasible session against that remaining budget.
    3. Subtract the session's nominal per-bin footprint from the remaining bins.
    4. Stop once every bin residual falls below the smallest total-data preset.
    """

    rng = np.random.default_rng(config.rng_seed)
    remaining_bits = target_load_table["target_bits_in_bin"].to_numpy(dtype=float).copy()
    smallest_total_data_bits = float(min(config.total_data_presets_bits))
    sessions = []
    session_id = 0

    while _has_remaining_demand(
        remaining_bits,
        smallest_total_data_bits=smallest_total_data_bits,
    ):
        session = _generate_synthetic_session(
            session_id=session_id,
            config=config,
            remaining_bits=remaining_bits,
            rng=rng,
        )
        _apply_nominal_session_to_remaining_bits(remaining_bits, session)
        _clip_small_negative_residuals(remaining_bits)
        sessions.append(session)
        session_id += 1

    return tuple(sessions)


def _generate_synthetic_session(
    session_id: int,
    config: SyntheticSessionGenerationConfig,
    remaining_bits: np.ndarray,
    rng: np.random.Generator,
) -> SyntheticSession:
    """Build one feasible session against the current remaining-demand vector."""

    nominal_duration_bins = _sample_nominal_duration_bins(
        config=config,
        remaining_bits=remaining_bits,
        rng=rng,
    )
    start_bin = _sample_start_bin(
        config=config,
        remaining_bits=remaining_bits,
        nominal_duration_bins=nominal_duration_bins,
        rng=rng,
    )
    total_data_bits = _sample_total_data_bits(
        config=config,
        remaining_bits=remaining_bits,
        start_bin=start_bin,
        nominal_duration_bins=nominal_duration_bins,
        rng=rng,
    )
    distance_m = _sample_distance_m(config=config, rng=rng)

    return SyntheticSession(
        session_id=session_id,
        distance_m=distance_m,
        total_data_bits=total_data_bits,
        start_bin=start_bin,
        nominal_duration_bins=nominal_duration_bins,
    )


def _build_scheduler_day_user_table_from_sessions(
    sessions: tuple[SyntheticSession, ...],
    *,
    bin_duration_s: float,
) -> pd.DataFrame:
    """Expand the synthetic day population onto the scheduler user-request contract."""

    rows = []
    for session in sessions:
        required_rate_bps = (
            float(session.total_data_bits)
            / float(session.nominal_duration_bins)
            / float(bin_duration_s)
        )
        stop_bin = int(session.start_bin) + int(session.nominal_duration_bins)
        rows.extend(
            {
                "bin_index": int(bin_index),
                "user_id": int(session.session_id),
                "distance_m": float(session.distance_m),
                "required_rate_bps": float(required_rate_bps),
            }
            for bin_index in range(int(session.start_bin), stop_bin)
        )

    return pd.DataFrame(rows, columns=SCHEDULER_DAY_USER_COLUMNS)


def _sample_nominal_duration_bins(
    config: SyntheticSessionGenerationConfig,
    remaining_bits: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Choose one duration preset that still supports at least one preset-sized session."""

    smallest_total_data_bits = float(min(config.total_data_presets_bits))
    feasible_duration_presets = [
        (int(duration_bins), float(weight))
        for duration_bins, weight in zip(
            config.nominal_duration_presets_bins,
            config.nominal_duration_weights,
        )
        if _duration_has_preset_feasible_window(
            remaining_bits=remaining_bits,
            duration_bins=int(duration_bins),
            smallest_total_data_bits=smallest_total_data_bits,
        )
    ]
    if feasible_duration_presets:
        feasible_durations, feasible_weights = zip(*feasible_duration_presets)
        return int(_weighted_choice(feasible_durations, feasible_weights, rng))

    raise RuntimeError("Could not find a duration that can host a preset-sized session.")


def _sample_start_bin(
    config: SyntheticSessionGenerationConfig,
    remaining_bits: np.ndarray,
    nominal_duration_bins: int,
    rng: np.random.Generator,
) -> int:
    """Choose one feasible session start, weighted by the window headroom it can absorb."""

    feasible_start_bins = []
    feasible_weights = []
    max_start_bin = int(config.day_bin_count) - int(nominal_duration_bins)
    smallest_total_data_bits = float(min(config.total_data_presets_bits))

    for start_bin in range(max_start_bin + 1):
        window_remaining_bits = remaining_bits[start_bin : start_bin + nominal_duration_bins]
        max_total_data_bits = _max_total_data_bits_for_window(window_remaining_bits)
        if max_total_data_bits + RESIDUAL_TOL < smallest_total_data_bits:
            continue
        feasible_start_bins.append(int(start_bin))
        feasible_weights.append(float(max_total_data_bits))

    if not feasible_start_bins:
        raise RuntimeError("Could not find a feasible start_bin for the remaining demand.")

    return int(_weighted_choice(feasible_start_bins, feasible_weights, rng))


def _sample_total_data_bits(
    config: SyntheticSessionGenerationConfig,
    remaining_bits: np.ndarray,
    start_bin: int,
    nominal_duration_bins: int,
    rng: np.random.Generator,
) -> float:
    """Choose one feasible total-data preset for the selected session window."""

    window_remaining_bits = remaining_bits[start_bin : start_bin + nominal_duration_bins]
    max_total_data_bits = _max_total_data_bits_for_window(window_remaining_bits)
    feasible_total_data_presets = [
        (float(total_data_bits), float(weight))
        for total_data_bits, weight in zip(
            config.total_data_presets_bits,
            config.total_data_weights,
        )
        if float(total_data_bits) <= float(max_total_data_bits) + RESIDUAL_TOL
    ]
    if feasible_total_data_presets:
        feasible_total_data_bits, feasible_weights = zip(*feasible_total_data_presets)
        return float(_weighted_choice(feasible_total_data_bits, feasible_weights, rng))

    raise RuntimeError("Selected session window could not host any total-data preset.")


def _sample_distance_m(
    config: SyntheticSessionGenerationConfig,
    rng: np.random.Generator,
) -> float:
    """Choose one global distance preset independently of the bin budget."""

    return float(_weighted_choice(config.distance_presets_m, config.distance_weights, rng))


def _duration_has_preset_feasible_window(
    remaining_bits: np.ndarray,
    duration_bins: int,
    smallest_total_data_bits: float,
) -> bool:
    """Return whether any window of this duration can still host one preset-sized session."""

    max_start_bin = len(remaining_bits) - int(duration_bins)
    return any(
        _max_total_data_bits_for_window(
            remaining_bits[start_bin : start_bin + duration_bins]
        )
        + RESIDUAL_TOL
        >= smallest_total_data_bits
        for start_bin in range(max_start_bin + 1)
    )


def _max_total_data_bits_for_window(window_remaining_bits: np.ndarray) -> float:
    """Return the largest uniform-over-window session that fits the window residual."""

    return float(len(window_remaining_bits) * float(np.min(window_remaining_bits)))


def _apply_nominal_session_to_remaining_bits(
    remaining_bits: np.ndarray,
    session: SyntheticSession,
) -> None:
    """Subtract one session's nominal uniform-over-window footprint from the remaining bins."""

    nominal_bits_per_bin = float(session.total_data_bits) / float(session.nominal_duration_bins)
    stop_bin = int(session.start_bin) + int(session.nominal_duration_bins)
    remaining_bits[int(session.start_bin) : stop_bin] -= nominal_bits_per_bin


def _has_remaining_demand(
    remaining_bits: np.ndarray,
    *,
    smallest_total_data_bits: float,
) -> bool:
    """Return whether any bin still carries at least one preset-sized residual."""

    return bool(np.any(remaining_bits + RESIDUAL_TOL >= float(smallest_total_data_bits)))


def _clip_small_negative_residuals(remaining_bits: np.ndarray) -> None:
    """Zero small floating-point negatives introduced by subtraction and weighting."""

    negative_mask = (remaining_bits < 0.0) & (np.abs(remaining_bits) <= RESIDUAL_TOL)
    remaining_bits[negative_mask] = 0.0


def _weighted_choice(values, weights, rng: np.random.Generator):
    """Draw one value from a weighted finite preset catalog."""

    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    choice_index = int(rng.choice(len(values), p=probabilities))
    return values[choice_index]


__all__ = [
    "SCHEDULER_DAY_USER_COLUMNS",
    "build_scheduler_day_user_table",
    "generate_synthetic_day_population",
]
