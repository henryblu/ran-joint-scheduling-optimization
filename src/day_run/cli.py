from __future__ import annotations

"""Build the small CLI surface for one synthetic day TDMA simulation run.

This module does only three things:
1. Parse the handful of supported command-line flags.
2. Resolve those flags onto the shared day-run config model.
3. Hand off execution and logging setup to the orchestration layer.
"""

import argparse
from pathlib import Path

from configs.day_run import (
    DEFAULT_DAY_RUN_CORES,
    DEFAULT_DAY_RUN_LOAD_CURVE_CSV,
    DEFAULT_DAY_RUN_SESSION_GENERATION_CONFIG,
    LOG_LEVEL_CHOICES,
)
from models import PASwitchPolicy
from models.day_run import DayRunConfig
from run_reporting import configure_run_logging

from .runner import run_day


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_from_cli(argv: list[str] | None = None):
    """Parse CLI inputs, configure reporting, and run one full day simulation."""

    args = parse_args(argv)
    config = build_day_run_config(
        switch_policy=PASwitchPolicy(args.switch_policy),
        cores=args.cores,
        load_curve_csv=args.load_curve_csv,
        log_level=args.log_level,
    )
    # Configure the parent process once here. Worker processes inherit the
    # chosen level later when the runner spins up the process pool.
    configure_run_logging(config.log_level)
    return run_day(config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the small CLI surface for one TDMA day simulation run."""

    parser = argparse.ArgumentParser(
        description="Thin orchestration entry point for one synthetic day TDMA simulation run."
    )
    parser.add_argument(
        "--switch-policy",
        choices=[policy.value for policy in PASwitchPolicy],
        default=PASwitchPolicy.DUAL_SWITCHABLE.value,
        help="PA switching scenario used by the TDMA scheduler.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        choices=range(1, 129),
        default=DEFAULT_DAY_RUN_CORES,
        help="Total core budget for the run. The bin worker count is capped from this budget.",
    )
    parser.add_argument(
        "--load-curve-csv",
        type=Path,
        default=None,
        help="Optional load-curve CSV override. Defaults to the shared day-cycle config path.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=None,
        help="Optional logging threshold for progress and solver-space diagnostics.",
    )
    return parser.parse_args(argv)


def build_day_run_config(
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
    cores: int = DEFAULT_DAY_RUN_CORES,
    load_curve_csv: Path | None = None,
    log_level: str | None = None,
) -> DayRunConfig:
    """Resolve the day-run config using the shared defaults owned by configs.

    Keep the config surface intentionally small: lower layers still own model
    validation, user-table normalization, and scheduler preparation details.
    """

    return DayRunConfig(
        load_curve_csv=DEFAULT_DAY_RUN_LOAD_CURVE_CSV if load_curve_csv is None else Path(load_curve_csv),
        session_generation_config=DEFAULT_DAY_RUN_SESSION_GENERATION_CONFIG,
        switch_policy=switch_policy,
        cores=int(cores),
        output_dir=REPO_ROOT / "outputs" / f"default_day_run_{switch_policy.value}",
        log_level=None if log_level is None else str(log_level).upper(),
    )


__all__ = [
    "build_day_run_config",
    "parse_args",
    "run_from_cli",
]
