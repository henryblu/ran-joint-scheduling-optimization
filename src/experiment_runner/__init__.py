"""Official experiment execution entrypoint and campaign contracts."""

from .cli import build_experiment_run_config, parse_args, run_from_cli
from .models import ExperimentRunConfig, ExperimentRunResult
from .runner import run_experiment_case


__all__ = [
    "ExperimentRunConfig",
    "ExperimentRunResult",
    "build_experiment_run_config",
    "parse_args",
    "run_experiment_case",
    "run_from_cli",
]
