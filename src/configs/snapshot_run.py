"""Shared defaults and runtime constants for finite-buffer snapshot runs."""

from finite_buffer_demand.models import FiniteBufferDemandSnapshotConfig
from models import SchedulerMode


DEFAULT_SNAPSHOT_RUN_CORES = 8
DEFAULT_FINITE_BUFFER_DEMAND_SNAPSHOT_CONFIG = FiniteBufferDemandSnapshotConfig(
    active_user_count=8,
    load_factor=1.0,
    distance_min_m=25.0,
    distance_max_m=500.0,
    reference_backlog_bits=100_000,
    frame_duration_s=0.010,
    distance_layout="area_uniform",
)

DEFAULT_TDMA_STRESS_ACTIVE_USER_COUNTS = (4, 8, 12, 16, 20, 24, 32)
DEFAULT_TDMA_STRESS_LOAD_FACTORS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_TDMA_STRESS_DISTANCE_LAYOUTS = ("area_uniform", "edge_heavy", "all_edge")

DEFAULT_OFDMA_MILP_TIME_LIMIT_S = None
DEFAULT_OFDMA_MILP_NODE_LIMIT = None
DEFAULT_OFDMA_MILP_REL_GAP = None
DEFAULT_OFDMA_MILP_MAX_USERS_PER_SLOT = None
DEFAULT_OFDMA_K2_ACCEPT_REL_GAP = 1e-3
DEFAULT_OFDMA_K2_CUTOFF_TIME_LIMIT_S = 60.0
DEFAULT_OFDMA_DUAL_ADMITTED_ROWS_PER_USER = 5
DEFAULT_OFDMA_ADMITTED_ROWS_PER_USER = 20

THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
WORKER_LOG_LEVEL_ENV_VAR = "THESIS_RUN_LOG_LEVEL"
ACTIVE_SNAPSHOT_SCOPE_ENV_VAR = "THESIS_ACTIVE_SNAPSHOT_SCOPE"
ACTIVE_RUN_SCOPE_ENV_VAR = "THESIS_ACTIVE_RUN_SCOPE"
MILP_TIME_LIMIT_ENV_VAR = "THESIS_MILP_TIME_LIMIT_S"
MILP_NODE_LIMIT_ENV_VAR = "THESIS_MILP_NODE_LIMIT"
MILP_REL_GAP_ENV_VAR = "THESIS_MILP_REL_GAP"
OFDMA_MILP_MAX_USERS_PER_SLOT_ENV_VAR = "THESIS_OFDMA_MILP_MAX_USERS_PER_SLOT"

LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def build_snapshot_run_result_filename(*, scheduler_mode, switch_policy) -> str:
    """Return a self-describing JSON filename for one resolved snapshot config."""

    scheduler_value = str(getattr(scheduler_mode, "value", scheduler_mode))
    switch_policy_value = str(getattr(switch_policy, "value", switch_policy))
    return f"snapshot_run_{scheduler_value}_{switch_policy_value}.json"


def supports_single_snapshot_run(scheduler_mode) -> bool:
    """Return whether the selected scheduler mode is valid for snapshot-run orchestration."""

    return SchedulerMode(str(getattr(scheduler_mode, "value", scheduler_mode))) in set(SchedulerMode)


__all__ = [
    "ACTIVE_SNAPSHOT_SCOPE_ENV_VAR",
    "ACTIVE_RUN_SCOPE_ENV_VAR",
    "DEFAULT_FINITE_BUFFER_DEMAND_SNAPSHOT_CONFIG",
    "DEFAULT_OFDMA_DUAL_ADMITTED_ROWS_PER_USER",
    "DEFAULT_OFDMA_ADMITTED_ROWS_PER_USER",
    "DEFAULT_OFDMA_K2_ACCEPT_REL_GAP",
    "DEFAULT_OFDMA_K2_CUTOFF_TIME_LIMIT_S",
    "DEFAULT_OFDMA_MILP_NODE_LIMIT",
    "DEFAULT_OFDMA_MILP_MAX_USERS_PER_SLOT",
    "DEFAULT_OFDMA_MILP_REL_GAP",
    "DEFAULT_OFDMA_MILP_TIME_LIMIT_S",
    "DEFAULT_SNAPSHOT_RUN_CORES",
    "DEFAULT_TDMA_STRESS_ACTIVE_USER_COUNTS",
    "DEFAULT_TDMA_STRESS_DISTANCE_LAYOUTS",
    "DEFAULT_TDMA_STRESS_LOAD_FACTORS",
    "FiniteBufferDemandSnapshotConfig",
    "LOG_LEVEL_CHOICES",
    "MILP_NODE_LIMIT_ENV_VAR",
    "OFDMA_MILP_MAX_USERS_PER_SLOT_ENV_VAR",
    "MILP_REL_GAP_ENV_VAR",
    "MILP_TIME_LIMIT_ENV_VAR",
    "THREAD_LIMIT_ENV_VARS",
    "WORKER_LOG_LEVEL_ENV_VAR",
    "build_snapshot_run_result_filename",
    "supports_single_snapshot_run",
]
