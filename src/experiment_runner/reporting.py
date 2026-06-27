from __future__ import annotations

"""Console reporting for completed experiment runs."""

from .models import ExperimentRunConfig, ExperimentRunResult


def print_experiment_result(config: ExperimentRunConfig, result: ExperimentRunResult) -> None:
    """Print the compact summary for a completed finite-frame experiment."""

    schedule_result = result.schedule_result
    solver_details = dict(schedule_result.solver_details)
    power_summary = schedule_result.power_summary
    active_slots = sum(1 for slot in schedule_result.slot_schedules if slot.active)
    allocation_count = sum(len(slot.allocations) for slot in schedule_result.slot_schedules)
    print(
        "EXPERIMENT_RUN",
        f"status={result.status}",
        f"scheduler={schedule_result.scheduler_mode.value}",
        f"algorithm={solver_details.get('algorithm', 'unknown')}",
        f"policy={config.switch_policy.value}",
        f"users={config.user_generation_config.active_user_count}",
        f"load={config.user_generation_config.load_factor:g}",
        f"distance_m={config.user_generation_config.distance_max_m:g}",
    )
    print(
        "EXPERIMENT_RESULT",
        f"feasible={schedule_result.feasible}",
        f"infeasible_reason={schedule_result.infeasible_reason}",
        f"active_slots={active_slots}",
        f"allocations={allocation_count}",
        f"avg_dc_w={power_summary.average_frame_dc_power_w:.9g}",
        f"frame_energy_j={power_summary.frame_energy_j:.9g}",
    )
    print(
        "EXPERIMENT_TIMINGS",
        f"candidate_table_s={result.candidate_table_elapsed_s:.3f}",
        f"user_generation_s={result.user_generation_elapsed_s:.3f}",
        f"candidate_lookup_s={result.candidate_lookup_elapsed_s:.3f}",
        f"scheduler_s={result.scheduler_elapsed_s:.3f}",
        f"total_s={result.total_elapsed_s:.3f}",
    )


__all__ = ["print_experiment_result"]
