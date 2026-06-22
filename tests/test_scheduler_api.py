from __future__ import annotations

import math

import pandas as pd

from candidate_table import build_batch_user_parameter_space
from models import MultiUserScheduleResult, PASwitchPolicy, SchedulerMode
from schedulers import run_scheduler
from schedulers.k_milp import run_k_milp_scheduler
from schedulers.round_robin import run_round_robin_scheduler


def _tiny_batch_space():
    user_table = pd.DataFrame(
        {
            "user_id": [1, 2],
            "distance_m": [25.0, 50.0],
            "required_rate_bps": [1.0e6, 1.5e6],
        }
    )
    return build_batch_user_parameter_space(user_table)


def _assert_scheduler_result_contract(
    result: MultiUserScheduleResult,
    mode: SchedulerMode,
    *,
    expected_slot_count: int,
) -> None:
    assert isinstance(result, MultiUserScheduleResult)
    assert result.scheduler_mode == mode
    assert isinstance(result.feasible, bool)
    assert len(result.slot_schedules) == int(expected_slot_count)
    assert tuple(slot.slot_index for slot in result.slot_schedules) == tuple(range(int(expected_slot_count)))
    assert math.isfinite(float(result.power_summary.frame_energy_j))
    assert math.isfinite(float(result.power_summary.average_frame_dc_power_w))
    assert result.user_summaries
    assert result.solver_details["scheduler_mode"] == mode.value


def test_scheduler_public_api_exports_final_names():
    import schedulers

    assert "run_scheduler" in schedulers.__all__
    assert SchedulerMode.ROUND_ROBIN.value == "round_robin"
    assert SchedulerMode.K_MILP.value == "k_milp"
    assert not hasattr(SchedulerMode, "TDMA")
    assert not hasattr(SchedulerMode, "OFDMA")


def test_round_robin_dispatch_contract():
    batch_space = _tiny_batch_space()

    direct = run_round_robin_scheduler(
        batch_space,
        switch_policy=PASwitchPolicy.DUAL_SWITCHABLE,
    )
    dispatched = run_scheduler(
        batch_space,
        scheduler_mode=SchedulerMode.ROUND_ROBIN,
        switch_policy=PASwitchPolicy.DUAL_SWITCHABLE,
    )

    _assert_scheduler_result_contract(direct, SchedulerMode.ROUND_ROBIN, expected_slot_count=batch_space.frame_n_slots)
    _assert_scheduler_result_contract(dispatched, SchedulerMode.ROUND_ROBIN, expected_slot_count=batch_space.frame_n_slots)
    assert dispatched.feasible == direct.feasible
    assert dispatched.power_summary.frame_energy_j == direct.power_summary.frame_energy_j


def test_k_milp_dispatch_contract():
    batch_space = _tiny_batch_space()

    direct = run_k_milp_scheduler(
        batch_space,
        switch_policy=PASwitchPolicy.DUAL_SWITCHABLE,
    )
    dispatched = run_scheduler(
        batch_space,
        scheduler_mode=SchedulerMode.K_MILP,
        switch_policy=PASwitchPolicy.DUAL_SWITCHABLE,
    )

    _assert_scheduler_result_contract(direct, SchedulerMode.K_MILP, expected_slot_count=batch_space.frame_n_slots)
    _assert_scheduler_result_contract(dispatched, SchedulerMode.K_MILP, expected_slot_count=batch_space.frame_n_slots)
    assert dispatched.feasible == direct.feasible
    assert dispatched.solver_details["algorithm"] in {
        "ofdma_k1_tdma_highs_milp",
        "ofdma_slot_pattern_count_milp",
    }