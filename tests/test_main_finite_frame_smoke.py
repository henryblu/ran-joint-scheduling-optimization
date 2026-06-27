from __future__ import annotations

import main


def test_main_default_finite_frame_smoke(capsys):
    result = main.main([])
    captured = capsys.readouterr()

    assert result.status == "solved"
    assert result.scheduler_user_table.shape[0] == 15
    assert result.schedule_result.feasible
    assert result.schedule_result.scheduler_mode.value == "k_milp"
    assert result.schedule_result.solver_details["algorithm"] in {
        "ofdma_k1_tdma_highs_milp",
        "ofdma_slot_pattern_count_milp",
    }
    assert "FINITE_FRAME_RUN status=solved scheduler=k_milp" in captured.out
    assert "users=15 load=0.4 distance_m=250" in captured.out
