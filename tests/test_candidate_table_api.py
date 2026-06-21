from __future__ import annotations

import pandas as pd

import candidate_table
from candidate_table import build_batch_user_parameter_space, load_candidate_table
from candidate_table.build import build_candidate_frontier_for_distance
from models import BatchUserParameterSpace
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS


def test_candidate_table_public_api_exports_final_names():
    expected_exports = {
        "build_batch_user_parameter_space",
        "build_candidate_frontier_for_distance",
        "build_candidate_table",
        "DISTANCE_BIN_GRID_M",
        "DistanceBinnedCandidateTable",
        "load_candidate_table",
        "load_or_build_candidate_table",
        "lookup_user_parameter_space",
        "save_candidate_table",
    }

    assert expected_exports <= set(candidate_table.__all__)
    assert "load_distance_binned_candidate_table" not in candidate_table.__all__
    assert "build_distance_binned_candidate_table" not in candidate_table.__all__


def test_load_candidate_table_restores_stored_frontier_contract():
    table = load_candidate_table("data/distance_binned_candidate_table.json")

    assert table.frontiers_by_distance_m
    assert min(table.frontiers_by_distance_m) == 25

    for frontier in table.frontiers_by_distance_m.values():
        assert list(frontier.columns) == BATCH_USER_PARAMETER_SPACE_COLUMNS
        assert not frontier.empty
        assert str(frontier["pa_id"].dtype) == "int64"
        assert str(frontier["n_prb"].dtype) == "int64"
        assert str(frontier["bits_per_slot"].dtype) == "float64"
        assert str(frontier["p_dc_active_w"].dtype) == "float64"


def test_build_batch_user_parameter_space_from_stored_table():
    user_table = pd.DataFrame(
        {
            "user_id": [1, 2],
            "distance_m": [25.0, 50.0],
            "required_rate_bps": [1.0e6, 2.0e6],
        }
    )

    batch = build_batch_user_parameter_space(user_table)

    assert isinstance(batch, BatchUserParameterSpace)
    assert list(batch.user_requirements.columns) == ["user_id", "required_rate_bps"]
    assert set(batch.user_parameter_spaces) == {1, 2}
    assert batch.frame_n_slots > 0
    assert batch.n_tx_chains > 0
    assert batch.pa_catalog
    for user_space in batch.user_parameter_spaces.values():
        assert list(user_space.columns) == BATCH_USER_PARAMETER_SPACE_COLUMNS
        assert not user_space.empty


def test_build_candidate_frontier_for_one_distance_bin():
    frontier = build_candidate_frontier_for_distance(25)

    assert list(frontier.columns) == BATCH_USER_PARAMETER_SPACE_COLUMNS
    assert not frontier.empty
