from __future__ import annotations

import json

import pandas as pd
from pandas.testing import assert_frame_equal

import candidate_table
from candidate_table.artifact import CANDIDATE_FRONTIER_SORT_COLUMNS
from candidate_table.build import build_candidate_frontier_for_distance
from candidate_table_generation import (
    load_distance_binned_candidate_table,
    save_distance_binned_candidate_table,
)
from candidate_table_generation.builder import (
    build_candidate_frontier_for_distance as build_old_candidate_frontier_for_distance,
)
from models.candidate_table import BATCH_USER_PARAMETER_SPACE_COLUMNS
from single_user_lookup import build_batch_user_parameter_space as build_old_batch_user_parameter_space


def _canonical_frontier(frontier):
    return (
        frontier.reindex(columns=BATCH_USER_PARAMETER_SPACE_COLUMNS)
        .sort_values(CANDIDATE_FRONTIER_SORT_COLUMNS)
        .reset_index(drop=True)
    )


def test_new_loader_matches_old_candidate_table_loader():
    old_table = load_distance_binned_candidate_table("data/distance_binned_candidate_table.json")
    new_table = candidate_table.load_candidate_table("data/distance_binned_candidate_table.json")

    assert old_table.frontiers_by_distance_m.keys() == new_table.frontiers_by_distance_m.keys()
    for distance_m in old_table.frontiers_by_distance_m:
        assert_frame_equal(
            _canonical_frontier(old_table.frontiers_by_distance_m[distance_m]),
            _canonical_frontier(new_table.frontiers_by_distance_m[distance_m]),
        )


def test_new_save_round_trip_matches_old_semantic_payload(tmp_path):
    old_table = load_distance_binned_candidate_table("data/distance_binned_candidate_table.json")
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"

    save_distance_binned_candidate_table(old_table, old_path)
    candidate_table.save_candidate_table(old_table, new_path)

    assert json.loads(old_path.read_text(encoding="utf-8")) == json.loads(new_path.read_text(encoding="utf-8"))


def test_new_frontier_build_matches_old_frontier_build_for_one_distance():
    old_frontier = build_old_candidate_frontier_for_distance(25)
    new_frontier = build_candidate_frontier_for_distance(25)

    assert_frame_equal(
        _canonical_frontier(old_frontier),
        _canonical_frontier(new_frontier),
    )


def test_new_lookup_matches_old_lookup_for_tiny_user_table():
    user_table = pd.DataFrame(
        {
            "user_id": [1, 2],
            "distance_m": [25.0, 50.0],
            "required_rate_bps": [1.0e6, 2.0e6],
        }
    )

    old_batch = build_old_batch_user_parameter_space(user_table)
    new_batch = candidate_table.build_batch_user_parameter_space(user_table)

    assert_frame_equal(old_batch.user_requirements, new_batch.user_requirements)
    assert old_batch.frame_n_slots == new_batch.frame_n_slots
    assert old_batch.n_tx_chains == new_batch.n_tx_chains
    assert len(old_batch.pa_catalog) == len(new_batch.pa_catalog)
    assert [pa.p_max_w for pa in old_batch.pa_catalog] == [pa.p_max_w for pa in new_batch.pa_catalog]
    assert [pa.p_idle_w for pa in old_batch.pa_catalog] == [pa.p_idle_w for pa in new_batch.pa_catalog]
    assert [pa.eta_max for pa in old_batch.pa_catalog] == [pa.eta_max for pa in new_batch.pa_catalog]
    assert old_batch.user_parameter_spaces.keys() == new_batch.user_parameter_spaces.keys()
    for user_id in old_batch.user_parameter_spaces:
        assert_frame_equal(
            _canonical_frontier(old_batch.user_parameter_spaces[user_id]),
            _canonical_frontier(new_batch.user_parameter_spaces[user_id]),
        )
