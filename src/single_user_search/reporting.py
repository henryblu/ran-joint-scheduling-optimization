from itertools import islice

import numpy as np
import pandas as pd

from downlink_candidate_evaluation import DownlinkProblemSpace


def summarize_single_user_problem(problem):
    """Return notebook-friendly tables that explain one built single-user problem."""
    built_problem = problem.problem
    deployment_summary = pd.DataFrame(
        [
            {
                "distance_m": float(built_problem.deployment.distance_m),
                "path_loss_db": float(built_problem.deployment.path_loss_db),
                "fc_hz": float(built_problem.deployment.fc_hz),
                "n_tx_chains": int(built_problem.deployment.n_tx_chains),
                "n_slots_win": int(built_problem.deployment.n_slots_win),
                "delta_f_hz": float(built_problem.rrc_catalog[0].delta_f_hz) if built_problem.rrc_catalog else np.nan,
            }
        ]
    )
    rrc_catalog = pd.DataFrame(
        [
            {
                "pa_id": int(rrc.active_pa_id),
                "bwp_idx": int(rrc.bwp_index),
                "bandwidth_hz": float(rrc.bwp_bw_hz),
                "prb_max_bwp": int(rrc.prb_max_bwp),
                "max_layers": int(rrc.max_layers),
                "max_mcs": int(rrc.max_mcs),
            }
            for rrc in built_problem.rrc_catalog
        ]
    ).sort_values(["pa_id", "bandwidth_hz"]).reset_index(drop=True)
    search_space_summary = pd.DataFrame(
        [
            DownlinkProblemSpace(problem.model_inputs["mcs_table"]).estimate_search_space(
                built_problem,
                scenario_count=1,
            )
        ]
    )
    return {
        "deployment_summary": deployment_summary,
        "rrc_catalog": rrc_catalog,
        "search_space_summary": search_space_summary,
    }


def preview_single_user_candidates(problem, limit=5):
    """Return the first few candidates from the built traversal order."""
    candidate_space = DownlinkProblemSpace(problem.model_inputs["mcs_table"])
    preview_rows = [
        candidate.__dict__
        for candidate in islice(candidate_space.iter_candidates(problem.problem), limit)
    ]
    return pd.DataFrame(preview_rows)
