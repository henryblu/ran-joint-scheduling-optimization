import numpy as np

from radio_core.pa_models import average_pa_power

from .mcs_requirements import McsRequirementModel
from .resource_grid import ResourceGridModel
from .sinr_chain import SinrChainModel


class DownlinkCandidateEvaluator:
    """Evaluate one downlink candidate against a fully-built problem.

    The class is organized around two public entry points:
    - `evaluate_candidate` runs the full end-to-end evaluation pipeline.
    - `evaluate_feasibility` exposes the staged feasibility checks used inside that pipeline.
    """

    def __init__(self, mcs_table):
        self.mcs_table = dict(mcs_table)
        self.grid_model = ResourceGridModel(self.mcs_table)
        self.mcs_model = McsRequirementModel(self.mcs_table)
        self.sinr_model = SinrChainModel(self.grid_model, self.mcs_model)

    def evaluate_candidate(self, problem, candidate, include_infeasible=False):
        """Run the full candidate-evaluation pipeline and return one result row.

        Steps:
        1. Resolve the candidate's RRC envelope and reject structurally invalid choices.
        2. Build scheduler variables and compute the rate implied by the candidate.
        3. Solve the minimum required source power for the requested MCS.
        4. Validate the solved powers against PA and PSD limits.
        5. Assemble the compact row consumed by the search and reporting layers.
        """
        deployment = problem.deployment
        rrc = self._find_rrc(problem, candidate)
        ok, reason = self.evaluate_feasibility(problem, candidate, rrc=rrc, stage="pre_solve")
        if not ok:
            return self._maybe_return_infeasible(
                include_infeasible,
                candidate,
                reason,
                problem=problem,
                rrc=rrc,
            )

        sched = self.grid_model.build_scheduler_vars(candidate)
        pa = problem.pa_catalog[candidate.pa_id]
        rate_ach_bps = self.grid_model.compute_rate(deployment, rrc, sched)

        ps_solution = self.sinr_model.solve_required_source_power(deployment, rrc, sched, pa)
        if ps_solution is None:
            return self._maybe_return_infeasible(
                include_infeasible,
                candidate,
                "sinr_infeasible",
                problem=problem,
                rrc=rrc,
                pa=pa,
                rate_ach_bps=rate_ach_bps,
            )

        rf_terms = self._compute_rf_terms(pa, sched, ps_solution)
        psd_w_per_hz = rf_terms["p_out_total_w"] / max(ps_solution["b_occ"], 1e-30)
        ok, reason = self.evaluate_feasibility(
            problem,
            candidate,
            rrc=rrc,
            sched=sched,
            pa=pa,
            ps_solution=ps_solution,
            p_out_ant=rf_terms["p_out_ant_w"],
            p_out_total=rf_terms["p_out_total_w"],
            psd=psd_w_per_hz,
            stage="post_solve",
        )
        if not ok:
            return self._maybe_return_infeasible(
                include_infeasible,
                candidate,
                reason,
                problem=problem,
                rrc=rrc,
                pa=pa,
                rate_ach_bps=rate_ach_bps,
            )

        dc_terms = self._compute_dc_terms(problem, pa, sched, rf_terms["p_out_ant_w"])
        row = self._assemble_candidate_result(
            problem,
            candidate,
            rrc,
            sched,
            pa,
            ps_solution,
            rf_terms,
            dc_terms,
            rate_ach_bps,
        )
        return self._mark_feasible_row(row, include_infeasible)

    @staticmethod
    def evaluate_feasibility(
        problem,
        candidate,
        rrc=None,
        sched=None,
        pa=None,
        ps_solution=None,
        p_out_ant=None,
        p_out_total=None,
        psd=None,
        stage="pre_solve",
    ):
        """Evaluate feasibility at a named stage of the pipeline.

        `pre_solve` only checks discrete candidate validity against the search envelope.
        `post_solve` assumes the source-power solve has run and checks physical limits.
        """
        if stage == "pre_solve":
            return DownlinkCandidateEvaluator._evaluate_pre_solve_feasibility(problem, candidate, rrc)
        if stage == "post_solve":
            return DownlinkCandidateEvaluator._evaluate_post_solve_feasibility(
                problem,
                candidate,
                sched=sched,
                pa=pa,
                ps_solution=ps_solution,
                p_out_ant=p_out_ant,
                p_out_total=p_out_total,
                psd=psd,
            )
        raise ValueError(f"Unknown feasibility stage: {stage}")

    @staticmethod
    def _evaluate_pre_solve_feasibility(problem, candidate, rrc):
        """Reject candidates that do not fit the discrete deployment and RRC limits."""
        deployment = problem.deployment
        if rrc is None:
            return False, "rrc_not_found"
        if not 1 <= candidate.layers <= rrc.max_layers:
            return False, "invalid_layer_count"
        if not candidate.layers <= candidate.n_active_tx <= deployment.n_tx_chains:
            return False, "invalid_active_tx_count"
        if candidate.mcs > rrc.max_mcs:
            return False, "invalid_mcs"
        if not 1 <= candidate.n_slots_on <= deployment.n_slots_win:
            return False, "invalid_slot_count"
        if candidate.n_prb > rrc.prb_max_bwp:
            return False, "insufficient_res"
        return True, "ok"

    @staticmethod
    def _evaluate_post_solve_feasibility(
        problem,
        _candidate,
        sched=None,
        pa=None,
        ps_solution=None,
        p_out_ant=None,
        p_out_total=None,
        psd=None,
    ):
        """Reject solved candidates that violate PHY, PA, or PSD constraints."""
        deployment = problem.deployment
        if ps_solution is None:
            return False, "sinr_infeasible"
        if p_out_total is not None and p_out_total < 0:
            return False, "nonphysical_negative_power"
        if p_out_ant is not None and pa is not None:
            if p_out_ant > pa.p_max_w:
                return False, "per_chain_pa_cap"
            if pa.curve_pout_w is not None and len(pa.curve_pout_w) and p_out_ant > float(pa.curve_pout_w[-1]):
                return False, "interpolation_out_of_range"
        if p_out_total is not None and sched is not None and pa is not None:
            if p_out_total > sched.n_active_tx * pa.p_max_w:
                return False, "total_pa_cap"
        if psd is not None and deployment.use_psd_constraint and psd > deployment.psd_max_w_per_hz:
            return False, "psd_violation"
        return True, "ok"

    @staticmethod
    def _find_rrc(problem, candidate):
        """Return the RRC envelope referenced by a candidate, if it exists."""
        return next(
            (
                rrc
                for rrc in problem.rrc_catalog
                if rrc.active_pa_id == candidate.pa_id and rrc.bwp_index == candidate.bwp_idx
            ),
            None,
        )

    @staticmethod
    def _maybe_return_infeasible(include_infeasible, candidate, reason, **context):
        """Return a populated infeasible row when requested, otherwise suppress it."""
        if not include_infeasible:
            return None
        return DownlinkCandidateEvaluator._build_infeasible_row(candidate, reason, **context)

    @staticmethod
    def _mark_feasible_row(row, include_infeasible):
        """Attach feasibility bookkeeping only when the caller requested a full ledger."""
        if include_infeasible:
            row["is_feasible"] = True
            row["infeasibility_reason"] = "ok"
        return row

    @staticmethod
    def _build_infeasible_row(candidate, reason, problem=None, rrc=None, pa=None, rate_ach_bps=None):
        """Build the standardized placeholder row for rejected candidates."""
        alpha_f = np.nan
        bandwidth_hz = np.nan
        if rrc is not None:
            alpha_f = float(candidate.n_prb / max(rrc.prb_max_bwp, 1))
            bandwidth_hz = float(rrc.bwp_bw_hz)
        return {
            "distance_m": float(problem.deployment.distance_m) if problem is not None else np.nan,
            "path_loss_db": float(problem.deployment.path_loss_db) if problem is not None else np.nan,
            "is_feasible": False,
            "infeasibility_reason": reason,
            "pa_id": int(candidate.pa_id),
            "pa_name": getattr(pa, "pa_name", ""),
            "bwp_idx": int(candidate.bwp_idx),
            "bandwidth_hz": bandwidth_hz,
            "n_prb": int(candidate.n_prb),
            "n_slots_on": int(candidate.n_slots_on),
            "layers": int(candidate.layers),
            "n_active_tx": int(candidate.n_active_tx),
            "mcs": int(candidate.mcs),
            "alpha_f": alpha_f,
            "alpha_t": np.nan,
            "p_dc_avg_total_w": np.nan,
            "p_rf_out_active_w": np.nan,
            "p_out_total_w": np.nan,
            "p_sig_out_active_w": np.nan,
            "p_sig_out_total_w": np.nan,
            "ps_total_w": np.nan,
            "gamma_req_lin": np.nan,
            "gamma_achieved": np.nan,
            "rho_ach_raw_linear": np.nan,
            "n_streams": np.nan,
            "g_bf_linear": np.nan,
            "sigma_e2": np.nan,
            "rate_ach_bps": np.nan if rate_ach_bps is None else float(rate_ach_bps),
        }

    def _compute_rf_terms(self, pa, sched, ps_solution):
        """Translate the solved source-power point into RF output powers."""
        n_streams = self.grid_model.get_n_streams(sched)
        ps_total_w = ps_solution["ps_min_w"]
        ps_stream_w = ps_total_w / n_streams
        p_dist_stream_w = self.sinr_model.sigma_z2(pa, ps_stream_w, sched)
        p_sig_out_stream_w = pa.g_pa_eff_linear * ps_stream_w
        p_sig_out_total_w = n_streams * p_sig_out_stream_w
        p_dist_total_w = n_streams * p_dist_stream_w
        p_out_total_w = p_sig_out_total_w + p_dist_total_w
        p_out_ant_w = p_out_total_w / sched.n_active_tx
        return {
            "ps_total_w": ps_total_w,
            "p_sig_out_active_w": p_sig_out_total_w,
            "p_sig_out_total_w": p_sig_out_total_w,
            "p_rf_out_active_w": p_out_total_w,
            "p_dist_total_w": p_dist_total_w,
            "p_out_total_w": p_out_total_w,
            "p_out_ant_w": p_out_ant_w,
        }

    def _compute_dc_terms(self, problem, pa, sched, p_out_ant_w):
        """Compute average PA DC draw including idle chains."""
        alpha_t = self.grid_model.scheduler_duty_cycle(problem.deployment, sched)
        p_dc_avg_total_w = (
            sched.n_active_tx * average_pa_power(pa, p_out_ant_w, alpha_t)
            + (problem.deployment.n_tx_chains - sched.n_active_tx) * pa.p_idle_w
        )
        return {"alpha_t": alpha_t, "p_dc_avg_total_w": p_dc_avg_total_w}

    @staticmethod
    def _assemble_candidate_result(problem, candidate, rrc, sched, pa, ps_solution, rf_terms, dc_terms, rate_ach_bps):
        """Assemble the compact row used by optimization and explanation tables."""
        alpha_f = sched.n_prb / max(rrc.prb_max_bwp, 1)
        return {
            "distance_m": float(problem.deployment.distance_m),
            "path_loss_db": float(problem.deployment.path_loss_db),
            "pa_id": int(candidate.pa_id),
            "pa_name": pa.pa_name,
            "bwp_idx": int(candidate.bwp_idx),
            "rate_ach_bps": float(rate_ach_bps),
            "p_dc_avg_total_w": float(dc_terms["p_dc_avg_total_w"]),
            "layers": int(sched.layers),
            "mcs": int(sched.mcs),
            "n_prb": int(sched.n_prb),
            "n_slots_on": int(sched.n_slots_on),
            "alpha_f": float(alpha_f),
            "alpha_t": float(dc_terms["alpha_t"]),
            "bandwidth_hz": float(rrc.bwp_bw_hz),
            "n_active_tx": int(sched.n_active_tx),
            "p_rf_out_active_w": float(rf_terms["p_out_total_w"]),
            "p_out_total_w": float(rf_terms["p_out_total_w"]),
            "p_sig_out_active_w": float(rf_terms["p_sig_out_total_w"]),
            "p_sig_out_total_w": float(rf_terms["p_sig_out_total_w"]),
            "ps_total_w": float(rf_terms["ps_total_w"]),
            "gamma_req_lin": float(ps_solution["rho_req_linear"]),
            "gamma_achieved": float(ps_solution["rho_achieved_linear"]),
            "rho_ach_raw_linear": float(ps_solution["rho_ach_raw_linear"]),
            "n_streams": int(ps_solution["n_streams"]),
            "g_bf_linear": float(ps_solution["g_bf_linear"]),
            "sigma_e2": float(ps_solution["sigma_e2"]),
        }
