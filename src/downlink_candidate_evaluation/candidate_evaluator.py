from radio_core import Candidate
from radio_core.pa_models import average_pa_power

from .mcs_requirements import McsRequirementModel
from .models import DynamicCandidateEvaluation
from .resource_grid import ResourceGridModel
from .sinr_chain import SinrChainModel


class DownlinkCandidateEvaluator:
    """Solve the dynamic feasibility and power state for one static candidate.

    The evaluator assumes the candidate's static metrics were computed upstream.
    It only owns:
    1. discrete feasibility checks against the built deployment problem,
    2. the source-power solve for an explicit SINR target,
    3. PA and PSD validation,
    4. dynamic RF/DC term assembly for result consumers.
    """

    def __init__(self, mcs_table):
        self.mcs_table = dict(mcs_table)
        self.grid_model = ResourceGridModel(self.mcs_table)
        self.mcs_model = McsRequirementModel(self.mcs_table)
        self.sinr_model = SinrChainModel(self.grid_model, self.mcs_model)

    def evaluate_static_candidate(self, problem, static_candidate):
        """Evaluate one static candidate against a concrete deployment problem."""
        candidate = self._build_candidate(static_candidate)
        rrc = self._find_rrc(problem, candidate)
        ok, reason = self._evaluate_pre_solve_feasibility(problem, candidate, rrc)
        if not ok:
            return self._build_infeasible_result(static_candidate, problem, reason)

        sched = self.grid_model.build_scheduler_vars(candidate)
        pa = problem.pa_catalog[candidate.pa_id]
        ps_solution = self.sinr_model.solve_required_source_power_for_target(
            float(static_candidate.gamma_req_lin),
            problem.deployment,
            rrc,
            sched,
            pa,
        )
        if ps_solution is None:
            return self._build_infeasible_result(static_candidate, problem, "sinr_infeasible")

        rf_terms = self._compute_rf_terms(pa, sched, ps_solution)
        psd_w_per_hz = rf_terms["p_out_total_w"] / max(ps_solution["b_occ"], 1e-30)
        ok, reason = self._evaluate_post_solve_feasibility(
            problem,
            sched=sched,
            pa=pa,
            ps_solution=ps_solution,
            p_out_ant=rf_terms["p_out_ant_w"],
            p_out_total=rf_terms["p_out_total_w"],
            psd=psd_w_per_hz,
        )
        if not ok:
            return self._build_infeasible_result(static_candidate, problem, reason)

        dc_terms = self._compute_dc_terms(problem, pa, sched, rf_terms["p_out_ant_w"])
        return DynamicCandidateEvaluation(
            candidate_ordinal=int(static_candidate.candidate_ordinal),
            is_feasible=True,
            infeasibility_reason="ok",
            distance_m=float(problem.deployment.distance_m),
            path_loss_db=float(problem.deployment.path_loss_db),
            p_dc_avg_total_w=float(dc_terms["p_dc_avg_total_w"]),
            p_rf_out_active_w=float(rf_terms["p_out_total_w"]),
            p_out_total_w=float(rf_terms["p_out_total_w"]),
            p_sig_out_active_w=float(rf_terms["p_sig_out_total_w"]),
            p_sig_out_total_w=float(rf_terms["p_sig_out_total_w"]),
            ps_total_w=float(rf_terms["ps_total_w"]),
            gamma_achieved=float(ps_solution["rho_achieved_linear"]),
            rho_ach_raw_linear=float(ps_solution["rho_ach_raw_linear"]),
            n_streams=int(ps_solution["n_streams"]),
            g_bf_linear=float(ps_solution["g_bf_linear"]),
            sigma_e2=float(ps_solution["sigma_e2"]),
        )

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
        *,
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
        return problem.rrc_lookup.get((int(candidate.pa_id), int(candidate.bwp_idx)))

    @staticmethod
    def _build_candidate(static_candidate):
        """Build a scheduler candidate from the static candidate specification."""
        candidate = getattr(static_candidate, "candidate", None)
        if candidate is not None:
            return candidate
        return Candidate(
            pa_id=int(static_candidate.pa_id),
            bwp_idx=int(static_candidate.bwp_idx),
            n_prb=int(static_candidate.n_prb),
            n_slots_on=int(static_candidate.n_slots_on),
            layers=int(static_candidate.layers),
            n_active_tx=int(static_candidate.n_active_tx),
            mcs=int(static_candidate.mcs),
        )

    @staticmethod
    def _build_infeasible_result(static_candidate, problem, reason):
        """Build the standardized infeasible dynamic result."""
        return DynamicCandidateEvaluation(
            candidate_ordinal=int(static_candidate.candidate_ordinal),
            is_feasible=False,
            infeasibility_reason=str(reason),
            distance_m=float(problem.deployment.distance_m),
            path_loss_db=float(problem.deployment.path_loss_db),
        )

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
