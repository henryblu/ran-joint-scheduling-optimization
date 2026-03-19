from itertools import product

import numpy as np
import pandas as pd

from radio_core.model_types import DeploymentParams, ModelOptions, Problem
from radio_core.pa_models import average_pa_power, build_pa_catalog

from .mcs_requirements import McsRequirementModel
from .path_loss_models import PathLossModel
from .resource_grid import ResourceGridModel
from .sinr_chain import SinrChainModel


class SingleUserResourceAllocationEngine:
    """Reusable single-user feasible-configuration engine."""

    def __init__(
        self,
        link_constants,
        phy_constants,
        scheduler_sweep,
        mcs_table,
        pa_catalog=None,
        pa_data_csv=None,
    ):
        self.link_constants = dict(link_constants)
        self.phy_constants = dict(phy_constants)
        self.scheduler_sweep = dict(scheduler_sweep)
        self.mcs_table = dict(mcs_table)
        self._pa_catalog = pa_catalog
        self.pa_data_csv = pa_data_csv
        self._candidate_table_cache = {}

        path_loss_model = self.link_constants.get("pl_model", "fspl")
        shadow_margin_db = self.link_constants.get("shadow_margin_db", 0.0)
        self.path_loss_model = PathLossModel(
            fc_hz=self.link_constants["fc_hz"],
            model=path_loss_model,
            g_tx_db=self.link_constants["g_tx_db"],
            g_rx_db=self.link_constants["g_rx_db"],
            shadow_margin_db=shadow_margin_db,
        )
        self.grid_model = ResourceGridModel(self.mcs_table)
        self.mcs_model = McsRequirementModel(self.mcs_table)
        self.sinr_model = SinrChainModel(self.grid_model, self.mcs_model)

    @staticmethod
    def _cache_key(problem, include_infeasible=False, required_rate_bps=None):
        rate_key = None if required_rate_bps is None else float(required_rate_bps)
        return (id(problem), bool(include_infeasible), rate_key)

    def clear_candidate_cache(self):
        """Clear cached candidate tables held by this engine instance."""
        self._candidate_table_cache.clear()

    def _resolve_pa_catalog(self, csv_path=None):
        if self._pa_catalog is not None:
            return self._pa_catalog
        resolved_csv = csv_path or self.pa_data_csv
        if not resolved_csv:
            raise ValueError("PA catalog not provided and pa_data_csv is not set.")
        self._pa_catalog = build_pa_catalog(resolved_csv)
        return self._pa_catalog

    @staticmethod
    def iter_scenarios(sweep_settings):
        """Expand a scenario sweep dictionary into explicit scenario rows."""
        keys = list(sweep_settings.keys())
        value_spaces = []
        for key in keys:
            values = sweep_settings[key]
            if np.isscalar(values):
                values = [values]
            value_spaces.append([float(v) for v in values])
        return [dict(zip(keys, values)) for values in product(*value_spaces)]

    def _build_deployment(self, distance_m, path_loss_db=None):
        if path_loss_db is None:
            path_loss_db = self.path_loss_model.effective_path_loss_db(distance_m)
        return DeploymentParams(
            fc_hz=self.link_constants["fc_hz"],
            channel_bw_hz=self.phy_constants["channel_bw_hz"],
            distance_m=float(distance_m),
            path_loss_db=float(path_loss_db),
            g_tx_db=self.link_constants["g_tx_db"],
            g_rx_db=self.link_constants["g_rx_db"],
            n0_dbm_per_hz=self.link_constants["n0_dbm_per_hz"],
            lna_noise_figure_db=self.link_constants["lna_noise_figure_db"],
            l_impl_db=self.phy_constants["l_impl_db"],
            mi_n_samples=self.phy_constants["mi_n_samples"],
            n_dmrs_sym=self.phy_constants["n_dmrs_sym"],
            dft_size_N=self.phy_constants["dft_size_N"],
            n_slots_win=self.phy_constants["n_slots_win"],
            t_slot_s=self.phy_constants["t_slot_s"],
            n_sym_data=self.phy_constants["n_sym_data"],
            n_sym_total=self.phy_constants["n_sym_total"],
            use_psd_constraint=self.phy_constants["use_psd_constraint"],
            psd_max_w_per_hz=self.phy_constants["psd_max_w_per_hz"],
            papr_db=self.phy_constants["papr_db"],
            g_phi=self.phy_constants["g_phi"],
            sigma_phi2=self.phy_constants["sigma_phi2"],
            sigma_q2=self.phy_constants["sigma_q2"],
            n_tx_chains=self.phy_constants["n_tx_chains"],
        )

    def build_single_user_problem(
        self,
        distance_m,
        fast_mode=False,
        prb_step=None,
        bandwidth_space=None,
        n_slots_on_space=None,
        csv_path=None,
        path_loss_db=None,
    ):
        """Build a complete Problem object from one explicit distance value."""
        deployment = self._build_deployment(distance_m=distance_m, path_loss_db=path_loss_db)
        pa_catalog = self._resolve_pa_catalog(csv_path=csv_path)
        bandwidth_space_source = (
            bandwidth_space if bandwidth_space is not None else self.scheduler_sweep["bandwidth_space_hz"]
        )
        if np.isscalar(bandwidth_space_source):
            bandwidth_space_hz = [float(bandwidth_space_source)]
        else:
            bandwidth_space_hz = [float(v) for v in bandwidth_space_source]

        rrc_catalog = self.grid_model.build_bwp_catalog(
            deployment=deployment,
            pa_catalog=pa_catalog,
            bandwidth_space=bandwidth_space_hz,
            delta_f_hz_default=self.phy_constants["delta_f_hz"],
        )
        search_space = self.grid_model.build_search_space(
            deployment=deployment,
            sweep_settings=self.scheduler_sweep,
            prb_step=prb_step,
            n_slots_on_space=n_slots_on_space,
        )
        options = ModelOptions(fast_mode=bool(fast_mode))
        return Problem(
            deployment=deployment,
            pa_catalog=pa_catalog,
            rrc_catalog=rrc_catalog,
            search_space=search_space,
            options=options,
        )

    def build_problem_from_deployment(
        self,
        deployment,
        fast_mode=False,
        prb_step=None,
        bandwidth_space=None,
        n_slots_on_space=None,
        csv_path=None,
    ):
        """Build a Problem object when the notebook provides the deployment explicitly."""
        pa_catalog = self._resolve_pa_catalog(csv_path=csv_path)
        bandwidth_space_source = (
            bandwidth_space if bandwidth_space is not None else self.scheduler_sweep["bandwidth_space_hz"]
        )
        if np.isscalar(bandwidth_space_source):
            bandwidth_space_hz = [float(bandwidth_space_source)]
        else:
            bandwidth_space_hz = [float(v) for v in bandwidth_space_source]

        rrc_catalog = self.grid_model.build_bwp_catalog(
            deployment=deployment,
            pa_catalog=pa_catalog,
            bandwidth_space=bandwidth_space_hz,
            delta_f_hz_default=self.phy_constants["delta_f_hz"],
        )
        search_space = self.grid_model.build_search_space(
            deployment=deployment,
            sweep_settings=self.scheduler_sweep,
            prb_step=prb_step,
            n_slots_on_space=n_slots_on_space,
        )
        return Problem(
            deployment=deployment,
            pa_catalog=pa_catalog,
            rrc_catalog=rrc_catalog,
            search_space=search_space,
            options=ModelOptions(fast_mode=bool(fast_mode)),
        )

    def describe_problem(self, problem):
        """Return compact reviewer-facing summary tables for one problem."""
        deployment_df = pd.DataFrame(
            [
                {
                    "distance_m": float(problem.deployment.distance_m),
                    "path_loss_db": float(problem.deployment.path_loss_db),
                    "fc_hz": float(problem.deployment.fc_hz),
                    "n_tx_chains": int(problem.deployment.n_tx_chains),
                    "n_slots_win": int(problem.deployment.n_slots_win),
                    "delta_f_hz": float(problem.rrc_catalog[0].delta_f_hz) if problem.rrc_catalog else np.nan,
                }
            ]
        )
        rrc_df = pd.DataFrame(
            [
                {
                    "pa_id": int(rrc.active_pa_id),
                    "bwp_idx": int(rrc.bwp_index),
                    "bandwidth_hz": float(rrc.bwp_bw_hz),
                    "prb_max_bwp": int(rrc.prb_max_bwp),
                    "max_layers": int(rrc.max_layers),
                    "max_mcs": int(rrc.max_mcs),
                }
                for rrc in problem.rrc_catalog
            ]
        ).sort_values(["pa_id", "bandwidth_hz"]).reset_index(drop=True)
        search_space_df = pd.DataFrame(
            [
                {
                    "n_slots_on_values": len(problem.search_space.n_slots_on_space),
                    "layers_values": len(problem.search_space.layers_space),
                    "n_active_tx_values": len(problem.search_space.n_active_tx_space),
                    "mcs_values": len(problem.search_space.mcs_space),
                    "prb_step": int(problem.search_space.prb_step),
                }
            ]
        )
        return {
            "deployment_summary": deployment_df,
            "rrc_catalog": rrc_df,
            "search_space_summary": search_space_df,
        }

    def estimate_search_space(self, problem, scenarios):
        """Return a compact raw search-space summary before evaluating candidates."""
        scenario_count = len(scenarios)
        pa_count = len(problem.pa_catalog)
        per_pa_counts = []
        for pa_id in range(pa_count):
            rrc_space = [rrc for rrc in problem.rrc_catalog if rrc.active_pa_id == pa_id]
            per_pa_counts.append(sum(self.grid_model.count_candidates_for_rrc(problem, rrc) for rrc in rrc_space))
        raw_configs_per_scenario = sum(per_pa_counts)
        return {
            "pa_count": pa_count,
            "scenario_count": scenario_count,
            "raw_configs_per_pa_per_scenario": int(per_pa_counts[0]) if per_pa_counts else 0,
            "raw_configs_per_scenario": int(raw_configs_per_scenario),
            "raw_total_configs": int(raw_configs_per_scenario * scenario_count),
        }

    def enumerate_scheduler_candidates(self, problem):
        return self.grid_model.enumerate_scheduler_candidates(problem)

    @staticmethod
    def _find_rrc(problem, candidate):
        return next(
            (
                rrc
                for rrc in problem.rrc_catalog
                if rrc.active_pa_id == candidate.pa_id and rrc.bwp_index == candidate.bwp_idx
            ),
            None,
        )

    @staticmethod
    def _build_infeasible_row(candidate, reason, problem=None, rrc=None, pa=None, rate_ach_bps=None):
        alpha_f = np.nan
        bandwidth_hz = np.nan
        if rrc is not None:
            alpha_f = float(candidate.n_prb / max(rrc.prb_max_bwp, 1))
            bandwidth_hz = float(rrc.bwp_bw_hz)
        row = {
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
        }
        if rate_ach_bps is not None:
            row["rate_ach_bps"] = float(rate_ach_bps)
        else:
            row["rate_ach_bps"] = np.nan
        return row

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
        """Evaluate structural feasibility first, then physical feasibility after the solve."""
        deployment = problem.deployment
        if rrc is None:
            return False, "rrc_not_found"
        if candidate.layers < 1 or candidate.layers > rrc.max_layers:
            return False, "invalid_layer_count"
        if candidate.n_active_tx <= 0 or candidate.n_active_tx < candidate.layers or candidate.n_active_tx > deployment.n_tx_chains:
            return False, "invalid_active_tx_count"
        if candidate.mcs > rrc.max_mcs:
            return False, "invalid_mcs"
        if not (1 <= candidate.n_slots_on <= deployment.n_slots_win):
            return False, "invalid_slot_count"
        if candidate.n_prb > rrc.prb_max_bwp:
            return False, "insufficient_res"
        if stage == "pre_solve":
            return True, "ok"
        if stage != "post_solve":
            raise ValueError(f"Unknown feasibility stage: {stage}")
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

    def _compute_rf_terms(self, pa, sched, ps_solution):
        """Compute RF powers from solved source power."""
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

    def evaluate_candidate(self, problem, candidate, include_infeasible=False, required_rate_bps=None):
        """Evaluate one candidate in the exact model order used by the notebook."""
        deployment = problem.deployment
        rrc = self._find_rrc(problem, candidate)
        ok, reason = self.evaluate_feasibility(problem, candidate, rrc=rrc, stage="pre_solve")
        if not ok:
            if include_infeasible:
                return self._build_infeasible_row(candidate, reason, problem=problem, rrc=rrc)
            return None

        sched = self.grid_model.build_scheduler_vars(candidate)
        pa = problem.pa_catalog[candidate.pa_id]
        rate_ach_bps = self.grid_model.compute_rate(deployment, rrc, sched)
        if required_rate_bps is not None and rate_ach_bps < float(required_rate_bps):
            if include_infeasible:
                return self._build_infeasible_row(
                    candidate,
                    "below_rate_target",
                    problem=problem,
                    rrc=rrc,
                    pa=pa,
                    rate_ach_bps=rate_ach_bps,
                )
            return None

        ps_solution = self.sinr_model.solve_required_source_power(deployment, rrc, sched, pa)
        if ps_solution is None:
            if include_infeasible:
                return self._build_infeasible_row(
                    candidate,
                    "sinr_infeasible",
                    problem=problem,
                    rrc=rrc,
                    pa=pa,
                    rate_ach_bps=rate_ach_bps,
                )
            return None

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
            if include_infeasible:
                return self._build_infeasible_row(
                    candidate,
                    reason,
                    problem=problem,
                    rrc=rrc,
                    pa=pa,
                    rate_ach_bps=rate_ach_bps,
                )
            return None

        dc_terms = self._compute_dc_terms(problem, pa, sched, rf_terms["p_out_ant_w"])
        row = self._assemble_candidate_result(problem, candidate, rrc, sched, pa, ps_solution, rf_terms, dc_terms, rate_ach_bps)
        if include_infeasible:
            row["is_feasible"] = True
            row["infeasibility_reason"] = "ok"
        return row

    def evaluate_candidate_table(self, problem, include_infeasible=False, required_rate_bps=None, use_cache=True):
        """Enumerate and evaluate all candidate configurations.

        If required_rate_bps is provided, candidates that cannot meet the target
        are pruned immediately after rate evaluation and do not enter the
        SINR/RF/PA power solve chain.
        """
        cache_key = self._cache_key(
            problem,
            include_infeasible=include_infeasible,
            required_rate_bps=required_rate_bps,
        )
        if use_cache and cache_key in self._candidate_table_cache:
            return self._candidate_table_cache[cache_key].copy()

        rows = []
        for candidate in self.enumerate_scheduler_candidates(problem):
            metrics = self.evaluate_candidate(
                problem,
                candidate,
                include_infeasible=include_infeasible,
                required_rate_bps=required_rate_bps,
            )
            if metrics is not None:
                rows.append(metrics)
        candidate_table = pd.DataFrame(rows)
        if use_cache:
            self._candidate_table_cache[cache_key] = candidate_table.copy()
        return candidate_table

    @staticmethod
    def filter_feasible_configurations(candidate_table, required_rate_bps=None):
        """Filter the candidate table down to feasible rows and optional rate threshold."""
        df = candidate_table.copy()
        if "is_feasible" in df.columns:
            df = df[df["is_feasible"]].copy()
        if required_rate_bps is not None and "rate_ach_bps" in df.columns:
            df = df[df["rate_ach_bps"] >= float(required_rate_bps)].copy()
        return df.reset_index(drop=True)
