from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicCandidateEvaluation:
    """Dynamic feasibility and power result for one static candidate."""

    candidate_ordinal: int
    is_feasible: bool
    infeasibility_reason: str
    distance_m: float
    path_loss_db: float
    p_dc_avg_total_w: float | None = None
    p_rf_out_active_w: float | None = None
    p_out_total_w: float | None = None
    p_sig_out_active_w: float | None = None
    p_sig_out_total_w: float | None = None
    ps_total_w: float | None = None
    gamma_achieved: float | None = None
    rho_ach_raw_linear: float | None = None
    n_streams: int | None = None
    g_bf_linear: float | None = None
    sigma_e2: float | None = None
