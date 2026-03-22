from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateRateResult:
    """Rate-side result for one resolved scheduler candidate."""

    rate_ach_bps: float
    n_re_data: float
    n_re_raw: float
    n_pilot: float
    b_occ_hz: float


@dataclass(frozen=True)
class CandidatePowerResult:
    """Power-side feasibility and operating point for one resolved candidate."""

    is_feasible: bool
    infeasibility_reason: str
    gamma_req_lin: float | None = None
    gamma_req_db: float | None = None
    gamma_achieved: float | None = None
    rho_ach_raw_linear: float | None = None
    sigma_e2: float | None = None
    n_streams: int | None = None
    g_bf_linear: float | None = None
    ps_total_w: float | None = None
    p_sig_out_total_w: float | None = None
    p_out_total_w: float | None = None
    p_out_ant_w: float | None = None
    p_dc_avg_total_w: float | None = None
