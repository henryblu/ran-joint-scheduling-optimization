import os
from enum import Enum

import numpy as np
import pandas as pd

from .model_types import PAParams
from .unit_conversions import dbm_to_w


class PAState(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    OFF = "off"


class PASwitchPolicy(Enum):
    STANDBY = "standby"
    HARD_OFF = "hard_off"


def _build_measured_pa_from_curves(pa_name, kappa_distortion, pin_dbm, pout_w, pdcin_w, source_tag):
    """Build PA model from measured Pin/Pout/PDCIN samples."""
    pin_dbm = np.asarray(pin_dbm, dtype=float)
    pout_w = np.asarray(pout_w, dtype=float)
    pdcin_w = np.asarray(pdcin_w, dtype=float)

    valid = np.isfinite(pin_dbm) & np.isfinite(pout_w) & np.isfinite(pdcin_w)
    pin_dbm = pin_dbm[valid]
    pout_w = pout_w[valid]
    pdcin_w = pdcin_w[valid]

    if len(pin_dbm) < 3:
        raise ValueError(f"Insufficient PA samples for {pa_name}")

    order = np.argsort(pin_dbm)
    pin_dbm = pin_dbm[order]
    pout_w = pout_w[order]
    pdcin_w = pdcin_w[order]

    pin_unique, idx = np.unique(pin_dbm, return_index=True)
    pout_unique_by_pin = pout_w[idx]
    pdc_unique_by_pin = pdcin_w[idx]

    n_grid = max(len(pin_unique), 64)
    pin_grid_dbm = np.linspace(float(pin_unique.min()), float(pin_unique.max()), n_grid)
    pin_grid_w = dbm_to_w(pin_grid_dbm)
    pout_grid_w = np.interp(pin_grid_dbm, pin_unique, pout_unique_by_pin)
    pdc_grid_w = np.interp(pin_grid_dbm, pin_unique, pdc_unique_by_pin)

    eta_samples = np.clip(pout_grid_w / np.clip(pdc_grid_w, 1e-12, None), 1e-4, 1.0)
    p_max_w = float(np.max(pout_grid_w))
    p_idle_w = float(np.min(pdc_grid_w))
    eta_max = float(np.max(eta_samples))

    n_gain = max(3, int(0.3 * len(pin_grid_w)))
    g_pa_eff_linear = float(np.median(pout_grid_w[:n_gain] / np.clip(pin_grid_w[:n_gain], 1e-12, None)))
    g_pa_eff_linear = max(g_pa_eff_linear, 1e-6)

    order_out = np.argsort(pout_grid_w)
    pout_sorted = pout_grid_w[order_out]
    pdc_sorted = pdc_grid_w[order_out]
    pout_unique, idx = np.unique(pout_sorted, return_index=True)
    pdc_unique = pdc_sorted[idx]

    return PAParams(
        p_max_w=p_max_w,
        p_idle_w=p_idle_w,
        eta_max=eta_max,
        g_pa_eff_linear=g_pa_eff_linear,
        kappa_distortion=kappa_distortion,
        backoff_db=6.0,
        pa_name=pa_name,
        curve_pout_w=pout_unique,
        curve_pdc_w=pdc_unique,
        curve_pin_w=pin_grid_w,
        source_csv=source_tag,
    )


def build_pa_catalog(csv_path):
    """Load measured PA profiles from CSV using measured PDCIN as the DC input."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing PA CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"pa_name", "Pin_dBm", "Pout_W", "PDCIN_W"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    pa_name_alias = {"8W": "8W PA (3.5GHz)", "4W": "4W PA (3.5GHz)"}
    pa_models = []
    for pa_name in sorted(df["pa_name"].dropna().unique()):
        sel = df[df["pa_name"] == pa_name].copy().sort_values("Pin_dBm")
        p_out = sel["Pout_W"].to_numpy(dtype=float)
        p_dc = sel["PDCIN_W"].to_numpy(dtype=float)

        kappa_guess = 0.03
        if len(p_out) >= 3 and np.nanmax(p_out) > 0:
            eta_obs = np.clip(p_out / np.clip(p_dc, 1e-12, None), 1e-4, 1.0)
            spread = float(np.nanmax(eta_obs) - np.nanmin(eta_obs))
            kappa_guess = float(np.clip(0.02 + 0.5 * spread, 0.01, 0.08))

        display_name = pa_name_alias.get(str(pa_name), str(pa_name))
        pa_models.append(
            _build_measured_pa_from_curves(
                pa_name=display_name,
                kappa_distortion=kappa_guess,
                pin_dbm=sel["Pin_dBm"].to_numpy(dtype=float),
                pout_w=sel["Pout_W"].to_numpy(dtype=float),
                pdcin_w=sel["PDCIN_W"].to_numpy(dtype=float),
                source_tag=str(csv_path),
            )
        )

    return sorted(pa_models, key=lambda pa: (-float(pa.p_max_w), str(pa.pa_name)))


def build_pa_characteristics_table(pa_catalog_or_problem):
    """Create a compact descriptive table for the PA catalog used in the optimization."""
    pa_catalog = getattr(pa_catalog_or_problem, "pa_catalog", pa_catalog_or_problem)
    rows = []
    for pa_id, pa in enumerate(pa_catalog):
        rows.append(
            {
                "pa_id": int(pa_id),
                "pa_name": pa.pa_name,
                "source_csv": getattr(pa, "source_csv", ""),
                "n_curve_points": int(len(np.asarray(getattr(pa, "curve_pout_w", []), dtype=float))),
                "p_max_w": float(pa.p_max_w),
                "p_idle_w": float(pa.p_idle_w),
                "eta_max": float(pa.eta_max),
                "g_pa_eff_linear": float(pa.g_pa_eff_linear),
                "g_pa_eff_db": float(10.0 * np.log10(max(pa.g_pa_eff_linear, 1e-12))),
                "kappa_distortion": float(pa.kappa_distortion),
                "backoff_db": float(pa.backoff_db),
            }
        )
    return pd.DataFrame(rows).sort_values("pa_id").reset_index(drop=True)


def pa_dc_power(pa, p_out):
    """Instantaneous DC power for a given RF output power."""
    if p_out <= 0.0:
        return pa.p_idle_w

    curve_pout = getattr(pa, "curve_pout_w", None)
    curve_pdc = getattr(pa, "curve_pdc_w", None)
    if curve_pout is not None and curve_pdc is not None and len(curve_pout) >= 2:
        if p_out <= float(curve_pout[0]):
            return pa.p_idle_w
        p_out_clip = min(float(p_out), float(curve_pout[-1]))
        return float(np.interp(p_out_clip, curve_pout, curve_pdc))

    loading = np.clip(p_out / pa.p_max_w, 1e-3, 1.0)
    eta = pa.eta_max * (loading ** 0.5)
    return pa.p_idle_w + p_out / eta


def inactive_pa_bank_power(pa, state, n_tx_chains):
    """Instantaneous inactive DC power for one PA bank across all TX chains."""
    if isinstance(state, PASwitchPolicy):
        state = PAState.IDLE if state == PASwitchPolicy.STANDBY else PAState.OFF
    if not isinstance(state, PAState):
        state = PAState(str(state))

    if state == PAState.IDLE:
        return int(n_tx_chains) * float(pa.p_idle_w)
    if state == PAState.OFF:
        return 0.0
    raise ValueError("inactive_pa_bank_power only supports IDLE or OFF states.")


def average_pa_power(pa, p_out, alpha_t):
    """Average DC power (W) for duty cycle alpha_t."""
    return alpha_t * pa_dc_power(pa, p_out) + (1.0 - alpha_t) * pa.p_idle_w
