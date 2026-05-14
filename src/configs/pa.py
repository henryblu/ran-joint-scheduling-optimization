"""Shared PA defaults, catalog loading, and active-state PA power helpers."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from models import PAParams


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PA_DATA_CSV = str(
    REPO_ROOT / "PA models" / "3.5Ghz_pas" / "4W_8W_NR_combined_NR_carrier_with_idle.csv"
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

    scenario_label_alias = {
        "8W": "8W PA",
        "4W": "4W PA",
        "QPA9942": "4W PA",
        "Bae et al. NR": "8W PA",
    }
    catalog = []
    for pa_name in sorted(df["pa_name"].dropna().unique()):
        sel = df[df["pa_name"] == pa_name].copy().sort_values("Pin_dBm")
        p_out = sel["Pout_W"].to_numpy(dtype=float)
        p_dc = sel["PDCIN_W"].to_numpy(dtype=float)

        kappa_guess = 0.03
        if len(p_out) >= 3 and np.nanmax(p_out) > 0:
            eta_obs = np.clip(p_out / np.clip(p_dc, 1e-12, None), 1e-4, 1.0)
            spread = float(np.nanmax(eta_obs) - np.nanmin(eta_obs))
            kappa_guess = float(np.clip(0.02 + 0.5 * spread, 0.01, 0.08))

        catalog.append(
            _build_measured_pa_from_curves(
                pa_name=str(pa_name),
                scenario_label=scenario_label_alias.get(str(pa_name), str(pa_name)),
                kappa_distortion=kappa_guess,
                pin_dbm=sel["Pin_dBm"].to_numpy(dtype=float),
                pout_w=sel["Pout_W"].to_numpy(dtype=float),
                pdcin_w=sel["PDCIN_W"].to_numpy(dtype=float),
                source_tag=str(csv_path),
            )
        )

    return sorted(catalog, key=lambda pa: (-float(pa.p_max_w), str(pa.pa_name)))


def build_pa_characteristics_table(pa_catalog_or_problem):
    """Create a compact descriptive table for the PA catalog used in the optimization."""

    pa_catalog = getattr(pa_catalog_or_problem, "pa_catalog", pa_catalog_or_problem)
    rows = []
    for pa_id, pa in enumerate(pa_catalog):
        rows.append(
            {
                "pa_id": int(pa_id),
                "scenario_label": getattr(pa, "scenario_label", ""),
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
    """Return the active-state DC power for one PA chain at the requested RF output."""

    if p_out <= 0.0:
        return 0.0

    curve_pout = getattr(pa, "curve_pout_w", None)
    curve_pdc = getattr(pa, "curve_pdc_w", None)
    if curve_pout is not None and curve_pdc is not None and len(curve_pout) >= 2:
        if p_out <= float(curve_pout[0]):
            return float(curve_pdc[0])
        p_out_clip = min(float(p_out), float(curve_pout[-1]))
        return float(np.interp(p_out_clip, curve_pout, curve_pdc))

    loading = np.clip(p_out / pa.p_max_w, 1e-3, 1.0)
    eta = pa.eta_max * (loading ** 0.5)
    return p_out / eta


def pa_slot_dc_power(pa, *, p_out_total_w, n_tx_chains, prb_fraction=1.0):
    """Return allocation-level active PA DC draw for a slot RF output.

    The PA curve is evaluated at the per-chain operating point. The optional
    PRB fraction scales precomputed allocation rows that occupy only part of
    the carrier; aggregate slot-level callers can use the default full carrier
    fraction.
    """

    if float(p_out_total_w) <= 0.0:
        return 0.0

    per_chain_p_out_w = float(p_out_total_w) / float(n_tx_chains)
    full_carrier_dc_w = float(n_tx_chains) * float(pa_dc_power(pa, per_chain_p_out_w))
    return float(prb_fraction) * full_carrier_dc_w


def _build_measured_pa_from_curves(pa_name, scenario_label, kappa_distortion, pin_dbm, pout_w, pdcin_w, source_tag):
    pin_dbm = np.asarray(pin_dbm, dtype=float)
    pout_w = np.asarray(pout_w, dtype=float)
    pdcin_w = np.asarray(pdcin_w, dtype=float)
    finite_pdcin_w = pdcin_w[np.isfinite(pdcin_w)]
    if len(finite_pdcin_w) == 0:
        raise ValueError(f"Missing PA DC samples for {pa_name}")
    p_idle_w = float(np.min(finite_pdcin_w))

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
    pin_grid_w = _dbm_to_w(pin_grid_dbm)
    pout_grid_w = np.interp(pin_grid_dbm, pin_unique, pout_unique_by_pin)
    pdc_grid_w = np.interp(pin_grid_dbm, pin_unique, pdc_unique_by_pin)

    eta_samples = np.clip(pout_grid_w / np.clip(pdc_grid_w, 1e-12, None), 1e-4, 1.0)
    p_max_w = float(np.max(pout_grid_w))
    if p_idle_w is None:
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
    positive_mask = np.asarray(pout_unique, dtype=float) > 0.0
    pout_curve_w = np.asarray(pout_unique, dtype=float)[positive_mask]
    pdc_curve_w = np.asarray(pdc_unique, dtype=float)[positive_mask]
    if len(pout_curve_w) == 0:
        raise ValueError(f"Missing positive-output PA samples for {pa_name}")

    return PAParams(
        p_max_w=p_max_w,
        p_idle_w=p_idle_w,
        eta_max=eta_max,
        g_pa_eff_linear=g_pa_eff_linear,
        kappa_distortion=kappa_distortion,
        backoff_db=6.0,
        pa_name=pa_name,
        scenario_label=scenario_label,
        curve_pout_w=pout_curve_w,
        curve_pdc_w=pdc_curve_w,
        curve_pin_w=pin_grid_w,
        source_csv=source_tag,
    )


def _dbm_to_w(x_dbm):
    return 10 ** ((np.asarray(x_dbm, dtype=float) - 30.0) / 10.0)


__all__ = [
    "DEFAULT_PA_DATA_CSV",
    "build_pa_catalog",
    "build_pa_characteristics_table",
    "pa_dc_power",
    "pa_slot_dc_power",
]
