from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np

from .model_types import (
    DeploymentParams,
    PAParams,
    ResolvedModelInputs,
    ResolvedMultiUserSystemCfg,
    ResolvedSearchShape,
)
from .pa_models import build_pa_catalog
from .path_loss_models import PathLossModel


_MODEL_INPUTS_CACHE = {}
_PA_CATALOG_CACHE = {}


def clear_config_builder_cache():
    """Clear resolved radio-config caches used by notebook and test helpers."""

    _MODEL_INPUTS_CACHE.clear()
    _PA_CATALOG_CACHE.clear()


def resolve_model_inputs(model_preset):
    """Resolve one frozen preset into the typed radio inputs shared by engine code."""

    cache_key = build_resolved_fingerprint(model_preset)
    cached = _MODEL_INPUTS_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    resolved_inputs = ResolvedModelInputs(
        fingerprint=cache_key,
        link=model_preset.link,
        phy=model_preset.phy,
        scheduler=model_preset.scheduler,
        mcs_table=deepcopy(model_preset.mcs_table),
        pa_data_csv=str(Path(model_preset.pa_data_csv).resolve()),
    )
    _MODEL_INPUTS_CACHE[cache_key] = resolved_inputs
    return deepcopy(resolved_inputs)


def resolve_search_shape(model_inputs, *, use_cache=True):
    """Resolve the concrete tuple-based search dimensions used by single-user search."""

    fingerprint_payload = {
        "bandwidth_space_hz": model_inputs.scheduler.bandwidth_space_hz,
        "n_slots_on_space": tuple(range(1, model_inputs.phy.n_slots_win + 1)),
        "layers_space": model_inputs.scheduler.layers_space,
        "mcs_space": model_inputs.scheduler.mcs_space,
        "prb_step": model_inputs.scheduler.prb_step,
    }
    return ResolvedSearchShape(
        bandwidth_space_hz=model_inputs.scheduler.bandwidth_space_hz,
        n_slots_on_space=fingerprint_payload["n_slots_on_space"],
        layers_space=model_inputs.scheduler.layers_space,
        mcs_space=model_inputs.scheduler.mcs_space,
        prb_step=model_inputs.scheduler.prb_step,
        fingerprint=build_resolved_fingerprint(fingerprint_payload),
        use_cache=bool(use_cache),
    )


def resolve_pa_catalog(model_inputs):
    """Load and freeze the measured PA catalog referenced by one resolved model bundle."""

    cache_key = model_inputs.pa_data_csv
    cached = _PA_CATALOG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    resolved_catalog = _freeze_pa_catalog(build_pa_catalog(model_inputs.pa_data_csv))
    _PA_CATALOG_CACHE[cache_key] = resolved_catalog
    return resolved_catalog


def resolve_path_loss_db(link_constants, distance_m):
    """Resolve one deployment's concrete path loss from distance and radio configuration."""

    return PathLossModel(
        fc_hz=link_constants.fc_hz,
        model=link_constants.pl_model,
        g_tx_db=link_constants.g_tx_db,
        g_rx_db=link_constants.g_rx_db,
        shadow_margin_db=link_constants.shadow_margin_db,
        h_bs_m=link_constants.h_bs_m,
        h_ut_m=link_constants.h_ut_m,
    ).effective_path_loss_db(distance_m)


def resolve_path_loss_db_values(link_constants, distance_values_m):
    """Resolve one concrete path-loss value per distance entry."""

    return [
        resolve_path_loss_db(link_constants, distance_m=distance_m)
        for distance_m in distance_values_m
    ]


def build_single_user_deployment(link_constants, phy_constants, distance_m):
    """Build a deployment object from typed resolved radio config and one distance."""

    return DeploymentParams(
        fc_hz=link_constants.fc_hz,
        channel_bw_hz=phy_constants.channel_bw_hz,
        distance_m=distance_m,
        path_loss_db=resolve_path_loss_db(link_constants, distance_m=distance_m),
        g_tx_db=link_constants.g_tx_db,
        g_rx_db=link_constants.g_rx_db,
        n0_dbm_per_hz=link_constants.n0_dbm_per_hz,
        lna_noise_figure_db=link_constants.lna_noise_figure_db,
        l_impl_db=phy_constants.l_impl_db,
        mi_n_samples=phy_constants.mi_n_samples,
        n_dmrs_sym=phy_constants.n_dmrs_sym,
        n_guard_sym=phy_constants.n_guard_sym,
        n_ul_sym=phy_constants.n_ul_sym,
        dft_size_N=phy_constants.dft_size_N,
        n_slots_win=phy_constants.n_slots_win,
        t_slot_s=phy_constants.t_slot_s,
        n_sym_data=phy_constants.n_sym_data,
        n_sym_total=phy_constants.n_sym_total,
        use_psd_constraint=phy_constants.use_psd_constraint,
        psd_max_w_per_hz=phy_constants.psd_max_w_per_hz,
        papr_db=phy_constants.papr_db,
        g_phi=phy_constants.g_phi,
        sigma_phi2=phy_constants.sigma_phi2,
        sigma_q2=phy_constants.sigma_q2,
        n_tx_chains=phy_constants.n_tx_chains,
    )


def build_multi_user_system_cfg(model_inputs, tdd_config):
    """Build the resolved mixed-slot TDMA system view from canonical radio config.

    Steps:
    1. Resolve the frame length directly from the shared PHY window.
    2. Validate that the mixed-slot TDD pattern matches the PHY symbol accounting.
    3. Expose one schedulable slot per frame slot, with reduced DL payload carried in the PHY symbols.
    4. Return the frozen system view reused across the multi-user study.
    """

    link = model_inputs.link
    phy = model_inputs.phy
    scheduler = model_inputs.scheduler

    frame_slots = phy.n_slots_win
    _validate_mixed_slot_pattern(phy, tdd_config)
    return ResolvedMultiUserSystemCfg(
        fc_hz=link.fc_hz,
        channel_bw_hz=phy.channel_bw_hz,
        bandwidth_space_hz=scheduler.bandwidth_space_hz,
        total_prbs=int(phy.channel_bw_hz // (12.0 * phy.delta_f_hz)),
        frame_slots=frame_slots,
        slot_dl_symbols=tdd_config.n_dl_symbols,
        slot_guard_symbols=tdd_config.n_guard_symbols,
        slot_ul_symbols=tdd_config.n_ul_symbols,
        slot_payload_symbols=tdd_config.n_dl_symbols - phy.n_dmrs_sym,
        total_slots=frame_slots,
        delta_f_hz=phy.delta_f_hz,
        g_tx_db=link.g_tx_db,
        g_rx_db=link.g_rx_db,
        noise_density_dbm_per_hz=link.n0_dbm_per_hz,
        noise_figure_db=link.lna_noise_figure_db,
        impl_loss_db=phy.l_impl_db,
        mi_n_samples=phy.mi_n_samples,
        n_dmrs_sym=phy.n_dmrs_sym,
        n_guard_sym=phy.n_guard_sym,
        n_ul_sym=phy.n_ul_sym,
        n_sym_data=phy.n_sym_data,
        n_sym_total=phy.n_sym_total,
        dft_size_N=phy.dft_size_N,
        t_slot_s=phy.t_slot_s,
        n_tx_chains=phy.n_tx_chains,
        use_psd_constraint=phy.use_psd_constraint,
        psd_max_w_per_hz=phy.psd_max_w_per_hz,
        papr_db=phy.papr_db,
        g_phi=phy.g_phi,
        sigma_phi2=phy.sigma_phi2,
        sigma_q2=phy.sigma_q2,
        layers_space=scheduler.layers_space,
        mcs_space=scheduler.mcs_space,
        prb_step=scheduler.prb_step,
    )


def build_multi_user_runtime_cfg(runtime_config):
    """Convert runtime policy into notebook-ready values."""

    return {
        "switch_policy": runtime_config.switch_policy,
        "max_configs_per_user": int(runtime_config.max_configs_per_user),
        "max_schedule_windows": int(runtime_config.max_schedule_windows),
    }


def build_resolved_fingerprint(value):
    """Build one stable SHA256 fingerprint for resolved config or engine state."""

    raw_payload = json.dumps(
        _normalize_fingerprint_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


def _validate_mixed_slot_pattern(phy, tdd_config):
    """Reject inconsistent mixed-slot TDD definitions before study code uses them."""

    if tdd_config.n_dl_symbols != phy.n_sym_data:
        raise ValueError("TDD DL-symbol count must match phy.n_sym_data.")
    if tdd_config.n_guard_symbols != phy.n_guard_sym:
        raise ValueError("TDD guard-symbol count must match phy.n_guard_sym.")
    if tdd_config.n_ul_symbols != phy.n_ul_sym:
        raise ValueError("TDD UL-symbol count must match phy.n_ul_sym.")
    if tdd_config.n_dl_symbols + tdd_config.n_guard_symbols + tdd_config.n_ul_symbols != phy.n_sym_total:
        raise ValueError("TDD slot symbols must sum to phy.n_sym_total.")
    if phy.n_dmrs_sym > tdd_config.n_dl_symbols:
        raise ValueError("DMRS symbols cannot exceed the DL-symbol region in one slot.")


def _freeze_pa_catalog(pa_catalog):
    """Copy one PA catalog into tuple-based engine state with read-only curves."""

    return tuple(_freeze_pa_params(pa) for pa in pa_catalog)


def _freeze_pa_params(pa):
    """Copy one PA profile into immutable process-shared primitives."""

    return PAParams(
        p_max_w=float(pa.p_max_w),
        p_idle_w=float(pa.p_idle_w),
        eta_max=float(pa.eta_max),
        g_pa_eff_linear=float(pa.g_pa_eff_linear),
        kappa_distortion=float(pa.kappa_distortion),
        backoff_db=float(pa.backoff_db),
        pa_name=str(pa.pa_name),
        curve_pout_w=_freeze_pa_curve(getattr(pa, "curve_pout_w", None)),
        curve_pdc_w=_freeze_pa_curve(getattr(pa, "curve_pdc_w", None)),
        curve_pin_w=_freeze_pa_curve(getattr(pa, "curve_pin_w", None)),
        source_csv=str(getattr(pa, "source_csv", "")),
    )


def _freeze_pa_curve(values):
    """Copy one PA curve into a read-only float array."""

    if values is None:
        return None
    frozen_curve = np.asarray(values, dtype=float).copy()
    frozen_curve.setflags(write=False)
    return frozen_curve


def _normalize_fingerprint_value(value):
    """Convert resolved values into stable JSON primitives for fingerprinting."""

    if is_dataclass(value):
        return _normalize_fingerprint_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_fingerprint_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize_fingerprint_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "build_multi_user_runtime_cfg",
    "build_multi_user_system_cfg",
    "build_resolved_fingerprint",
    "build_single_user_deployment",
    "clear_config_builder_cache",
    "resolve_model_inputs",
    "resolve_pa_catalog",
    "resolve_path_loss_db",
    "resolve_path_loss_db_values",
    "resolve_search_shape",
]
