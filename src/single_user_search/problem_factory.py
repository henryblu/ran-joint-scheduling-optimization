import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from radio_core import build_model_inputs, build_pa_catalog, build_single_user_deployment

from .candidate_space import build_discrete_problem
from .models import PreparedSingleUserContext, SingleUserSearchOptions


_MODEL_INPUTS_CACHE = {}
_PA_CATALOG_CACHE = {}


def prepare_single_user_problem(request, preset, *, pa_catalog=None, options=None):
    """Build the reusable single-user context for one deployment request.

    Steps:
    1. Resolve the runtime options and preset-derived radio inputs.
    2. Build the deployment for the requested distance and optional path loss.
    3. Materialize the discrete downlink problem for that deployment.
    4. Return the compact prepared context reused by active-table and study helpers.
    """

    resolved_options = _resolve_options(options)
    model_inputs = resolve_model_inputs(preset)
    deployment = build_deployment_for_request(request, model_inputs)
    return prepare_single_user_problem_from_deployment(
        deployment,
        preset=preset,
        pa_catalog=pa_catalog,
        options=resolved_options,
        model_inputs=model_inputs,
    )


def prepare_single_user_problem_from_deployment(
    deployment,
    *,
    preset,
    pa_catalog=None,
    options=None,
    model_inputs=None,
):
    """Build the single-user context when deployment state is already known."""

    resolved_options = _resolve_options(options)
    resolved_model_inputs = resolve_model_inputs(preset) if model_inputs is None else model_inputs
    bandwidth_space, n_slots_on_space, prb_step = resolve_discrete_search_inputs(
        resolved_model_inputs,
        resolved_options,
    )
    built_problem = build_discrete_problem(
        deployment,
        pa_catalog=resolve_pa_catalog(resolved_model_inputs, pa_catalog),
        scheduler_sweep=resolved_model_inputs["scheduler_sweep"],
        delta_f_hz_default=resolved_model_inputs["phy_constants"]["delta_f_hz"],
        bandwidth_space=bandwidth_space,
        n_slots_on_space=n_slots_on_space,
        prb_step=prb_step,
    )
    return PreparedSingleUserContext(
        model_inputs=resolved_model_inputs,
        deployment=deployment,
        built_problem=built_problem,
        pa_catalog=built_problem.pa_catalog,
        mcs_table=resolved_model_inputs["mcs_table"],
        rrc_lookup=built_problem.rrc_lookup,
        options=resolved_options,
    )


def build_deployment_for_request(request, model_inputs):
    """Build deployment state from a request and already-resolved model inputs."""

    return build_single_user_deployment(
        model_inputs["link_constants"],
        model_inputs["phy_constants"],
        distance_m=float(request.distance_m),
        path_loss_db=request.path_loss_db,
    )

def resolve_discrete_search_inputs(model_inputs, options):
    """Resolve the discrete search-shaping inputs without leaking option objects downstream."""

    scheduler_sweep = model_inputs["scheduler_sweep"]
    bandwidth_space = (
        [float(v) for v in options.bandwidth_space_hz]
        if options.bandwidth_space_hz is not None
        else [float(v) for v in scheduler_sweep["bandwidth_space_hz"]]
    )
    n_slots_on_space = (
        [int(v) for v in options.n_slots_on_space]
        if options.n_slots_on_space is not None
        else None
    )
    prb_step = int(options.prb_step) if options.prb_step is not None else int(scheduler_sweep["prb_step"])
    return bandwidth_space, n_slots_on_space, prb_step


def resolve_model_inputs(preset):
    """Materialize the preset inputs required to build one single-user problem."""

    cache_key = _build_model_inputs_cache_key(preset)
    cached = _MODEL_INPUTS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model_inputs = build_model_inputs(preset)
    _MODEL_INPUTS_CACHE[cache_key] = model_inputs
    return model_inputs


def resolve_pa_catalog(model_inputs, pa_catalog=None):
    """Load the preset PA catalog unless the caller supplied one explicitly."""

    if pa_catalog is not None:
        return pa_catalog

    cache_key = _build_pa_catalog_cache_key(model_inputs["pa_data_csv"])
    cached = _PA_CATALOG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    resolved_pa_catalog = build_pa_catalog(model_inputs["pa_data_csv"])
    _PA_CATALOG_CACHE[cache_key] = resolved_pa_catalog
    return resolved_pa_catalog


def clear_problem_factory_cache():
    """Clear all memoized model inputs and PA catalogs for the current process."""

    _MODEL_INPUTS_CACHE.clear()
    _PA_CATALOG_CACHE.clear()


def _resolve_options(options):
    """Default missing search options without mutating caller-provided objects."""

    return SingleUserSearchOptions() if options is None else options


def _build_model_inputs_cache_key(preset):
    """Build a stable process-local key for preset-derived model inputs."""

    normalized_preset = _normalize_cache_value(preset)
    return json.dumps(normalized_preset, sort_keys=True, separators=(",", ":"))


def _build_pa_catalog_cache_key(pa_data_csv):
    """Build a stable process-local key for preset-sourced PA catalogs."""

    return str(Path(pa_data_csv).resolve())


def _normalize_cache_value(value):
    """Convert nested preset values into stable cache-key primitives."""

    if is_dataclass(value):
        return _normalize_cache_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize_cache_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value
