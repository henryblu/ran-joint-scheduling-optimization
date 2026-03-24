from copy import deepcopy
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


def prepare_single_user_problem(request, preset, *, pa_catalog=None):
    """Build the reusable single-user context for one deployment request.

    Steps:
    1. Resolve the preset-derived radio inputs and one concrete search-space policy.
    2. Build the concrete deployment, including the path loss derived from distance in radio_core.
    3. Build the discrete single-user problem from the concrete deployment and PA catalog.
    4. Return the prepared context used by the search and study layers.
    """

    model_inputs = resolve_model_inputs(preset)
    resolved_options = _build_resolved_search_options(model_inputs)
    resolved_pa_catalog = resolve_pa_catalog(model_inputs, pa_catalog)
    deployment = build_single_user_deployment(
        model_inputs["link_constants"],
        model_inputs["phy_constants"],
        distance_m=float(request.distance_m),
    )
    built_problem = build_discrete_problem(
        deployment,
        pa_catalog=resolved_pa_catalog,
        scheduler_sweep=model_inputs["scheduler_sweep"],
        delta_f_hz_default=model_inputs["phy_constants"]["delta_f_hz"],
        bandwidth_space=resolved_options.bandwidth_space_hz,
        n_slots_on_space=resolved_options.n_slots_on_space,
        prb_step=resolved_options.prb_step,
    )
    return PreparedSingleUserContext(
        model_inputs=model_inputs,
        deployment=deployment,
        built_problem=built_problem,
        pa_catalog=built_problem.pa_catalog,
        mcs_table=model_inputs["mcs_table"],
        rrc_lookup=built_problem.rrc_lookup,
        options=resolved_options,
    )


def resolve_model_inputs(preset):
    """Materialize the preset inputs required to build one single-user problem."""

    cache_key = _build_model_inputs_cache_key(preset)
    cached = _MODEL_INPUTS_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    model_inputs = build_model_inputs(preset)
    _MODEL_INPUTS_CACHE[cache_key] = deepcopy(model_inputs)
    return deepcopy(_MODEL_INPUTS_CACHE[cache_key])


def resolve_pa_catalog(model_inputs, pa_catalog=None):
    """Load the preset PA catalog unless the caller supplied one explicitly."""

    if pa_catalog is not None:
        return pa_catalog

    cache_key = _build_pa_catalog_cache_key(model_inputs["pa_data_csv"])
    cached = _PA_CATALOG_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    resolved_pa_catalog = build_pa_catalog(model_inputs["pa_data_csv"])
    _PA_CATALOG_CACHE[cache_key] = deepcopy(resolved_pa_catalog)
    return deepcopy(_PA_CATALOG_CACHE[cache_key])


def clear_problem_factory_cache():
    """Clear all memoized model inputs and PA catalogs for the current process."""

    _MODEL_INPUTS_CACHE.clear()
    _PA_CATALOG_CACHE.clear()


def _build_resolved_search_options(model_inputs):
    """Resolve the concrete search-space dimensions stored in the prepared context."""

    scheduler_sweep = model_inputs["scheduler_sweep"]
    phy_constants = model_inputs["phy_constants"]
    return SingleUserSearchOptions(
        prb_step=int(scheduler_sweep["prb_step"]),
        bandwidth_space_hz=tuple(float(v) for v in scheduler_sweep["bandwidth_space_hz"]),
        n_slots_on_space=tuple(
            int(v)
            for v in range(1, int(phy_constants["n_slots_win"]) + 1)
        ),
        use_cache=True,
    )


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
