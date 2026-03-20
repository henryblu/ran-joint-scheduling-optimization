import copy
import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum

import numpy as np


_RESULT_CACHE = {}


def build_cache_key(request, model_inputs, options, pa_catalog=None):
    """Build a stable cache key for one search request and model configuration."""
    payload = {
        "request": _normalize(request),
        "model_inputs": _normalize(model_inputs),
        "options": _normalize(options),
        "pa_catalog": _normalize(pa_catalog),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_result(cache_key):
    """Return a defensive copy of the cached result, if present."""
    cached = _RESULT_CACHE.get(cache_key)
    return None if cached is None else copy.deepcopy(cached)


def store_cached_result(cache_key, value):
    """Store a defensive copy of the computed search result."""
    _RESULT_CACHE[cache_key] = copy.deepcopy(value)


def clear_cache():
    """Clear all memoized search results for the current Python process."""
    _RESULT_CACHE.clear()


def _normalize(value):
    """Convert nested search inputs into stable JSON-serializable primitives."""
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_normalize(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value
