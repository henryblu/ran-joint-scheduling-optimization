"""Stable fingerprint helpers for resolved configs and runtime state."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np


def build_resolved_fingerprint(value):
    """Build one stable SHA256 fingerprint for resolved config or engine state."""

    raw_payload = json.dumps(
        _normalize_fingerprint_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


def _normalize_fingerprint_value(value):
    """Convert resolved values into stable JSON primitives for fingerprinting."""

    if is_dataclass(value):
        return {
            field.name: _normalize_fingerprint_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
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


__all__ = ["build_resolved_fingerprint"]
