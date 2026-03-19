import numpy as np


def db_to_linear(db):
    """Convert dB -> linear power ratio."""
    return 10 ** (db / 10.0)


def linear_to_dbm(x):
    """Convert linear power (W) -> dBm."""
    return 10.0 * np.log10(x * 1000.0) if x > 0 else -np.inf


def dbm_to_w(x_dbm):
    """Convert dBm -> linear power (W)."""
    return 10 ** ((np.asarray(x_dbm, dtype=float) - 30.0) / 10.0)
