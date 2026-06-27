def slot_level_re_counts(deployment, candidate):
    """Per-slot NR resource-element accounting with explicit pilot subtraction."""

    n_re_raw = candidate.n_prb * 12 * deployment.n_sym_data
    n_dmrs_re_per_prb = 12 * deployment.n_dmrs_sym
    n_pilot = candidate.n_prb * n_dmrs_re_per_prb
    n_re_data = max(n_re_raw - n_pilot, 1.0)
    return {
        "n_re_raw": float(n_re_raw),
        "n_dmrs_re_per_prb": float(n_dmrs_re_per_prb),
        "n_pilot": float(n_pilot),
        "n_re_data": float(n_re_data),
    }


def occupied_bandwidth_hz(rrc, candidate):
    """Occupied bandwidth from allocated PRBs."""

    return float(candidate.n_prb * 12.0 * rrc.delta_f_hz)


def get_n_streams(candidate):
    """Current single-user stream count."""

    return max(int(candidate.layers), 1)
