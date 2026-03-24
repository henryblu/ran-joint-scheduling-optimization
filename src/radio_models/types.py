"""Shared derived radio-domain types, physics, and shared utilities."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DeploymentParams:
    """Physical deployment parameters derived from shared radio config and distance."""

    fc_hz: float
    channel_bw_hz: float
    distance_m: float
    path_loss_db: float
    g_tx_db: float
    g_rx_db: float
    n0_dbm_per_hz: float
    lna_noise_figure_db: float
    l_impl_db: float
    mi_n_samples: int
    n_dmrs_sym: int
    n_guard_sym: int
    n_ul_sym: int
    dft_size_N: int
    n_slots_win: int
    t_slot_s: float
    n_sym_data: int
    n_sym_total: int
    use_psd_constraint: bool
    psd_max_w_per_hz: float
    papr_db: float
    g_phi: float
    sigma_phi2: float
    sigma_q2: float
    n_tx_chains: int


class PathLossModel:
    """Large-scale path-loss and channel-gain helpers."""

    C_M_PER_S = 299_792_458.0

    def __init__(
        self,
        fc_hz,
        model="umi_sc_los",
        g_tx_db=0.0,
        g_rx_db=0.0,
        shadow_margin_db=4.0,
        h_bs_m=10.0,
        h_ut_m=1.5,
    ):
        self.fc_hz = float(fc_hz)
        self.model = model
        self.g_tx_db = float(g_tx_db)
        self.g_rx_db = float(g_rx_db)
        self.shadow_margin_db = float(shadow_margin_db)
        self.h_bs_m = float(h_bs_m)
        self.h_ut_m = float(h_ut_m)

    @staticmethod
    def _validate_link_geometry(fc_hz, distance_m, h_bs_m, h_ut_m):
        if float(fc_hz) <= 0.0:
            raise ValueError("fc_hz must be > 0")
        if float(distance_m) <= 0.0:
            raise ValueError("distance_m must be > 0")
        if float(h_bs_m) <= 0.0 or float(h_ut_m) <= 0.0:
            raise ValueError("Antenna heights must be > 0")

    @staticmethod
    def distance_3d_m(distance_m, h_bs_m=10.0, h_ut_m=1.5):
        PathLossModel._validate_link_geometry(1.0, distance_m, h_bs_m, h_ut_m)
        d_2d = float(distance_m)
        return float(np.sqrt(d_2d**2 + (float(h_bs_m) - float(h_ut_m)) ** 2))

    @classmethod
    def umi_breakpoint_distance_m(cls, fc_hz, h_bs_m=10.0, h_ut_m=1.5):
        cls._validate_link_geometry(fc_hz, 1.0, h_bs_m, h_ut_m)
        return float(4.0 * float(h_bs_m) * float(h_ut_m) * float(fc_hz) / cls.C_M_PER_S)

    @staticmethod
    def free_space_path_loss_db(fc_hz, distance_m):
        PathLossModel._validate_link_geometry(fc_hz, distance_m, 1.0, 1.0)
        fc_mhz = float(fc_hz) / 1e6
        distance_km = float(distance_m) / 1e3
        return float(32.44 + 20.0 * np.log10(fc_mhz) + 20.0 * np.log10(distance_km))

    @classmethod
    def umi_sc_los_pl1_db(cls, fc_hz, distance_m, h_bs_m=10.0, h_ut_m=1.5):
        cls._validate_link_geometry(fc_hz, distance_m, h_bs_m, h_ut_m)
        fc_ghz = float(fc_hz) / 1e9
        d_3d = cls.distance_3d_m(distance_m, h_bs_m, h_ut_m)
        return float(32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz))

    @classmethod
    def umi_sc_los_pl2_db(cls, fc_hz, distance_m, h_bs_m=10.0, h_ut_m=1.5):
        cls._validate_link_geometry(fc_hz, distance_m, h_bs_m, h_ut_m)
        fc_ghz = float(fc_hz) / 1e9
        d_3d = cls.distance_3d_m(distance_m, h_bs_m, h_ut_m)
        d_bp = cls.umi_breakpoint_distance_m(fc_hz, h_bs_m, h_ut_m)
        height_delta = float(h_bs_m) - float(h_ut_m)
        return float(
            32.4
            + 40.0 * np.log10(d_3d)
            + 20.0 * np.log10(fc_ghz)
            - 9.5 * np.log10(d_bp**2 + height_delta**2)
        )

    @classmethod
    def umi_sc_los_path_loss_db(cls, fc_hz, distance_m, h_bs_m=10.0, h_ut_m=1.5):
        cls._validate_link_geometry(fc_hz, distance_m, h_bs_m, h_ut_m)
        d_bp = cls.umi_breakpoint_distance_m(fc_hz, h_bs_m, h_ut_m)
        if float(distance_m) <= d_bp:
            return cls.umi_sc_los_pl1_db(fc_hz, distance_m, h_bs_m, h_ut_m)
        return cls.umi_sc_los_pl2_db(fc_hz, distance_m, h_bs_m, h_ut_m)

    @classmethod
    def umi_sc_nlos_prime_path_loss_db(cls, fc_hz, distance_m, h_bs_m=10.0, h_ut_m=1.5):
        cls._validate_link_geometry(fc_hz, distance_m, h_bs_m, h_ut_m)
        fc_ghz = float(fc_hz) / 1e9
        d_3d = cls.distance_3d_m(distance_m, h_bs_m, h_ut_m)
        return float(
            35.3 * np.log10(d_3d)
            + 22.4
            + 21.3 * np.log10(fc_ghz)
            - 0.3 * (float(h_ut_m) - 1.5)
        )

    @classmethod
    def umi_sc_nlos_path_loss_db(cls, fc_hz, distance_m, h_bs_m=10.0, h_ut_m=1.5):
        cls._validate_link_geometry(fc_hz, distance_m, h_bs_m, h_ut_m)
        pl_los = cls.umi_sc_los_path_loss_db(fc_hz, distance_m, h_bs_m, h_ut_m)
        pl_nlos_prime = cls.umi_sc_nlos_prime_path_loss_db(fc_hz, distance_m, h_bs_m, h_ut_m)
        return float(max(pl_los, pl_nlos_prime))

    def effective_path_loss_db(self, distance_m):
        if self.model == "fspl":
            base_pl_db = self.free_space_path_loss_db(self.fc_hz, distance_m)
        elif self.model == "umi_sc_los":
            base_pl_db = self.umi_sc_los_path_loss_db(self.fc_hz, distance_m, self.h_bs_m, self.h_ut_m)
        elif self.model == "umi_sc_nlos":
            base_pl_db = self.umi_sc_nlos_path_loss_db(self.fc_hz, distance_m, self.h_bs_m, self.h_ut_m)
        else:
            raise ValueError(f"Unknown path loss model: {self.model}")
        return float(base_pl_db + self.shadow_margin_db)

    def channel_gain_linear(self, path_loss_db):
        g_tx_linear = 10 ** (self.g_tx_db / 10.0)
        g_rx_linear = 10 ** (self.g_rx_db / 10.0)
        path_loss_linear = 10 ** (float(path_loss_db) / 10.0)
        return (g_tx_linear * g_rx_linear) / path_loss_linear


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


__all__ = ["DeploymentParams", "PathLossModel", "build_resolved_fingerprint"]
