import numpy as np


class McsRequirementModel:
    """Mutual-information-based MCS requirement lookup.

    The file is ordered from the public query methods down to the lower-level MI
    estimation helpers so readers can understand the lookup workflow first.
    """

    def __init__(self, mcs_table, rho_db_grid=None, seed=12345):
        self.mcs_table = mcs_table
        self.rho_db_grid = np.asarray(
            rho_db_grid if rho_db_grid is not None else np.linspace(-10.0, 35.0, 91),
            dtype=float,
        )
        self.seed = int(seed)
        self._mi_curve_cache = {}
        self._sinr_req_table_cache = {}

    def get_required_sinr(self, deployment, sched):
        """Fetch the required SINR for the candidate scheduler MCS."""
        return self.get_required_sinr_table(deployment)[sched.mcs]["rho_req_linear"]

    def get_required_sinr_table(self, deployment):
        """Build and cache SINR requirements for all MCS entries."""
        key = (
            round(float(deployment.l_impl_db), 6),
            int(len(self.rho_db_grid)),
            float(self.rho_db_grid[0]),
            float(self.rho_db_grid[-1]),
            int(deployment.mi_n_samples),
            tuple(sorted(self.mcs_table.keys())),
        )
        if key in self._sinr_req_table_cache:
            return self._sinr_req_table_cache[key]

        table = {}
        for mcs_idx, mcs_row in self.mcs_table.items():
            table[mcs_idx] = self.required_sinr_from_mcs(
                mcs_row,
                l_impl_db=deployment.l_impl_db,
                n_samples=deployment.mi_n_samples,
            )
        self._sinr_req_table_cache[key] = table
        return table

    def current_required_sinr_table(self, deployment):
        """Expose the active MCS requirement table for notebook inspection."""
        table = self.get_required_sinr_table(deployment)
        rows = []
        for mcs_idx, values in table.items():
            row = {"mcs": int(mcs_idx)}
            row.update(values)
            rows.append(row)
        return rows

    def required_sinr_from_mcs(self, mcs_row, l_impl_db, n_samples=1500):
        """Compute the SINR requirement for one MCS row including implementation loss."""
        qm = int(mcs_row["qm"])
        M = 2 ** qm
        se_target = float(mcs_row["eta"])
        rho_req_mi = self.invert_mi_for_se(M, se_target, n_samples=n_samples)
        if not np.isfinite(rho_req_mi):
            return {
                "rho_req_linear": np.inf,
                "rho_req_db": np.inf,
                "rho_req_mi_linear": np.inf,
                "rho_req_mi_db": np.inf,
            }

        impl_loss_linear = 10 ** (l_impl_db / 10.0)
        rho_req = rho_req_mi * impl_loss_linear
        return {
            "rho_req_linear": float(rho_req),
            "rho_req_db": float(10.0 * np.log10(rho_req)),
            "rho_req_mi_linear": float(rho_req_mi),
            "rho_req_mi_db": float(10.0 * np.log10(rho_req_mi)),
        }

    def invert_mi_for_se(self, M, se_target, n_samples=1500):
        """Invert the cached MI curve to find the SINR for a target efficiency."""
        if se_target <= 0.0:
            return 0.0
        curve = self.build_mi_curve(M, n_samples=n_samples)
        rho_db_grid = curve["rho_db_grid"]
        i_grid = curve["i_grid"]
        if se_target > i_grid[-1]:
            return np.inf
        rho_req_db = np.interp(se_target, i_grid, rho_db_grid)
        return float(10 ** (rho_req_db / 10.0))

    def build_mi_curve(self, M, n_samples=1500):
        """Build and cache the MI-vs-SINR curve used for MCS requirement lookup."""
        key = (
            int(M),
            float(self.rho_db_grid[0]),
            float(self.rho_db_grid[-1]),
            int(len(self.rho_db_grid)),
            int(n_samples),
            int(self.seed),
        )
        if key in self._mi_curve_cache:
            return self._mi_curve_cache[key]

        rng = np.random.default_rng(self.seed + int(M) * 1000)
        constellation = self.qam_constellation(M)
        i_grid = np.array(
            [
                self.mi_qam_awgn(
                    M,
                    10 ** (rho_db / 10.0),
                    n_samples=n_samples,
                    rng=rng,
                    constellation=constellation,
                )
                for rho_db in self.rho_db_grid
            ]
        )
        i_grid = np.maximum.accumulate(i_grid)
        curve = {"rho_db_grid": self.rho_db_grid, "i_grid": i_grid}
        self._mi_curve_cache[key] = curve
        return curve

    def mi_qam_awgn(self, M, rho_linear, n_samples=1500, rng=None, constellation=None):
        """Estimate MI in bits/symbol for M-QAM in AWGN at a given SINR."""
        if rho_linear <= 0.0:
            return 0.0
        if constellation is None:
            constellation = self.qam_constellation(M)
        if rng is None:
            rng = np.random.default_rng()

        n0 = 1.0 / rho_linear
        idx = rng.integers(0, M, size=int(n_samples))
        x = constellation[idx]
        noise = np.sqrt(n0 / 2.0) * (
            rng.standard_normal(len(idx)) + 1j * rng.standard_normal(len(idx))
        )
        y = x + noise
        metric = -(np.abs(y[:, None] - constellation[None, :]) ** 2) / n0
        metric_x = metric[np.arange(len(idx)), idx]
        metric_max = np.max(metric, axis=1, keepdims=True)
        logsumexp_metric = np.squeeze(metric_max) + np.log(np.sum(np.exp(metric - metric_max), axis=1))
        mi_nats = np.log(M) + metric_x - logsumexp_metric
        mi_bits = np.mean(mi_nats) / np.log(2.0)
        return float(np.clip(mi_bits, 0.0, np.log2(M)))

    @staticmethod
    def qam_constellation(M):
        """Generate unit-power square QAM constellation points for order M."""
        m_side = int(np.sqrt(M))
        if m_side * m_side != M:
            raise ValueError(f"M must be square QAM, got {M}")
        levels = np.arange(-(m_side - 1), m_side + 1, 2, dtype=float)
        xv, yv = np.meshgrid(levels, levels)
        const = (xv + 1j * yv).reshape(-1)
        return const / np.sqrt(np.mean(np.abs(const) ** 2))
