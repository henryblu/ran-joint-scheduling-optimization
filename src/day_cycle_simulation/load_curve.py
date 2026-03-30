"""Load-curve ingestion and 15-minute target-bin expansion."""

import pandas as pd


BINS_PER_HOUR = 4
BITS_PER_GB = 8e9


def load_hourly_load_curve(csv_path) -> pd.DataFrame:
    """Load one hourly load-curve CSV as-is for the day-cycle pipeline."""

    return pd.read_csv(csv_path)


def build_15_minute_target_load_table(hourly_load_table: pd.DataFrame) -> pd.DataFrame:
    """Expand one hourly GB/h load table into piecewise-constant 15-minute bins."""

    rows = []
    for hour_row in hourly_load_table.itertuples(index=False):
        hour = int(hour_row.hour)
        total_load_gbph = float(hour_row.total_load_gbph)
        target_bits_in_bin = total_load_gbph * BITS_PER_GB / float(BINS_PER_HOUR)

        for quarter_index in range(BINS_PER_HOUR):
            rows.append(
                {
                    "bin_index": BINS_PER_HOUR * (hour - 1) + quarter_index,
                    "hour": hour,
                    "total_load_gbph": total_load_gbph,
                    "target_bits_in_bin": target_bits_in_bin,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["bin_index", "hour", "total_load_gbph", "target_bits_in_bin"],
    )
