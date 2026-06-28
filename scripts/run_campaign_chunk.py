from __future__ import annotations

"""Run one finite-frame experiment campaign chunk."""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment_runner.campaign_builder import (  # noqa: E402
    DEFAULT_CAMPAIGN_CHUNK_COUNT,
    build_default_campaign_points,
)
from experiment_runner.campaign_builder.run_chunk import run_campaign_chunk  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "campaign_chunks"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_campaign_chunk(
        build_default_campaign_points(),
        output_root=Path(args.output_root),
        chunk_index=int(args.chunk_index),
        chunk_count=int(args.chunk_count),
        cores=int(args.cores),
        limit=args.limit,
        dry_run=bool(args.dry_run),
        argv=sys.argv[1:] if argv is None else argv,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunk-count", type=int, default=DEFAULT_CAMPAIGN_CHUNK_COUNT)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
