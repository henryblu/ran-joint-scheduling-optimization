from __future__ import annotations

"""Local extraction boundary for the scheduler-comparison ZIP artifact."""

from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile


DEFAULT_ARTIFACT_ZIP = Path("outputs/scheduler_comparison_hpc_sweep.zip")
DEFAULT_EXTRACTION_ROOT = Path("outputs/scheduler_comparison_hpc_sweep")
DEFAULT_ANALYSIS_ROOT = Path("outputs/scheduler_comparison_hpc_sweep_analysis")
CHUNK_NAME_PATTERN = re.compile(r"^chunk_(?P<index>\d{2})_of_(?P<count>\d+)$")


def resolve_scheduler_comparison_input_root(
    *,
    input_root: Path | None = None,
    artifact_zip: Path | None = None,
    extraction_root: Path | None = None,
    force_extract: bool = False,
) -> Path:
    """Resolve the extracted scheduler-comparison chunk root for preprocessing.

    The preferred thesis artifact is the tracked ZIP. Callers may pass an
    already-extracted directory for local iteration, or pass the ZIP and let
    this boundary extract it into an ignored `outputs/` directory. Downstream
    preprocessing sees only an extracted root with chunk CSVs.
    """

    if input_root is not None:
        return Path(input_root)

    resolved_artifact_zip = Path(artifact_zip) if artifact_zip is not None else DEFAULT_ARTIFACT_ZIP
    resolved_extraction_root = Path(extraction_root) if extraction_root is not None else DEFAULT_EXTRACTION_ROOT

    if force_extract or not extracted_chunks_present(resolved_extraction_root):
        return extract_scheduler_comparison_artifact(
            artifact_zip=resolved_artifact_zip,
            extraction_root=resolved_extraction_root,
        )

    return resolve_extracted_chunk_root(resolved_extraction_root)


def extract_scheduler_comparison_artifact(*, artifact_zip: Path, extraction_root: Path) -> Path:
    """Extract the tracked scheduler-comparison ZIP and return the chunk root.

    The final HPC artifact may already contain a single top-level directory
    named like the ZIP stem. In that case extraction targets `outputs/`, so the
    resulting chunk root is `outputs/scheduler_comparison_hpc_sweep` rather
    than `outputs/scheduler_comparison_hpc_sweep/scheduler_comparison_hpc_sweep`.
    """

    resolved_artifact_zip = Path(artifact_zip)
    resolved_extraction_root = Path(extraction_root)

    with ZipFile(resolved_artifact_zip) as archive:
        archive_members = archive.namelist()
        extraction_target = extraction_target_for_archive(
            archive_members=archive_members,
            artifact_zip=resolved_artifact_zip,
            extraction_root=resolved_extraction_root,
        )
        extraction_target.mkdir(parents=True, exist_ok=True)
        archive.extractall(extraction_target)

    return resolve_extracted_chunk_root(resolved_extraction_root)


def extraction_target_for_archive(
    *,
    archive_members: list[str],
    artifact_zip: Path,
    extraction_root: Path,
) -> Path:
    top_level_dirs = archive_top_level_dirs(archive_members)
    if top_level_dirs == {Path(artifact_zip).stem}:
        return Path(extraction_root).parent
    return Path(extraction_root)


def archive_top_level_dirs(archive_members: list[str]) -> set[str]:
    roots = set()
    for member in archive_members:
        parts = PurePosixPath(member).parts
        if parts:
            roots.add(parts[0])
    return roots


def resolve_extracted_chunk_root(root: Path) -> Path:
    resolved_root = Path(root)
    if chunks_directly_under(resolved_root):
        return resolved_root

    candidate_roots = sorted(
        {path.parent for path in resolved_root.rglob("*") if path.is_dir() and CHUNK_NAME_PATTERN.match(path.name)}
    )
    if len(candidate_roots) == 1:
        return candidate_roots[0]
    if not candidate_roots:
        raise FileNotFoundError(f"No scheduler-comparison chunk directories found under {resolved_root}")
    raise ValueError(
        "Ambiguous scheduler-comparison chunk roots found: "
        + ", ".join(str(candidate) for candidate in candidate_roots)
    )


def extracted_chunks_present(root: Path) -> bool:
    try:
        resolve_extracted_chunk_root(root)
    except (FileNotFoundError, ValueError):
        return False
    return True


def chunks_directly_under(root: Path) -> bool:
    resolved_root = Path(root)
    if not resolved_root.exists():
        return False
    return any(path.is_dir() and CHUNK_NAME_PATTERN.match(path.name) for path in resolved_root.iterdir())
