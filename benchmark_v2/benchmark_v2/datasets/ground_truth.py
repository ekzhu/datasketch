"""Ground-truth utilities for benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .cache import ensure_dataset


@dataclass(frozen=True)
class GroundTruthRecord:
    query_id: int
    neighbor_id: int
    score: float | None = None


def ensure_ground_truth(
    dataset_name: str,
    ground_truth_name: str,
    *,
    cache_root: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Ensure a specific ground-truth artifact is available locally."""

    result = ensure_dataset(
        dataset_name,
        cache_root=cache_root,
        include_queries=False,
        include_ground_truth=True,
        force_download=force_download,
    )
    try:
        return result.ground_truth_paths[ground_truth_name]
    except KeyError as exc:
        raise KeyError(
            f"Dataset '{dataset_name}' does not define ground truth '{ground_truth_name}'"
        ) from exc


def iter_ground_truth(path: Path) -> Iterator[GroundTruthRecord]:
    """Yield normalized ground-truth entries from *path*."""

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Ground truth row must contain at least query and neighbor ids: {stripped}"
                )
            query_id = int(parts[0])
            neighbor_id = int(parts[1])
            score = float(parts[2]) if len(parts) > 2 else None
            yield GroundTruthRecord(
                query_id=query_id, neighbor_id=neighbor_id, score=score
            )
