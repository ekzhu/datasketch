"""Utilities to construct and load numpy memmap representations of datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..progress import make_progress


@dataclass(frozen=True)
class CSRMemmap:
    """Paths to the memmapped CSR-like representation."""

    tokens_path: Path
    indptr_path: Path
    metadata_path: Path

    def open(self) -> "LoadedCSR":
        """Open the memmaps in read-only mode."""

        metadata = json.loads(self.metadata_path.read_text())
        tokens = np.memmap(
            self.tokens_path, dtype=np.dtype(metadata["dtype"]), mode="r"
        )
        indptr = np.memmap(self.indptr_path, dtype=np.int64, mode="r")
        return LoadedCSR(tokens=tokens, indptr=indptr, metadata=metadata)


@dataclass(frozen=True)
class LoadedCSR:
    tokens: np.memmap
    indptr: np.memmap
    metadata: dict

    def iter_rows(self) -> Iterator[np.ndarray]:
        """Yield each set as a view over the underlying memmap."""

        indptr = self.indptr
        tokens = self.tokens
        for idx in range(len(indptr) - 1):
            start = int(indptr[idx])
            end = int(indptr[idx + 1])
            yield tokens[start:end]


def ensure_memmap(
    dataset_path: Path,
    *,
    dtype: str = "int32",
    force: bool = False,
) -> CSRMemmap:
    """Ensure the memmap representation for *dataset_path* exists."""

    memmap = _memmap_paths(dataset_path, dtype)
    if force or not memmap.tokens_path.exists() or not memmap.metadata_path.exists():
        build_memmap(dataset_path, dtype=dtype)
    return memmap


def build_memmap(dataset_path: Path, *, dtype: str = "int32") -> CSRMemmap:
    """Build memmaps directly, overwriting any previous files."""

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    memmap = _memmap_paths(dataset_path, dtype)
    memmap.tokens_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _scan_dataset(dataset_path)
    np_dtype = np.dtype(dtype)

    tokens_mm = np.memmap(
        memmap.tokens_path, dtype=np_dtype, mode="w+", shape=(stats.total_tokens,)
    )
    indptr_mm = np.memmap(
        memmap.indptr_path, dtype=np.int64, mode="w+", shape=(stats.num_sets + 1,)
    )
    indptr_mm[0] = 0

    offset = 0
    row = 0
    progress_total = stats.num_sets if stats.num_sets > 0 else None
    desc = f"Memmap {dataset_path.name}"
    with make_progress(desc=desc, total=progress_total, unit="set") as progress:
        with dataset_path.open("r", encoding="utf-8") as fh:
            for tokens in _iter_token_rows(fh):
                count = int(len(tokens))
                if count:
                    tokens_mm[offset : offset + count] = tokens
                offset += count
                row += 1
                indptr_mm[row] = offset
                progress.update(1)

    tokens_mm.flush()
    indptr_mm.flush()

    metadata = {
        "dtype": str(np_dtype),
        "num_sets": stats.num_sets,
        "num_tokens": stats.total_tokens,
        "source": str(dataset_path),
    }
    memmap.metadata_path.write_text(json.dumps(metadata, indent=2))
    return memmap


@dataclass(frozen=True)
class _DatasetStats:
    num_sets: int
    total_tokens: int


def _scan_dataset(path: Path) -> _DatasetStats:
    num_sets = 0
    total_tokens = 0
    with path.open("r", encoding="utf-8") as fh:
        for tokens in _iter_token_rows(fh):
            num_sets += 1
            total_tokens += len(tokens)
    return _DatasetStats(num_sets=num_sets, total_tokens=total_tokens)


def _iter_token_rows(lines: Iterable[str]) -> Iterator[np.ndarray]:
    header_skipped = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        size = int(parts[0])
        tokens = _parse_token_fields(parts[1:])
        if size != len(tokens):
            # The published datasets sometimes begin with a metadata line
            # containing dataset statistics. Allow skipping the first mismatch
            # but still fail fast on any later inconsistencies.
            if not header_skipped:
                header_skipped = True
                continue
            raise ValueError(
                f"Line claims {size} tokens but contains {len(tokens)} entries: '{stripped[:80]}'"
            )
        if size == 0:
            yield np.empty(0, dtype=np.int64)
        else:
            yield np.asarray(tokens, dtype=np.int64)


def _parse_token_fields(parts: Iterable[str]) -> list[int]:
    tokens: list[int] = []
    for part in parts:
        if part.startswith("#"):
            break
        for chunk in part.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            tokens.append(int(chunk))
    return tokens


def _memmap_paths(dataset_path: Path, dtype: str) -> CSRMemmap:
    base_dir = dataset_path.parent / "memmap" / dataset_path.stem
    base_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = base_dir / f"tokens.{dtype}.dat"
    indptr_path = base_dir / "indptr.int64.dat"
    metadata_path = base_dir / f"metadata.{dtype}.json"
    return CSRMemmap(
        tokens_path=tokens_path, indptr_path=indptr_path, metadata_path=metadata_path
    )
