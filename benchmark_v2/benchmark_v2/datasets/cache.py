"""Dataset caching helpers."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping
from urllib import parse, request

from ..progress import make_progress
from .registry import DatasetAsset, DatasetEntry, get_dataset

CACHE_ROOT_ENV = "DATASKETCH_BENCHMARK_CACHE"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "datasketch" / "benchmark_v2"
_BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DatasetCacheResult:
    """Result of ensuring that a dataset exists in the local cache."""

    entry: DatasetEntry
    corpus_path: Path
    query_paths: Mapping[str, Path]
    ground_truth_paths: Mapping[str, Path]
    raw_paths: Mapping[str, Path]


class DatasetCacheError(RuntimeError):
    """Raised when caching fails."""


def get_cache_root(cache_root: Path | None = None) -> Path:
    """Return the cache root, honoring environment overrides."""

    if cache_root is not None:
        return cache_root
    override = os.environ.get(CACHE_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CACHE_ROOT


def ensure_dataset(
    dataset_name: str,
    *,
    cache_root: Path | None = None,
    include_queries: bool = True,
    include_ground_truth: bool = False,
    force_download: bool = False,
) -> DatasetCacheResult:
    """Ensure every requested artifact for *dataset_name* exists locally."""

    entry = get_dataset(dataset_name)
    root = get_cache_root(cache_root)
    dataset_dir = root / entry.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: Dict[str, Path] = {}
    corpus_path = _materialize_asset(
        entry.corpus, dataset_dir, raw_paths, force_download
    )

    query_paths: Dict[str, Path] = {}
    if include_queries:
        for query_name, asset in entry.queries.items():
            query_paths[query_name] = _materialize_asset(
                asset, dataset_dir, raw_paths, force_download
            )

    ground_truth_paths: Dict[str, Path] = {}
    if include_ground_truth:
        for gt_name, asset in entry.ground_truth.items():
            ground_truth_paths[gt_name] = _materialize_asset(
                asset, dataset_dir, raw_paths, force_download
            )

    return DatasetCacheResult(
        entry=entry,
        corpus_path=corpus_path,
        query_paths=query_paths,
        ground_truth_paths=ground_truth_paths,
        raw_paths=raw_paths,
    )


def _materialize_asset(
    asset: DatasetAsset,
    dataset_dir: Path,
    raw_paths: Dict[str, Path],
    force: bool,
) -> Path:
    url_basename = _filename_from_url(asset.url)
    download_path = dataset_dir / url_basename
    raw_paths[asset.name] = download_path

    if force or not download_path.exists():
        _stream_download(asset.url, download_path)
        _verify_checksum(download_path, asset.checksum)
    elif asset.checksum:
        _verify_checksum(download_path, asset.checksum)

    if asset.compressed:
        target_path = dataset_dir / download_path.with_suffix("").name
        if force or not target_path.exists():
            _decompress_gzip(download_path, target_path)
        return target_path
    return download_path


def _filename_from_url(url: str) -> str:
    parsed = parse.urlparse(url)
    if parsed.scheme in {"", "file"}:
        return Path(parsed.path).name
    return Path(parsed.path).name or "download.bin"


def _stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    with request.urlopen(url) as response, open(tmp_path, "wb") as fh:
        try:
            total = int(response.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            total = 0
        desc = f"Downloading {dest.name}"
        progress_total = total if total > 0 else None
        with make_progress(
            desc=desc, total=progress_total, unit="B", unit_scale=True
        ) as progress:
            while True:
                chunk = response.read(_BUFFER_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
                progress.update(len(chunk))
    shutil.move(tmp_path, dest)


def _verify_checksum(path: Path, checksum: str | None) -> None:
    if not checksum:
        return
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_BUFFER_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != checksum.lower():
        raise DatasetCacheError(
            f"Checksum mismatch for {path}. Expected {checksum}, got {digest.hexdigest()}"
        )


def _decompress_gzip(source: Path, dest: Path) -> None:
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    desc = f"Extracting {dest.name}"
    with gzip.open(source, "rb") as src, open(tmp_path, "wb") as target:
        with make_progress(desc=desc, unit="B", unit_scale=True) as progress:
            while True:
                chunk = src.read(_BUFFER_SIZE)
                if not chunk:
                    break
                target.write(chunk)
                progress.update(len(chunk))
    shutil.move(tmp_path, dest)
