from __future__ import annotations

import gzip
from pathlib import Path
from uuid import uuid4

from benchmark_v2.datasets import cache, registry
from benchmark_v2.datasets.registry import DatasetAsset, DatasetEntry


def _register_local_dataset(
    tmp_path: Path, *, compressed: bool
) -> tuple[DatasetEntry, Path]:
    corpus_source = tmp_path / ("corpus.inp.gz" if compressed else "corpus.inp")
    raw_payload = b"3 1 2 3\n1 99\n"
    if compressed:
        with gzip.open(corpus_source, "wb") as fh:
            fh.write(raw_payload)
    else:
        corpus_source.write_bytes(raw_payload)

    query_source = tmp_path / "query.inp"
    query_source.write_text("1 7\n", encoding="utf-8")

    entry = DatasetEntry(
        name=f"cache_local_{uuid4().hex}",
        corpus=DatasetAsset(
            name="corpus",
            url=corpus_source.as_uri(),
            compressed=compressed,
        ),
        queries={
            "q": DatasetAsset(
                name="query",
                url=query_source.as_uri(),
                compressed=False,
            )
        },
    )
    registry.register_dataset(entry, overwrite=True)
    return entry, corpus_source


def test_cache_download_and_reuse(tmp_path: Path) -> None:
    entry, source_path = _register_local_dataset(tmp_path, compressed=False)
    cache_root = tmp_path / "cache"

    result = cache.ensure_dataset(entry.name, cache_root=cache_root)
    assert result.corpus_path.read_text() == "3 1 2 3\n1 99\n"
    assert result.query_paths["q"].read_text() == "1 7\n"

    # Modify source and ensure cache does not change unless forced.
    source_path.write_text("1 5\n", encoding="utf-8")
    cached_contents = result.corpus_path.read_text()

    result_again = cache.ensure_dataset(entry.name, cache_root=cache_root)
    assert result_again.corpus_path.read_text() == cached_contents

    # Force download reflects new contents.
    forced = cache.ensure_dataset(
        entry.name, cache_root=cache_root, force_download=True
    )
    assert forced.corpus_path.read_text() == "1 5\n"


def test_cache_handles_gzip(tmp_path: Path) -> None:
    entry, _ = _register_local_dataset(tmp_path, compressed=True)
    cache_root = tmp_path / "cache"

    result = cache.ensure_dataset(entry.name, cache_root=cache_root)
    assert result.corpus_path.suffix == ".inp"
    assert result.corpus_path.read_text() == "3 1 2 3\n1 99\n"
