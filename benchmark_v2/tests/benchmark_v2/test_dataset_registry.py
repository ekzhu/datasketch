from __future__ import annotations

from uuid import uuid4

from benchmark_v2.datasets import registry
from benchmark_v2.datasets.registry import DatasetAsset, DatasetEntry


def test_register_and_fetch_dataset() -> None:
    dataset_name = f"local_{uuid4().hex}"
    entry = DatasetEntry(
        name=dataset_name,
        corpus=DatasetAsset(
            name="corpus",
            url="https://example.com/corpus.inp.gz",
        ),
        queries={},
        tags=("test",),
    )
    registry.register_dataset(entry, overwrite=True)

    fetched = registry.get_dataset(dataset_name)
    assert fetched.name == dataset_name
    assert "test" in fetched.tags


def test_datasets_dict_returns_copy() -> None:
    snapshot = registry.datasets_dict()
    assert snapshot
    snapshot.clear()
    assert registry.datasets_dict()
