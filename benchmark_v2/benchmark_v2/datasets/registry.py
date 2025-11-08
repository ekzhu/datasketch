"""Dataset registry for benchmark v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping


@dataclass(frozen=True, slots=True)
class DatasetAsset:
    """Single downloadable asset such as a corpus or query file."""

    name: str
    url: str
    checksum: str | None = None
    compressed: bool = True
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """Collection of assets that belong to a logical dataset."""

    name: str
    corpus: DatasetAsset
    queries: Mapping[str, DatasetAsset] = field(default_factory=dict)
    ground_truth: Mapping[str, DatasetAsset] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    notes: str | None = None


_DATASETS: MutableMapping[str, DatasetEntry] = {}


def _add_builtin(entry: DatasetEntry) -> None:
    _DATASETS[entry.name] = entry


# Core benchmark datasets from the original implementation that reference this repository:
# https://github.com/ekzhu/set-similarity-search-benchmarks
_add_builtin(
    DatasetEntry(
        name="kosarak",
        corpus=DatasetAsset(
            name="kosarak",
            url="https://storage.googleapis.com/set-similarity-search/KOSARAK_dup_dr.inp.gz",
            description="Frequent itemset dataset from the FIMI repository",
        ),
        tags=("sketches", "indexes"),
    )
)
_add_builtin(
    DatasetEntry(
        name="flickr",
        corpus=DatasetAsset(
            name="flickr",
            url="https://storage.googleapis.com/set-similarity-search/FLICKR-london2y_dup_dr.inp.gz",
            description="Flickr London geotagged data",
        ),
        tags=("sketches",),
    )
)
_add_builtin(
    DatasetEntry(
        name="netflix",
        corpus=DatasetAsset(
            name="netflix",
            url="https://storage.googleapis.com/set-similarity-search/NETFLIX_dup_dr.inp.gz",
            description="Netflix prize style implicit feedback dataset",
        ),
        tags=("sketches", "indexes"),
    )
)
_add_builtin(
    DatasetEntry(
        name="orkut",
        corpus=DatasetAsset(
            name="orkut",
            url="https://storage.googleapis.com/set-similarity-search/orkut_ge10.inp.gz",
            description="MPI-SWS Orkut social network graph",
        ),
        tags=("indexes",),
    )
)
_add_builtin(
    DatasetEntry(
        name="canada_us_uk_opendata",
        corpus=DatasetAsset(
            name="canada_us_uk_opendata",
            url="https://storage.googleapis.com/set-similarity-search/canada_us_uk_opendata.inp.gz",
            description="Canada/US/UK open data collection",
        ),
        queries={
            "1k": DatasetAsset(
                name="canada_us_uk_opendata_queries_1k",
                url="https://storage.googleapis.com/set-similarity-search/canada_us_uk_opendata_queries_1k.inp.gz",
            ),
            "10k": DatasetAsset(
                name="canada_us_uk_opendata_queries_10k",
                url="https://storage.googleapis.com/set-similarity-search/canada_us_uk_opendata_queries_10k.inp.gz",
            ),
            "100k": DatasetAsset(
                name="canada_us_uk_opendata_queries_100k",
                url="https://storage.googleapis.com/set-similarity-search/canada_us_uk_opendata_queries_100k.inp.gz",
            ),
        },
        tags=("sketches", "indexes"),
    )
)
_add_builtin(
    DatasetEntry(
        name="wdc_webtables_2015",
        corpus=DatasetAsset(
            name="wdc_webtables_2015_english_relational",
            url="https://storage.googleapis.com/set-similarity-search/wdc_webtables_2015_english_relational.inp.gz",
            description="WDC Web Tables 2015 English relational subset",
        ),
        queries={
            "100": DatasetAsset(
                name="wdc_webtables_2015_english_relational_queries_100",
                url="https://storage.googleapis.com/set-similarity-search/wdc_webtables_2015_english_relational_queries_100.inp.gz",
            ),
            "1k": DatasetAsset(
                name="wdc_webtables_2015_english_relational_queries_1k",
                url="https://storage.googleapis.com/set-similarity-search/wdc_webtables_2015_english_relational_queries_1k.inp.gz",
            ),
            "10k": DatasetAsset(
                name="wdc_webtables_2015_english_relational_queries_10k",
                url="https://storage.googleapis.com/set-similarity-search/wdc_webtables_2015_english_relational_queries_10k.inp.gz",
            ),
        },
        tags=("indexes",),
    )
)


def list_datasets() -> Iterable[DatasetEntry]:
    """Return the currently registered datasets."""

    return tuple(_DATASETS.values())


def get_dataset(name: str) -> DatasetEntry:
    """Look up a dataset by name, raising KeyError when missing."""

    try:
        return _DATASETS[name]
    except KeyError as exc:  # pragma: no cover - defensive detail
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {', '.join(sorted(_DATASETS))}"
        ) from exc


def register_dataset(entry: DatasetEntry, *, overwrite: bool = False) -> None:
    """Register an additional dataset entry (used by tests and extensions)."""

    if not overwrite and entry.name in _DATASETS:
        raise ValueError(f"Dataset '{entry.name}' already registered")
    _DATASETS[entry.name] = entry


def datasets_dict() -> Dict[str, DatasetEntry]:
    """Return a shallow copy of the dataset mapping for read-only consumers."""

    return dict(_DATASETS)
