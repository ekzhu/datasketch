"""Dataset utilities for benchmark v2."""

from . import cache, memmap, registry
from .cache import DatasetCacheResult, ensure_dataset, get_cache_root
from .ground_truth import GroundTruthRecord, ensure_ground_truth, iter_ground_truth
from .memmap import CSRMemmap, build_memmap, ensure_memmap
from .registry import (
    DatasetAsset,
    DatasetEntry,
    datasets_dict,
    get_dataset,
    list_datasets,
    register_dataset,
)

__all__ = [
    "cache",
    "memmap",
    "registry",
    "DatasetAsset",
    "DatasetEntry",
    "DatasetCacheResult",
    "CSRMemmap",
    "ensure_dataset",
    "ensure_memmap",
    "build_memmap",
    "get_cache_root",
    "list_datasets",
    "get_dataset",
    "register_dataset",
    "datasets_dict",
    "GroundTruthRecord",
    "ensure_ground_truth",
    "iter_ground_truth",
]
