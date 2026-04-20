"""Typer-based CLI for benchmark v2."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import typer

from .datasets import cache, memmap, registry

app = typer.Typer(help="Benchmark v2 orchestration")
datasets_app = typer.Typer(help="Manage benchmark datasets")
app.add_typer(datasets_app, name="datasets")


@datasets_app.command("list")
def list_datasets() -> None:
    """Display the datasets available for benchmarking."""

    entries = sorted(registry.list_datasets(), key=lambda entry: entry.name)
    for entry in entries:
        typer.echo(f"- {entry.name}: corpus={entry.corpus.url}")
        if entry.queries:
            query_names = ", ".join(entry.queries.keys())
            typer.echo(f"    queries: {query_names}")
        if entry.ground_truth:
            gt_names = ", ".join(entry.ground_truth.keys())
            typer.echo(f"    ground truth: {gt_names}")
        if entry.tags:
            typer.echo(f"    tags: {', '.join(entry.tags)}")
        if entry.notes:
            typer.echo(f"    notes: {entry.notes}")


def _sync_single_dataset(
    name: str,
    cache_root: Optional[Path],
    include_queries: bool,
    include_ground_truth: bool,
    dtype: str,
    build_memmap: bool,
    force_download: bool,
    force_memmap: bool,
    write: Callable[[str], None],
) -> None:
    """Ensure dataset assets are cached locally and optionally memmapped."""

    result = cache.ensure_dataset(
        name,
        cache_root=cache_root,
        include_queries=include_queries,
        include_ground_truth=include_ground_truth,
        force_download=force_download,
    )

    write(f"Corpus stored at {result.corpus_path}")
    if build_memmap:
        corpus_memmap = memmap.ensure_memmap(
            result.corpus_path, dtype=dtype, force=force_memmap
        )
        write(f"Corpus memmap ready at {corpus_memmap.tokens_path}")

    for query_name, path in result.query_paths.items():
        write(f"Query '{query_name}' stored at {path}")
        if build_memmap:
            q_memmap = memmap.ensure_memmap(path, dtype=dtype, force=force_memmap)
            write(f"  memmap: {q_memmap.tokens_path}")

    for gt_name, path in result.ground_truth_paths.items():
        write(f"Ground truth '{gt_name}' stored at {path}")


@datasets_app.command("sync")
def sync_dataset(
    name: str = typer.Argument(..., help="Dataset identifier"),
    cache_root: Optional[Path] = typer.Option(
        None, "--cache-root", help="Override cache directory"
    ),
    include_queries: bool = typer.Option(
        True, "--queries/--no-queries", help="Include query assets"
    ),
    include_ground_truth: bool = typer.Option(
        False,
        "--with-ground-truth/--without-ground-truth",
        help="Download ground-truth artifacts when available",
    ),
    dtype: str = typer.Option("int32", help="dtype for memmapped integer tokens"),
    build_memmap: bool = typer.Option(
        True, "--memmap/--no-memmap", help="Construct memmaps"
    ),
    force_download: bool = typer.Option(
        False, "--force", help="Force re-download of assets"
    ),
    force_memmap: bool = typer.Option(
        False, "--force-memmap", help="Rebuild memmaps from scratch"
    ),
) -> None:
    """Ensure dataset assets are cached locally and optionally memmapped."""

    _sync_single_dataset(
        name=name,
        cache_root=cache_root,
        include_queries=include_queries,
        include_ground_truth=include_ground_truth,
        dtype=dtype,
        build_memmap=build_memmap,
        force_download=force_download,
        force_memmap=force_memmap,
        write=typer.echo,
    )


@datasets_app.command("sync-all")
def sync_all_datasets(
    cache_root: Optional[Path] = typer.Option(
        None, "--cache-root", help="Override cache directory"
    ),
    include_queries: bool = typer.Option(
        True, "--queries/--no-queries", help="Include query assets"
    ),
    include_ground_truth: bool = typer.Option(
        False,
        "--with-ground-truth/--without-ground-truth",
        help="Download ground-truth artifacts when available",
    ),
    dtype: str = typer.Option("int32", help="dtype for memmapped integer tokens"),
    build_memmap: bool = typer.Option(
        True, "--memmap/--no-memmap", help="Construct memmaps"
    ),
    force_download: bool = typer.Option(
        False, "--force", help="Force re-download of assets"
    ),
    force_memmap: bool = typer.Option(
        False, "--force-memmap", help="Rebuild memmaps from scratch"
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        min=1,
        help="Maximum concurrent dataset downloads (default: number of datasets)",
    ),
) -> None:
    """Ensure every registered dataset is cached locally."""

    entries = sorted(registry.list_datasets(), key=lambda entry: entry.name.lower())
    if not entries:
        typer.echo("No datasets registered.")
        return

    default_workers = os.cpu_count() or 4
    worker_count = max_workers or min(len(entries), default_workers)
    results: dict[str, list[str]] = {}

    def _run(entry_name: str) -> tuple[str, list[str]]:
        buffer: list[str] = [f"=== {entry_name} ==="]
        _sync_single_dataset(
            name=entry_name,
            cache_root=cache_root,
            include_queries=include_queries,
            include_ground_truth=include_ground_truth,
            dtype=dtype,
            build_memmap=build_memmap,
            force_download=force_download,
            force_memmap=force_memmap,
            write=buffer.append,
        )
        return entry_name, buffer

    executor = ThreadPoolExecutor(max_workers=worker_count)
    future_map = {executor.submit(_run, entry.name): entry.name for entry in entries}
    try:
        for future in as_completed(future_map):
            entry_name, buffer = future.result()
            results[entry_name] = buffer
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    for entry in entries:
        for line in results.get(entry.name, []):
            typer.echo(line)
