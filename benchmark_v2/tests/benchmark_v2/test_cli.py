from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from benchmark_v2 import get_cli_app
from benchmark_v2.datasets import registry as dataset_registry
from benchmark_v2.datasets.registry import DatasetAsset, DatasetEntry, register_dataset


runner = CliRunner()


def _register_cli_dataset(tmp_path: Path) -> str:
    dataset_path = tmp_path / "cli_dataset.inp"
    dataset_path.write_text("1 42\n", encoding="utf-8")
    entry = DatasetEntry(
        name=f"cli_{uuid4().hex}",
        corpus=DatasetAsset(
            name="cli_corpus", url=dataset_path.as_uri(), compressed=False
        ),
    )
    register_dataset(entry, overwrite=True)
    return entry.name


def test_cli_list_shows_dataset(tmp_path: Path) -> None:
    dataset_name = _register_cli_dataset(tmp_path)
    app = get_cli_app()

    result = runner.invoke(app, ["datasets", "list"])
    assert result.exit_code == 0
    assert dataset_name in result.stdout


def test_cli_sync_downloads_and_memmaps(tmp_path: Path) -> None:
    dataset_name = _register_cli_dataset(tmp_path)
    app = get_cli_app()
    cache_root = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "datasets",
            "sync",
            dataset_name,
            "--cache-root",
            str(cache_root),
            "--no-queries",
            "--memmap",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert "memmap" in result.stdout


def test_cli_sync_all_uses_registry(monkeypatch, tmp_path: Path) -> None:
    dataset_name = _register_cli_dataset(tmp_path)
    entry = dataset_registry.get_dataset(dataset_name)
    monkeypatch.setattr(dataset_registry, "list_datasets", lambda: [entry])
    app = get_cli_app()
    cache_root = tmp_path / "cache_all"

    result = runner.invoke(
        app,
        [
            "datasets",
            "sync-all",
            "--cache-root",
            str(cache_root),
            "--no-queries",
            "--memmap",
            "--force",
            "--max-workers",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert f"=== {dataset_name} ===" in result.stdout
