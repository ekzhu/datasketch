from __future__ import annotations

from pathlib import Path

from benchmark_v2.datasets import memmap


def test_build_memmap(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.inp"
    dataset_path.write_text("3 1 2 3\n1 9\n0\n", encoding="utf-8")

    csr = memmap.build_memmap(dataset_path, dtype="int32")
    loaded = csr.open()

    assert loaded.metadata["num_sets"] == 3
    assert loaded.metadata["num_tokens"] == 4

    rows = [row.tolist() for row in loaded.iter_rows()]
    assert rows == [[1, 2, 3], [9], []]

    # Ensure the memmaps are truly backed by files.
    assert csr.tokens_path.exists()
    assert csr.indptr_path.exists()


def test_ensure_memmap_no_rebuild(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy2.inp"
    dataset_path.write_text("1 1\n", encoding="utf-8")

    csr = memmap.ensure_memmap(dataset_path, dtype="int16", force=True)
    mtime_before = csr.tokens_path.stat().st_mtime
    csr_again = memmap.ensure_memmap(dataset_path, dtype="int16", force=False)
    assert csr_again.tokens_path.stat().st_mtime == mtime_before


def test_memmap_skips_header_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "with_header.inp"
    dataset_path.write_text("3 100 200\n2 5 6\n1 9\n", encoding="utf-8")

    csr = memmap.build_memmap(dataset_path)
    loaded = csr.open()
    rows = [row.tolist() for row in loaded.iter_rows()]
    assert rows == [[5, 6], [9]]


def test_memmap_handles_comma_delimited_tokens(tmp_path: Path) -> None:
    dataset_path = tmp_path / "comma.inp"
    dataset_path.write_text("2 1,2\n3 3,4,5\n", encoding="utf-8")

    csr = memmap.build_memmap(dataset_path)
    loaded = csr.open()
    rows = [row.tolist() for row in loaded.iter_rows()]
    assert rows == [[1, 2], [3, 4, 5]]
