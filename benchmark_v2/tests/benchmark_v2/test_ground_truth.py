from __future__ import annotations

from pathlib import Path

from benchmark_v2.datasets import ground_truth


def test_iter_ground_truth_parses_rows(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text("1 2 0.9\n2 3\n", encoding="utf-8")

    rows = list(ground_truth.iter_ground_truth(gt_path))
    assert rows[0].score == 0.9
    assert rows[1].score is None
