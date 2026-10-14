"""Progress bar helpers for benchmark tooling."""

from __future__ import annotations

import sys
from tqdm.auto import tqdm


def make_progress(
    *,
    desc: str,
    total: int | None = None,
    unit: str | None = None,
    unit_scale: bool = False,
    leave: bool = False,
) -> "tqdm":
    """Return a configured tqdm progress bar if the session supports it."""

    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        unit_scale=unit_scale,
        dynamic_ncols=True,
        leave=leave,
        disable=not sys.stderr.isatty(),
    )
