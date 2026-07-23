"""Consistent low-overhead progress reporting for CLI and notebook use."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress_disabled(no_progress: bool = False) -> bool:
    """Disable animation explicitly or when stderr is not an interactive terminal."""
    return bool(no_progress or not sys.stderr.isatty())


def track_progress(
    iterable: Iterable[T], *, description: str, total: int | None = None,
    unit: str = "item", disabled: bool = False,
) -> Iterator[T]:
    """Iterate with a Windows-friendly tqdm bar configured in one place."""
    return iter(tqdm(
        iterable, desc=description, total=total, unit=unit,
        dynamic_ncols=True, mininterval=0.25, smoothing=0.1,
        disable=disabled, leave=True,
    ))


def create_progress(
    *, description: str, total: int | None = None, unit: str = "item",
    disabled: bool = False,
) -> tqdm:
    """Create a manually updated bar for loops that maintain counters."""
    return tqdm(
        total=total, desc=description, unit=unit, dynamic_ncols=True,
        mininterval=0.25, smoothing=0.1, disable=disabled, leave=True,
    )


def stage(message: str) -> None:
    """Emit an immediate stage message without buffering."""
    print(message, flush=True)
