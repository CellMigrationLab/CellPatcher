"""One opened preprocessing run per CellPatcher call (layer: orchestration).

A CellPatcher call (``run_patching``, a folder or tree run) opens the
preprocessing run it reads with ``RunInputs.open`` once, and every step of
that call (the checks, each leaf's normalization, ``run_config.json``) reuses
it. ``RunInputs.open`` checks every image and mask of the run against the
manifest, and the brick checks each file again whenever it is read, so within
a call nothing is read unchecked.

Nothing is kept beyond the call: the next call opens the run again, so an
image or mask changed in between, patched or not, is refused then (design
v0.4 §6; CP-2a). Before CP-2a an opened run was cached for the whole session,
keyed on the table and manifest only, and a mask or an unpatched image changed
after the first open was not checked again.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from cellpatcher.statistics import PlaneStatistics

_SESSION: ContextVar[dict[str, PlaneStatistics] | None] = ContextVar("cellpatcher_run_session", default=None)
_P = ParamSpec("_P")
_R = TypeVar("_R")


@contextmanager
def statistics_session() -> Iterator[None]:
    """Keep the runs opened inside the block for the block only; a nested block shares the outer one."""
    if _SESSION.get() is not None:
        yield
        return
    token = _SESSION.set({})
    try:
        yield
    finally:
        _SESSION.reset(token)


def one_session(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run *function* inside :func:`statistics_session`."""

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with statistics_session():
            return function(*args, **kwargs)

    return wrapper


def session_statistics(csv_path: Path, load: Callable[[], PlaneStatistics]) -> PlaneStatistics:
    """Return the statistics of *csv_path* opened in this call, opening them with *load* if needed.

    Outside a session nothing is kept: *load* runs on every call.
    """
    session = _SESSION.get()
    if session is None:
        return load()
    key = str(csv_path.expanduser().resolve())
    if key not in session:
        session[key] = load()
    return session[key]


def keep_in_session(statistics: PlaneStatistics) -> None:
    """Make *statistics*, opened by a file run with its own roots, the ones its table gives in this call.

    Raises
    ------
    RuntimeError
        Outside :func:`statistics_session`.
    """
    session = _SESSION.get()
    if session is None:
        raise RuntimeError("keep_in_session needs an open statistics_session().")
    session[str(statistics.plane_histograms.expanduser().resolve())] = statistics
