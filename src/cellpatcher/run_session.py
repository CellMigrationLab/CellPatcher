"""One opened preprocessing run per CellPatcher call (layer: orchestration).

A CellPatcher call (``run_patching``, a folder or tree run) opens the
preprocessing run it reads with ``RunInputs.open`` once, and every step of
that call (the checks, each leaf's normalization, ``run_config.json``) reuses
it. ``RunInputs.open`` checks every image and mask of the run against the
manifest, and the brick checks each file again whenever it is read, so within
a call nothing is read unchecked.

The call also keeps what ``run_config.json`` records about the run (:func:`remember`): the
SHA-256 of its manifest when it was opened, whether it was read as a folder or tree run, and
each image's layout (:func:`remember_layouts`).

Nothing is kept beyond the call: the next call opens the run again, so an
image or mask changed in between, patched or not, is refused then (design
v0.4 §6; CP-2a). Before CP-2a an opened run was cached for the whole session,
keyed on the table and manifest only, and a mask or an unpatched image changed
after the first open was not checked again.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from cellpatcher.statistics import PlaneStatistics

_Layouts = dict[str, dict[str, str]]
_SESSION: ContextVar[dict[str, PlaneStatistics | str | _Layouts] | None] = ContextVar(
    "cellpatcher_run_session", default=None
)
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



def remember(kind: str, output_folder: str | Path, value: str) -> None:
    """Record *value* of *kind* (``"manifest_sha256"``, ``"file_run"``) for the run in *output_folder*, for this call.

    Outside :func:`statistics_session` nothing is kept.
    """
    session = _SESSION.get()
    if session is not None:
        session[f"{kind}:{Path(output_folder).expanduser().resolve()}"] = value


def recalled(kind: str, output_folder: str | Path) -> str | None:
    """Return the value of *kind* recorded in this call for the run in *output_folder*, or None."""
    session = _SESSION.get()
    if session is None:
        return None
    value = session.get(f"{kind}:{Path(output_folder).expanduser().resolve()}")
    return value if isinstance(value, str) else None


def remember_layouts(output_folder: str | Path, images: Mapping[str, str], masks: Mapping[str, str]) -> None:
    """Record each image's layout and mask layout (keyed by image path) of the file run in *output_folder*.

    Outside :func:`statistics_session` nothing is kept.
    """
    session = _SESSION.get()
    if session is not None:
        session[f"layouts:{Path(output_folder).expanduser().resolve()}"] = {
            path: {"image": layout, "mask": masks.get(path, "YX")} for path, layout in sorted(images.items())
        }


def recalled_layouts(output_folder: str | Path) -> dict[str, dict[str, str]] | None:
    """Return the layouts :func:`remember_layouts` recorded in this call for the run in *output_folder*, or None."""
    session = _SESSION.get()
    if session is None:
        return None
    value = session.get(f"layouts:{Path(output_folder).expanduser().resolve()}")
    return value if isinstance(value, dict) else None
