"""Write a run's files to a staging folder; move them into ``output_dir`` only when the run succeeds (layer: io).

Issue #107: a run that stops (a refusal, a failing image, a full disk) leaves no patch file
without the ``patches.csv`` and ``run_config.json`` that make it traceable. Inside
:func:`staged_output`, every file the run exports under ``output_dir`` is written to a hidden
staging folder in it (``output_dir/.cellpatcher-partial-<random>``, so on the same file system);
when the run returns, its files are moved to their places in ``output_dir`` (``os.replace``,
replacing a file of the same name as an export always did), and when it raises the staging
folder is deleted, with ``output_dir`` and its parents if the run created them, so ``output_dir``
is left as it was. Nothing is created until the run writes its first file.

The paths a run records (``PatchingResult.exported_files``, ``metadata_file``, the
``output_dir`` in ``run_config.json``) are those in ``output_dir``, never in the staging folder.
A run inside another's staging (each leaf of a tree run) writes to that run's staging folder.
"""

from __future__ import annotations

import os
import shutil
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

STAGING_PREFIX = ".cellpatcher-partial-"


@dataclass
class _Stage:
    """A run's ``output_dir`` (absolute), its staging folder once created, and the folders it created."""

    final: Path
    folder: Path | None = None
    created: list[Path] = field(default_factory=list)

    def staging_folder(self) -> Path:
        """Return the staging folder, creating it (and ``output_dir``) on first use."""
        if self.folder is None:
            self.created = [path for path in (self.final, *self.final.parents) if not path.exists()]
            self.final.mkdir(parents=True, exist_ok=True)
            self.folder = self.final / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
            self.folder.mkdir()
        return self.folder

    def discard(self) -> None:
        """Delete the staging folder and the empty folders the run created for it."""
        if self.folder is None:
            return
        # Never raised over the run's own error, which says why it stopped: a folder left is reported.
        shutil.rmtree(self.folder, ignore_errors=True)
        if self.folder.exists():
            warnings.warn(f"The stopped run's staging folder {self.folder} could not be deleted.", stacklevel=4)
            return
        for path in self.created:  # deepest first
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    def commit(self) -> None:
        """Move every staged file to its place in ``output_dir``, then delete the staging folder."""
        if self.folder is None:
            return
        for source in sorted(self.folder.rglob("*")):  # a folder before its files
            target = self.final / source.relative_to(self.folder)
            if source.is_dir():
                target.mkdir(exist_ok=True)
            else:
                os.replace(source, target)
        shutil.rmtree(self.folder)


_STAGE: ContextVar[_Stage | None] = ContextVar("cellpatcher_stage", default=None)


def _absolute(path: str | os.PathLike) -> Path:
    return Path(os.path.abspath(path))


def _enclosing(path: Path) -> _Stage | None:
    """Return the active stage whose ``output_dir`` holds *path*, or None."""
    stage = _STAGE.get()
    return stage if stage is not None and path.is_relative_to(stage.final) else None


@contextmanager
def staged_output(output_dir: str | os.PathLike | None) -> Iterator[None]:
    """Stage the files written under *output_dir* (:func:`staged_folder`) until the block succeeds.

    A no-op when *output_dir* is None or already inside an enclosing staged run.
    """
    if output_dir is None or _enclosing(_absolute(output_dir)) is not None:
        yield
        return
    stage = _Stage(_absolute(output_dir))
    token = _STAGE.set(stage)
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        _STAGE.reset(token)
        if not succeeded:
            stage.discard()
    stage.commit()


def staged_folder(folder: str | os.PathLike) -> Path:
    """Return where to write the files of *folder*, created: its place in the staging folder of a staged run.

    *folder* itself (created) outside :func:`staged_output`.
    """
    path = _absolute(folder)
    stage = _enclosing(path)
    target = path if stage is None else stage.staging_folder() / path.relative_to(stage.final)
    target.mkdir(parents=True, exist_ok=True)
    return target
