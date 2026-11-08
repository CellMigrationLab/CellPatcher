"""The per-position table ``positions.csv``: one row per FOV, grouping its source images (layer: io).

``images.csv`` has one row per input image; a position split over several time-point files has
several. This module groups the :class:`~cellpatcher.types.ImageRecord` of every source image,
plane and channel of one FOV key and writes the rows. It imports nothing of the other bricks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from cellpatcher.types import no_patch_reason

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cellpatcher.types import ImageRecord

# The per-position table written beside every images.csv (see save_position_table_csv).
POSITION_TABLE_FILENAME = "positions.csv"

# Version of the per-position table (``positions.csv``), written in every row's last column. It
# changes only when a column is renamed, removed or reinterpreted; appended columns keep it.
POSITION_TABLE_VERSION = "1"

# Statuses of a position (:attr:`PositionRecord.status`): what CellPatcher observed of it.
POSITION_STATUSES = (
    "analysed",  # at least one patch
    "analysed_no_regions",  # patched, and the patch mode generated no region
    "all_regions_filtered",  # regions were generated, the filters removed all
    "all_rejected_by_normalization",  # regions were kept, normalization discarded all
    "skipped",  # every source image had nothing to patch (skip_reason)
    "partly_skipped",  # some source images were skipped, the rest are counted
)

# Columns of the per-position table (``positions.csv``), written beside ``images.csv``. The first
# four are the canonical FOV key in the order of the preprocessing run's ``Fov.key`` and with the
# names ``patches.csv`` uses, so the two tables join on them; the lists are JSON arrays.
POSITION_TABLE_COLUMNS = (
    "condition",
    "biological_repeat",
    "technical_repeat",
    "position_id",
    "repeat_group",
    "condition_id",
    "technical_repeat_id",
    "n_source_images",
    "source_relative_paths",
    "image_files",
    "mask_files",
    "status",
    "reason",
    "n_regions",
    "n_kept",
    "n_normalized",
    "n_patches",
    "schema_version",
)


@dataclass
class PositionRecord:
    """One position (FOV) of a run and what became of all its source images: a row of ``positions.csv``.

    The key is the run's FOV key, ``(condition, biological_repeat, technical_repeat,
    position_id)`` (``Fov.key`` of a preprocessing run); ``repeat_group``,
    ``condition_id`` and ``technical_repeat_id`` are the run's canonical IDs when it has a
    description, else None. The counts are the sums of the position's :class:`ImageRecord`
    counts. ``source_relative_paths``, ``image_files`` and ``mask_files`` list the source
    images (provenance only, never the key). ``status`` is one of :data:`POSITION_STATUSES`,
    from the counts and from which source images were skipped.
    """

    condition: str | None
    biological_repeat: str | None
    technical_repeat: str | None
    position_id: str | None
    repeat_group: str | None = None
    condition_id: str | None = None
    technical_repeat_id: str | None = None
    source_relative_paths: list[str] = field(default_factory=list)
    image_files: list[str] = field(default_factory=list)
    mask_files: list[str] = field(default_factory=list)
    n_source_images: int = 0
    n_skipped: int = 0
    skip_reasons: list[str] = field(default_factory=list)
    n_regions: int = 0
    n_kept: int = 0
    n_normalized: int = 0
    n_patches: int = 0

    @property
    def status(self) -> str:
        """One of :data:`POSITION_STATUSES`, from the counts and the skipped source images."""
        if self.n_skipped == self.n_source_images:
            return "skipped"
        if self.n_patches > 0:
            return "partly_skipped" if self.n_skipped else "analysed"
        if self.n_skipped:
            return "partly_skipped"
        if self.n_regions == 0:
            return "analysed_no_regions"
        if self.n_kept == 0:
            return "all_regions_filtered"
        return "all_rejected_by_normalization"

    @property
    def reason(self) -> str | None:
        """Why the position is not simply analysed with patches; ``None`` when it is."""
        skipped = ""
        if self.n_skipped:
            skipped = f"{self.n_skipped} of {self.n_source_images} source image(s) skipped: " + "; ".join(
                dict.fromkeys(self.skip_reasons)
            )
        if self.n_skipped == self.n_source_images:
            return skipped
        counted = no_patch_reason(self.n_regions, self.n_kept, self.n_normalized, self.n_patches)
        return "; ".join(part for part in (skipped, counted) if part) or None

    def to_dict(self) -> dict:
        """Return the ``positions.csv`` row: :data:`POSITION_TABLE_COLUMNS` only (lists as JSON arrays)."""
        row = {column: getattr(self, column, None) for column in POSITION_TABLE_COLUMNS}
        for column in ("source_relative_paths", "image_files", "mask_files"):
            row[column] = json.dumps(getattr(self, column))
        row["status"], row["reason"], row["schema_version"] = self.status, self.reason, POSITION_TABLE_VERSION
        return row


def _own(value: str | None, existing: str | None, what: str, record: PositionRecord) -> str | None:
    """Keep the first ID of a position; refuse a source image that names another."""
    if value is not None and existing is not None and value != existing:
        raise ValueError(
            f"Source images of position {record.position_id!r} (technical repeat "
            f"{record.technical_repeat!r}) name two {what} values, {existing!r} and {value!r}."
        )
    return existing if value is None else value


def position_records(image_records: Iterable[ImageRecord]) -> list[PositionRecord]:
    """One :class:`PositionRecord` per FOV key of *image_records*, in order of first appearance.

    The key is ``(condition, biological_repeat, technical_repeat, position_id)``: images of one
    position (told apart by their T/Z/C planes) are grouped, equal basenames of different
    wells or repeats stay apart, and no label, ``image_id`` or file name stands in for it. A
    record with no ``position_id`` (only from hand-built patches) is its own row with an empty
    one: it is never merged with another.
    """
    positions: dict[tuple, PositionRecord] = {}
    for index, image in enumerate(image_records):
        key = (image.condition, image.biological_repeat, image.technical_repeat, image.position_id)
        if image.position_id is None:
            key = (*key, index)
        record = positions.get(key)
        if record is None:
            record = positions[key] = PositionRecord(*key[:4])
        for name in ("repeat_group", "condition_id", "technical_repeat_id"):
            setattr(record, name, _own(getattr(image, name), getattr(record, name), name, record))
        record.n_source_images += 1
        for values, value in (
            (record.source_relative_paths, image.source_relative_path),
            (record.image_files, image.image_file),
            (record.mask_files, image.mask_file),
        ):
            if value is not None and value not in values:
                values.append(value)
        if image.skip_reason is not None:
            record.n_skipped += 1
            record.skip_reasons.append(image.skip_reason)
        for counter in ("n_regions", "n_kept", "n_normalized", "n_patches"):
            setattr(record, counter, getattr(record, counter) + getattr(image, counter))
    return list(positions.values())


def save_position_table_csv(image_records: Iterable[ImageRecord], csv_path: str | Path) -> None:
    """Write the per-position table: one ``positions.csv`` row per FOV of *image_records*.

    Groups the records of every source image (plane, channel or file) of one FOV key,
    ``(condition, biological_repeat, technical_repeat, position_id)``, into one row with the
    position's status, reason, source files and summed counts (:func:`position_records`,
    :data:`POSITION_TABLE_COLUMNS`). Positions CellPatcher never saw (failed or never
    discovered acquisitions) are not rows: it cannot observe them.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(POSITION_TABLE_COLUMNS))
        writer.writeheader()
        for record in position_records(image_records):
            writer.writerow(record.to_dict())


def position_table_beside(image_table_file: str | Path | None) -> str | None:
    """Return the ``positions.csv`` beside an ``images.csv`` path, or None for no table."""
    return None if image_table_file is None else str(Path(image_table_file).with_name(POSITION_TABLE_FILENAME))
