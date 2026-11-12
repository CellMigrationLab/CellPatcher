"""Link CellPatcher's planes to the plane statistics of the preprocessing brick.

Reads a clean preprocessing run through the brick's ``RunInputs`` (design
v0.4 §6) and runs the checks of ``docs/normalization_contract.md`` §5, which
block a run before any patch is normalized:

- :func:`load_plane_statistics` (from :mod:`cellpatcher.run_open`) opens the
  run that wrote ``plane_histograms.csv`` with ``RunInputs.open``, which
  refuses a run before v0.4.0, a table or design whose SHA-256 is not the
  manifest's, and an image or mask changed since the run;
- :func:`check_images` reports every image without rows in one message and
  checks each image's size, modification time and axes against the manifest;
- :func:`check_planes` links each patched plane to its row and compares the
  labels, ``dtype``, pixel count, exact finite extrema and finite pixel count.

Reading, histograms, plane identifiers and messages all come from the
MorphoBricks bricks, core dependencies of CellPatcher: the plane table,
histograms and plane identifiers from ``morphobricks_normalization`` (where
the preprocessing brick moved them), the run, images and messages from
``morphobricks_preprocessing``. This module only links and compares.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from cellpatcher import run_open as _run_open
from cellpatcher.axes import normalize_axes
from cellpatcher.io import SINGLE_CONDITION_LABEL as SINGLE_LABEL

# Opening the run moved to cellpatcher.run_open (CP-2a); re-exported here.
load_plane_statistics = _run_open.load_plane_statistics
open_run = _run_open.open_run
recorded_image_type = _run_open.recorded_image_type
statistics_of = _run_open.statistics_of

if TYPE_CHECKING:
    from morphobricks_normalization import PlaneRecord
    from morphobricks_preprocessing import RunManifest, ValidationIssue
    from morphobricks_preprocessing.run_inputs import RunInputs

# The optional part of the preprocessing brick: its OME-Zarr reader (zarr).
# The brick itself is a core dependency (CP-6 of the D86 audit), so this extra,
# named ``preprocessing`` before, is now ``zarr``.
INSTALL_COMMAND = 'pip install "cellpatcher[zarr]"'

_PLANE_ROLES = ("Z", "C", "T")

_RERUN_ACTION = "Run the preprocessing brick again so its statistics describe the current images."


class PreprocessingUnavailableError(ImportError):
    """The ``zarr`` extra is missing for an OME-Zarr image."""


class PreprocessingCheckError(ValueError):
    """A check of the preprocessing outputs failed, so the run is blocked.

    Attributes
    ----------
    issues
        The errors found, as ``morphobricks_preprocessing.ValidationIssue``
        objects: what happened, the affected paths or planes, and a suggested
        action. The message lists all of them.
    """

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


def _preprocessing() -> ModuleType:
    """Return the ``morphobricks_preprocessing`` package.

    The file versions it reads are its own: ``RunInputs`` refuses a run it
    cannot use (design v0.4 §6), so CellPatcher pins no table or manifest
    version of its own.
    """
    import morphobricks_preprocessing

    return morphobricks_preprocessing


def require_normalization() -> ModuleType:
    """Return the ``morphobricks_normalization`` package.

    Plane histograms, the ``plane_histograms.csv`` reader, plane identifiers,
    pooling and normalization parameters moved there from the preprocessing
    brick. CellPatcher declares it at the tag the preprocessing brick pins.
    """
    import morphobricks_normalization

    return morphobricks_normalization


@dataclass(frozen=True)
class SourcePlane:
    """One 2D plane of a source image that CellPatcher patches.

    Attributes
    ----------
    source_relative_path
        The image's path relative to the experiment root, with ``/``
        separators (``PatchRegion.source_relative_path``).
    condition, biological_repeat, technical_repeat
        CellPatcher's labels for the image; None is read as ``"single"``.
    z, c, t
        Zero-based plane coordinates (CellPatcher's ``z0`` as ``z``). An axis
        the image does not have, or has with length 1, has coordinate 0; None
        is stored as 0, as in the preprocessing table (contract §3). A patch
        that keeps several channels is one ``SourcePlane`` per channel; its
        row in the table holds every channel (schema version 4).
    axis_lengths
        Length of each Z, C and T axis the image has, e.g. ``{"Z": 5}``. An
        axis left out has length 1. None when the lengths are not known (for
        a plane already split from its image): the coordinates are then
        taken as given, None as 0.
    position_id
        CellPatcher's ``position_id`` for the image, compared with the row's
        by :func:`check_planes`; None when not known (not compared).

    Raises
    ------
    ValueError
        If a coordinate is missing for an axis longer than 1, or lies outside
        its axis.
    """

    source_relative_path: str
    condition: str | None
    biological_repeat: str | None
    technical_repeat: str | None
    # None means "not given": allowed only for an axis of length 1, then stored as 0.
    z: int | None = None
    c: int | None = None
    t: int | None = None
    axis_lengths: Mapping[str, int] | None = field(default_factory=dict)
    position_id: str | None = None

    def __post_init__(self) -> None:
        # A missing label is "single", as everywhere else in CellPatcher.
        for name in ("condition", "biological_repeat", "technical_repeat"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, SINGLE_LABEL)
        unknown = set(self.axis_lengths or {}) - set(_PLANE_ROLES)
        if unknown:
            raise ValueError(f"axis_lengths may only name the axes {_PLANE_ROLES}; got {sorted(unknown)}.")
        for role in _PLANE_ROLES:
            value = self.coordinate(role)
            if self.axis_lengths is None:
                if value is None:
                    object.__setattr__(self, role.lower(), 0)
                elif value < 0:
                    raise ValueError(f"{self.source_relative_path}: {role.lower()}={value} is negative.")
                continue
            length = self.axis_length(role)
            if value is None and length != 1:
                raise ValueError(
                    f"{self.source_relative_path}: a {role.lower()} coordinate is needed, because "
                    f"the image has {length} planes along {role}."
                )
            if value is not None and not 0 <= value < length:
                raise ValueError(
                    f"{self.source_relative_path}: {role.lower()}={value} is outside the image, "
                    f"which has {length} planes along {role}."
                )
            if value is None:
                object.__setattr__(self, role.lower(), 0)

    def coordinate(self, role: str) -> int:
        """The coordinate along ``role`` (``"Z"``, ``"C"`` or ``"T"``)."""
        return {"Z": self.z, "C": self.c, "T": self.t}[role]

    def axis_length(self, role: str) -> int:
        """The image's length along ``role``; 1 for an axis it does not have."""
        return int((self.axis_lengths or {}).get(role, 1))

    @property
    def labels(self) -> tuple[str | None, str | None, str | None]:
        """``(condition, biological_repeat, technical_repeat)``."""
        return (self.condition, self.biological_repeat, self.technical_repeat)

    @property
    def plane_id(self) -> str:
        """The preprocessing ``plane_id`` of this plane's row, e.g. ``"control/img.tif::z=0,t=0"``.

        The row is the XYC plane: every channel of the image at this Z and T.
        """
        return require_normalization().make_plane_id(self.source_relative_path, z=self.z, t=self.t)

    @property
    def name(self) -> str:
        """The plane and its channel, as named in messages, e.g. ``"img.tif::z=0,t=0 c=1"``."""
        return f"{self.plane_id} c={self.c}"


@dataclass(frozen=True)
class ChannelRecord:
    """One channel of one row of ``plane_histograms.csv``.

    A row (schema version 5) is an XYC plane holding one histogram per
    channel, named by canonical channel ID: the row's ``channels`` lists the IDs in
    source order, entry ``i`` naming source channel ``C<i>`` (ADR 0003 HV1). CellPatcher
    links one channel at a time by its C index and pools it by its ID.

    Attributes
    ----------
    record
        The row, a ``morphobricks_normalization.PlaneRecord``.
    c
        The zero-based channel (C index in the source image).
    """

    record: PlaneRecord
    c: int

    @property
    def channel(self) -> str:
        """The channel's canonical ID: the row's ``channels`` entry at its C index."""
        return self.record.channels[self.c]

    @property
    def histogram(self):
        """The channel's ``PlaneHistogram``."""
        return self.record.channel_histogram(self.channel)

    @property
    def source_relative_path(self) -> str:
        return self.record.source_relative_path

    @property
    def condition(self) -> str:
        return self.record.condition

    @property
    def biological_repeat(self) -> str:
        return self.record.biological_repeat

    @property
    def technical_repeat(self) -> str:
        return self.record.technical_repeat

    @property
    def z(self) -> int:
        return self.record.z

    @property
    def t(self) -> int:
        return self.record.t

    @property
    def plane_id(self) -> str:
        """The row's ``plane_id``."""
        return self.record.plane_id

    def group_key(self, level: str) -> tuple[str, ...]:
        """The row's labels at ``level`` (``PlaneRecord.group_key``)."""
        return self.record.group_key(level)


@dataclass(frozen=True, eq=False)
class PlaneStatistics:
    """The preprocessing outputs of one experiment, as loaded by :func:`load_plane_statistics`.

    Attributes
    ----------
    plane_histograms
        Path of ``plane_histograms.csv``.
    records
        Its rows, as ``morphobricks_preprocessing.PlaneRecord`` objects.
    csv_sha256
        SHA-256 of the table, as the run's manifest records it.
    manifest
        The run's ``run_manifest.json``, or None (only for statistics built
        without a run, to link rows).
    manifest_path
        Its path, or None.
    inputs
        The run, opened with the preprocessing brick's ``RunInputs``, or None
        as for ``manifest``.
    """

    plane_histograms: Path
    records: tuple[PlaneRecord, ...]
    csv_sha256: str
    manifest: RunManifest | None = None
    manifest_path: Path | None = None
    inputs: RunInputs | None = None
    _by_plane_id: dict[str, PlaneRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_plane_id", {record.plane_id: record for record in self.records})

    @property
    def source_relative_paths(self) -> frozenset[str]:
        """The images that have rows in the table."""
        return frozenset(record.source_relative_path for record in self.records)

    def records_for_image(self, source_relative_path: str) -> tuple[PlaneRecord, ...]:
        """All rows of one image, in table order."""
        return tuple(r for r in self.records if r.source_relative_path == source_relative_path)

    def find_record(self, plane: SourcePlane) -> ChannelRecord | None:
        """``plane``'s channel of the row with its ``plane_id``, or None if the table has neither."""
        record = self._by_plane_id.get(plane.plane_id)
        if record is None or not 0 <= plane.c < len(record.channels):
            return None
        return ChannelRecord(record, plane.c)

    def channel_records(self) -> Iterable[ChannelRecord]:
        """Every channel of every row, in table order."""
        for record in self.records:
            for c in range(len(record.channels)):
                yield ChannelRecord(record, c)


def check_images(
    statistics: PlaneStatistics,
    image_axes: Mapping[str, str],
    *,
    experiment_root=None,
) -> None:
    """Run the per-image checks of contract §5.

    Every image must have rows in the table; all images without rows are
    reported in one message. With a manifest, the images with rows go through
    the preprocessing brick's ``check_sources`` (D86, formerly this
    function's own checks): each is in the manifest, still exists, is
    unchanged by ``source_fingerprint`` (an OME-Zarr folder: its files' total
    size and newest time), its recorded ``position_id`` is the one its name
    gives, and CellPatcher's axes give every axis the role the run recorded
    for the file. A TIFF image's axes are ``image_axes``; an OME-Zarr image's
    are its own OME-NGFF axes (matched by role), so both are checked.

    Parameters
    ----------
    statistics
        From :func:`load_plane_statistics`.
    image_axes
        CellPatcher's axes of each image as read (``common.image_axes``,
        e.g. ``"ZYX"``), keyed by ``source_relative_path``.
    experiment_root
        The experiment root the paths are relative to; required when
        ``statistics`` has a manifest.

    Raises
    ------
    PreprocessingCheckError
        If any check fails; the message lists every failure.
    """
    mbp = _preprocessing()
    from morphobricks_preprocessing.validation import error, plural

    csv_name = statistics.plane_histograms.name
    known = statistics.source_relative_paths
    without_rows = [path for path in image_axes if path not in known]
    issues = []
    if without_rows:
        issues.append(error(
            "images_without_rows",
            f"{plural(len(without_rows), 'image')} found by CellPatcher "
            f"{'has' if len(without_rows) == 1 else 'have'} no rows in {csv_name}: excluded or "
            "skipped by the preprocessing brick, or added after it ran.",
            without_rows,
            "Run the preprocessing brick again on the same experiment root, remove these images "
            "from the CellPatcher run, or use the patch scope, which does not need the table.",
        ))

    manifest = statistics.manifest
    if manifest is not None:
        if experiment_root is None:
            raise ValueError("experiment_root is required to check images against run_manifest.json.")
        root = Path(experiment_root).expanduser()
        with_rows = [path for path in image_axes if path in known]
        axes = {path: _checked_axes(mbp, root, path, image_axes[path]) for path in with_rows}
        issues += mbp.check_sources(manifest, root, with_rows, image_axes=axes)
    if issues:
        raise PreprocessingCheckError(issues)


def _checked_axes(mbp: ModuleType, root: Path, path: str, axes: str) -> str:
    """The axes CellPatcher reads *path* with, one letter per stored dimension.

    A TIFF image is read in ``common.image_axes``. An OME-Zarr image is read
    in its own OME-NGFF axes, matched by role, so those are the axes to
    compare with the run's roles (an unreadable store gives no axes and fails
    the comparison).
    """
    if not path.lower().endswith(".zarr"):
        return normalize_axes(axes)
    header = mbp.inspect_image(root / path, path)
    return header.axes or ""


def check_planes(
    statistics: PlaneStatistics,
    planes: Iterable[tuple[SourcePlane, np.ndarray]],
) -> list[ChannelRecord]:
    """Link each patched plane to its row and run the per-plane checks of contract §5.

    For every plane, a row with its ``plane_id`` holding its channel must exist, with the same
    ``condition``, ``biological_repeat`` and ``technical_repeat`` (and ``position_id``, when both the
    plane and the row have one), the same
    ``dtype`` and ``total_pixel_count`` (Y × X), and the plane's exact finite
    minimum, maximum and finite pixel count equal to ``pixel_min``,
    ``pixel_max`` and ``valid_pixel_count``. Run :func:`check_images` first:
    it reports images without any rows in one message.

    Parameters
    ----------
    statistics
        From :func:`load_plane_statistics`.
    planes
        ``(plane, pixels)`` pairs, ``pixels`` being the plane's 2D (Y, X)
        array as CellPatcher patches it. May be a generator, so only one
        plane needs to be in memory at a time.

    Returns
    -------
    list
        The matched :class:`ChannelRecord` of each plane, in order.

    Raises
    ------
    PreprocessingCheckError
        If any check fails for any plane; the message lists every failure.
    ValueError
        If ``pixels`` is not a 2D array.
    """
    mbn = require_normalization()
    failures: dict[str, list[str]] = defaultdict(list)
    matched: list[ChannelRecord] = []
    for plane, pixels in planes:
        pixels = np.asarray(pixels)
        if pixels.ndim != 2:
            raise ValueError(
                f"{plane.name}: a plane must be a 2D (Y, X) array; got shape {pixels.shape}."
            )
        record = statistics.find_record(plane)
        if record is None:
            failures["missing_plane_rows"].append(plane.name)
            continue
        matched.append(record)
        histogram = record.histogram
        record_labels = (record.condition, record.biological_repeat, record.technical_repeat)
        if plane.labels != record_labels:
            failures["label_mismatch"].append(
                f"{plane.name} (CellPatcher {_labels_text(plane.labels)}, "
                f"table {_labels_text(record_labels)})"
            )
        # The FOV key (D85) must be the table's too. A row written before the
        # preprocessing brick filled position_id has none, and a plane given
        # without one is not compared: neither is guessed.
        record_position = record.record.position_id
        if plane.position_id is not None and record_position is not None and plane.position_id != record_position:
            failures["position_mismatch"].append(
                f"{plane.name} (CellPatcher {plane.position_id!r}, table {record_position!r})"
            )
        if pixels.dtype.name != histogram.dtype.name:
            failures["dtype_mismatch"].append(
                f"{plane.name} (CellPatcher {pixels.dtype.name}, table {histogram.dtype.name})"
            )
        elif pixels.size != histogram.total_pixel_count:
            failures["pixel_count_mismatch"].append(
                f"{plane.name} (CellPatcher {pixels.size:,} pixels, table "
                f"{histogram.total_pixel_count:,})"
            )
        else:
            # The histogram the table was written with gives its definitions
            # of the finite extrema and finite pixel count.
            current = mbn.compute_plane_histogram(pixels, bin_count=histogram.bin_count)
            if (current.pixel_min, current.pixel_max, current.valid_pixel_count) != (
                histogram.pixel_min, histogram.pixel_max, histogram.valid_pixel_count
            ):
                failures["pixel_values_changed"].append(plane.name)
    if failures:
        raise PreprocessingCheckError(_plane_issues(statistics.plane_histograms.name, failures))
    return matched


def _labels_text(labels) -> str:
    return "/".join("(none)" if label is None else label for label in labels)


def _plane_issues(csv_name: str, failures: Mapping[str, list[str]]) -> list[ValidationIssue]:
    from morphobricks_preprocessing.validation import error, plural

    texts = {
        "missing_plane_rows": (
            "{have} no row in {csv}, or a row without their channel. Their images were read with "
            "other axes, or changed, since the preprocessing run.",
            "Set common.image_axes to the roles confirmed in the preprocessing brick, or run the "
            "preprocessing brick again.",
        ),
        "label_mismatch": (
            "{have} other condition, biological repeat or technical repeat labels than their rows "
            "in {csv}.",
            "Give CellPatcher the experiment root that was given to the preprocessing brick, so "
            "both read the same folder levels.",
        ),
        "position_mismatch": (
            "{have} another position_id than their rows in {csv}.",
            (
                "Leave position_id to the image name, as the preprocessing brick names it, or give the "
                "position_id its rows record."
            ),
        ),
        "dtype_mismatch": ("{have} another pixel type than their rows in {csv}.", _RERUN_ACTION),
        "pixel_count_mismatch": ("{have} another size (Y × X) than their rows in {csv}.", _RERUN_ACTION),
        "pixel_values_changed": (
            "{have} another finite minimum, maximum or finite pixel count than their rows in {csv}: "
            "the images changed after the preprocessing run.",
            _RERUN_ACTION,
        ),
    }
    issues = []
    for code, (message, action) in texts.items():
        if failures.get(code):
            planes = failures[code]
            one = len(planes) == 1
            text = message.format(csv=csv_name, have="has" if one else "have")
            issues.append(error(code, f"{plural(len(planes), 'plane')} {text}", planes, action))
    return issues
