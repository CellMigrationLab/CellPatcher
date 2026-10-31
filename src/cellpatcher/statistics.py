"""Link CellPatcher's planes to the plane statistics of the preprocessing brick.

Loads ``plane_histograms.csv`` and the optional ``run_manifest.json`` written by
`morphobricks-preprocessing <https://github.com/CellMigrationLab/morphobricks-preprocessing>`_
and runs the checks of ``docs/normalization_contract.md`` §5, which block a
run before any patch is normalized:

- :func:`load_plane_statistics` reads the table and the manifest (given, or
  found next to the table) and, with a manifest, checks ``csv_sha256``;
- :func:`check_images` reports every image without rows in one message and,
  with a manifest, checks each image's size, modification time and axes;
- :func:`check_planes` links each patched plane to its row and compares the
  labels, ``dtype``, pixel count, exact finite extrema and finite pixel count.

Reading, histograms, plane identifiers and messages all come from the
MorphoBricks bricks, core dependencies of CellPatcher: the plane table,
histograms and plane identifiers from ``morphobricks_normalization`` (where
the preprocessing brick moved them), images, the manifest and messages from
``morphobricks_preprocessing``. This module only links and compares. Each
function raises :class:`PreprocessingUnavailableError` when the installed
bricks read other file versions than this CellPatcher.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from cellpatcher.axes import normalize_axes
from cellpatcher.io import SINGLE_CONDITION_LABEL as SINGLE_LABEL

if TYPE_CHECKING:
    from morphobricks_normalization import PlaneRecord
    from morphobricks_preprocessing import AxisMapping, RunManifest, ValidationIssue

# The optional part of the preprocessing brick: its OME-Zarr reader (zarr).
# The brick itself is a core dependency (CP-6 of the D86 audit), so this extra,
# named ``preprocessing`` before, is now ``zarr``.
INSTALL_COMMAND = 'pip install "cellpatcher[zarr]"'

# Versions of plane_histograms.csv and run_manifest.json this CellPatcher reads
# (contract §1). The installed preprocessing package must read the same ones.
# Table version 4 has one row per XYC plane (an image at one Z and T), each
# channel's histogram in its own columns (with the underflow and overflow of
# the robust histogram range, which the brick's pooling and fitting use), and
# always writes z and t, so plane_id matches exactly; a channel is found
# inside its row. Version 3 lacked the underflow and overflow columns.
SUPPORTED_SCHEMA_VERSION = "4"
SUPPORTED_MANIFEST_VERSION = "1"

_PLANE_ROLES = ("Z", "C", "T")

_RERUN_ACTION = "Run the preprocessing brick again so its statistics describe the current images."


class PreprocessingUnavailableError(ImportError):
    """The ``zarr`` extra is missing for an OME-Zarr image, or a brick is installed in a version CellPatcher cannot use."""


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
    """Return the ``morphobricks_preprocessing`` package, checked for the file versions CellPatcher reads.

    Raises
    ------
    PreprocessingUnavailableError
        If the installed bricks read other versions of ``plane_histograms.csv``
        or ``run_manifest.json`` than this CellPatcher (an install other than
        the pinned tags).
    """
    import morphobricks_preprocessing
    from morphobricks_normalization.csv_io import SCHEMA_VERSION
    from morphobricks_preprocessing.manifest import MANIFEST_VERSION

    if (SCHEMA_VERSION, MANIFEST_VERSION) != (SUPPORTED_SCHEMA_VERSION, SUPPORTED_MANIFEST_VERSION):
        raise PreprocessingUnavailableError(
            f"The installed morphobricks-preprocessing reads plane_histograms.csv version "
            f"{SCHEMA_VERSION!r} and run_manifest.json version {MANIFEST_VERSION!r}, but this "
            f"CellPatcher reads version {SUPPORTED_SCHEMA_VERSION!r} of both. Reinstall CellPatcher, "
            "which installs the versions it is tested with."
        )
    return morphobricks_preprocessing


def require_preprocessing() -> ModuleType:
    """Return the ``morphobricks_preprocessing`` package (deprecated).

    The preprocessing brick is a core dependency of CellPatcher, so there is
    nothing to require: import ``morphobricks_preprocessing`` directly. This
    function warns and will be removed in CellPatcher v0.2.0.

    Raises
    ------
    PreprocessingUnavailableError
        If the installed bricks read other file versions than this CellPatcher.
    """
    warnings.warn(
        "cellpatcher.statistics.require_preprocessing is deprecated and will be removed in CellPatcher "
        "v0.2.0: morphobricks-preprocessing is a core dependency, import morphobricks_preprocessing directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _preprocessing()


def require_normalization() -> ModuleType:
    """Return the ``morphobricks_normalization`` package, checked like :func:`_preprocessing`.

    Plane histograms, the ``plane_histograms.csv`` reader, plane identifiers,
    pooling and normalization parameters moved there from the preprocessing
    brick, whose old import paths only warn and forward. CellPatcher declares
    it at the tag the preprocessing brick pins.

    Raises
    ------
    PreprocessingUnavailableError
        As :func:`_preprocessing`.
    """
    _preprocessing()
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

    A row (schema version 4) is an XYC plane holding one histogram per
    channel; CellPatcher links and pools one channel at a time.

    Attributes
    ----------
    record
        The row, a ``morphobricks_normalization.PlaneRecord``.
    c
        The zero-based channel.
    """

    record: PlaneRecord
    c: int

    @property
    def histogram(self):
        """The channel's ``PlaneHistogram``."""
        return self.record.channel_histogram(self.c)

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
        SHA-256 of the table.
    manifest
        The ``run_manifest.json`` used for the extra checks, or None.
    manifest_path
        Its path, or None.
    """

    plane_histograms: Path
    records: tuple[PlaneRecord, ...]
    csv_sha256: str
    manifest: RunManifest | None = None
    manifest_path: Path | None = None
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
        if record is None or not 0 <= plane.c < record.channel_count:
            return None
        return ChannelRecord(record, plane.c)

    def channel_records(self) -> Iterable[ChannelRecord]:
        """Every channel of every row, in table order."""
        for record in self.records:
            for c in range(record.channel_count):
                yield ChannelRecord(record, c)


def load_plane_statistics(plane_histograms, manifest=None) -> PlaneStatistics:
    """Read ``plane_histograms.csv`` and the optional ``run_manifest.json``.

    Parameters
    ----------
    plane_histograms
        Path of the table written by the preprocessing brick.
    manifest
        Path of its ``run_manifest.json``. When None, a manifest next to the
        table is used if there is one; without one, the manifest checks are
        skipped.

    Returns
    -------
    PlaneStatistics

    Raises
    ------
    PreprocessingUnavailableError
        If the installed bricks read other file versions than this CellPatcher.
    FileNotFoundError
        If the table, or an explicitly given manifest, does not exist.
    PreprocessingCheckError
        If the table or manifest cannot be read, has an unsupported version,
        has duplicate planes or no rows, or (with a manifest) the table is
        not the one the manifest was written with.
    """
    mbp = _preprocessing()
    mbn = require_normalization()
    from morphobricks_normalization.csv_io import CsvFormatError
    from morphobricks_preprocessing.manifest import file_sha256
    from morphobricks_preprocessing.validation import error, plural

    csv_path = Path(plane_histograms).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"The plane histogram table {csv_path} does not exist. Give the path of the "
            f"{mbn.OUTPUT_FILENAME} written by the preprocessing brick, or run the preprocessing "
            "brick on this experiment first."
        )
    try:
        records = tuple(mbn.read_plane_histograms(csv_path))
    except CsvFormatError as exc:
        raise PreprocessingCheckError([
            error(
                "unreadable_plane_histograms",
                f"The plane histogram table cannot be read: {exc}",
                [str(csv_path)],
                f"Give a {mbn.OUTPUT_FILENAME} written by the preprocessing brick (version "
                f"{SUPPORTED_SCHEMA_VERSION}), or run the preprocessing brick again.",
            )
        ]) from exc

    issues = []
    if not records:
        issues.append(error(
            "empty_plane_histograms",
            "The plane histogram table has no rows.",
            [str(csv_path)],
            "Run the preprocessing brick on this experiment again.",
        ))
    counts = defaultdict(int)
    for record in records:
        counts[record.plane_id] += 1
    duplicates = [plane_id for plane_id, count in counts.items() if count > 1]
    if duplicates:
        issues.append(error(
            "duplicate_plane_rows",
            f"The plane histogram table {csv_path.name} has several rows for "
            f"{plural(len(duplicates), 'plane')}, so they cannot be linked to "
            "their patches.",
            duplicates,
            "Use the table the preprocessing brick wrote, without combining or editing it.",
        ))

    csv_sha256 = file_sha256(csv_path)
    manifest_path = _manifest_path(csv_path, manifest, mbp.MANIFEST_FILENAME)
    run_manifest = None
    if manifest_path is not None:
        run_manifest = _read_manifest(mbp, manifest_path)
        if run_manifest.csv_sha256 != csv_sha256:
            issues.append(error(
                "plane_histograms_changed",
                f"{csv_path.name} is not the table {manifest_path.name} was written with: it was "
                "changed or replaced after the preprocessing run.",
                [str(csv_path), str(manifest_path)],
                "Give the table and manifest of the same preprocessing run, or run the "
                "preprocessing brick again.",
            ))
    if issues:
        raise PreprocessingCheckError(issues)
    return PlaneStatistics(
        plane_histograms=csv_path,
        records=records,
        csv_sha256=csv_sha256,
        manifest=run_manifest,
        manifest_path=manifest_path,
    )


def check_images(
    statistics: PlaneStatistics,
    image_axes: Mapping[str, str],
    *,
    experiment_root=None,
) -> None:
    """Run the per-image checks of contract §5.

    Every image must have rows in the table; all images without rows are
    reported in one message. With a manifest, each image with rows must also
    have the size and modification time recorded for it
    (``manifest.source_fingerprint``), and CellPatcher's axes must give every
    axis the role of the manifest's ``axis_mapping``.

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
    from morphobricks_preprocessing.manifest import source_fingerprint
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
        not_in_manifest, changed, axes_differ = [], [], []
        for path, axes in image_axes.items():
            if path not in known:
                continue
            entry = manifest.file(path)
            if entry is None:
                not_in_manifest.append(path)
                continue
            if source_fingerprint(root / path) != (entry.size, entry.mtime_ns):
                changed.append(path)
            # The preprocessing brick's own resolution of the experiment
            # mapping for this image (OME-Zarr images may store another axis
            # order); reads the header only.
            if path.lower().endswith(".zarr"):
                # CellPatcher takes an OME-Zarr image's axes by role from the
                # same metadata the preprocessing brick read (io.read_image).
                continue
            file_mapping = mbp.file_mapping_for(
                mbp.inspect_image(root / path, path),
                manifest.mapping,
                missing_as_singleton=manifest.settings.missing_as_singleton,
            )
            if not _axes_agree(normalize_axes(axes), file_mapping):
                preprocessing_roles = "none fit" if file_mapping is None else file_mapping.roles_string
                axes_differ.append(f"{path} (CellPatcher {normalize_axes(axes)}, preprocessing {preprocessing_roles})")
        manifest_name = statistics.manifest_path.name if statistics.manifest_path else "the manifest"
        if not_in_manifest:
            issues.append(error(
                "images_not_in_manifest",
                f"{plural(len(not_in_manifest), 'image')} with rows in {csv_name} "
                f"{'is' if len(not_in_manifest) == 1 else 'are'} missing from {manifest_name}.",
                not_in_manifest,
                "Give the table and manifest of the same preprocessing run.",
            ))
        if changed:
            issues.append(error(
                "images_changed",
                f"{plural(len(changed), 'image')} changed after the preprocessing brick read "
                f"{'it' if len(changed) == 1 else 'them'}: the size or modification time differs "
                f"from {manifest_name}.",
                changed,
                _RERUN_ACTION,
            ))
        if axes_differ:
            issues.append(error(
                "axis_mapping_mismatch",
                f"CellPatcher's axes give some axes of {plural(len(axes_differ), 'image')} another "
                f"role than the axis mapping confirmed in the preprocessing brick.",
                axes_differ,
                "Set common.image_axes to the roles confirmed in the preprocessing brick.",
            ))
    if issues:
        raise PreprocessingCheckError(issues)


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


def recorded_image_type(plane_histograms, manifest=None) -> str | None:
    """The image type the preprocessing run recorded in its ``run_manifest.json``.

    Reads only the manifest (not the table), so a notebook can show the type as
    soon as a table is chosen. ``normalization.image_type`` must equal it
    (normalization brick decision N34).

    Parameters
    ----------
    plane_histograms
        Path of ``plane_histograms.csv``; its manifest is the one next to it
        unless ``manifest`` is given.
    manifest
        Path of the ``run_manifest.json``, or None to look next to the table.

    Returns
    -------
    str or None
        ``"fluorescence"`` or ``"rgb"``; None without a manifest or when the
        manifest records no image type (written before the preprocessing brick
        asked for it), so the configuration decides.

    Raises
    ------
    PreprocessingUnavailableError
        If the installed bricks read other file versions than this CellPatcher.
    FileNotFoundError
        If an explicitly given manifest does not exist.
    PreprocessingCheckError
        If the manifest cannot be read or has an unsupported version.
    """
    mbp = _preprocessing()
    csv_path = Path(plane_histograms).expanduser()
    manifest_path = _manifest_path(csv_path, manifest, mbp.MANIFEST_FILENAME)
    if manifest_path is None:
        return None
    return getattr(_read_manifest(mbp, manifest_path), "image_type", None)


def _manifest_path(csv_path: Path, manifest, default_name: str) -> Path | None:
    if manifest is not None:
        path = Path(manifest).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"The run manifest {path} does not exist. Give the {default_name} written next to "
                f"{csv_path.name}, or leave the manifest out to skip its checks."
            )
        return path
    beside = csv_path.parent / default_name
    return beside if beside.is_file() else None


def _read_manifest(mbp: ModuleType, path: Path) -> RunManifest:
    from morphobricks_preprocessing.validation import error

    try:
        manifest = mbp.read_manifest(path)
    except mbp.ManifestFormatError as exc:
        raise PreprocessingCheckError([
            error(
                "unreadable_manifest",
                f"The run manifest cannot be read: {exc}",
                [str(path)],
                f"Give a {mbp.MANIFEST_FILENAME} written by the preprocessing brick (version "
                f"{SUPPORTED_MANIFEST_VERSION}), or leave the manifest out to skip its checks.",
            )
        ]) from exc
    if manifest.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise PreprocessingCheckError([
            error(
                "unsupported_schema_version",
                f"The run manifest describes a plane histogram table of version "
                f"{manifest.schema_version!r}; CellPatcher reads version {SUPPORTED_SCHEMA_VERSION!r}.",
                [str(path)],
                "Run the preprocessing brick again with the version CellPatcher is tested with.",
            )
        ])
    return manifest


def _axes_agree(image_axes: str, mapping: AxisMapping | None) -> bool:
    """Whether CellPatcher's axes give every dimension the manifest's role.

    A dimension without a role in the manifest has length 1 (preprocessing
    contract §2), so its planes have coordinate 0 whichever role CellPatcher
    gives it (contract §3), and any role fits it.
    """
    if mapping is None or len(mapping.roles) != len(image_axes):
        return False
    return all(role is None or role == axis for role, axis in zip(mapping.roles, image_axes))


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
