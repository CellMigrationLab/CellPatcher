"""Normalize patches with the statistics of the preprocessing brick.

Implements ``docs/normalization_contract.md`` §6–§9 for ``config.normalization``:

- **Scopes.** ``"condition"``, ``"biological_repeat"``, ``"technical_repeat"``
  and ``"image"`` pool the rows of ``plane_histograms.csv`` of the patch's
  group (``group_records`` levels, with the channel, and with
  ``per_time_point`` the time point, added to the key); ``"patch"`` builds a
  histogram of the patch's own pixels (``"mask"``, ``"bbox"`` or ``"patch"``
  region), never including padding.
- **Methods.** ``min_max``, ``percentile_range``, ``z_score`` and
  ``robust_z_score`` of the normalization brick turn the pooled histogram
  into ``NormalizationParameters``; a patch is normalized per channel as
  ``(pixels - offset) / scale`` in float32, without clipping.
- **Checks.** Every patched plane is linked to its CSV row with
  :func:`cellpatcher.statistics.check_planes` before anything is normalized.

Histograms, pooling and parameters all come from ``morphobricks_normalization``
(installed by the ``preprocessing`` extra); this module chooses the pixels and
applies the result. A group or patch with no finite pixels or zero spread stops the run
(contract §7), and a patch whose own pixels have no finite values or zero spread
is discarded and reported (decision D8).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellpatcher.axes import normalize_axes
from cellpatcher.masking import _label_to_2d
from cellpatcher.statistics import (
    PlaneStatistics,
    SourcePlane,
    check_images,
    check_planes,
    load_plane_statistics,
    require_normalization,
    require_preprocessing,
)

CSV_SCOPES = ("condition", "biological_repeat", "technical_repeat", "image")
_GROUP_SCOPES = ("condition", "biological_repeat", "technical_repeat")
_STATISTICS_CACHE: dict[tuple, PlaneStatistics] = {}


class NormalizationWarning(UserWarning):
    """A normalization runs, but with something the user should know (contract §14)."""


def resolved_patch_region(config) -> str:
    """``normalization.patch_region``, with None resolved: ``"mask"`` in ``cell`` mode, else ``"patch"``."""
    region = config.normalization.patch_region
    if region is None:
        return "mask" if config.mode == "cell" else "patch"
    return region


def validate_normalization(config) -> None:
    """Check that ``config.normalization`` can run with this config (contract §13–§14).

    Raises
    ------
    cellpatcher.statistics.PreprocessingUnavailableError
        If normalization is enabled without the ``preprocessing`` extra.
    ValueError
        If a CSV scope has no ``plane_histograms``, or a ``"mask"``/``"bbox"``
        patch region is used outside ``cell`` mode.
    """
    settings = config.normalization
    if not settings.enabled:
        return
    require_preprocessing()
    if settings.scope in CSV_SCOPES and settings.plane_histograms is None:
        raise ValueError(
            f"normalization.scope={settings.scope!r} needs the plane_histograms.csv written by the "
            "preprocessing brick (normalization.plane_histograms). Run the preprocessing brick on this "
            "experiment and give its table, or use normalization.scope='patch', which does not need it."
        )
    region = resolved_patch_region(config)
    if settings.scope == "patch" and region in ("mask", "bbox") and config.mode != "cell":
        raise ValueError(
            f"normalization.patch_region={region!r} uses each patch's own label, which only mode='cell' "
            "has. Use patch_region='patch' (all pixels of the patch) or another scope."
        )


def cached_statistics(plane_histograms, manifest=None) -> PlaneStatistics:
    """:func:`~cellpatcher.statistics.load_plane_statistics`, loaded once per unchanged table.

    A tree run normalizes every leaf with the same table, which can be large;
    the cache is keyed by the table's path, size and modification time.
    """
    csv_path = Path(plane_histograms).expanduser()
    if not csv_path.is_file():
        return load_plane_statistics(csv_path, manifest)  # raises the plain-language error
    stat = csv_path.resolve().stat()
    key = (str(csv_path.resolve()), stat.st_size, stat.st_mtime_ns, None if manifest is None else str(manifest))
    if key not in _STATISTICS_CACHE:
        if len(_STATISTICS_CACHE) >= 4:
            _STATISTICS_CACHE.pop(next(iter(_STATISTICS_CACHE)))
        _STATISTICS_CACHE[key] = load_plane_statistics(csv_path, manifest)
    return _STATISTICS_CACHE[key]


def run_inputs(config) -> dict | None:
    """The ``normalization_inputs`` of ``run_config.json`` (contract §11).

    None when normalization is off or uses no CSV (the patch scope). Paths
    are not recorded: they are machine-specific.
    """
    settings = config.normalization
    if not settings.enabled or settings.scope not in CSV_SCOPES or settings.plane_histograms is None:
        return None
    mbp = require_preprocessing()
    mbn = require_normalization()
    statistics = cached_statistics(settings.plane_histograms, settings.manifest)
    manifest = statistics.manifest
    return {
        "plane_histograms_sha256": statistics.csv_sha256,
        "schema_version": mbn.csv_io.SCHEMA_VERSION,
        "manifest_used": manifest is not None,
        "manifest_version": None if manifest is None else manifest.manifest_version,
        "preprocessing_version": mbp.__version__,
    }


def format_numbers(values) -> str:
    """Per-channel numbers for ``patches.csv``: full precision (``repr``), separated by ``;``."""
    return ";".join(repr(float(value)) for value in values)


@dataclass(frozen=True)
class PatchNormalization:
    """The normalization of one patch: one set of parameters per channel.

    Attributes
    ----------
    method, scope
        The preprocessing method name (e.g. ``"percentile 1-99.8"``) and
        ``normalization.scope``.
    group
        The group key (contract §7); None for the patch scope, whose key
        needs the final ``patch_id`` (set by :func:`finish_patch_groups`).
    parameters
        ``NormalizationParameters`` per channel, in channel order.
    exact
        True when every channel's parameters are exact (no histogram
        approximation).
    """

    method: str
    scope: str
    group: str | None
    parameters: tuple
    exact: bool

    @classmethod
    def from_patch(cls, patch) -> PatchNormalization | None:
        """The normalization recorded in a patch's ``normalization_*`` fields, or None when there is none.

        The offsets and scales are read back exactly (they are written with
        ``repr``), so the result applies the same parameters; nothing is
        recomputed.
        """
        if not patch.normalization_offset:
            return None
        mbn = require_normalization()
        offsets = [float(value) for value in str(patch.normalization_offset).split(";")]
        scales = [float(value) for value in str(patch.normalization_scale).split(";")]
        if len(offsets) != len(scales):
            raise ValueError(f"{patch.patch_id}: {len(offsets)} normalization offsets but {len(scales)} scales.")
        method = patch.normalization_method or ""
        parameters = tuple(mbn.NormalizationParameters(method, o, s) for o, s in zip(offsets, scales))
        exact = patch.normalization_exact in (True, "true", "True")
        return cls(method, patch.normalization_scope or "", patch.normalization_group, parameters, exact)

    def unapply(self, array, axes: str) -> np.ndarray:
        """Map normalized values back to raw units per channel, as float32: ``values * scale + offset``."""
        array = np.asarray(array)
        axes = normalize_axes(axes, array.ndim)
        if "C" not in axes:
            only = self.parameters[0]
            return (array.astype(np.float64) * only.scale + only.offset).astype(np.float32)
        channel_axis = axes.index("C")
        out = np.empty(array.shape, dtype=np.float32)
        for channel, parameters in enumerate(self.parameters):
            index = (slice(None),) * channel_axis + (channel,)
            out[index] = array[index].astype(np.float64) * parameters.scale + parameters.offset
        return out

    def stamp(self, patch) -> None:
        """Record the normalization in the patch's ``normalization_*`` fields."""
        patch.normalization_method = self.method
        patch.normalization_scope = self.scope
        patch.normalization_group = self.group
        patch.normalization_offset = format_numbers(p.offset for p in self.parameters)
        patch.normalization_scale = format_numbers(p.scale for p in self.parameters)
        patch.normalization_exact = self.exact

    def apply(self, array, axes: str, clip=None) -> np.ndarray:
        """Normalize *array* (with *axes*) per channel into a new float32 array.

        *clip* (``config.normalization.clip``) limits the normalized values,
        through the brick's own ``NormalizationParameters.apply``.
        """
        array = np.asarray(array)
        axes = normalize_axes(axes, array.ndim)
        if "C" not in axes:
            return self.parameters[0].apply(array, clip=clip, dtype=np.float32)
        channel_axis = axes.index("C")
        if array.shape[channel_axis] != len(self.parameters):
            raise ValueError(
                f"The patch has {array.shape[channel_axis]} channels but {len(self.parameters)} sets of "
                "normalization parameters."
            )
        out = np.empty(array.shape, dtype=np.float32)
        for channel, parameters in enumerate(self.parameters):
            index = (slice(None),) * channel_axis + (channel,)
            out[index] = parameters.apply(array[index], clip=clip, dtype=np.float32)
        return out


def group_parameters(config) -> list[dict]:
    """The normalization parameters of every group in ``plane_histograms.csv``, for review before a run.

    For a CSV scope (``condition``, ``biological_repeat``, ``technical_repeat``
    or ``image``), one entry per group and channel (and time point with
    ``per_time_point``), in sorted order: ``group`` (as in
    ``normalization_group``), ``channel``, ``t``, ``planes`` (CSV rows
    pooled), ``offset``, ``scale`` and ``exact``, or ``problem`` when the
    group has no finite pixels or zero spread (which stops a run).

    Raises
    ------
    ValueError
        For the patch scope, whose parameters come from each patch.
    """
    normalizer = Normalizer.for_config(config)
    if normalizer is None or normalizer.statistics is None:
        raise ValueError("Per-group parameters exist for the condition, repeat and image scopes only.")
    normalizer._ensure_index()
    rows = []
    for key in sorted(normalizer._index, key=lambda k: (k[0], k[1] if k[1] is not None else -1, k[2] or 0)):
        records = normalizer._index[key]
        result = normalizer._group_parameters(key)
        row = {
            "group": _group_text(records[0], normalizer.settings.scope, normalizer.settings.per_time_point),
            "channel": key[1],
            "t": key[2],
            "planes": len(records),
        }
        if isinstance(result, str):
            row["problem"] = result.split(": ", 1)[-1]
        else:
            parameters, exact = result
            row.update(offset=parameters.offset, scale=parameters.scale, exact=exact)
        rows.append(row)
    return rows


def finish_patch_groups(patches, per_time_point: bool) -> None:
    """Set the patch-scope ``normalization_group``, ``<source_relative_path>::<patch_id>``, once ids are final."""
    for patch in patches:
        if patch.normalization_scope != "patch":
            continue
        source = patch.source_relative_path or patch.image_file or patch.image_id
        patch.normalization_group = f"{source}::{patch.patch_id}" + _time_suffix(patch.t, per_time_point)


def split_channel_normalization(patch, channel: int) -> None:
    """Keep only *channel*'s parameters on a split channel patch (one value per channel of the patch)."""
    if patch.normalization_offset is None:
        return
    patch.normalization_offset = patch.normalization_offset.split(";")[channel]
    patch.normalization_scale = patch.normalization_scale.split(";")[channel]


class Normalizer:
    """The normalization of one run: statistics loaded once, group parameters computed once.

    Create it with :meth:`for_config`, which returns None when normalization
    is off; then :meth:`normalize` gives every patch its parameters.
    """

    def __init__(self, config, statistics: PlaneStatistics | None):
        self.config = config
        self.settings = config.normalization
        self.statistics = statistics
        self.region = resolved_patch_region(config)
        self._mbn = require_normalization()
        self._index: dict[tuple, list] | None = None
        self._groups: dict[tuple, tuple] = {}

    @classmethod
    def for_config(cls, config) -> Normalizer | None:
        """The run's normalizer, or None when ``config.normalization.enabled`` is False.

        Validates the settings, loads the statistics for a CSV scope, and
        warns about the scope's consequences (contract §14).
        """
        settings = config.normalization
        if not settings.enabled:
            return None
        validate_normalization(config)
        statistics = None
        if settings.scope in CSV_SCOPES:
            statistics = cached_statistics(settings.plane_histograms, settings.manifest)
            warnings.warn(
                "The group and image statistics of the preprocessing brick come from whole planes, "
                "background included; in sparse images they describe the background more than the "
                "cells. Use normalization.scope='patch' with patch_region='mask' for parameters from "
                "the cells only.",
                NormalizationWarning,
                stacklevel=2,
            )
            if settings.scope in _GROUP_SCOPES:
                warnings.warn(
                    f"normalization.scope={settings.scope!r} puts every {settings.scope.replace('_', ' ')} "
                    "on the same intensity scale, which removes real intensity differences between them.",
                    NormalizationWarning,
                    stacklevel=2,
                )
        return cls(config, statistics)

    def normalize(self, images, image_axes, masks, mask_axes, metadata, labels, patches) -> dict[int, PatchNormalization]:
        """Compute and record every patch's normalization; return it keyed by ``id(patch)``.

        A patch missing from the result was discarded: at the patch scope, its
        own pixels have no finite values or zero spread (decision D8), which a
        :class:`NormalizationWarning` reports.

        Parameters
        ----------
        images, masks
            The run's images (2D planes, possibly with channels) and masks
            (None entries allowed), by image index.
        image_axes, mask_axes
            Their axes.
        metadata
            Per image: ``source_relative_path``, ``z``, ``t``, ``c`` and
            ``image_file`` as in ``run_patching(image_metadata=...)`` (empty
            dicts allowed).
        labels
            Per image: ``(condition, biological_repeat, technical_repeat)``.
        patches
            The patches, with ``image_id`` ``image_<index>``.

        Raises
        ------
        cellpatcher.statistics.PreprocessingCheckError
            If a check of contract §5 fails.
        ValueError
            If a group or patch has no finite pixels or zero spread, or a
            CSV scope lacks the images' ``source_relative_path``.
        """
        by_image = defaultdict(list)
        for patch in patches:
            by_image[int(patch.image_id.rsplit("_", 1)[1])].append(patch)
        channel_planes = {index: _channel_planes(images[index], image_axes) for index in by_image}

        if self.statistics is not None:
            records = self._check(channel_planes, metadata, labels, image_axes)
            normalizations = self._group_normalizations(records, by_image)
        else:
            normalizations = self._patch_normalizations(channel_planes, masks, mask_axes, by_image)
        for patch in patches:
            if id(patch) in normalizations:  # patches discarded under decision D8 are absent
                normalizations[id(patch)].stamp(patch)
        if not all(normalization.exact for normalization in normalizations.values()):
            warnings.warn(
                "Some normalization parameters are histogram approximations (always for float images, "
                "and for integer images binned with fewer bins than values); see normalization_exact "
                "in patches.csv.",
                NormalizationWarning,
                stacklevel=2,
            )
        return normalizations

    # CSV scopes --------------------------------------------------------------------------

    def _check(self, channel_planes, metadata, labels, image_axes) -> dict[tuple[int, int], object]:
        """Run the checks of contract §5; return each (image, channel) plane's CSV row."""
        missing = sorted({index for index in channel_planes if not metadata[index].get("source_relative_path")})
        if missing:
            raise ValueError(
                f"normalization.scope={self.settings.scope!r} links each image to its rows in "
                "plane_histograms.csv by its path relative to the experiment root, but "
                f"{len(missing)} image(s) have none (image_metadata 'source_relative_path'). Folder and "
                "tree runs fill it; for run_patching, pass image_metadata=[{'source_relative_path': ...}]."
            )
        # Images without any row are reported together first; the manifest's
        # file checks need the files and run in folder and tree runs.
        without_manifest = PlaneStatistics(
            self.statistics.plane_histograms, self.statistics.records, self.statistics.csv_sha256
        )
        check_images(
            without_manifest, {metadata[index]["source_relative_path"]: image_axes for index in channel_planes}
        )
        has_channels = "C" in normalize_axes(image_axes)
        keys, planes = [], []
        for index, channels in channel_planes.items():
            meta = metadata[index]
            for channel, pixels in enumerate(channels):
                c = channel if has_channels else meta.get("c")
                plane = SourcePlane(
                    meta["source_relative_path"], *labels[index], z=meta.get("z"), c=c, t=meta.get("t"),
                    axis_lengths=None,
                )
                keys.append((index, channel))
                planes.append((plane, pixels))
        return dict(zip(keys, check_planes(self.statistics, planes)))

    def _group_normalizations(self, records, by_image) -> dict[int, PatchNormalization]:
        level = self.settings.scope
        per_image = defaultdict(list)
        for (index, _), record in sorted(records.items()):
            per_image[index].append(record)
        failures, normalizations = [], {}
        for index, image_patches in by_image.items():
            channel_records = per_image[index]
            parameters, exact = [], True
            for record in channel_records:
                key = self._group_key(record)
                if key not in self._groups:
                    self._groups[key] = self._group_parameters(key)
                result = self._groups[key]
                if isinstance(result, str):
                    failures.append(result)
                    continue
                parameters.append(result[0])
                exact = exact and result[1]
            if len(parameters) != len(channel_records):
                continue
            group = _group_text(channel_records[0], level, self.settings.per_time_point)
            normalization = PatchNormalization(parameters[0].method, level, group, tuple(parameters), exact)
            for patch in image_patches:
                normalizations[id(patch)] = normalization
        if failures:
            raise ValueError(
                "Normalization parameters cannot be computed for "
                f"{len(set(failures))} group(s):\n" + "\n".join(f"  - {f}" for f in sorted(set(failures)))
                + "\nChoose a larger scope or another method (contract §7)."
            )
        return normalizations

    def _ensure_index(self) -> None:
        """Group every CSV row by its key (the scope's labels, channel and optional t), once."""
        if self._index is None:
            self._index = defaultdict(list)
            for record in self.statistics.records:
                self._index[self._group_key(record)].append(record)

    def _group_key(self, record) -> tuple:
        t = record.t if self.settings.per_time_point else None
        return (record.group_key(self.settings.scope), record.c, t)

    def _group_parameters(self, key):
        """``(NormalizationParameters, exact)`` of a group, or a failure description."""
        self._ensure_index()
        name = "/".join(key[0]) + f" (channel {key[1]}" + ("" if key[2] is None else f", t={key[2]}") + ")"
        try:
            pooled = self._mbn.PooledHistogram(self._index[key])
            return self._parameters(pooled), pooled.max_representative_error == 0
        except self._mbn.NoValidPixelsError:
            return f"{name}: no finite pixels"
        except ValueError:
            return f"{name}: zero intensity spread for {self.settings.method}"

    # Patch scope -------------------------------------------------------------------------

    def _patch_normalizations(self, channel_planes, masks, mask_axes, by_image) -> dict[int, PatchNormalization]:
        failures, discarded, normalizations = [], [], {}
        for index, image_patches in by_image.items():
            channels = channel_planes[index]
            labels_2d = None
            if self.region in ("mask", "bbox"):
                labels_2d = _label_plane(masks[index], mask_axes, self.config.cell.mask_channel)
            for patch in image_patches:
                try:
                    region = _region_selection(patch, self.region, labels_2d, channels[0].shape)
                except ValueError as exc:
                    failures.append(f"{_describe(patch)}: {exc}")
                    continue
                parameters, exact, problem = [], True, None
                for pixels in channels:
                    values = pixels[region]
                    histogram = self._mbn.compute_plane_histogram(values)
                    try:
                        pooled = self._mbn.PooledHistogram([histogram])
                        parameters.append(self._parameters(pooled))
                    except self._mbn.NoValidPixelsError:
                        problem = "no finite pixels in the region"
                        break
                    except ValueError:
                        problem = f"zero intensity spread for {self.settings.method}"
                        break
                    exact = exact and pooled.max_representative_error == 0
                if problem:
                    discarded.append(f"{_describe(patch)}: {problem}")
                    continue
                normalizations[id(patch)] = PatchNormalization(
                    parameters[0].method, "patch", None, tuple(parameters), exact
                )
        if failures:
            raise ValueError(
                f"{len(failures)} patch(es) have no pixels for patch_region={self.region!r}:\n"
                + "\n".join(f"  - {f}" for f in failures[:20]) + ("\n  ..." if len(failures) > 20 else "")
            )
        if discarded:
            # Decision D8 (October 2026): a patch that cannot be normalized from
            # its own pixels is discarded, never kept unnormalized, and reported.
            warnings.warn(
                f"{len(discarded)} patch(es) were discarded because they cannot be normalized from their own "
                f"pixels (patch_region={self.region!r}, method={self.settings.method!r}):\n"
                + "\n".join(f"  - {d}" for d in discarded[:20]) + ("\n  ..." if len(discarded) > 20 else "")
                + "\nUse another method, patch_region or scope (for example scope='image') to keep them.",
                NormalizationWarning,
                stacklevel=3,
            )
        return normalizations

    def _parameters(self, pooled):
        method = self.settings.method
        if method == "minmax":
            return self._mbn.min_max(pooled)
        if method == "percentile":
            low, high = self.settings.percentile
            return self._mbn.percentile_range(pooled, low, high)
        if method == "zscore":
            return self._mbn.z_score(pooled)
        return self._mbn.robust_z_score(pooled)


def _time_suffix(t, per_time_point: bool) -> str:
    return f"::t={0 if t is None else t}" if per_time_point else ""


def _group_text(record, level: str, per_time_point: bool) -> str:
    """``normalization_group`` of a CSV scope (contract §7)."""
    if level == "image":
        text = record.source_relative_path
    else:
        text = "/".join(record.group_key(level))
    return text + _time_suffix(record.t, per_time_point)


def _channel_planes(image, axes: str) -> list[np.ndarray]:
    """The (Y, X) plane of each channel of a 2D image (one plane without a C axis).

    Axes other than Y, X and C must have length 1 here: normalization needs one
    plane per patch (contract §3), which the plane policies ensure.
    """
    array = np.asarray(image)
    axes = normalize_axes(axes, array.ndim)
    for axis, length in zip(axes, array.shape):
        if axis not in "YXC" and length != 1:
            raise ValueError(
                f"Normalization needs one plane per patch, but the images still have {length} planes "
                f"along {axis}; set common.{axis.lower()}_policy to 'all', 'first' or 'select'."
            )
    order = [axes.index(a) for a in "CYX" if a in axes] + [i for i, a in enumerate(axes) if a not in "CYX"]
    moved = np.transpose(array, order)
    channels = moved.shape[0] if "C" in axes else 1
    height, width = array.shape[axes.index("Y")], array.shape[axes.index("X")]
    return list(moved.reshape(channels, height, width))


def _label_plane(mask, mask_axes: str, mask_channel) -> np.ndarray:
    """The (Y, X) label plane of a mask (selecting ``cell.mask_channel`` when it has channels)."""
    if mask is None:
        raise ValueError("normalization.patch_region 'mask'/'bbox' needs the masks of cell mode.")
    array = np.asarray(mask)
    axes = normalize_axes(mask_axes, array.ndim)
    if "C" in axes and array.shape[axes.index("C")] != 1:
        if mask_channel is None:
            raise ValueError("Masks with several channels need cell.mask_channel to choose the label channel.")
        array = np.take(array, int(mask_channel), axis=axes.index("C"))
        axes = axes.replace("C", "")
    return _label_to_2d(array, axes)


def _region_selection(patch, region: str, labels_2d, shape) -> tuple:
    """Index into a (Y, X) plane selecting the pixels of *patch*'s region; never padding."""
    height, width = shape
    rows = slice(max(patch.y0, 0), min(patch.y1, height))
    cols = slice(max(patch.x0, 0), min(patch.x1, width))
    if region == "patch":
        return rows, cols
    if patch.label_id is None:
        raise ValueError("the patch has no label_id")
    if region == "mask":
        inside = np.zeros(shape, dtype=bool)
        inside[rows, cols] = labels_2d[rows, cols] == patch.label_id
        return np.nonzero(inside)
    found_rows, found_cols = np.nonzero(labels_2d == patch.label_id)
    if found_rows.size == 0:
        raise ValueError(f"label {patch.label_id} is not in the mask")
    return slice(found_rows.min(), found_rows.max() + 1), slice(found_cols.min(), found_cols.max() + 1)


def _describe(patch) -> str:
    """A patch in messages, by source and position (ids change once planes are merged)."""
    source = patch.source_relative_path or patch.image_file or patch.image_id
    where = f"y={patch.y0}:{patch.y1}, x={patch.x0}:{patch.x1}"
    plane = "".join(f", {axis}={value}" for axis, value in (("z", patch.z0), ("t", patch.t)) if value is not None)
    label = "" if patch.label_id is None else f", label {patch.label_id}"
    return f"{source} ({where}{plane}{label})"
