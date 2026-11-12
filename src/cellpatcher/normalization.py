"""Normalize patches with the statistics of the preprocessing brick.

Implements ``docs/normalization_contract.md`` §6–§9 for ``config.normalization``:

- **Scopes.** ``"biological_repeat"`` and ``"image"`` pool the rows of
  ``plane_histograms.csv`` of the patch's group, grouped by the normalization
  brick's ``scope_groups`` (with ``per_time_point`` the time point added to
  the key). A ``biological_repeat`` group is one repeat group: the same
  biological repeat in every condition, technical repeats merged, paired by
  the run's published description (``Fov.repeat_group`` through ``RunInputs``,
  never the derived ``experiment_design.csv``). ``"patch"`` builds a
  histogram of the patch's own pixels (``"mask"``, ``"bbox"`` or ``"patch"``
  region), never including padding.
- **Methods.** The normalization brick's ``fit`` turns a group's rows, or a
  patch's pixels, into ``Parameters`` (one offset and scale per channel), and
  ``Parameters.apply`` normalizes a patch per channel as
  ``(pixels - offset) / scale`` in float32. ``normalization.clip`` is applied
  here, after the brick's ``apply``: the brick never clips (its decision N19).
- **Image types.** ``"fluorescence"`` normalizes each channel separately;
  ``"rgb"`` uses StarDist's percentile method, one offset and scale from the
  R, G and B pixels pooled together (recorded for the channels ``R``, ``G`` and
  ``B``), applied with CSBDeep's float32 arithmetic by the brick.
- **Channels by ID** (normalization v0.5.0, Spec ADR 0003 HV2). Every channel is
  named by its canonical channel ID, never by its C index: in a file run the ID of
  each C index of an image is the image's own (``image_metadata["channel_ids"]``,
  ``Fov.channel_ids``), and at a CSV scope its ``plane_histograms.csv`` row's, which
  must agree. An in-memory image without channel IDs has no description: its
  channels are named by position (``C0``, ``C1``, ...; ``R``, ``G``, ``B`` for
  ``"rgb"``) at the patch scope only, whose parameters come from the patch itself.
  The parameters hold the IDs in sorted order; :class:`PatchNormalization` keeps the
  patch's own C order (``normalization_channels``) and applies each channel with
  the parameters of its ID.
- **Checks.** Every patched plane is linked to its CSV row with
  :func:`cellpatcher.statistics.check_planes` before anything is normalized.

Histograms, pooling, fitting and the parameter columns all come from
``morphobricks_normalization`` (a core dependency); this
module chooses the groups and pixels and records the result. A group with no
finite pixels or zero spread stops the run (contract §7), and a patch whose own
pixels have no finite values or zero spread is discarded and reported
(decision D8).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from morphobricks_normalization.records import SINGLE

from cellpatcher.axes import normalize_axes
from cellpatcher.masking import _label_to_2d
from cellpatcher.patch_normalization import (  # PatchNormalization and the split are re-exported (their old home)
    PatchNormalization,
    channel_order_from_row,
    image_channel_ids,
)
from cellpatcher.patch_normalization import split_channel_normalization as split_channel_normalization
from cellpatcher.run_open import check_image_type, settings_statistics
from cellpatcher.statistics import (
    PlaneStatistics,
    PreprocessingCheckError,
    SourcePlane,
    _preprocessing,
    check_images,
    check_planes,
    require_normalization,
)

if TYPE_CHECKING:
    from cellpatcher.config import CellPatcherConfig

CSV_SCOPES = ("biological_repeat", "image")

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
        If normalization is enabled and the installed bricks read other
        file versions than this CellPatcher.
    ValueError
        If a CSV scope has no ``plane_histograms``, a ``"mask"``/``"bbox"``
        patch region is used outside ``cell`` mode, or ``"rgb"`` images are
        split into channels.
    """
    settings = config.normalization
    if not settings.enabled:
        return
    _preprocessing()
    if settings.image_type == "rgb" and config.extraction.split_channels is not False:
        raise ValueError(
            "normalization.image_type='rgb' normalizes the R, G and B channels together and keeps them in one "
            "patch, so extraction.split_channels must be False. Use image_type='fluorescence' to get one patch "
            "per channel."
        )
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


def repeat_groups(statistics: PlaneStatistics) -> dict[tuple[str, str], str]:
    """Return the repeat group of every ``(condition, biological_repeat)`` of the run, from its description.

    The ``biological_repeat`` scope pools one repeat group: ``Fov.repeat_group``, the repeat ID the
    published description gives each FOV through ``RunInputs`` (Spec ADR 0002 AR11 (3)), never the
    derived ``experiment_design.csv`` (an output only) nor a folder name. The keys are the FOV key's
    condition and biological-repeat labels, as the normalization brick's ``scope_groups`` takes them.

    Raises
    ------
    PreprocessingCheckError
        ``missing_repeat_group`` when a FOV of the run has none, or ``repeat_group_conflict``
        when the images of one condition and biological-repeat folder name two repeat groups.
    """
    from morphobricks_preprocessing.validation import error

    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing = []
    for fov in statistics.inputs.fovs():
        if not fov.repeat_group:
            missing.append(fov.relative_path)
            continue
        groups[(fov.key[0], fov.key[1])].add(fov.repeat_group)
    if missing:
        raise PreprocessingCheckError([
            error(
                "missing_repeat_group",
                "normalization.scope='biological_repeat' pools the biological repeats as the run's experiment "
                "description pairs them, but these images of the run have no repeat group.",
                missing,
                "Run preprocessing again on the experiment with its published description, and give CellPatcher "
                "that run.",
            )
        ])
    conflicts = sorted(key for key, names in groups.items() if len(names) > 1)
    if conflicts:
        raise PreprocessingCheckError([
            error(
                "repeat_group_conflict",
                "The run's description gives several repeat groups to the images of one condition and "
                "biological-repeat folder ("
                + "; ".join(f"{'/'.join(key)}: {', '.join(sorted(groups[key]))}" for key in conflicts)
                + "), so the biological_repeat scope cannot pool them.",
                [str(statistics.manifest_path)],
                "Correct the experiment description in the preprocessing notebook, publish it and run "
                "preprocessing again.",
            )
        ])
    return {key: next(iter(names)) for key, names in sorted(groups.items())}


def normalization_inputs(config: CellPatcherConfig) -> dict | None:
    """The ``normalization_inputs`` of ``run_config.json`` (contract §11), from the opened run.

    None when normalization is off or uses no CSV (the patch scope). The
    values come from the run ``RunInputs`` opened: the table's SHA-256 and
    the manifest version its manifest records. Paths are not recorded: they
    are machine-specific. The ``biological_repeat`` scope also records the
    SHA-256 of the published description its repeat groups come from.
    """
    settings = config.normalization
    if not settings.enabled or settings.scope not in CSV_SCOPES or settings.plane_histograms is None:
        return None
    statistics = settings_statistics(settings)
    manifest = statistics.manifest
    inputs = {
        "plane_histograms_sha256": manifest.csv_sha256,
        "schema_version": manifest.schema_version,
        "manifest_used": True,
        "manifest_version": manifest.manifest_version,
        "preprocessing_version": _preprocessing().__version__,
    }
    if settings.scope == "biological_repeat":
        repeat_groups(statistics)
        inputs.update(description_sha256=manifest.experiment_description["sha256"])
    return inputs


def fit_settings(settings):
    """The normalization brick's ``NormalizationSettings`` for ``config.normalization``."""
    mbn = require_normalization()
    return mbn.NormalizationSettings(
        image_type=settings.image_type,
        method=settings.method,
        percentile=tuple(settings.percentile),
        scope=settings.scope,
        pooling=settings.pooling,
        per_time_point=settings.per_time_point,
    )


def run_settings(config) -> dict | None:
    """The brick's ``NormalizationSettings.to_dict()``, ``normalization_settings`` of ``run_config.json``; None when off."""
    settings = config.normalization
    if not settings.enabled:
        return None
    return fit_settings(settings).to_dict()


def parameters_from_row(row: Mapping, settings: Mapping) -> object | None:
    """Read the brick ``Parameters`` of a ``patches.csv`` row, at its run's scope (brick #37); None if it has none.

    *settings*: the row's ``run_config.json`` ``normalization``. ``Normalization_*_<ID>`` columns at a scope removed
    since (N29), which the brick refuses, are read as image-scope ones, keeping their recorded scope (it must be the
    run's) and values (contract §7). Rows written before normalization v0.5.0 (``Normalization_*_C<i>`` columns
    without ``normalization_parameters_version``, or the ``;``-separated ``normalization_offset`` of earlier tables)
    are refused, never converted (Spec ADR 0002 AR10): run CellPatcher again.

    Raises
    ------
    ValueError
        If the row's scope is not its run's, or its parameters cannot be read.
    """
    from cellpatcher.setting_checks import REMOVED_NORMALIZATION_SCOPES

    mbn = require_normalization()
    prefixes = ("Normalization_Offset_", "Normalization_Scale_")
    pairs = any(str(name).startswith(prefixes) and value not in (None, "") for name, value in row.items())
    if not pairs and row.get("normalization_offset"):
        raise ValueError(
            "The row has the ;-separated normalization_offset/normalization_scale columns of a patches.csv written "
            "before the per-channel columns; such rows are not read (no conversion). Run CellPatcher again."
        )
    if not pairs:
        return None
    removed = (scope := settings.get("scope", "image")) in REMOVED_NORMALIZATION_SCOPES
    if removed and (recorded := row.get("normalization_scope")) != scope:
        raise ValueError(f"Column normalization_scope is {recorded!r}, but its run normalized at {scope!r}.")
    brick = mbn.NormalizationSettings(
        image_type=settings.get("image_type", "fluorescence"), method=settings.get("method", "percentile"),
        percentile=tuple(settings.get("percentile", (1.0, 99.8))), scope="image" if removed else scope,
    )
    parameters = mbn.Parameters.from_columns({**row, "normalization_scope": "image"} if removed else row, brick)
    return replace(parameters, scope=scope)


def normalization_fields(row: Mapping, settings: Mapping) -> dict:
    """Return a ``patches.csv`` row's ``normalization_parameters`` and ``normalization_channels`` (its C order)."""
    return {
        "normalization_parameters": parameters_from_row(row, settings),
        "normalization_channels": channel_order_from_row(row),
    }


def group_parameters(config) -> list[dict]:
    """The normalization parameters of every group in ``plane_histograms.csv``, for review before a run.

    For a CSV scope (``biological_repeat`` or ``image``), one entry per group
    and channel (and time point with ``per_time_point``), in sorted order:
    ``group`` (as in ``normalization_group``), ``channel`` (its canonical ID), ``t``, ``planes``
    (CSV rows pooled), ``offset``, ``scale`` and ``exact``, or ``problem``
    when the group has no finite pixels or zero spread (which stops a run).

    Raises
    ------
    ValueError
        For the patch scope, whose parameters come from each patch.
    """
    normalizer = Normalizer.for_config(config)
    if normalizer is None or normalizer.statistics is None:
        raise ValueError("Per-group parameters exist for the biological-repeat and image scopes only.")
    normalizer._ensure_index()
    rows = []
    for key in normalizer._index:  # sorted by the brick's scope_groups
        records = normalizer._index[key]
        result = normalizer._group_parameters(key)
        group = normalizer._mbn.group_name(key)
        t = key[1] if normalizer.settings.per_time_point else None
        for channel in sorted({channel for record in records for channel in record.channels}):
            row = {"group": group, "channel": channel, "t": t, "planes": len(records)}
            if isinstance(result, str):
                row["problem"] = result.split(": ", 1)[-1]
            else:
                index = result.channels.index(channel)
                row.update(offset=result.offset[index], scale=result.scale[index], exact=result.exact)
            rows.append(row)
    return rows


def finish_patch_groups(patches, per_time_point: bool) -> None:
    """Set the patch-scope ``normalization_group``, ``<source_relative_path>::<patch_id>``, once ids are final."""
    for patch in patches:
        if patch.normalization_scope != "patch":
            continue
        source = patch.source_relative_path or patch.image_file or patch.image_id
        patch.normalization_group = f"{source}::{patch.patch_id}" + _time_suffix(patch.t, per_time_point)
        if patch.normalization_parameters is not None:
            patch.normalization_parameters = replace(patch.normalization_parameters, group=patch.normalization_group)


class Normalizer:
    """The normalization of one run: statistics loaded once, group parameters computed once.

    Create it with :meth:`for_config`, which returns None when normalization
    is off; then :meth:`normalize` gives every patch its parameters.
    """

    def __init__(self, config, statistics: PlaneStatistics | None, repeat_groups: dict | None = None):
        self.config = config
        self.settings = config.normalization
        self.statistics = statistics
        self.repeat_groups = repeat_groups
        self.region = resolved_patch_region(config)
        self._mbn = require_normalization()
        self._fit_settings = fit_settings(self.settings)
        self._index: dict[tuple, list] | None = None
        self._groups: dict[tuple, object] = {}

    @classmethod
    def for_config(cls, config: CellPatcherConfig) -> Normalizer | None:
        """The run's normalizer, or None when ``config.normalization.enabled`` is False.

        Validates the settings, loads the statistics for a CSV scope (and,
        for ``biological_repeat``, takes the repeat groups of the run's description
        and groups the rows, so a missing repeat group stops the run before any patch
        is normalized), and warns about the scope's consequences (contract §14).
        """
        settings = config.normalization
        if not settings.enabled:
            return None
        validate_normalization(config)
        statistics, groups = None, None
        if settings.scope in CSV_SCOPES:
            statistics = settings_statistics(settings)
            check_image_type(settings, statistics.inputs)
            warnings.warn(
                "The group and image statistics of the preprocessing brick come from whole planes, "
                "background included; in sparse images they describe the background more than the "
                "cells. Use normalization.scope='patch' with patch_region='mask' for parameters from "
                "the cells only.",
                NormalizationWarning,
                stacklevel=2,
            )
            if settings.scope == "biological_repeat":
                groups = repeat_groups(statistics)
        mbn = require_normalization()
        for issue in mbn.check_settings(fit_settings(settings), statistics_file=statistics is not None):
            warnings.warn(issue.message, NormalizationWarning, stacklevel=2)
        normalizer = cls(config, statistics, groups)
        if groups is not None:
            normalizer._ensure_index()
        return normalizer

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
            If a group or patch has no finite pixels or zero spread, a channel
            only some planes of a group have (``A_SCOPE_INCOMPLETE``), a CSV
            scope lacks the images' ``source_relative_path``, an image's channel
            IDs are not its rows' IDs, the images have different numbers of
            channels, or ``"rgb"`` images are not three channels of an integer type.
        """
        by_image = defaultdict(list)
        for patch in patches:
            by_image[int(patch.image_id.rsplit("_", 1)[1])].append(patch)
        channel_planes = {index: _channel_planes(images[index], image_axes) for index in by_image}
        self._check_channel_counts(channel_planes, metadata)
        if self.settings.image_type == "rgb":
            self._check_rgb(channel_planes)
        has_channels = "C" in normalize_axes(image_axes)
        rgb = self.settings.image_type == "rgb"
        channel_ids = {
            index: image_channel_ids(metadata[index], len(channels), has_channels, rgb)
            for index, channels in channel_planes.items()
        }

        if self.statistics is not None:
            records = self._check(channel_planes, metadata, labels, image_axes)
            normalizations = self._group_normalizations(records, by_image, metadata)
        else:
            normalizations = self._patch_normalizations(channel_planes, masks, mask_axes, by_image, channel_ids)
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
        if self.repeat_groups is not None:
            self._check_repeat_labels(channel_planes, metadata, labels)
        has_channels = "C" in normalize_axes(image_axes)
        keys, planes = [], []
        for index, channels in channel_planes.items():
            meta = metadata[index]
            for channel, pixels in enumerate(channels):
                c = channel if has_channels else meta.get("c")
                plane = SourcePlane(
                    meta["source_relative_path"], *labels[index], z=meta.get("z"), c=c, t=meta.get("t"),
                    axis_lengths=None, position_id=meta.get("position_id"),
                )
                keys.append((index, channel))
                planes.append((plane, pixels))
        return dict(zip(keys, check_planes(self.statistics, planes)))

    def _group_normalizations(
        self, records: Mapping, by_image: Mapping, metadata: list
    ) -> dict[int, PatchNormalization]:
        """Give each image's patches its group's parameters, selected by the IDs of its rows in its C order."""
        level = self.settings.scope
        per_image = defaultdict(list)
        for (index, _), record in sorted(records.items()):
            per_image[index].append(record)
        for index, channel_records in per_image.items():
            _check_row_channel_ids(metadata[index], channel_records)
        failures, normalizations = [], {}
        for index, image_patches in by_image.items():
            channel_records = per_image[index]
            # Every channel of an image is in the same row, so in the same group.
            key = self._group_key(channel_records[0].record)
            if key not in self._groups:
                self._groups[key] = self._group_parameters(key)
            result = self._groups[key]
            if isinstance(result, str):
                failures.append(result)
                continue
            group = self._mbn.group_name(key)
            order = tuple(record.channel for record in channel_records)  # the image's C order, by its rows' IDs
            parameters = replace(result.select(order), scope=level, group=group)
            normalization = PatchNormalization(parameters, order)
            for patch in image_patches:
                normalizations[id(patch)] = normalization
        if failures:
            raise ValueError(
                "Normalization parameters cannot be computed for "
                f"{len(set(failures))} group(s):\n" + "\n".join(f"  - {f}" for f in sorted(set(failures)))
                + "\nChoose a larger scope or another method (contract §7)."
            )
        return normalizations

    def _check_repeat_labels(self, channel_planes: Mapping, metadata: list, labels: list) -> None:
        """Refuse images whose condition and biological repeat no FOV of the run's description has."""
        from morphobricks_preprocessing.validation import error

        unknown = sorted({
            (metadata[index].get("source_relative_path"), tuple(labels[index][:2]))
            for index in channel_planes
            if tuple(label or SINGLE for label in labels[index][:2]) not in self.repeat_groups
        })
        if unknown:
            pairs = sorted({"/".join(label or SINGLE for label in pair) for _, pair in unknown})
            raise PreprocessingCheckError([
                error(
                    "image_not_in_description",
                    f"The preprocessing run's description has no repeat group for {', '.join(pairs)}, the "
                    "condition and biological repeat of these images.",
                    [path for path, _ in unknown if path],
                    "Patch the images of the run with the labels the run recorded, or describe these images in "
                    "the preprocessing notebook and run preprocessing again.",
                )
            ])

    def _ensure_index(self) -> None:
        """Group every CSV row (an XYC plane) with the brick's ``scope_groups``, once; warn about its issues."""
        if self._index is None:
            groups, issues = self._mbn.scope_groups(self.statistics.records, self._fit_settings, self.repeat_groups)
            for issue in issues:
                warnings.warn(issue.message, NormalizationWarning, stacklevel=3)
            self._index = groups

    def _group_key(self, record) -> tuple:
        return self._mbn.group_key(record, self._fit_settings, self.repeat_groups)

    def _group_parameters(self, key):
        """The brick ``Parameters`` of a group (every channel), or a failure description."""
        self._ensure_index()
        name = self._mbn.group_name(key)
        try:
            return self._mbn.fit(self._index[key], self._fit_settings)
        except self._mbn.NoValidPixelsError as exc:
            return f"{name}: no finite pixels ({exc})"
        except self._mbn.ChannelScopeIncompleteError as exc:
            # A channel only some planes of the group have: refused, never pooled over the planes with it
            # (CellPatcher offers no partial_channels choice; Spec A_SCOPE_INCOMPLETE).
            return f"{name}: {exc}"
        except ValueError as exc:
            return f"{name}: zero intensity spread for {self.settings.method} ({exc})"

    def _check_channel_counts(self, channel_planes, metadata) -> None:
        """Refuse images with different numbers of channels in one normalized run (as before normalization v0.5.0).

        Normalization v0.5.0 pools each channel by its ID, but CellPatcher does not yet patch images of different
        channel counts in one normalized run (Spec Stage 3, gate G3): refused, never padded or dropped.
        """
        counts = [
            (metadata[index].get("source_relative_path") or metadata[index].get("image_file") or f"image_{index}",
             len(channels))
            for index, channels in sorted(channel_planes.items())
        ]
        by_count = defaultdict(list)
        for name, count in counts:
            by_count[count].append(name)
        if len(by_count) > 1:
            listed = "; ".join(
                f"{count} channel(s): {', '.join(names[:3])}{' ...' if len(names) > 3 else ''}"
                for count, names in sorted(by_count.items())
            )
            raise ValueError(
                f"The images of this run have different numbers of channels ({listed}). CellPatcher normalizes images "
                "of one channel count per run; patch each group of images in its own run."
            )

    def _check_rgb(self, channel_planes) -> None:
        """Refuse ``"rgb"`` images that are not three integer channels, before any fit.

        The brick's ``fit`` checks this too, but at the patch scope its error
        would read as a patch without spread and discard the patch (decision
        D8) instead of stopping the run.
        """
        from morphobricks_normalization.rgb import check_rgb

        for channels in channel_planes.values():
            check_rgb(len(channels), [pixels.dtype for pixels in channels])

    # Patch scope -------------------------------------------------------------------------

    def _patch_normalizations(
        self, channel_planes, masks, mask_axes, by_image, channel_ids
    ) -> dict[int, PatchNormalization]:
        failures, discarded, normalizations = [], [], {}
        for index, image_patches in by_image.items():
            channels = channel_planes[index]
            order = channel_ids[index]  # the ID of each channel plane, in the image's C order
            labels_2d = None
            if self.region in ("mask", "bbox"):
                labels_2d = _label_plane(masks[index], mask_axes, self.config.cell.mask_channel)
            for patch in image_patches:
                try:
                    region = _region_selection(patch, self.region, labels_2d, channels[0].shape)
                except ValueError as exc:
                    failures.append(f"{_describe(patch)}: {exc}")
                    continue
                values = np.stack([pixels[region] for pixels in channels])
                try:
                    parameters = self._mbn.fit(values, self._fit_settings, channel_axis=0, channels=order)
                except self._mbn.NoValidPixelsError:
                    discarded.append(f"{_describe(patch)}: no finite pixels in the region")
                    continue
                except ValueError:
                    discarded.append(f"{_describe(patch)}: zero intensity spread for {self.settings.method}")
                    continue
                normalizations[id(patch)] = PatchNormalization(parameters, order)
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


def _check_row_channel_ids(meta: Mapping, channel_records: list) -> None:
    """Refuse an image whose own channel IDs are not those of its ``plane_histograms.csv`` rows, C index by C index."""
    from morphobricks_preprocessing.validation import error

    ids = meta.get("channel_ids")
    if ids is None:
        return
    for record in channel_records:
        if not 0 <= record.c < len(ids) or ids[record.c] != record.channel:
            given = ids[record.c] if 0 <= record.c < len(ids) else None
            raise PreprocessingCheckError([
                error(
                    "channel_id_mismatch",
                    f"Channel C{record.c} of {meta.get('source_relative_path')} is {given!r} for the image but "
                    f"{record.channel!r} in plane_histograms.csv ({record.plane_id}).",
                    [str(meta.get("source_relative_path"))],
                    "Run preprocessing again so its table describes the run's images.",
                )
            ])


def _time_suffix(t, per_time_point: bool) -> str:
    return f"::t={0 if t is None else t}" if per_time_point else ""


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
        array = np.take(array, mask_channel, axis=axes.index("C"))
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
