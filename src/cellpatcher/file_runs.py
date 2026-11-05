"""Folder and tree runs, read through a clean preprocessing run (layer: orchestration).

Design v0.4 §6 (CP-2a): a file run (:func:`cellpatcher.run_patching_pipeline`,
:func:`cellpatcher.run_patching_condition_repeat_tree`) needs a clean preprocessing run,
``common.preprocessing_run``, opened with ``RunInputs.open`` with the experiment root and
masks folder CellPatcher is given (so an experiment moved or copied since its run is checked
and read where it is now). Then:

- the images patched are the run's, with their FOV key (condition, biological and technical
  repeat, ``position_id``) as the run recorded it, never parsed from a name;
- each image is read with ``RunInputs.read_plane`` in the roles the run recorded, and each mask
  with ``RunInputs.read_mask`` by the run's pairing (:mod:`cellpatcher.run_reading`);
- the CSV normalization scopes read the run's own table, design and image type.

The settings that told CellPatcher how to read files (``common.image_axes``, ``mask_axes`` and
the labels) are refused for file runs with a migration message; in-memory ``run_patching``
keeps them. ``mask_source`` and ``mask_label_name`` were removed in v0.2.0 (CP-2b).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from cellpatcher.ids import object_set_from_mask_type, validate_object_set
from cellpatcher.io import (
    SINGLE_CONDITION_LABEL,
    ConditionRepeatFolder,
    refuse_removed_matching,
)
from cellpatcher.normalization import (
    CSV_SCOPES,
    check_image_type,
    load_experiment_design,
    validate_normalization,
)
from cellpatcher.pipeline import (
    PatchingResult,
    _patch_planes,
    check_config_identity,
)
from cellpatcher.run_open import _refuse_other_file, open_run, statistics_of
from cellpatcher.run_reading import (
    fov_layout,
    iter_fov_planes,
    mask_file_name,
    mask_layout,
    mask_record,
    read_fov_mask,
    union_layout,
)
from cellpatcher.run_session import keep_in_session, remember, remember_layouts
from cellpatcher.setting_checks import check_run_settings
from cellpatcher.statistics import (
    PlaneStatistics,
    PreprocessingCheckError,
    check_images,
)

if TYPE_CHECKING:
    import numpy as np
    from morphobricks_preprocessing.run_inputs import Fov, RunInputs

    from cellpatcher.config import CellPatcherConfig, NormalizationConfig

PREPROCESSING_COMMAND = (
    "morphobricks-preprocessing check <experiment> --masks <masks folder>, then "
    "morphobricks-preprocessing design --output <output folder> --confirm GROUP (or --confirm-all, once the "
    "repeat groups are comparable), then "
    "morphobricks-preprocessing run <experiment> --axes <roles> --masks <masks folder> "
    "(leave out --masks without masks), or the preprocessing notebook"
)

_DEFAULT_AXES = "YX"
_LABELS = ("condition", "biological_repeat", "technical_repeat")


@dataclass(frozen=True)
class FileRun:
    """A clean preprocessing run opened for a folder or tree run.

    Attributes
    ----------
    inputs
        The run, opened with ``RunInputs.open``.
    layouts
        Each image's canonical layout, keyed by its path (:func:`cellpatcher.run_reading.fov_layout`):
        its own roles, whatever the other images of the run have (audit A-M1).
    mask_type
        The mask type read, or None without masks.
    mask_layouts
        Each image's mask layout, keyed by the image's path
        (:func:`cellpatcher.run_reading.mask_layout`); empty without masks.
    statistics
        The run's table for a CSV normalization scope, else None.
    """

    inputs: RunInputs
    layouts: Mapping[str, str]
    mask_type: str | None
    mask_layouts: Mapping[str, str]
    statistics: PlaneStatistics | None

    @property
    def fovs(self) -> tuple[Fov, ...]:
        """Every image of the run, sorted by path."""
        return self.inputs.fovs()

    @property
    def layout(self) -> str:
        """The T, Z and C axes any image's layout has, then ``YX`` (recorded in ``run_config.json``)."""
        return union_layout(self.layouts.values())

    @property
    def mask_axes(self) -> str:
        """The T and Z any mask's layout has, then ``YX`` (``"YX"`` without masks)."""
        return union_layout(self.mask_layouts.values())

    def image_layout(self, fov: Fov) -> str:
        """Return the canonical layout *fov* is read in."""
        return self.layouts[fov.relative_path]

    def mask_layout(self, fov: Fov) -> str:
        """Return the layout *fov*'s mask is read in (``"YX"`` without masks)."""
        return self.mask_layouts.get(fov.relative_path, _DEFAULT_AXES)


def refuse_file_run_settings(config: CellPatcherConfig, where: str) -> None:
    """Refuse a file run without a preprocessing run, or with a setting it no longer reads.

    Raises
    ------
    PreprocessingCheckError
        ``no_preprocessing_run`` when ``common.preprocessing_run`` is not set.
    ValueError
        Naming every setting a file run reads from the preprocessing run instead, and what
        to do (the migration message).
    """
    common = config.common
    problems = [
        f"common.{name}={getattr(common, name)!r}: each image's (and mask's) axes are the roles the "
        "preprocessing run recorded for it; leave it at its default 'YX'"
        for name in ("image_axes", "mask_axes")
        if getattr(common, name) != _DEFAULT_AXES
    ]
    problems += [
        f"normalization.{name}={getattr(config.normalization, name)!r}: where the experiment and masks are now "
        "is the image folder or root and mask_folder_path or mask_root_path of the run; leave it unset"
        for name in ("experiment_root", "masks_root")
        if getattr(config.normalization, name) is not None
    ]
    problems += [
        f"common.{name}={getattr(common, name)!r}: each image's labels are its folders, as the preprocessing "
        "run recorded them; leave it unset"
        for name in _LABELS
        if getattr(common, name) not in (None, SINGLE_CONDITION_LABEL)
    ]
    if problems:
        raise ValueError(
            f"{where} reads the images, masks and labels through the preprocessing run since CellPatcher "
            "v0.2.0 (design v0.4 §6), so these settings are not used for files: " + "; ".join(problems) + ". "
            "The axes and labels still describe in-memory arrays given to run_patching."
        )
    if common.preprocessing_run is None:
        _refuse_without_run(where)


def _refuse_without_run(where: str, setting: str = "common.preprocessing_run") -> None:
    """Raise ``no_preprocessing_run`` (*setting* not set), with the preprocessing command."""
    from morphobricks_preprocessing.validation import error

    raise PreprocessingCheckError([
        error(
            "no_preprocessing_run",
            f"{where} reads files only through a clean preprocessing run (v0.4), and {setting} is not set.",
            [],
            f"Run preprocessing on the experiment first: {PREPROCESSING_COMMAND}. Then set "
            f"{setting} to its output folder (by default <experiment>_MorphoBricks_Preprocessing, "
            "beside the experiment).",
        )
    ])


def masks_used(config: CellPatcherConfig, masks_root: str | os.PathLike | None) -> bool:
    """Return whether the run reads masks: always in ``cell`` mode, in ``random`` mode when asked.

    ``random`` mode reads them when a masks folder or ``matching.mask_type`` is given. ``grid``
    and ``whole_fov`` read none: a masks folder given for them only says where the run's masks
    folder is now (owner decision of 2026-11-02), and the run checks its masks there.
    """
    if config.mode in {"grid", "whole_fov"}:
        return False
    return config.mode == "cell" or masks_root is not None or config.matching.mask_type is not None


def open_file_run(
    config: CellPatcherConfig,
    experiment_root: str | os.PathLike,
    masks_root: str | os.PathLike | None,
    *,
    use_masks: bool,
    where: str,
) -> FileRun:
    """Open ``common.preprocessing_run`` with where the experiment and masks are now, before any read.

    *use_masks* (:func:`masks_used`) says whether masks are read; *masks_root* is where the
    run's masks folder is now (None: where the run recorded it) either way. Must be called
    inside :func:`cellpatcher.run_session.statistics_session`: the run's table is kept there
    for the call's normalization.

    Raises
    ------
    PreprocessingCheckError
        Without a preprocessing run, when ``RunInputs.open`` refuses it, or when a check of
        the normalization inputs fails.
    ValueError
        For a setting a file run does not read, or masks that cannot be used.
    """
    refuse_file_run_settings(config, where)
    check_run_settings(config)
    check_config_identity(config)
    inputs = open_run(config.common.preprocessing_run, experiment_root=experiment_root, masks_root=masks_root)
    remember("file_run", config.common.preprocessing_run, "folder or tree run")
    fovs = inputs.fovs()
    if not fovs:
        raise ValueError(f"The preprocessing run in {inputs.output_folder} has no images.")
    layouts = {fov.relative_path: fov_layout(fov, config.common) for fov in fovs}
    mask_type = _mask_type(config.matching.mask_type, inputs, f"mode={config.mode!r}") if use_masks else None
    mask_layouts = {}
    if mask_type is not None:
        mask_layouts = {fov.relative_path: mask_layout(_record(inputs, fov, mask_type)) for fov in fovs}
        _check_object_set(config, mask_type)
    remember_layouts(config.common.preprocessing_run, layouts, mask_layouts)
    statistics = _csv_statistics(config, inputs, layouts)
    return FileRun(inputs, layouts, mask_type, mask_layouts, statistics)


def _mask_type(wanted: str | None, inputs: RunInputs, reader: str) -> str:
    """Return the mask type *reader* reads: *wanted* (``matching.mask_type``), or the run's only type."""
    types = inputs.mask_types()
    if not types:
        raise ValueError(
            f"{reader} reads masks, but the preprocessing run in {inputs.output_folder} recorded none. "
            f"Run preprocessing with the masks folder ({PREPROCESSING_COMMAND})."
        )
    if wanted is None and len(types) > 1:
        raise ValueError(
            f"The preprocessing run recorded {len(types)} mask types ({', '.join(types)}); set "
            "matching.mask_type to the one to use."
        )
    if wanted is not None and wanted not in types:
        raise ValueError(
            f"matching.mask_type={wanted!r} is not a mask type of the preprocessing run, which recorded "
            f"{', '.join(map(repr, types))}."
        )
    return wanted if wanted is not None else types[0]


def _record(inputs: RunInputs, fov: Fov, mask_type: str) -> Mapping:
    """Return an image's recorded mask of *mask_type*, refusing an image the run recorded none for."""
    record = mask_record(inputs, fov, mask_type)
    if record is None:
        raise ValueError(
            f"The preprocessing run recorded no {mask_type!r} mask for {fov.relative_path}; every image needs one. "
            "Add its mask and run preprocessing again."
        )
    if not record.get("roles"):
        # rc1 writes a run whose mask header was refused (mask_layout) with no axes for it.
        raise ValueError(
            f"The preprocessing run recorded no axes for the mask {record['path']}: its header was not usable (the "
            "run's report says why, for example X before Y). Fix the mask and run preprocessing again."
        )
    return record


def _check_object_set(config: CellPatcherConfig, mask_type: str) -> None:
    """Refuse an object set the mask type gives that cannot be one, before any read (issue #42)."""
    if config.mode == "cell" and config.common.object_set is None:
        validate_object_set(object_set_from_mask_type(mask_type), name=f"The object set from mask type {mask_type!r}")


def _csv_statistics(
    config: CellPatcherConfig, inputs: RunInputs, layouts: Mapping[str, str]
) -> PlaneStatistics | None:
    """Check a CSV normalization scope against the opened run, before any image is read.

    The run's own table, image type and (for ``biological_repeat``) design; every image must
    have rows in the table.
    """
    settings = normalization_settings(config, inputs)
    if not settings.enabled or settings.scope not in CSV_SCOPES:
        return None
    validate_normalization(replace(config, normalization=settings))
    statistics = statistics_of(inputs)
    keep_in_session(statistics)
    check_image_type(settings, statistics)
    if settings.scope == "biological_repeat":
        load_experiment_design(settings, statistics)
    rows_only = PlaneStatistics(statistics.plane_histograms, statistics.records, statistics.csv_sha256)
    check_images(rows_only, dict(layouts))
    return statistics


def normalization_settings(config: CellPatcherConfig, inputs: RunInputs) -> NormalizationConfig:
    """Return ``config.normalization`` reading the run's own table at a CSV scope.

    ``normalization.plane_histograms`` may be left unset (the run's table is used) or name
    the run's own table.

    Raises
    ------
    PreprocessingCheckError
        ``not_the_run_output`` when it names another table.
    """
    from morphobricks_normalization import OUTPUT_FILENAME

    settings = config.normalization
    if not settings.enabled or settings.scope not in CSV_SCOPES:
        return settings
    table = inputs.output_folder / OUTPUT_FILENAME
    if settings.plane_histograms is not None:
        _refuse_other_file("normalization.plane_histograms", Path(settings.plane_histograms).expanduser(), table)
    return replace(settings, plane_histograms=str(table))


def leaves(run: FileRun) -> list[tuple[ConditionRepeatFolder, list[Fov]]]:
    """Group the run's images by condition, biological and technical repeat, sorted by those labels.

    Each group's folder is where its images are now (the run's folders, under the experiment
    root it was opened with).
    """
    groups: dict[tuple[str, str, str], list[Fov]] = {}
    for fov in run.fovs:
        groups.setdefault(fov.key[:3], []).append(fov)
    result = []
    for (condition, biological_repeat, technical_repeat), fovs in sorted(groups.items()):
        folder = run.inputs.experiment_root / PurePosixPath(fovs[0].relative_path).parent
        leaf = ConditionRepeatFolder(condition, biological_repeat, folder, technical_repeat=technical_repeat)
        result.append((leaf, fovs))
    return result


def run_config_for(config: CellPatcherConfig, run: FileRun, labels: Sequence[str]) -> CellPatcherConfig:
    """Return the config the patching core runs with: the run's layouts, labels and table."""
    common = replace(
        config.common,
        image_axes=run.layout,
        mask_axes=run.mask_axes,
        **dict(zip(_LABELS, labels)),
    )
    return replace(config, common=common, normalization=normalization_settings(config, run.inputs))


def patch_fovs(run: FileRun, fovs: Sequence[Fov], config: CellPatcherConfig) -> PatchingResult:
    """Patch images of the run with *config* (from :func:`run_config_for`), reading them through ``RunInputs``.

    One image is read at a time, in its own layout (:meth:`FileRun.image_layout`), and only the
    planes ``z_policy``/``t_policy`` choose; with ``"keep"`` for both, the whole image is its one
    plane. Each image is patched as it would be alone, so its ``patch_id`` and rectangles do not
    depend on the other images of the run (audit A-M1).
    """
    metadata = [fov_metadata(run, fov, config) for fov in fovs]
    return _patch_planes(config, _plane_sources(run, fovs, metadata, config))


def _plane_sources(run: FileRun, fovs: Sequence[Fov], metadata: Sequence[dict], config: CellPatcherConfig) -> Iterator:
    """Yield ``(metadata, planes)`` per image, reading one image (and its mask) at a time."""
    common = config.common
    for fov, meta in zip(fovs, metadata, strict=True):
        mask = None if run.mask_type is None else read_run_mask(run, fov)
        planes = iter_fov_planes(
            run.inputs,
            fov,
            run.image_layout(fov),
            mask,
            run.mask_layout(fov),
            common,
            single_planes=config.normalization.enabled,
        )
        yield meta, planes


def read_run_mask(run: FileRun, fov: Fov, layout: str | None = None) -> np.ndarray:
    """Read an image's mask of the run's type, through ``RunInputs.read_mask``.

    In *layout*, or the mask's own (:meth:`FileRun.mask_layout`) when None.
    """
    record = _record(run.inputs, fov, run.mask_type)
    return read_fov_mask(run.inputs, fov, record, run.mask_layout(fov) if layout is None else layout)


def fov_metadata(run: FileRun, fov: Fov, config: CellPatcherConfig) -> dict:
    """Return an image's ``run_patching(image_metadata=...)`` entry, from the run's records.

    The FOV key is the run's (``position_id`` and ``technical_repeat`` included); the mask
    file and, in ``cell`` mode without ``common.object_set``, the object set come from the
    recorded mask and its type.
    """
    meta = {
        "image_file": PurePosixPath(fov.relative_path).name,
        "source_relative_path": fov.relative_path,
        "position_id": fov.key[3],
        "technical_repeat": fov.key[2],
    }
    if run.mask_type is not None:
        meta["mask_file"] = mask_file_name(_record(run.inputs, fov, run.mask_type))
        if config.mode == "cell" and config.common.object_set is None:
            meta["object_set"] = object_set_from_mask_type(run.mask_type)
    return meta


def validate_image_and_mask_folders(
    image_folder_path: str | os.PathLike,
    mask_folder_path: str | os.PathLike,
    *,
    preprocessing_run: str | os.PathLike | None = None,
    mask_type: str | None = None,
    **removed_options: object,
) -> dict[Path, Path]:
    """Return the mask of every image of a clean preprocessing run, as the run paired them.

    Since v0.2.0 (design v0.4 §6) CellPatcher pairs no file itself: the run is opened with
    ``RunInputs.open`` with *image_folder_path* as the experiment root and *mask_folder_path* as
    its masks folder (where they are now), so every image and mask is checked against the run.

    Parameters
    ----------
    image_folder_path
        The experiment root, where the preprocessed experiment is now.
    mask_folder_path
        Where the run's masks folder is now.
    preprocessing_run
        The run's output folder (``common.preprocessing_run`` of a file run).
    mask_type
        The mask type (``matching.mask_type``); None: the run's only type.
    **removed_options
        The matching options removed in v0.1.1, refused by name.

    Returns
    -------
    dict
        Each image's path mapped to its mask's path (a label image's is inside its OME-Zarr image).

    Raises
    ------
    PreprocessingCheckError
        ``no_preprocessing_run`` without *preprocessing_run*, or when ``RunInputs`` refuses the run.
    ValueError
        For a mask type the run did not record, several types without *mask_type*, or an image
        the run recorded no mask of the type for.
    """
    from morphobricks_preprocessing.masks import FOLDER

    refuse_removed_matching("validate_image_and_mask_folders", removed_options)
    if preprocessing_run is None:
        _refuse_without_run("validate_image_and_mask_folders", "preprocessing_run=")
    inputs = open_run(preprocessing_run, experiment_root=image_folder_path, masks_root=mask_folder_path)
    chosen = _mask_type(mask_type, inputs, "validate_image_and_mask_folders")
    pairs = {}
    for fov in inputs.fovs():
        record = _record(inputs, fov, chosen)
        root = inputs.masks_root if record["source"] == FOLDER else inputs.experiment_root
        pairs[inputs.experiment_root / fov.relative_path] = root / str(record["path"])
    return pairs
