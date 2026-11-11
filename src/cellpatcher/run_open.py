"""Open a clean preprocessing run through the brick's ``RunInputs`` (layer: io).

Design v0.4 §6: CellPatcher reads a preprocessing run only through ``RunInputs.open``, which
refuses a run before v0.4.0, a table or design whose SHA-256 is not the manifest's, an
image or mask changed or missing since the run and (since preprocessing v0.4.0rc2) an image or
mask in the experiment that the run did not record. An experiment moved or copied since its run is
opened with where it is now (``experiment_root``, ``masks_root``). Re-exported by
:mod:`cellpatcher.statistics`, where these functions were before CP-2a. :func:`cached_statistics`
(moved from :mod:`cellpatcher.normalization` in CP-2b) keeps an opened run for one call.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from cellpatcher.run_session import recalled, remember, session_statistics

if TYPE_CHECKING:
    from morphobricks_normalization import PlaneRecord
    from morphobricks_preprocessing import ValidationIssue
    from morphobricks_preprocessing.run_inputs import Fov, RunInputs

    from cellpatcher.config import NormalizationConfig
    from cellpatcher.statistics import PlaneStatistics


def load_plane_statistics(
    plane_histograms: str | os.PathLike,
    manifest: str | os.PathLike | None = None,
    *,
    experiment_root: str | os.PathLike | None = None,
    masks_root: str | os.PathLike | None = None,
) -> PlaneStatistics:
    """Open the preprocessing run that wrote ``plane_histograms.csv``, through ``RunInputs``.

    The run's output folder is the table's folder, opened with :func:`open_run`. CellPatcher
    reads no other table, manifest or design than the run's own.

    Parameters
    ----------
    plane_histograms
        Path of the ``plane_histograms.csv`` written by the preprocessing brick.
    manifest
        Path of its ``run_manifest.json``, or None. Only the run's own manifest,
        beside the table, can be given.
    experiment_root, masks_root
        Where the experiment and its masks folder are now, for an experiment moved or copied
        since the run; None: where the run recorded them.

    Returns
    -------
    PlaneStatistics

    Raises
    ------
    FileNotFoundError
        If the table does not exist.
    PreprocessingCheckError
        If the table or manifest is not the run's own, ``RunInputs`` refuses
        the run, or the table cannot be read, has no rows or duplicate planes.
    """
    from morphobricks_preprocessing import MANIFEST_FILENAME

    from cellpatcher.statistics import require_normalization

    mbn = require_normalization()

    csv_path = Path(plane_histograms).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"The plane histogram table {csv_path} does not exist. Give the path of the "
            f"{mbn.OUTPUT_FILENAME} written by the preprocessing brick, or run the preprocessing "
            "brick on this experiment first."
        )
    folder = csv_path.parent
    _refuse_other_file("normalization.plane_histograms", csv_path, folder / mbn.OUTPUT_FILENAME)
    if manifest is not None:
        _refuse_other_file("normalization.manifest", Path(manifest).expanduser(), folder / MANIFEST_FILENAME)
    return statistics_of(open_run(folder, experiment_root=experiment_root, masks_root=masks_root))


def cached_statistics(
    plane_histograms: str | os.PathLike,
    manifest: str | os.PathLike | None = None,
    *,
    experiment_root: str | os.PathLike | None = None,
    masks_root: str | os.PathLike | None = None,
) -> PlaneStatistics:
    """:func:`~cellpatcher.statistics.load_plane_statistics`, opened once per CellPatcher call.

    Within one call (:mod:`cellpatcher.run_session`: a tree run normalizes every leaf with
    the same, possibly large, table) the run is opened once; outside a call, or in the next
    one, it is opened again, so ``RunInputs.open`` checks every image and mask of the run each
    time. Nothing is reused across calls.
    """
    csv_path = Path(plane_histograms).expanduser()
    statistics = session_statistics(
        csv_path,
        lambda: load_plane_statistics(csv_path, manifest, experiment_root=experiment_root, masks_root=masks_root),
    )
    if manifest is not None:
        _refuse_other_file("normalization.manifest", Path(manifest).expanduser(), statistics.manifest_path)
    return statistics


def settings_statistics(settings: NormalizationConfig) -> PlaneStatistics:
    """Return :func:`cached_statistics` of ``settings.plane_histograms``, opened where the experiment is now.

    ``normalization.experiment_root``/``masks_root`` say where the run's experiment and masks
    folder are when they moved since the run (owner decision of 2026-11-02); a folder or tree
    run gives them as its arguments instead.
    """
    return cached_statistics(
        settings.plane_histograms,
        settings.manifest,
        experiment_root=settings.experiment_root,
        masks_root=settings.masks_root,
    )


def statistics_of(inputs: RunInputs) -> PlaneStatistics:
    """Return the :class:`PlaneStatistics` of a run opened with :func:`open_run`.

    Raises
    ------
    PreprocessingCheckError
        If the table has no rows, or several rows for one plane.
    """
    from morphobricks_preprocessing import MANIFEST_FILENAME

    from cellpatcher.statistics import PlaneStatistics, require_normalization

    mbn = require_normalization()

    csv_path = inputs.output_folder / mbn.OUTPUT_FILENAME
    records = inputs.histograms()
    _refuse_table_rows(csv_path, records)
    return PlaneStatistics(
        plane_histograms=csv_path,
        records=records,
        csv_sha256=inputs.manifest.csv_sha256,
        manifest=inputs.manifest,
        manifest_path=inputs.output_folder / MANIFEST_FILENAME,
        inputs=inputs,
    )


def open_run(
    output_folder: str | os.PathLike,
    *,
    experiment_root: str | os.PathLike | None = None,
    masks_root: str | os.PathLike | None = None,
) -> RunInputs:
    """Open a clean preprocessing run with the brick's ``RunInputs.open``.

    ``RunInputs.open`` refuses a run written before v0.4.0, a table or design whose SHA-256 is
    not the manifest's, any image or mask of the run changed or missing (design v0.4 §6) and,
    since preprocessing v0.4.0rc2, any image or mask of the experiment the run did not record.
    *experiment_root* and *masks_root* say where the experiment and its masks folder are now,
    for an experiment moved or copied since the run (None: where the run recorded them); the
    images and masks there are the ones checked and read.

    Raises
    ------
    PreprocessingCheckError
        ``unrecorded_files`` (images or masks the run did not record),
        ``zarr_extra_missing`` or ``preprocessing_run_refused``, with the brick's
        reason, when ``RunInputs`` refuses the run, or
        ``unreadable_plane_histograms`` when its table cannot be read.
    """
    from morphobricks_normalization.csv_io import CsvFormatError
    from morphobricks_preprocessing.manifest import ManifestFormatError
    from morphobricks_preprocessing.run_inputs import RunInputs, RunInputsError
    from morphobricks_preprocessing.validation import error

    from cellpatcher.statistics import PreprocessingCheckError

    folder = Path(output_folder).expanduser()
    roots = [Path(root).expanduser() if root is not None else None for root in (experiment_root, masks_root)]
    try:
        before = _manifest_sha256(folder)
        if recalled("manifest_sha256", folder) not in {None, before}:
            raise RunInputsError("its run_manifest.json changed since this call first opened it.")
        inputs = RunInputs.open(folder, *roots)
        if _manifest_sha256(folder) != before:
            raise RunInputsError("its run_manifest.json changed while it was opened.")
    except (RunInputsError, ManifestFormatError) as exc:
        # ManifestFormatError: a manifest of another version (preprocessing v0.5.1 reads version 4 only,
        # older runs are refused, never converted) or not a manifest at all.
        raise PreprocessingCheckError([_refusal_issue(exc, folder)]) from exc
    except CsvFormatError as exc:
        raise PreprocessingCheckError([
            error(
                "unreadable_plane_histograms",
                f"The plane histogram table cannot be read: {exc}",
                [str(folder)],
                "Give the plane_histograms.csv the preprocessing brick wrote, or run it again.",
            )
        ]) from exc
    remember("manifest_sha256", folder, before)
    return inputs


DESCRIPTION_REQUIRED = "A_DESCRIPTION_REQUIRED"
"""The code of a run without a published experiment description (Spec ADR 0002 AR10, AR11 (1))."""


def _refusal_issue(exc: Exception, folder: Path) -> ValidationIssue:
    """Return the issue for ``RunInputs.open`` refusing the run in *folder*, by its reason.

    Since preprocessing v0.5.1 ``RunInputs.open`` refuses a run without a published description
    (``DescriptionRequiredError``: ``A_DESCRIPTION_REQUIRED``, with CellPatcher's fix). Since
    preprocessing v0.4.0rc2 the brick refuses a run whose experiment holds images or
    masks it did not record (``UnrecordedFilesError``: none is ever left out), and a run
    with OME-Zarr images when the zarr package is missing. Their messages say what to do;
    every other refusal gets the generic advice.
    """
    from morphobricks_preprocessing.run_inputs import (
        DescriptionRequiredError,
        UnrecordedFilesError,
        ZarrUnavailableError,
    )
    from morphobricks_preprocessing.validation import error

    if isinstance(exc, DescriptionRequiredError):
        # Every run needs one (Spec ADR 0002 AR10, owner policy of 2026-11-10): its conditions, biological
        # repeats (repeat_group), technical repeats, channels and image type are read from it through
        # RunInputs, never from folder names or the derived experiment_design.csv. Nothing falls back.
        return error(
            DESCRIPTION_REQUIRED,
            f"The preprocessing run in {folder} has no published experiment description (morphobricks.toml), "
            "and CellPatcher reads a run's conditions, biological repeats, technical repeats and their IDs "
            f"only from its description ({exc}).",
            [str(folder)],
            "Describe the experiment and publish its description in the preprocessing notebook "
            "(morphobricks-preprocessing describe lists what is still open), run preprocessing again and "
            "give CellPatcher that run.",
        )
    if isinstance(exc, UnrecordedFilesError):
        files = [*exc.files.images, *exc.files.masks]
        return error(
            "unrecorded_files",
            f"The preprocessing run in {folder} cannot be used: {exc}",
            files,
            "CellPatcher patches only the images the run recorded, and never leaves one out without a word: run "
            "preprocessing again on the experiment as it is, or exclude those images in preprocessing.",
        )
    if isinstance(exc.__cause__, ZarrUnavailableError):
        return error(
            "zarr_extra_missing",
            f"The preprocessing run in {folder} cannot be used: {exc}",
            [str(folder)],
            'Install CellPatcher with its zarr extra (pip install "cellpatcher[zarr]"; the notebook installs it '
            "on Colab), then run CellPatcher again.",
        )
    return error(
        "preprocessing_run_refused",
        f"The preprocessing run in {folder} cannot be used: {exc}",
        [str(folder)],
        "Run preprocessing again on the experiment as it is now and give CellPatcher that run. For an "
        "experiment only moved or copied since its run, give CellPatcher the experiment and masks "
        "folder where they are now (the image folder or root, and mask_folder_path or mask_root_path).",
    )


def refuse_changed_manifest(output_folder: str | os.PathLike, opened_sha256: str) -> None:
    """Refuse a run whose ``run_manifest.json`` changed since this call opened it.

    ``run_config.json`` records the manifest's SHA-256 taken at the open; a manifest rewritten
    during the run (preprocessing run again on the same output folder) no longer names what was
    read, so nothing is recorded.

    Raises
    ------
    PreprocessingCheckError
        ``preprocessing_run_refused`` when the manifest's SHA-256 is not *opened_sha256*.
    """
    from morphobricks_preprocessing.validation import error

    from cellpatcher.statistics import PreprocessingCheckError

    folder = Path(output_folder).expanduser()
    if _manifest_sha256(folder) != opened_sha256:
        raise PreprocessingCheckError([
            error(
                "preprocessing_run_refused",
                f"The preprocessing run in {folder} cannot be used: its run_manifest.json changed during this "
                "CellPatcher run, so the run read is no longer the one in that folder.",
                [str(folder)],
                "Wait until preprocessing has finished writing its run, then run CellPatcher again.",
            )
        ])


def _manifest_sha256(folder: Path) -> str | None:
    """Return the SHA-256 of the run's ``run_manifest.json`` (recorded in ``run_config.json``), None if absent."""
    from morphobricks_preprocessing import MANIFEST_FILENAME, file_sha256

    path = folder / MANIFEST_FILENAME
    return file_sha256(path) if path.is_file() else None


def _refuse_other_file(setting: str, given: Path, expected: Path) -> None:
    """Refuse a file *setting* names that is not the preprocessing run's own *expected* file."""
    from morphobricks_preprocessing.validation import error

    from cellpatcher.statistics import PreprocessingCheckError

    if given.resolve() == expected.resolve():
        return
    raise PreprocessingCheckError([
        error(
            "not_the_run_output",
            f"{setting} names {given}, but CellPatcher reads a preprocessing run only through its "
            f"output folder, where this file is {expected.name}.",
            [str(given)],
            f"Give the {expected.name} the preprocessing brick wrote in its output folder, unrenamed "
            "and in place.",
        )
    ])


def _refuse_table_rows(csv_path: Path, records: tuple[PlaneRecord, ...]) -> None:
    """Refuse a table without rows, or with several rows for one plane."""
    from morphobricks_preprocessing.validation import error, plural

    from cellpatcher.statistics import PreprocessingCheckError

    issues = []
    if not records:
        issues.append(error(
            "empty_plane_histograms",
            "The plane histogram table has no rows.",
            [str(csv_path)],
            "Run the preprocessing brick on this experiment again.",
        ))
    counts: dict[str, int] = defaultdict(int)
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
    if issues:
        raise PreprocessingCheckError(issues)


IMAGE_TYPES = {"channels": "fluorescence", "rgb": "rgb"}
"""``normalization.image_type`` of each declared ``Fov.image_type`` (experiment-v1 §4.4b): an ordinary
channel image (``channels``) is normalized per channel (``fluorescence``), an ``rgb`` unit as RGB."""


def declared_image_type(fov: Fov) -> str:
    """Return the ``normalization.image_type`` of *fov*'s declared image type (``Fov.image_type``).

    The run's published description declares it per imaging unit, never a header (experiment-v1
    §4.4b); CellPatcher takes it from ``RunInputs`` and nowhere else.

    Raises
    ------
    PreprocessingCheckError
        ``image_type_not_declared`` for a ``Fov`` without a declared type (a ``Fov`` built by hand):
        nothing is guessed.
    """
    from morphobricks_preprocessing.validation import error

    from cellpatcher.statistics import PreprocessingCheckError

    if fov.image_type not in IMAGE_TYPES:
        raise PreprocessingCheckError([
            error(
                "image_type_not_declared",
                f"The run gives no declared image type for {fov.relative_path} (got {fov.image_type!r}), "
                "and CellPatcher takes the image type only from the run's description.",
                [fov.relative_path],
                "Open the run with RunInputs.open on its published description; never build a Fov by hand.",
            )
        ])
    return IMAGE_TYPES[fov.image_type]


def run_image_type(inputs: RunInputs) -> str:
    """Return the one ``normalization.image_type`` every position of the run declares.

    Raises
    ------
    PreprocessingCheckError
        ``image_types_mixed`` when the positions declare different types (rgb and ordinary channel
        units in one run), or as :func:`declared_image_type`.
    """
    from morphobricks_preprocessing.validation import error

    from cellpatcher.statistics import PreprocessingCheckError

    by_type: dict[str, list[str]] = defaultdict(list)
    for fov in inputs.fovs():
        by_type[declared_image_type(fov)].append(fov.relative_path)
    if len(by_type) == 1:
        return next(iter(by_type))
    raise PreprocessingCheckError([
        error(
            "image_types_mixed",
            "The run's description declares different image types for its positions ("
            + "; ".join(f"{kind}: {len(paths)} images" for kind, paths in sorted(by_type.items()))
            + "), so no one normalization.image_type applies to the run.",
            sorted(path for paths in by_type.values() for path in paths),
            "Patch the rgb and the other units in separate preprocessing runs.",
        )
    ])


def check_image_type(settings: NormalizationConfig, inputs: RunInputs) -> None:
    """Refuse a ``normalization.image_type`` that contradicts the run's declared image type (N34, P24).

    The image type of each position is the one its unit declares in the run's published description
    (``Fov.image_type``: ``rgb``, or ``channels`` for an ordinary channel image, which is
    normalized per channel, ``"fluorescence"``). The declared type wins: a setting that says
    otherwise is refused, naming both, and never overrides it.

    Raises
    ------
    PreprocessingCheckError
        ``image_type_differs`` when a position declares another image type, or as
        :func:`declared_image_type`.
    """
    from morphobricks_preprocessing.validation import error

    from cellpatcher.statistics import PreprocessingCheckError

    differing: dict[tuple[str, str], list[str]] = {}
    for fov in inputs.fovs():
        declared = declared_image_type(fov)
        if declared != settings.image_type:
            differing.setdefault((fov.image_type, declared), []).append(fov.relative_path)
    if not differing:
        return
    files = sorted(path for paths in differing.values() for path in paths)
    declared_text = " and ".join(f"{unit!r} (normalized as {kind!r})" for unit, kind in sorted(differing))
    settings_text = " or ".join(f"normalization.image_type={kind!r}" for kind in sorted({k for _, k in differing}))
    raise PreprocessingCheckError([
        error(
            "image_type_differs",
            f"The run's description declares the image type {declared_text} for {len(files)} of its "
            f"{len(inputs.fovs())} images, but normalization.image_type is {settings.image_type!r}. The "
            "declared image type decides; a setting never overrides it.",
            files,
            f"Set {settings_text}, or correct the image type in the experiment description and run "
            "preprocessing again if the description is wrong.",
        )
    ])


def recorded_image_type(
    plane_histograms: str | os.PathLike,
    manifest: str | os.PathLike | None = None,
    *,
    experiment_root: str | os.PathLike | None = None,
    masks_root: str | os.PathLike | None = None,
) -> str:
    """Return the image type the run's description declares, read through ``RunInputs`` (``Fov.image_type``).

    ``normalization.image_type`` must equal it (normalization brick decision
    N34; the declared type wins, a contradicting setting is refused). The run is
    opened as :func:`load_plane_statistics` opens it.

    Parameters
    ----------
    plane_histograms
        Path of the run's ``plane_histograms.csv``.
    manifest
        Path of its ``run_manifest.json``, or None; only the run's own.
    experiment_root, masks_root
        As in :func:`load_plane_statistics`.

    Returns
    -------
    str
        ``"fluorescence"`` (the description's ordinary channel units) or ``"rgb"``.

    Raises
    ------
    FileNotFoundError
        If the table does not exist.
    PreprocessingCheckError
        As :func:`load_plane_statistics` and :func:`run_image_type`.
    """
    statistics = load_plane_statistics(plane_histograms, manifest, experiment_root=experiment_root, masks_root=masks_root)
    return run_image_type(statistics.inputs)
