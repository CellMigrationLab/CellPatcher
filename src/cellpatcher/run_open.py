"""Open a clean preprocessing run through the brick's ``RunInputs`` (layer: io).

Design v0.4 §6: CellPatcher reads a preprocessing run only through ``RunInputs.open``, which
refuses a run before v0.4.0, a table or design whose SHA-256 is not the manifest's, and an
image or mask changed or missing since the run. An experiment moved or copied since its run is
opened with where it is now (``experiment_root``, ``masks_root``). Re-exported by
:mod:`cellpatcher.statistics`, where these functions were before CP-2a.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from morphobricks_normalization import PlaneRecord
    from morphobricks_preprocessing.run_inputs import RunInputs

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
    not the manifest's, and any image or mask of the run changed or missing (design v0.4 §6).
    *experiment_root* and *masks_root* say where the experiment and its masks folder are now,
    for an experiment moved or copied since the run (None: where the run recorded them); the
    images and masks there are the ones checked and read.

    Raises
    ------
    PreprocessingCheckError
        ``preprocessing_run_refused`` with the brick's reason when ``RunInputs``
        refuses the run, or ``unreadable_plane_histograms`` when its table
        cannot be read.
    """
    from morphobricks_normalization.csv_io import CsvFormatError
    from morphobricks_preprocessing.run_inputs import RunInputs, RunInputsError
    from morphobricks_preprocessing.validation import error

    from cellpatcher.statistics import PreprocessingCheckError

    folder = Path(output_folder).expanduser()
    roots = [Path(root).expanduser() if root is not None else None for root in (experiment_root, masks_root)]
    try:
        return RunInputs.open(folder, *roots)
    except RunInputsError as exc:
        raise PreprocessingCheckError([
            error(
                "preprocessing_run_refused",
                f"The preprocessing run in {folder} cannot be used: {exc}",
                [str(folder)],
                "Run preprocessing v0.4 on the experiment as it is now and give CellPatcher that run. For an "
                "experiment only moved or copied since its run, give CellPatcher the experiment and masks "
                "folder where they are now (the image folder or root, and mask_folder_path or mask_root_path).",
            )
        ]) from exc
    except CsvFormatError as exc:
        raise PreprocessingCheckError([
            error(
                "unreadable_plane_histograms",
                f"The plane histogram table cannot be read: {exc}",
                [str(folder)],
                "Give the plane_histograms.csv the preprocessing brick wrote, or run it again.",
            )
        ]) from exc


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


def recorded_image_type(
    plane_histograms: str | os.PathLike,
    manifest: str | os.PathLike | None = None,
    *,
    experiment_root: str | os.PathLike | None = None,
    masks_root: str | os.PathLike | None = None,
) -> str | None:
    """Return the image type the preprocessing run recorded, read through ``RunInputs``.

    ``normalization.image_type`` must equal it (normalization brick decision
    N34). The run is opened as :func:`load_plane_statistics` opens it.

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
    str or None
        ``"fluorescence"`` or ``"rgb"``; None when the run recorded no image
        type, so the configuration decides.

    Raises
    ------
    FileNotFoundError
        If the table does not exist.
    PreprocessingCheckError
        As :func:`load_plane_statistics`.
    """
    statistics = load_plane_statistics(plane_histograms, manifest, experiment_root=experiment_root, masks_root=masks_root)
    return statistics.inputs.image_type()
