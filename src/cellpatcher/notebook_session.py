"""What the CellPatcher notebook does with a preprocessing run, without any widget (layer: orchestration).

The notebook (:mod:`cellpatcher.notebook_ui`) only shows these results. Since v0.2.0 every
file is read through the preprocessing run's ``RunInputs`` (design v0.4 §6): the user points
the notebook to the run's output folder and, only for an experiment moved or copied since its
run, to where the experiment and its masks folder are now. Nothing here discovers, pairs or
reads a file itself: :func:`summarize_run` opens the run, :func:`preview_image` patches one
of its images in memory with the file-run code (with normalization, also without it, to show
the patches before and after), :func:`group_parameters_of_run` gives the normalization
parameters of every group of the run's table, :func:`scan_object_sizes` measures the objects
of its masks, and :func:`export_patches_of_run` is
:func:`cellpatcher.run_patching_condition_repeat_tree` with one combined table.
"""

from __future__ import annotations

import os
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from cellpatcher.cell_scan import scan_cell_patch_size
from cellpatcher.file_runs import (
    masks_used,
    normalization_settings,
    open_file_run,
    patch_fovs,
    read_run_mask,
    run_config_for,
)
from cellpatcher.mask_policy import resolved_masks
from cellpatcher.normalization import CSV_SCOPES, NormalizationWarning, group_parameters
from cellpatcher.normalization_review import BeforeAfter, before_after
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.run_open import open_run, run_image_type
from cellpatcher.run_reading import run_layout
from cellpatcher.run_session import statistics_session

if TYPE_CHECKING:
    import numpy as np

    from cellpatcher.cell_scan import CellPatchSizeSummary
    from cellpatcher.config import CellPatcherConfig
    from cellpatcher.orchestration import ConditionRepeatResult
    from cellpatcher.pipeline import PatchingResult

PREVIEW_WHERE = "The notebook preview"
"""How the preview names itself in a refusal."""


@dataclass(frozen=True)
class RunSummary:
    """A clean preprocessing run, as the notebook shows it before patching.

    Attributes
    ----------
    run
        The run's output folder.
    experiment_root, masks_root
        Where the experiment and its masks folder are read from (given, or where the run
        recorded them); ``masks_root`` is None for a run without a masks folder.
    moved_experiment, moved_masks
        Whether the user gave where the experiment or its masks folder is now; only then are
        they passed to the run (otherwise the run's recorded places are used).
    images
        The run's images, relative to the experiment root, sorted.
    groups
        ``((condition, biological_repeat, technical_repeat), number of images)``, sorted.
    channels
        The canonical channel IDs of the run's description (``Fov.channel_ids``), every image's, in order of
        first appearance: the channels ``extraction.split_channels`` can name. Empty when no image has any.
    layout
        The T and C axes some image of the run has (:func:`cellpatcher.run_reading.run_layout`); each
        image is read in its own roles.
    mask_types
        The mask types the run recorded.
    image_type
        The ``normalization.image_type`` of the image type the run's description declares
        (``Fov.image_type``: ``"fluorescence"`` for ordinary channel units, or ``"rgb"``).
    """

    run: Path
    experiment_root: Path
    masks_root: Path | None
    moved_experiment: bool
    moved_masks: bool
    images: tuple[str, ...]
    groups: tuple[tuple[tuple[str, str, str], int], ...]
    channels: tuple[str, ...]
    layout: str
    mask_types: tuple[str, ...]
    image_type: str

    @property
    def experiment_argument(self) -> Path:
        """The experiment root to give a file run: where it is now (the run needs one)."""
        return self.experiment_root

    @property
    def masks_argument(self) -> Path | None:
        """The masks folder to give a file run: only when the user said where it is now."""
        return self.masks_root if self.moved_masks else None


def _path_or_none(text: str | os.PathLike | None) -> Path | None:
    """Return *text* as a path with ``~`` expanded, or None for an empty value."""
    if text is None or not str(text).strip():
        return None
    return Path(str(text).strip()).expanduser()


def summarize_run(
    preprocessing_run: str | os.PathLike,
    experiment_root: str | os.PathLike | None = None,
    masks_root: str | os.PathLike | None = None,
) -> RunSummary:
    """Open a clean preprocessing run with ``RunInputs.open`` and summarize what it recorded.

    Parameters
    ----------
    preprocessing_run
        The run's output folder.
    experiment_root, masks_root
        Where the experiment and its masks folder are now, only for an experiment moved or
        copied since its run; empty or None: where the run recorded them.

    Returns
    -------
    RunSummary

    Raises
    ------
    PreprocessingCheckError
        When ``RunInputs`` refuses the run: a run before v0.4, a changed, missing or
        unrecorded image or mask (``unrecorded_files``), and so on.
    ValueError
        For a run without images.
    """
    run, given_root, given_masks = (_path_or_none(path) for path in (preprocessing_run, experiment_root, masks_root))
    if run is None:
        raise ValueError("Give the preprocessing run's output folder (the folder holding run_manifest.json).")
    with statistics_session():
        inputs = open_run(run, experiment_root=given_root, masks_root=given_masks)
    fovs = inputs.fovs()
    if not fovs:
        raise ValueError(f"The preprocessing run in {run} has no images.")
    groups = Counter(fov.key[:3] for fov in fovs)
    return RunSummary(
        run=run,
        experiment_root=inputs.experiment_root,
        masks_root=inputs.masks_root,
        moved_experiment=given_root is not None,
        moved_masks=given_masks is not None,
        images=tuple(fov.relative_path for fov in fovs),
        groups=tuple(sorted(groups.items())),
        channels=tuple(dict.fromkeys(channel for fov in fovs for channel in fov.channel_ids or ())),
        layout=run_layout(fovs),
        mask_types=inputs.mask_types(),
        image_type=run_image_type(inputs),
    )


def file_run_config(config: CellPatcherConfig, summary: RunSummary) -> CellPatcherConfig:
    """Return *config* set to read *summary*'s run: ``common.preprocessing_run`` and no file axes."""
    common = replace(config.common, preprocessing_run=str(summary.run), image_axes="YX", mask_axes="YX")
    return replace(config, common=common)


@dataclass(frozen=True)
class Preview:
    """One image of the run patched in memory, nothing written.

    Attributes
    ----------
    image
        The image's relative path in the experiment.
    plane
        Its first plane (first T and Z, every channel), ``CYX``, as ``RunInputs.read_plane``
        gives it.
    result
        The patches of the whole image, with their arrays.
    before_after
        With normalization, the first patches without and with it
        (:func:`cellpatcher.normalization_review.before_after`); empty without normalization.
    """

    image: str
    plane: np.ndarray
    result: PatchingResult
    before_after: tuple[BeforeAfter, ...] = ()


def preview_image(config: CellPatcherConfig, summary: RunSummary, image: str) -> Preview:
    """Patch one image of the run in memory, through the run, as a tree run would.

    Parameters
    ----------
    config
        The settings (``export`` is turned off and arrays are kept).
    summary
        The opened run (:func:`summarize_run`).
    image
        One of ``summary.images``.

    Returns
    -------
    Preview
        With normalization, also the first patches as the same run gives them without it
        (the image is patched twice; the patches are paired by ``patch_id``).
    """
    config = file_run_config(config, summary)
    config = replace(config, export=replace(config.export, enabled=False))
    config.common.extract_arrays = True
    with statistics_session():
        use_masks = masks_used(config, summary.masks_argument)
        run = open_file_run(
            config, summary.experiment_argument, summary.masks_argument, use_masks=use_masks, where=PREVIEW_WHERE
        )
        chosen = [fov for fov in run.fovs if fov.relative_path == image]
        if not chosen:
            raise ValueError(f"Choose an image of the preprocessing run to preview; {image!r} is not one of them.")
        (fov,) = chosen
        result = patch_fovs(run, [fov], run_config_for(config, run, fov.key[:3]))
        pairs = ()
        if config.normalization.enabled:
            raw_config = replace(config, normalization=replace(config.normalization, enabled=False))
            raw = patch_fovs(run, [fov], run_config_for(raw_config, run, fov.key[:3]))
            pairs = before_after(raw.patches, raw.patch_arrays, result.patches, result.patch_arrays)
        plane = run.inputs.read_plane(fov, 0, 0)
    return Preview(image=image, plane=plane, result=result, before_after=pairs)


@dataclass(frozen=True)
class GroupParameters:
    """The normalization parameters of every group of the run's table, before any patch is made.

    Attributes
    ----------
    method, scope
        ``normalization.method`` and ``normalization.scope``.
    rows
        :func:`cellpatcher.normalization.group_parameters`: one entry per group and channel
        (and time point with ``per_time_point``): ``group``, ``channel``, ``t``, ``planes``,
        then ``offset``, ``scale`` and ``exact``, or ``problem`` for a group that stops a run.
    notes
        The normalization warnings these settings give (for example that plane statistics
        include the background), shown with the table rather than dropped.
    """

    method: str
    scope: str
    rows: tuple[dict, ...]
    notes: tuple[str, ...]


def group_parameters_of_run(config: CellPatcherConfig, summary: RunSummary) -> GroupParameters:
    """Return the normalization parameters of every group of the run, from its ``plane_histograms.csv``.

    The run is opened and checked as an export would open it; the groups and their
    parameters are the ones the run's patches get (:func:`cellpatcher.normalization.group_parameters`,
    fitted by the normalization brick).

    Raises
    ------
    ValueError
        When normalization is off, or for the patch scope, whose parameters come from each patch.
    PreprocessingCheckError
        When the run is refused, or its table or design does not fit the settings.
    """
    config = file_run_config(config, summary)
    settings = config.normalization
    if not settings.enabled:
        raise ValueError("Normalization is off: tick 'Normalize the patches' in step 2 first.")
    if settings.scope not in CSV_SCOPES:
        raise ValueError(
            "The patch scope has no parameters per group: each patch gets them from its own pixels. "
            "Show the preview to see some, or choose the image or biological-repeat scope."
        )
    with statistics_session(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", NormalizationWarning)
        run = open_file_run(
            config, summary.experiment_argument, summary.masks_argument, use_masks=False,
            where="The parameters per group",
        )
        rows = group_parameters(replace(config, normalization=normalization_settings(config, run.inputs)))
    notes = []
    for item in caught:
        if issubclass(item.category, NormalizationWarning):
            notes.append(str(item.message))
        else:  # not the table's to show: passed on unchanged
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    notes = tuple(dict.fromkeys(notes))
    return GroupParameters(method=settings.method, scope=settings.scope, rows=tuple(rows), notes=notes)


def scan_object_sizes(config: CellPatcherConfig, summary: RunSummary) -> CellPatchSizeSummary:
    """Measure the objects of every mask of the run's mask type, with the cell settings of *config*.

    Each mask is read through the run (``RunInputs.read_mask``); the sizes are those
    :func:`cellpatcher.scan_cell_patch_size` gives, from which a dataset patch size is chosen.
    """
    config = file_run_config(replace(config, mode="cell"), summary)
    with statistics_session():
        run = open_file_run(
            config, summary.experiment_argument, summary.masks_argument, use_masks=True, where="The object scan"
        )
        masks = [read_run_mask(run, fov, run.mask_axes) for fov in run.fovs]  # one layout for the scan
    shapes = [mask.shape for mask in masks]
    with resolved_masks():  # read through the resolver: a binary mask was labelled when read
        return scan_cell_patch_size(
            shapes, masks, config=config.cell, image_axes=run.mask_axes, mask_axes=run.mask_axes
        )


def refuse_output_inside_inputs(output_dir: Path, summary: RunSummary) -> None:
    """Refuse an output folder inside the experiment, its masks folder or the preprocessing run.

    Raises
    ------
    ValueError
        Naming the folder it is inside.
    """
    output = output_dir.resolve()
    for name, folder in (("experiment", summary.experiment_root), ("masks", summary.masks_root),
                         ("preprocessing run", summary.run)):
        if folder is not None and (output == folder.resolve() or folder.resolve() in output.parents):
            raise ValueError(
                f"The output folder {output_dir} is inside the {name} folder {folder}. Choose a folder outside it, "
                "for example next to the experiment."
            )


def export_patches_of_run(
    config: CellPatcherConfig,
    summary: RunSummary,
    output_dir: str | os.PathLike,
    progress: Callable | None = None,
) -> list[ConditionRepeatResult]:
    """Patch every image of the run and export them, with one ``patches.csv`` for the experiment.

    :func:`cellpatcher.run_patching_condition_repeat_tree` with ``combine_metadata=True``:
    ``patches.csv``, ``images.csv`` and ``run_config.json`` in *output_dir*, the patch files in
    ``<condition>/<biological_repeat>[/<technical_repeat>]`` below it.

    Raises
    ------
    ValueError
        For an output folder inside the inputs (:func:`refuse_output_inside_inputs`), or as
        the tree run.
    PreprocessingCheckError
        As the tree run (the run is opened again, so a file changed since the summary is refused).
    """
    output = _path_or_none(output_dir)
    if output is None:
        raise ValueError("Give the output folder for the patches.")
    refuse_output_inside_inputs(output, summary)
    config = file_run_config(config, summary)
    config = replace(config, export=replace(config.export, enabled=True))
    config.common.output_dir = str(output)
    return run_patching_condition_repeat_tree(
        summary.experiment_argument,
        config=config,
        mask_root_path=summary.masks_argument,
        combine_metadata=True,
        progress_callback=progress,
    )

