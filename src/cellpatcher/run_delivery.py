"""Deliver the patches of a folder or tree run lazily, read through its preprocessing run (layer: orchestration).

Lazy delivery of ``run_patching`` takes the images as arrays (:func:`cellpatcher.iter_patch_arrays`).
The images of a folder or tree run are files, and since v0.2.0 (CP-2b, design v0.4 §6) a file is
read only through the clean preprocessing run the patches were made from: the run is opened with
``RunInputs.open`` (where the experiment and masks folder are now), each patch's image is the run's
image at its ``source_relative_path`` with its recorded FOV key, and only the patch's plane is read
with ``RunInputs.read_plane``, its mask with ``RunInputs.read_mask``, in the image's own layout. Nothing
here names an axis, pairs a mask or reads a file itself.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

from cellpatcher.delivery import _finish_patch
from cellpatcher.file_runs import open_file_run, read_run_mask, run_config_for
from cellpatcher.io import SINGLE_CONDITION_LABEL
from cellpatcher.planes import patch_plane_choice, select_mask_plane
from cellpatcher.run_reading import read_fov
from cellpatcher.run_session import statistics_session

if TYPE_CHECKING:
    import numpy as np
    from morphobricks_preprocessing.run_inputs import Fov

    from cellpatcher.config import CellPatcherConfig
    from cellpatcher.types import PatchRegion


def iter_run_patch_arrays(
    patches: Iterable[PatchRegion],
    config: CellPatcherConfig,
    *,
    experiment_root: str | os.PathLike | None = None,
    masks_root: str | os.PathLike | None = None,
) -> Iterator[tuple[PatchRegion, np.ndarray]]:
    """Yield ``(patch, array)`` for patches of a folder or tree run, one at a time.

    Each array equals the one eager delivery gave: the patch's plane is read through the
    preprocessing run, then extracted, normalized with the patch's recorded parameters,
    masked and resized as the run did. The run is opened when iteration starts.

    Parameters
    ----------
    patches
        ``PatchRegion`` objects of a folder or tree run (its ``result.patches``, any leaves).
    config
        The run's config (the one given to the run, or ``load_config`` of its
        ``run_config.json``), with ``common.preprocessing_run``.
    experiment_root, masks_root
        Where the experiment and the run's masks folder are now; None: where the run recorded
        them. Masks are read when ``extraction.apply_mask`` is on.

    Yields
    ------
    tuple
        ``(patch, array)``.

    Raises
    ------
    PreprocessingCheckError
        Without ``common.preprocessing_run``, or when ``RunInputs`` refuses the run.
    ValueError
        For a patch whose image is not an image of the run, or whose FOV key is not the one the
        run recorded for it.
    """
    with statistics_session():  # closed before the first patch is yielded
        run = open_file_run(
            config, experiment_root, masks_root, use_masks=config.extraction.apply_mask, where="iter_run_patch_arrays"
        )
    delivered = run_config_for(config, run, (SINGLE_CONDITION_LABEL,) * 3)
    fovs = {fov.relative_path: fov for fov in run.fovs}
    mask: tuple[str, np.ndarray] | None = None
    for patch in patches:
        fov = _patch_fov(fovs, patch)
        layout = run.image_layout(fov)
        choice = patch_plane_choice(patch, layout, delivered.common)
        plane, axes = read_fov(run.inputs, fov, layout, choice)
        mask_plane, mask_axes = None, None
        if run.mask_type is not None:
            if mask is None or mask[0] != fov.relative_path:
                mask = (fov.relative_path, read_run_mask(run, fov))  # one image's mask in memory at a time
            mask_plane, mask_axes = select_mask_plane(mask[1], run.mask_layout(fov), choice)
        yield patch, _finish_patch(patch, plane, axes, mask_plane, mask_axes, delivered)


def _patch_fov(fovs: Mapping[str, Fov], patch: PatchRegion) -> Fov:
    """Return the run's image of *patch*, refusing a patch of another image or with another FOV key."""
    fov = fovs.get(str(patch.source_relative_path))
    if fov is None:
        raise ValueError(
            f"Patch {patch.patch_id!r} comes from {patch.source_relative_path!r}, which is not an image of the "
            "preprocessing run: give the patches of a run made from this preprocessing run."
        )
    key = (patch.condition, patch.biological_repeat, patch.technical_repeat, patch.position_id)
    if key != tuple(fov.key):
        raise ValueError(
            f"Patch {patch.patch_id!r} has the FOV key {key}, but the preprocessing run recorded {tuple(fov.key)} "
            f"for {fov.relative_path}: it was not made from this run."
        )
    return fov

