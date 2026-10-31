"""Deliver patches one at a time, or from exported files (``docs/normalization_contract.md`` §12).

Besides the eager arrays of ``run_patching(... extract_arrays=True)`` and the
exported files, patches can be delivered:

- **lazily**, with :func:`iter_patch_arrays` (or :func:`patch_array` for one
  patch): each patch is rebuilt from its source image with the pipeline's own
  steps (plane, extract, normalize with the patch's recorded parameters, mask,
  resize, channel), so only one patch, and one image or plane, is in memory at
  a time; ``cellpatcher.torch_dataset.PatchTorchDataset`` is built on it;
- **from exported files**, with :func:`iter_exported_patches` (or
  :func:`load_exported_patch`): files exported with
  ``normalization.export_values="raw"`` are normalized on load with each row's
  ``Normalization_Offset_C<i>``/``Normalization_Scale_C<i>`` columns (or the
  ``;``-separated ``normalization_offset``/``normalization_scale`` of earlier
  tables); parameters are never recomputed.

All delivery modes give the same arrays for the same configuration (raw files
of resized or masked patches up to float32 rounding).
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cellpatcher.axes import normalize_axes
from cellpatcher.extract import extract_patch
from cellpatcher.io import (
    read_array,
    read_image,
    refuse_image_header_axes,
    refuse_mask_header_axes,
    refuse_transposed_mask_axes,
)
from cellpatcher.masking import apply_label_mask_to_patch
from cellpatcher.normalization import PatchNormalization, parameters_from_row
from cellpatcher.planes import apply_plane_choice, patch_plane_choice, read_file_plane, select_mask_plane
from cellpatcher.resize import resize_patch


def patch_axes(config, patch) -> str:
    """Axes of a patch's delivered array: the image's, without resolved Z/T planes or a split channel."""
    common = config.common
    choice = patch_plane_choice(patch, common.image_axes, common)
    axes = "".join(axis for axis in normalize_axes(common.image_axes) if choice.value(axis) is None)
    if _splits_channel(config, patch, axes):
        axes = axes.replace("C", "")
    return axes


def patch_array(patch, image, config, *, mask=None) -> np.ndarray:
    """Rebuild one patch's array from its source image, as the pipeline delivers it.

    Parameters
    ----------
    patch
        A ``PatchRegion`` from a run with *config* (its ``z0``/``t``, ``c``,
        ``label_id`` and ``normalization_*`` fields are used).
    image
        Its source image, in ``config.common.image_axes``: an array, or a
        file path (for a TIFF or OME-Zarr image, only the patch's plane is
        read).
    config
        The run's ``CellPatcherConfig``.
    mask
        The image's mask (array or path) when ``extraction.apply_mask`` is on.

    Returns
    -------
    numpy.ndarray
        The patch, normalized (float32) when the patch records a
        normalization, with :func:`patch_axes`.
    """
    common = config.common
    choice = patch_plane_choice(patch, common.image_axes, common)
    plane, axes = _image_plane(image, choice, config)
    mask_plane, mask_axes = _mask_plane(mask, choice, config)
    return _finish_patch(patch, plane, axes, mask_plane, mask_axes, config)


def iter_patch_arrays(patches, images: Sequence, config, *, masks: Sequence | None = None) -> Iterator[tuple]:
    """Yield ``(patch, array)`` for every patch, one at a time (lazy delivery).

    Parameters
    ----------
    patches
        ``PatchRegion`` objects of one run (``run_patching`` or one leaf of a
        folder/tree run); their ``image_id`` (``image_<index>``) indexes
        *images*. A run with ``extract_arrays=False`` and no export gives
        them, with their normalization recorded, without holding any array.
    images
        The run's source images, in image order: arrays or file paths (for a
        folder run, ``cellpatcher.list_supported_files(folder)``).
    config
        The run's ``CellPatcherConfig``.
    masks
        The images' masks, in the same order, when ``extraction.apply_mask``
        is on.

    Yields
    ------
    tuple
        ``(patch, array)``; each array equals the one eager delivery gives.
    """
    common = config.common
    plane_cache: dict = {}
    mask_cache: dict = {}
    for patch in patches:
        index = int(patch.image_id.rsplit("_", 1)[1])
        choice = patch_plane_choice(patch, common.image_axes, common)
        key = (index, choice.t, choice.z)
        if key not in plane_cache:
            plane_cache.clear()  # one plane in memory at a time
            plane_cache[key] = _image_plane(images[index], choice, config)
        plane, axes = plane_cache[key]
        mask_plane, mask_axes = None, None
        if masks is not None and masks[index] is not None:
            if key not in mask_cache:
                mask_cache.clear()
                mask_cache[key] = _mask_plane(masks[index], choice, config)
            mask_plane, mask_axes = mask_cache[key]
        yield patch, _finish_patch(patch, plane, axes, mask_plane, mask_axes, config)


def load_exported_patch(row: Mapping, *, root_dir, run_config: Mapping | None = None) -> np.ndarray:
    """Load one exported patch named by a ``patches.csv`` row, normalized.

    Files exported with ``normalization.export_values="raw"`` (from
    *run_config*, the run's ``run_config.json``) are normalized on load with
    the row's parameter columns; normalized
    files and runs without normalization are returned as stored.
    """
    array = read_array(Path(root_dir) / str(row["patch_file"]))
    return _normalize_on_load(row, array, run_config)


def iter_exported_patches(metadata_file, *, root_dir=None) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield ``(row, array)`` for every patch of an exported ``patches.csv``, normalized on load.

    *root_dir* is where ``patch_file`` paths are relative to: the CSV's
    folder by default. Each row's ``config_file`` (``run_config.json``) says
    whether its files hold raw values; see :func:`load_exported_patch`.
    """
    metadata_file = Path(metadata_file)
    root = Path(root_dir) if root_dir is not None else metadata_file.parent
    run_configs: dict[str, dict | None] = {}
    with metadata_file.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        name = row.get("config_file") or ""
        if name not in run_configs:
            path = root / name if name else None
            run_configs[name] = json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else None
        yield row, load_exported_patch(row, root_dir=root, run_config=run_configs[name])


def _normalize_on_load(row: Mapping, array, run_config: Mapping | None) -> np.ndarray:
    """Normalize a loaded patch file when its run exported raw values (else return it as stored)."""
    settings = (run_config or {}).get("normalization") or {}
    if settings.get("export_values") != "raw":
        return array
    parameters = parameters_from_row(row, settings)
    if parameters is None:
        return array
    patch = _row_patch(row)
    clip = settings.get("clip")
    return PatchNormalization(parameters).apply(
        array, patch_axes(_run_config(run_config), patch), clip=tuple(clip) if clip else None
    )


def raw_export_array(patch, plane, plane_axes: str, array, config) -> np.ndarray:
    """The file content of a patch exported with ``export_values="raw"`` (contract §12).

    The raw crop in the source dtype when the patch is only cropped; with
    resizing or ``apply_mask``, the delivered (normalized) *array* mapped back
    to raw units as float32 (owner decision), so normalizing on load gives
    the delivered array up to float32 rounding, a NaN mask fill included.
    """
    normalization = PatchNormalization.from_patch(patch)
    axes = plane_axes
    if config.extraction.apply_mask or config.extraction.output_size is not None:
        if _splits_channel(config, patch, axes):
            axes = axes.replace("C", "")
        return normalization.unapply(array, axes)
    raw = extract_patch(
        plane, patch, axes=axes, pad_mode=config.extraction.pad_mode, constant_values=config.extraction.constant_values
    )
    if _splits_channel(config, patch, axes):
        raw = np.take(raw, patch.c, axis=axes.index("C"))
    return raw


def _splits_channel(config, patch, axes: str) -> bool:
    return bool(config.extraction.split_channels and patch.c is not None and "C" in axes)


def _image_plane(image, choice, config):
    common = config.common
    if isinstance(image, (str, os.PathLike)):
        refuse_image_header_axes([image], common.image_axes)
        return read_file_plane(image, common.image_axes, choice, common)
    return apply_plane_choice(image, common.image_axes, choice)


def _mask_plane(mask, choice, config):
    if mask is None:
        return None, None
    common = config.common
    if isinstance(mask, (str, os.PathLike)):
        refuse_mask_header_axes([mask], common.mask_axes)
        mask = read_image(mask, common.mask_axes)
    else:
        refuse_transposed_mask_axes(common.mask_axes)
        mask = np.asarray(mask)
    return select_mask_plane(mask, normalize_axes(common.mask_axes, mask.ndim), choice)


def _finish_patch(patch, plane, axes, mask_plane, mask_axes, config) -> np.ndarray:
    """Extract, (split the channel,) normalize, mask and resize one patch, as the pipeline does."""
    extraction = config.extraction
    array = extract_patch(
        plane, patch, axes=axes, pad_mode=extraction.pad_mode, constant_values=extraction.constant_values
    )
    image_axes = axes
    if _splits_channel(config, patch, axes):
        # Channels are independent at every step, so taking the channel
        # first gives the pipeline's split-after-processing values.
        array = np.take(array, patch.c, axis=axes.index("C"))
        image_axes = axes.replace("C", "")
    normalization = PatchNormalization.from_patch(patch)
    if normalization is not None:
        array = normalization.apply(array, image_axes, clip=config.normalization.clip)
    if extraction.apply_mask:
        if mask_plane is None:
            raise ValueError("extraction.apply_mask=True needs the images' masks (masks=...).")
        labels = extract_patch(mask_plane, patch, axes=mask_axes, pad_mode=extraction.pad_mode, constant_values=0)
        array = apply_label_mask_to_patch(
            array,
            labels,
            label_id=patch.label_id,
            image_axes=image_axes,
            label_axes=mask_axes,
            fill_value=extraction.mask_fill_value,
        )
    if extraction.output_size is not None:
        array = resize_patch(array, axes=image_axes, output_size=extraction.output_size, order=extraction.resize_order)
    return array


def _row_patch(row: Mapping):
    """The fields of a ``patches.csv`` row that :func:`patch_axes` needs."""

    def number(name):
        value = row.get(name)
        return None if value in (None, "") else int(value)

    return SimpleNamespace(
        patch_id=row.get("patch_id"),
        z0=number("z0"),
        t=number("t"),
        c=number("c"),
    )


def _run_config(data: Mapping):
    """The config of a ``run_config.json``, for the axes of its patches.

    A run normalized at a scope removed since (``condition``,
    ``technical_repeat``; normalization brick N29) still loads here: its
    parameters are in ``patches.csv``, and only the patch axes are needed.
    """
    from cellpatcher.config import REMOVED_NORMALIZATION_SCOPES
    from cellpatcher.orchestration import _config_from_dict

    data = dict(data)
    normalization = data.get("normalization")
    if isinstance(normalization, Mapping) and normalization.get("scope") in REMOVED_NORMALIZATION_SCOPES:
        data["normalization"] = {**normalization, "scope": "image"}
    return _config_from_dict(data)
