import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from cellpatcher.axes import normalize_axes, require_spatial_axes
from cellpatcher.config import CellPatcherConfig
from cellpatcher.extract import extract_patches
from cellpatcher.export import export_patches
from cellpatcher.filters import filter_regions
from cellpatcher.masking import apply_label_masks_to_patches
from cellpatcher.resize import resize_patches
from cellpatcher.strategies.cell import CellSegmentationStrategy
from cellpatcher.strategies.grid import GridStrategy
from cellpatcher.strategies.random import RandomPatchStrategy
from cellpatcher.strategies.whole_fov import WholeFOVStrategy
from cellpatcher.types import PatchRegion

# Filename for the full run-config JSON written alongside the metadata CSV;
# lets a single PatchRegion CSV row (via its `config_file` column) point back
# to every parameter needed to deterministically reproduce the run.
RUN_CONFIG_FILENAME = "run_config.json"


def _stamp_run_context(config: CellPatcherConfig, patches: list[PatchRegion]) -> None:
    for patch in patches:
        patch.condition = config.common.condition
        patch.biological_repeat = config.common.biological_repeat
        patch.technical_repeat = config.common.technical_repeat


def _stamp_image_metadata(image_list, image_metadata, patches: list[PatchRegion]) -> None:
    """Stamp per-image provenance: plane selection and source filenames.

    *image_metadata* is one dict per image. Plane-selection keys (``"t"``,
    ``"z"``, ``"c"``) match :class:`~cellpatcher.preprocess.PreparedPlane`/
    ``PreparedPlanePair`` ``.metadata`` directly — this is what closes the
    gap where that metadata was available on preprocessing helpers but never
    reached ``PatchRegion``/the export CSV. ``"position_id"`` optionally
    identifies the stable XY acquisition position shared by multiple Z/C/T
    planes. ``"technical_repeat"`` sets the well/dish of that image,
    overriding ``config.common.technical_repeat`` (a folder of one
    repeat can hold images from several wells). ``"image_file"``/
    ``"mask_file"`` stamp source-file provenance, and
    ``"source_relative_path"`` the image's path relative to the experiment
    root; ``run_patching_pipeline``
    supplies these automatically from the folders it read.
    """
    if image_metadata is None:
        return
    if len(image_metadata) != len(image_list):
        raise ValueError("image_metadata must have exactly one entry per image (or be None).")

    per_image = {f"image_{idx:04d}": meta for idx, meta in enumerate(image_metadata)}
    for patch in patches:
        meta = per_image.get(patch.image_id)
        if not meta:
            continue
        if meta.get("t") is not None:
            patch.t = int(meta["t"])
        if meta.get("z") is not None:
            z = int(meta["z"])
            patch.z0, patch.z1 = z, z + 1
        if meta.get("c") is not None:
            patch.c = int(meta["c"])
        if meta.get("position_id") is not None:
            patch.position_id = str(meta["position_id"])
        if meta.get("technical_repeat") is not None:
            patch.technical_repeat = str(meta["technical_repeat"])
        if meta.get("image_file") is not None:
            patch.image_file = str(meta["image_file"])
        if meta.get("mask_file") is not None:
            patch.mask_file = str(meta["mask_file"])
        if meta.get("source_relative_path") is not None:
            patch.source_relative_path = str(meta["source_relative_path"])


def _patch_array_axes(config: CellPatcherConfig) -> str:
    """Axis order of the final patch arrays (split channel patches have no C axis)."""
    axes = normalize_axes(config.common.image_axes)
    return axes.replace("C", "") if config.extraction.split_channels else axes


def _default_position_id(patch: PatchRegion) -> str:
    # One source image is one XY acquisition position, so all of its channel
    # patches share this ID. Condition and biological repeat keep it unique across a tree run.
    source = Path(patch.image_file).stem if patch.image_file else patch.image_id
    return "/".join(str(part) for part in (patch.condition, patch.biological_repeat, source) if part)


def _split_channel_patches(config: CellPatcherConfig, image_list, patches: list[PatchRegion], patch_arrays):
    """Replace each region by one region (and array) per channel of its source image.

    Every channel patch keeps the parent's coordinates and provenance, gets
    ``c`` set to its channel index and ``_c<index>`` appended to its
    ``patch_id``, and shares one ``position_id`` (the supplied one, or a
    default derived from condition/biological repeat/image file).
    """
    axes = normalize_axes(config.common.image_axes)
    if "C" not in axes:
        raise ValueError(
            f"extraction.split_channels=True needs a 'C' axis in common.image_axes (got '{axes}')."
        )
    channel_axis = axes.index("C")
    image_lookup = {f"image_{idx:04d}": image for idx, image in enumerate(image_list)}

    split_patches: list[PatchRegion] = []
    split_arrays = [] if patch_arrays is not None else None
    for index, patch in enumerate(patches):
        image_shape = image_lookup[patch.image_id].shape
        normalize_axes(axes, ndim=len(image_shape))
        position_id = patch.position_id or _default_position_id(patch)
        for channel in range(int(image_shape[channel_axis])):
            split_patches.append(
                replace(patch, patch_id=f"{patch.patch_id}_c{channel}", c=channel, position_id=position_id)
            )
            if split_arrays is not None:
                # Extraction, masking and resizing keep the image's axis
                # order, so the channel axis is at the same position here.
                split_arrays.append(np.take(patch_arrays[index], channel, axis=channel_axis))
    return split_patches, split_arrays


def _stamp_output_dimensions(config: CellPatcherConfig, patches: list[PatchRegion], patch_arrays) -> None:
    if patch_arrays is None or len(patch_arrays) != len(patches):
        return
    for patch, array in zip(patches, patch_arrays):
        axes = normalize_axes(_patch_array_axes(config), ndim=array.ndim)
        y_idx, x_idx = require_spatial_axes(axes)
        patch.output_height = int(array.shape[y_idx])
        patch.output_width = int(array.shape[x_idx])


@dataclass
class PatchingResult:
    patches: list[PatchRegion]
    patch_arrays: list | None = None
    exported_files: list[str] = field(default_factory=list)
    metadata_file: str | None = None


def _normalize_images(images):
    if isinstance(images, (list, tuple)):
        normalized = list(images)
    else:
        normalized = [images]

    if len(normalized) == 0:
        raise ValueError("images must contain at least one image.")

    for image in normalized:
        if not hasattr(image, "shape"):
            raise ValueError("Each image must provide a .shape attribute.")
    return normalized


def _normalize_masks(masks):
    if isinstance(masks, (list, tuple)):
        return list(masks)
    return [masks]


def _apply_filters(config: CellPatcherConfig, image_list, patches: list[PatchRegion]) -> list[PatchRegion]:
    # No intensity/foreground filters configured: skip regrouping and
    # re-extracting patches just to keep every one of them unchanged.
    if (
        config.filters.min_foreground_fraction is None
        and config.filters.foreground_threshold is None
        and config.filters.min_mean_intensity is None
    ):
        return patches

    # filter_regions needs one call per source image (it reads pixel values
    # from that specific image), so patches are grouped by image_id first.
    image_lookup = {f"image_{idx:04d}": image for idx, image in enumerate(image_list)}
    grouped: dict[str, list[PatchRegion]] = {}
    for patch in patches:
        if patch.image_id is None or patch.image_id not in image_lookup:
            raise ValueError(f"Unknown image_id for filtering: {patch.image_id}")
        grouped.setdefault(patch.image_id, []).append(patch)

    kept = []
    # Sorted order keeps output patch order stable/reproducible across runs.
    for image_id in sorted(grouped.keys()):
        image = image_lookup[image_id]
        kept.extend(
            filter_regions(
                image,
                grouped[image_id],
                config.filters,
                axes=config.common.image_axes,
                pad_mode=config.extraction.pad_mode,
                constant_values=config.extraction.constant_values,
            )
        )
    return kept


def _maybe_extract_arrays(config: CellPatcherConfig, image_list, patches: list[PatchRegion], masks=None):
    # Arrays are only materialized when actually needed: either the caller
    # asked for them directly, or export needs pixel data to write files.
    need_arrays = config.common.extract_arrays or config.export.enabled
    if not need_arrays:
        return None

    image_lookup = {f"image_{idx:04d}": image for idx, image in enumerate(image_list)}
    mask_lookup = None
    if config.extraction.apply_mask:
        if config.mode != "cell":
            raise ValueError("extraction.apply_mask=True is only supported in mode='cell'.")
        if masks is None:
            raise ValueError("extraction.apply_mask=True requires masks to be provided.")
        if any(mask is None for mask in masks):
            raise ValueError("extraction.apply_mask=True requires one non-None mask per image.")
        if len(masks) != len(image_list):
            raise ValueError("Number of images and masks must match when extraction.apply_mask=True.")
        mask_lookup = {f"image_{idx:04d}": mask for idx, mask in enumerate(masks)}

    # Group patches by source image so extract_patches can be called once
    # per image rather than once per patch.
    grouped: dict[str, list[PatchRegion]] = {}
    for patch in patches:
        if patch.image_id is None or patch.image_id not in image_lookup:
            raise ValueError(f"Unknown image_id for extraction: {patch.image_id}")
        grouped.setdefault(patch.image_id, []).append(patch)

    extracted = []
    for image_id in sorted(grouped.keys()):
        image = image_lookup[image_id]
        image_patches = grouped[image_id]
        # Step 1: crop each region's pixels out of the source image.
        per_image_arrays = extract_patches(
            image,
            image_patches,
            axes=config.common.image_axes,
            pad_mode=config.extraction.pad_mode,
            constant_values=config.extraction.constant_values,
        )

        if config.extraction.apply_mask:
            if mask_lookup is None or image_id not in mask_lookup:
                raise ValueError(f"Unknown image_id for mask extraction: {image_id}")

            # Step 2 (cell mode only): crop the matching mask region too,
            # then zero out pixels that don't belong to the patch's own
            # label so only the segmented object remains.
            label_patches = extract_patches(
                mask_lookup[image_id],
                image_patches,
                axes=config.common.mask_axes,
                pad_mode=config.extraction.pad_mode,
                constant_values=0,
            )
            per_image_arrays = apply_label_masks_to_patches(
                per_image_arrays,
                label_patches,
                image_patches,
                image_axes=config.common.image_axes,
                label_axes=config.common.mask_axes,
                fill_value=config.extraction.mask_fill_value,
            )

        # Step 3: optionally resize every patch to a uniform output size.
        if config.extraction.output_size is not None:
            per_image_arrays = resize_patches(
                per_image_arrays,
                axes=config.common.image_axes,
                output_size=config.extraction.output_size,
                order=config.extraction.resize_order,
            )
        extracted.extend(per_image_arrays)
    return extracted


def _build_result(
    config: CellPatcherConfig, image_list, patches: list[PatchRegion], masks=None, image_metadata=None
) -> PatchingResult:
    _stamp_run_context(config, patches)
    _stamp_image_metadata(image_list, image_metadata, patches)
    patch_arrays_internal = _maybe_extract_arrays(config, image_list, patches, masks=masks)
    if config.extraction.split_channels:
        # Split after extraction/masking/resizing so each region is only
        # cropped once; filters were already applied to the whole region.
        patches, patch_arrays_internal = _split_channel_patches(config, image_list, patches, patch_arrays_internal)
    _stamp_output_dimensions(config, patches, patch_arrays_internal)

    exported_files: list[str] = []
    metadata_file: str | None = None
    if config.export.enabled:
        if config.common.output_dir is None:
            raise ValueError("common.output_dir is required when export.enabled is True.")

        # Write the full run config first and stamp every patch's
        # config_file before export_patches runs, so the patches.csv it
        # writes below already includes that column (see
        # _stamp_image_metadata's docstring for the same ordering concern).
        out_dir = Path(config.common.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if config.export.save_run_config:
            config_path = out_dir / RUN_CONFIG_FILENAME
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(config), f, indent=2, default=str)
            for patch in patches:
                patch.config_file = RUN_CONFIG_FILENAME

        exported_files = export_patches(
            patches,
            patch_arrays_internal,
            config.common.output_dir,
            image_format=config.export.image_format,
            save_metadata=config.export.save_metadata_csv,
            metadata_filename="patches.csv",
        )
        if config.export.save_metadata_csv:
            metadata_file = str(out_dir / "patches.csv")

    patch_arrays_result = patch_arrays_internal if config.common.extract_arrays else None
    return PatchingResult(
        patches=patches,
        patch_arrays=patch_arrays_result,
        exported_files=exported_files,
        metadata_file=metadata_file,
    )


def run_patching(images=None, masks=None, config=None, image_metadata=None) -> PatchingResult:
    """Generate patch regions (and optionally extract arrays) from in-memory images.

    This is the low-level entry point.  Use :func:`~cellpatcher.run_patching_pipeline`
    if you want to read images and masks from disk folders automatically.

    Parameters
    ----------
    images:
        A single array or a list/tuple of arrays.  Each array must expose a
        ``.shape`` attribute (NumPy, Dask, etc.).  Required for all modes
        except when only coordinate-generation is needed from shape alone.
    masks:
        A single label array or a list/tuple of label arrays, one per image.

        * ``mode='cell'`` — **required**.  Integer label map where each unique
          non-zero value is a separate cell.
        * ``mode='random'`` — **optional**.  When provided, patches are
          restricted to the spatial region covered by the mask.
        * ``mode='grid'`` / ``'whole_fov'`` — not consumed; pass ``None``.
    image_metadata:
        Optional list of dicts, one per image, matching
        :class:`~cellpatcher.preprocess.PreparedPlane`/``PreparedPlanePair``
        ``.metadata`` (keys ``"t"``, ``"z"``, ``"c"``), optionally augmented
        with ``"position_id"`` for the stable spatial acquisition position. When supplied, the
        selected plane indices are stamped onto every resulting patch's
        ``t``/``z0``/``z1``/``c`` fields so the exported metadata CSV records
        exactly which C/Z/T plane each patch came from. The provenance
        keys ``"technical_repeat"``, ``"image_file"``, ``"mask_file"`` and
        ``"source_relative_path"`` are stamped onto the ``PatchRegion``
        fields of the same name.
    config:
        A :class:`~cellpatcher.CellPatcherConfig` instance.  Defaults to
        ``CellPatcherConfig()`` (cell mode with all defaults).

        The easiest way to customise behaviour is to edit attributes directly::

            cfg = CellPatcherConfig(mode="grid")
            cfg.grid.tile_size = 64
            cfg.export.enabled = True
            cfg.common.output_dir = "./patches"
            result = run_patching(images=my_images, config=cfg)

    Returns
    -------
    PatchingResult
        * ``patches`` — list of :class:`~cellpatcher.PatchRegion` objects.
        * ``patch_arrays`` — list of extracted arrays when
          ``config.common.extract_arrays=True``; otherwise ``None``.
        * ``exported_files`` — list of written file paths when
          ``config.export.enabled=True``; otherwise an empty list.
        * ``metadata_file`` — path to the metadata CSV, or ``None``.

    Raises
    ------
    TypeError
        If *config* is not a ``CellPatcherConfig`` instance.
    ValueError
        If required arguments are missing for the chosen mode, or if
        images/masks counts do not match.
    """
    if config is None:
        config = CellPatcherConfig()
    if not isinstance(config, CellPatcherConfig):
        raise TypeError("config must be a CellPatcherConfig instance.")

    if config.mode in {"whole_fov", "grid", "random"} and images is None:
        raise ValueError("images is required for mode 'whole_fov', 'grid', and 'random'.")

    if config.mode == "whole_fov":
        strategy = WholeFOVStrategy()
        image_list = _normalize_images(images)
        patches: list[PatchRegion] = []
        for idx, image in enumerate(image_list):
            image_id = f"image_{idx:04d}"
            patches.extend(
                strategy.generate_regions(
                    image_shape=image.shape, labels=None, image_id=image_id, axes=config.common.image_axes
                )
            )
        patches = _apply_filters(config, image_list, patches)
        return _build_result(config, image_list, patches, masks=None, image_metadata=image_metadata)

    if config.mode == "grid":
        strategy = GridStrategy(config.grid)
        image_list = _normalize_images(images)
        patches: list[PatchRegion] = []
        for idx, image in enumerate(image_list):
            image_id = f"image_{idx:04d}"
            patches.extend(
                strategy.generate_regions(
                    image_shape=image.shape, labels=None, image_id=image_id, axes=config.common.image_axes
                )
            )
        patches = _apply_filters(config, image_list, patches)
        return _build_result(config, image_list, patches, masks=None, image_metadata=image_metadata)

    if config.mode == "cell":
        if images is None:
            raise ValueError("images is required for mode 'cell'.")
        if masks is None:
            raise ValueError("masks is required for mode 'cell'.")

        image_list = _normalize_images(images)
        mask_list = _normalize_masks(masks)

        if len(image_list) != len(mask_list):
            raise ValueError("Number of images and masks must match for mode 'cell'.")

        strategy = CellSegmentationStrategy(config.cell, mask_axes=config.common.mask_axes)
        patches: list[PatchRegion] = []
        for idx, (image, mask) in enumerate(zip(image_list, mask_list)):
            image_id = f"image_{idx:04d}"
            patches.extend(
                strategy.generate_regions(
                    image_shape=image.shape, labels=mask, image_id=image_id, axes=config.common.image_axes
                )
            )
        patches = _apply_filters(config, image_list, patches)
        return _build_result(config, image_list, patches, masks=mask_list, image_metadata=image_metadata)

    if config.mode == "random":
        image_list = _normalize_images(images)

        if masks is None:
            mask_list = [None] * len(image_list)
        elif isinstance(masks, (list, tuple)):
            mask_list = list(masks)
            if len(mask_list) != len(image_list):
                raise ValueError("Number of images and masks must match for mode 'random'.")
        else:
            if len(image_list) > 1:
                raise ValueError("A single mask cannot be used with multiple images in mode 'random'.")
            mask_list = [masks]

        patches: list[PatchRegion] = []
        # Offset the seed by image index so a multi-image batch run is both
        # deterministic (same seed -> same patches) and non-degenerate
        # (different images don't get identical sampling patterns).
        for idx, (image, mask) in enumerate(zip(image_list, mask_list)):
            image_id = f"image_{idx:04d}"
            image_random_config = (
                replace(config.random, seed=int(config.random.seed) + idx)
                if config.random.seed is not None
                else config.random
            )
            strategy = RandomPatchStrategy(image_random_config)
            patches.extend(
                strategy.generate_regions(
                    image_shape=image.shape, labels=mask, image_id=image_id, axes=config.common.image_axes
                )
            )
        patches = _apply_filters(config, image_list, patches)
        return _build_result(config, image_list, patches, masks=mask_list, image_metadata=image_metadata)
