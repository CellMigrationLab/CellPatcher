from dataclasses import dataclass, field, replace

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
    if (
        config.filters.min_foreground_fraction is None
        and config.filters.foreground_threshold is None
        and config.filters.min_mean_intensity is None
    ):
        return patches

    image_lookup = {f"image_{idx:04d}": image for idx, image in enumerate(image_list)}
    grouped: dict[str, list[PatchRegion]] = {}
    for patch in patches:
        if patch.image_id is None or patch.image_id not in image_lookup:
            raise ValueError(f"Unknown image_id for filtering: {patch.image_id}")
        grouped.setdefault(patch.image_id, []).append(patch)

    kept = []
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
    need_arrays = config.common.extract_arrays or config.export.enabled
    if not need_arrays:
        return None

    image_lookup = {f"image_{idx:04d}": image for idx, image in enumerate(image_list)}
    mask_lookup = None
    if config.extraction.apply_mask:
        if masks is None:
            raise ValueError("extraction.apply_mask=True requires masks to be provided.")
        if len(masks) != len(image_list):
            raise ValueError("Number of images and masks must match when extraction.apply_mask=True.")
        mask_lookup = {f"image_{idx:04d}": mask for idx, mask in enumerate(masks)}

    grouped: dict[str, list[PatchRegion]] = {}
    for patch in patches:
        if patch.image_id is None or patch.image_id not in image_lookup:
            raise ValueError(f"Unknown image_id for extraction: {patch.image_id}")
        grouped.setdefault(patch.image_id, []).append(patch)

    extracted = []
    for image_id in sorted(grouped.keys()):
        image = image_lookup[image_id]
        image_patches = grouped[image_id]
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

        if config.extraction.output_size is not None:
            per_image_arrays = resize_patches(
                per_image_arrays,
                axes=config.common.image_axes,
                output_size=config.extraction.output_size,
                order=config.extraction.resize_order,
            )
        extracted.extend(per_image_arrays)
    return extracted


def _build_result(config: CellPatcherConfig, image_list, patches: list[PatchRegion], masks=None) -> PatchingResult:
    patch_arrays_internal = _maybe_extract_arrays(config, image_list, patches, masks=masks)

    exported_files: list[str] = []
    metadata_file: str | None = None
    if config.export.enabled:
        if config.common.output_dir is None:
            raise ValueError("common.output_dir is required when export.enabled is True.")

        exported_files = export_patches(
            patches,
            patch_arrays_internal,
            config.common.output_dir,
            image_format=config.export.image_format,
            save_metadata=config.export.save_metadata_csv,
            metadata_filename="patches.csv",
        )
        if config.export.save_metadata_csv:
            from pathlib import Path

            metadata_file = str(Path(config.common.output_dir) / "patches.csv")

    patch_arrays_result = patch_arrays_internal if config.common.extract_arrays else None
    return PatchingResult(
        patches=patches,
        patch_arrays=patch_arrays_result,
        exported_files=exported_files,
        metadata_file=metadata_file,
    )


def run_patching(images=None, masks=None, config=None) -> PatchingResult:
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
        return _build_result(config, image_list, patches, masks=None)

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
        return _build_result(config, image_list, patches, masks=None)

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
        return _build_result(config, image_list, patches, masks=mask_list)

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
        return _build_result(config, image_list, patches, masks=mask_list)
