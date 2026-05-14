from dataclasses import dataclass, field

from cellpatcher.config import CellPatcherConfig
from cellpatcher.extract import extract_patches
from cellpatcher.export import export_patches
from cellpatcher.filters import filter_regions
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
    warnings: list[str] = field(default_factory=list)
    exported_files: list[str] = field(default_factory=list)
    metadata_file: str | None = None


def _normalize_images(images):
    if isinstance(images, (list, tuple)):
        normalized = list(images)
    else:
        normalized = [images]

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

def _maybe_extract_arrays(config: CellPatcherConfig, image_list, patches: list[PatchRegion]):
    need_arrays = config.common.extract_arrays or config.export.enabled
    if not need_arrays:
        return None

    image_lookup = {f"image_{idx:04d}": image for idx, image in enumerate(image_list)}
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
        if config.extraction.output_size is not None:
            per_image_arrays = resize_patches(
                per_image_arrays,
                axes=config.common.image_axes,
                output_size=config.extraction.output_size,
                order=config.extraction.resize_order,
                keep_range=config.extraction.keep_range,
            )
        extracted.extend(per_image_arrays)
    return extracted


def _build_result(config: CellPatcherConfig, image_list, patches: list[PatchRegion]) -> PatchingResult:
    patch_arrays_internal = _maybe_extract_arrays(config, image_list, patches)

    exported_files: list[str] = []
    metadata_file: str | None = None
    if config.export.enabled:
        if config.common.output_dir is None:
            raise ValueError("common.output_dir is required when export.enabled is True.")
        if patch_arrays_internal is None:
            raise ValueError("Export requires extracted patch arrays.")

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
        warnings=[],
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
            patches.extend(strategy.generate_regions(image_shape=image.shape, labels=None, image_id=image_id))
        patches = _apply_filters(config, image_list, patches)
        return _build_result(config, image_list, patches)

    if config.mode == "grid":
        strategy = GridStrategy(config.grid)
        image_list = _normalize_images(images)
        patches: list[PatchRegion] = []
        for idx, image in enumerate(image_list):
            image_id = f"image_{idx:04d}"
            patches.extend(strategy.generate_regions(image_shape=image.shape, labels=None, image_id=image_id))
        patches = _apply_filters(config, image_list, patches)
        return _build_result(config, image_list, patches)

    if config.mode == "cell":
        if images is None:
            raise ValueError("images is required for mode 'cell'.")
        if masks is None:
            raise ValueError("masks is required for mode 'cell'.")

        image_list = _normalize_images(images)
        mask_list = _normalize_masks(masks)

        if len(image_list) != len(mask_list):
            raise ValueError("Number of images and masks must match for mode 'cell'.")

        strategy = CellSegmentationStrategy(config.cell)
        patches: list[PatchRegion] = []
        for idx, (image, mask) in enumerate(zip(image_list, mask_list)):
            image_id = f"image_{idx:04d}"
            patches.extend(strategy.generate_regions(image_shape=image.shape, labels=mask, image_id=image_id))
        patches = _apply_filters(config, image_list, patches)
        return _build_result(config, image_list, patches)

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

        strategy = RandomPatchStrategy(config.random)
        patches: list[PatchRegion] = []
        for idx, (image, mask) in enumerate(zip(image_list, mask_list)):
            image_id = f"image_{idx:04d}"
            patches.extend(strategy.generate_regions(image_shape=image.shape, labels=mask, image_id=image_id))
        patches = _apply_filters(config, image_list, patches)
        return _build_result(config, image_list, patches)

    raise NotImplementedError(f"Mode '{config.mode}' is not implemented yet.")
