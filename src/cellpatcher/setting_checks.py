"""Strict re-check of the scientific settings at the point a run uses them (pure; no I/O).

The config dataclasses are mutable and documented as edited in place
(``cfg.random.min_mask_fraction = 0.5``), so a check in ``__post_init__`` alone is bypassed by
an assignment after construction. The checks here are shared by ``__post_init__`` and by every
route that executes a run, and use the helpers of :mod:`cellpatcher.fractions` (never ``float()``,
``int()`` or ``bool()``, which turn ``True`` into 1.0 and ``"0.5"`` into 0.5).

Every scientific setting of every section is checked here, whatever the mode:
:func:`check_run_settings` is the one entry point of the routes that execute a run, and each
section's ``__post_init__`` calls the same ``check_*_settings`` function.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath

from cellpatcher.fractions import (
    check_choice,
    check_integer,
    check_real_number,
    check_strict_bool,
    check_unit_fraction,
)
from cellpatcher.ids import validate_object_set
from cellpatcher.planes import validate_plane_policies

MODES = ("cell", "random", "whole_fov", "grid")
CENTER_ON = ("centroid", "bbox_center")
BINARY_MASK_POLICIES = ("connected_components",)  # cellpatcher.object_geometry; no default (None: no policy)
SEED_SCOPES = ("plane_channel", "plane", "position")
GRID_PARTIAL_MODES = ("flush", "clipped")
PAD_MODES = ("clip", "constant")
IMAGE_FORMATS = ("tif", "tiff", "npy")
NORMALIZATION_METHODS = ("minmax", "percentile", "zscore", "robust_zscore")
NORMALIZATION_SCOPES = ("biological_repeat", "image", "patch")
# Removed without a deprecation period (normalization brick decision N29).
REMOVED_NORMALIZATION_SCOPES = ("condition", "technical_repeat")
POOLING_WEIGHTS = ("pixels", "conditions")
IMAGE_TYPES = ("fluorescence", "rgb")
# StarDist's percentile normalization is the one RGB method (normalization brick decision N33).
RGB_METHODS = ("percentile",)
PATCH_REGIONS = ("mask", "bbox", "patch")
EXPORT_VALUES = ("normalized", "raw")


def unknown_scope_message(scope: object) -> str:
    """Return the message for a ``normalization.scope`` that does not exist, as the normalization brick words it."""
    message = (
        f"Scope '{scope}' does not exist. Choose 'biological_repeat' (one biological repeat across all "
        "conditions), 'image' or 'patch'."
    )
    if scope in REMOVED_NORMALIZATION_SCOPES:
        message += (
            f" The '{scope}' scope was removed (October 2026): normalizing per {scope.replace('_', ' ')} "
            "removed real differences between them. Raw patches exported with it still normalize on load."
        )
    return message


def _check_path(value: object, name: str) -> None:
    """Refuse a path setting that is not None, text or a ``pathlib`` path."""
    if value is not None and not isinstance(value, (str, PurePath)):
        raise ValueError(f"{name} must be None or a path; got {value!r}.")


def check_random_settings(random: object) -> None:
    """Refuse a ``RandomConfig`` with a bad ``seed_scope``, a non-integer size, count or seed, or a bad fraction."""
    check_choice(random.seed_scope, SEED_SCOPES, "seed_scope")
    check_integer(random.patch_size, "patch_size", minimum=1)
    check_integer(random.n_patches, "n_patches", minimum=1)
    if random.seed is not None:
        check_integer(random.seed, "seed")
    check_unit_fraction(random.min_mask_fraction, "min_mask_fraction")


def check_grid_settings(grid: object) -> None:
    """Refuse a ``GridConfig`` with a non-integer ``tile_size``, a bad ``overlap``, flag or ``partial_mode``."""
    check_integer(grid.tile_size, "tile_size", minimum=1)
    check_real_number(grid.overlap, "overlap")
    if not 0 <= grid.overlap < 1:
        raise ValueError(f"overlap must satisfy 0 <= overlap < 1; got {grid.overlap!r}.")
    check_strict_bool(grid.include_partial, "include_partial")
    check_partial_mode(grid)


def check_partial_mode(grid: object) -> None:
    """Refuse an unknown ``partial_mode``, or ``"clipped"`` without ``include_partial``."""
    check_choice(grid.partial_mode, GRID_PARTIAL_MODES, "partial_mode")
    if grid.partial_mode == "clipped" and not grid.include_partial:
        raise ValueError(
            "partial_mode='clipped' chooses which edge tiles include_partial adds; "
            "set include_partial=True with it (include_partial=False keeps full tiles only)."
        )


def check_threshold_text(text: str, name: str = "foreground_threshold") -> float | None:
    """Return the percentile of ``"percentile:<0-100>"``, None for a named statistic; refuse other text.

    The named statistics are ``"mean"``, ``"median"`` and ``"otsu"``.

    The one reader of the text threshold grammar: the number after ``percentile:`` is parsed from
    its text here (a value of the setting is never converted), and NaN or a value outside [0, 100]
    is refused.
    """
    spec = text.strip().lower()
    if spec in ("mean", "median", "otsu"):
        return None
    if not spec.startswith("percentile:"):
        raise ValueError(
            f"Unsupported threshold string: {text!r} ({name} is a number, 'mean', 'median', 'otsu' or "
            "'percentile:<0-100>')."
        )
    try:
        percentile = float(spec.split(":", 1)[1])
    except ValueError:
        raise ValueError(f"Invalid percentile threshold format in {name}. Use 'percentile:<0-100>'.") from None
    if not 0 <= percentile <= 100:
        raise ValueError(f"Percentile threshold of {name} must be in [0, 100].")
    return percentile


def check_filter_settings(filters: object) -> None:
    """Refuse a ``FilterConfig`` whose fraction, threshold or mean is a bool, text (a threshold name aside) or NaN."""
    check_unit_fraction(filters.min_foreground_fraction, "min_foreground_fraction")
    if filters.min_mean_intensity is not None:
        check_real_number(filters.min_mean_intensity, "min_mean_intensity")
    if isinstance(filters.foreground_threshold, str):
        check_threshold_text(filters.foreground_threshold)
    elif filters.foreground_threshold is not None:
        check_real_number(filters.foreground_threshold, "foreground_threshold")


def check_cell_settings(cell: object) -> None:
    """Refuse a ``CellConfig`` with a bad choice, ``mask_channel``, size or area bound, or ``exclude_border``.

    ``mask_channel`` is None or an integer >= 0 (an ``int`` or NumPy integer, never a bool, text or
    a float such as 1.9); that it is a channel of the mask is checked where the mask is read.
    """
    check_choice(cell.center_on, CENTER_ON, "center_on")
    _check_binary_mask_policy(cell.binary_mask_policy)
    check_integer(cell.binary_mask_connectivity, "binary_mask_connectivity", minimum=1)
    if cell.binary_mask_connectivity > 2:
        raise ValueError(
            "binary_mask_connectivity must be 1 (4-connected) or 2 (8-connected); "
            f"got {cell.binary_mask_connectivity!r}."
        )
    if cell.mask_channel is not None:
        check_integer(cell.mask_channel, "mask_channel", minimum=0)
    if not isinstance(cell.patch_size, str) or cell.patch_size != "auto":
        check_integer(cell.patch_size, "patch_size", minimum=1)
    check_integer(cell.min_area, "min_area", minimum=1)
    if cell.max_area is not None:
        check_integer(cell.max_area, "max_area", minimum=cell.min_area)
    check_strict_bool(cell.exclude_border, "exclude_border")


def check_channel_selection(split_channels: object) -> None:
    """Refuse a ``split_channels`` that is neither a bool nor a non-empty list of distinct canonical channel IDs.

    Channels are chosen by their canonical channel ID only, never by position (owner decision of
    13 November 2026): each ID is resolved to its C index **in each image** by the image's own channel IDs.

    Raises
    ------
    ValueError
        For any other value, an empty ID or an ID listed twice.
    """
    if isinstance(split_channels, bool):
        return
    if (
        not isinstance(split_channels, (list, tuple))
        or not split_channels
        or not all(isinstance(channel, str) and channel for channel in split_channels)
    ):
        raise ValueError(
            "split_channels must be True (one patch per channel), False (one multichannel patch) or a non-empty "
            f"list of canonical channel IDs (text, one patch per channel named); got {split_channels!r}."
        )
    if len(set(split_channels)) != len(split_channels):
        raise ValueError(f"split_channels lists a channel more than once: {list(split_channels)!r}.")


def check_extraction_settings(extraction: object) -> None:
    """Refuse an ``ExtractionConfig`` with a bad ``pad_mode``, size, order, flag, fill or channel selection."""
    check_choice(extraction.pad_mode, PAD_MODES, "pad_mode")
    size = extraction.output_size
    if size is not None:
        if isinstance(size, (tuple, list)) and len(size) == 2:
            for side in size:
                check_integer(side, "output_size", minimum=1)
        elif isinstance(size, (tuple, list)):
            raise ValueError("output_size must be None, a positive int, or tuple/list of two positive ints.")
        else:
            check_integer(size, "output_size", minimum=1)
    check_integer(extraction.resize_order, "resize_order", minimum=0)
    check_strict_bool(extraction.apply_mask, "apply_mask")
    check_real_number(extraction.constant_values, "constant_values", finite=False)
    check_real_number(extraction.mask_fill_value, "mask_fill_value", finite=False)
    check_channel_selection(extraction.split_channels)


def check_export_settings(export: object) -> None:
    """Refuse an ``ExportConfig`` with a flag that is not literally ``True`` or ``False``, or a bad ``image_format``."""
    check_strict_bool(export.enabled, "enabled")
    check_strict_bool(export.save_metadata_csv, "save_metadata_csv")
    check_strict_bool(export.save_run_config, "save_run_config")
    if not isinstance(export.image_format, str):
        raise ValueError(f"image_format must be one of: 'tif', 'tiff', 'npy'; got {export.image_format!r}.")
    check_choice(export.image_format.lower(), IMAGE_FORMATS, "image_format")


def _check_binary_mask_policy(policy: object) -> None:
    """Refuse a ``binary_mask_policy`` that is neither None (no policy) nor a P27 policy, naming ``single_object``."""
    if policy == "single_object":
        raise ValueError(
            "binary_mask_policy='single_object' was removed (CP-2): there is no default policy. Leave it None, so "
            "a binary mask is refused as an object set, or set 'connected_components' to label it."
        )
    if policy is not None:
        check_choice(policy, BINARY_MASK_POLICIES, "binary_mask_policy")


def check_match_settings(matching: object) -> None:
    """Refuse a ``MatchConfig`` whose ``mask_type`` or ``mask_id`` is neither None nor a non-empty name."""
    for name in ("mask_type", "mask_id"):
        value = getattr(matching, name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"matching.{name} must be None or a non-empty name; got {value!r}.")


def _number_pair(value: object, name: str, form: str) -> tuple[float, float]:
    """Return the two numbers of *value* (a tuple or list; a JSON round trip makes lists), refusing others."""
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"{name} must be {form}two numbers (low, high); got {value!r}.")
    for number in value:
        check_real_number(number, name)
    return value[0], value[1]


def check_normalization_ranges(normalization: object) -> None:
    """Refuse ``percentile`` and ``clip`` unless each is two finite numbers in order (never via ``float()``)."""
    low, high = _number_pair(normalization.percentile, "normalization.percentile", "")
    if not 0 <= low < high <= 100:
        raise ValueError(f"normalization.percentile must satisfy 0 <= low < high <= 100; got ({low}, {high}).")
    if normalization.clip is not None:
        low, high = _number_pair(normalization.clip, "normalization.clip", "None or ")
        if not low < high:
            raise ValueError(f"normalization.clip must be finite with low < high; got ({low}, {high}).")


def _check_normalization_choices(normalization: object) -> None:
    """Refuse a ``NormalizationConfig`` text setting outside its choices, or an RGB method that is not percentile."""
    check_choice(normalization.method, NORMALIZATION_METHODS, "normalization.method")
    if not isinstance(normalization.scope, str) or normalization.scope not in NORMALIZATION_SCOPES:
        raise ValueError(f"normalization.scope: {unknown_scope_message(normalization.scope)}")
    check_choice(normalization.pooling, POOLING_WEIGHTS, "normalization.pooling")
    if not isinstance(normalization.image_type, str) or normalization.image_type not in IMAGE_TYPES:
        raise ValueError(
            "normalization.image_type must be 'fluorescence' (each channel normalized separately) or 'rgb' "
            f"(colour images); got {normalization.image_type!r}."
        )
    if normalization.image_type == "rgb" and normalization.method not in RGB_METHODS:
        raise ValueError(
            "normalization.image_type='rgb' is normalized with StarDist's percentile method only, which scales "
            f"the three colour channels together; got method={normalization.method!r}. Use method='percentile'."
        )
    if normalization.patch_region is not None:
        check_choice(normalization.patch_region, PATCH_REGIONS, "normalization.patch_region")
    check_choice(normalization.export_values, EXPORT_VALUES, "normalization.export_values")


def check_normalization_settings(normalization: object) -> None:
    """Refuse a ``NormalizationConfig`` with any setting that is not what its field says, enabled or not.

    Flags are literally ``True`` or ``False``, ``memory_limit`` is an integer >= 1 (bytes), the
    ranges are numbers (:func:`check_normalization_ranges`) and the rest are choices or paths.
    """
    check_strict_bool(normalization.enabled, "normalization.enabled")
    check_strict_bool(normalization.per_time_point, "normalization.per_time_point")
    _check_normalization_choices(normalization)
    check_normalization_ranges(normalization)
    check_integer(normalization.memory_limit, "normalization.memory_limit", minimum=1)
    for name in ("plane_histograms", "manifest", "experiment_root", "masks_root"):
        _check_path(getattr(normalization, name), f"normalization.{name}")


def check_whole_fov_settings(whole_fov: object) -> None:
    """Refuse a ``WholeFOVConfig`` whose ``square_cover`` is not literally ``True`` or ``False``."""
    check_strict_bool(whole_fov.square_cover, "square_cover")


def check_common_settings(common: object) -> None:
    """Refuse a ``CommonConfig`` with a non-bool flag, non-text axes, bad plane policy or object set.

    The axes strings are read against the arrays' dimensions where a run uses them. The identity
    labels (``condition``, ``biological_repeat``, ``technical_repeat``) are checked, never
    converted, by ``_run_labels`` in ``pipeline.py`` where a run stamps or reads them: a
    ``run_config.json`` holding a bad one still loads, and the run refuses it with the folder message.
    """
    check_strict_bool(common.extract_arrays, "extract_arrays")
    for name in ("image_axes", "mask_axes"):
        if not isinstance(getattr(common, name), str):
            raise ValueError(f"{name} must be an axes string such as 'YX'; got {getattr(common, name)!r}.")
    _check_path(common.output_dir, "output_dir")
    _check_path(common.preprocessing_run, "preprocessing_run")
    validate_plane_policies(common)
    validate_object_set(common.object_set)


def check_run_settings(config: object) -> None:
    """Re-check every scientific setting of a ``CellPatcherConfig`` before a run uses them.

    Called by ``run_patching``, ``iter_patch_arrays``, ``patch_array`` and the folder and tree routes
    (``open_file_run``) before any file is read, so a setting edited after construction is
    refused as at construction. Every section is checked, whatever the mode.
    """
    check_choice(config.mode, MODES, "mode")
    check_common_settings(config.common)
    check_whole_fov_settings(config.whole_fov)
    check_random_settings(config.random)
    check_grid_settings(config.grid)
    check_filter_settings(config.filters)
    check_cell_settings(config.cell)
    check_export_settings(config.export)
    check_extraction_settings(config.extraction)
    check_match_settings(config.matching)
    check_normalization_settings(config.normalization)
