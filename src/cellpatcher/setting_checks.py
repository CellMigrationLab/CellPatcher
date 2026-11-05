"""Strict re-check of the scientific settings at the point a run uses them (pure; no I/O).

The config dataclasses are mutable and documented as edited in place
(``cfg.random.min_mask_fraction = 0.5``), so a check in ``__post_init__`` alone is bypassed by
an assignment after construction. The checks here are shared by ``__post_init__`` and by every
route that executes a run, and use the helpers of :mod:`cellpatcher.fractions` (never ``float()``,
``int()`` or ``bool()``, which turn ``True`` into 1.0 and ``"0.5"`` into 0.5).
"""

from __future__ import annotations

from collections.abc import Sequence

from cellpatcher.fractions import check_integer, check_real_number, check_strict_bool, check_unit_fraction


def check_random_settings(random: object) -> None:
    """Refuse a ``RandomConfig`` with a non-integer size, count or seed, or a bad ``min_mask_fraction``."""
    check_integer(random.patch_size, "patch_size", minimum=1)
    check_integer(random.n_patches, "n_patches", minimum=1)
    if random.seed is not None:
        check_integer(random.seed, "seed")
    check_unit_fraction(random.min_mask_fraction, "min_mask_fraction")


def check_grid_settings(grid: object) -> None:
    """Refuse a ``GridConfig`` with a non-integer ``tile_size``, an ``overlap`` outside [0, 1) or a non-bool flag."""
    check_integer(grid.tile_size, "tile_size", minimum=1)
    check_real_number(grid.overlap, "overlap")
    if not 0 <= grid.overlap < 1:
        raise ValueError(f"overlap must satisfy 0 <= overlap < 1; got {grid.overlap!r}.")
    check_strict_bool(grid.include_partial, "include_partial")


def check_filter_settings(filters: object) -> None:
    """Refuse a ``FilterConfig`` whose fraction, threshold or mean is a bool, text (a threshold name aside) or NaN."""
    check_unit_fraction(filters.min_foreground_fraction, "min_foreground_fraction")
    if filters.min_mean_intensity is not None:
        check_real_number(filters.min_mean_intensity, "min_mean_intensity")
    if filters.foreground_threshold is not None and not isinstance(filters.foreground_threshold, str):
        check_real_number(filters.foreground_threshold, "foreground_threshold")


def check_cell_settings(cell: object) -> None:
    """Refuse a ``CellConfig`` with a fixed size or area bound that is no integer, or a non-bool ``exclude_border``."""
    if cell.patch_size != "auto":
        check_integer(cell.patch_size, "patch_size", minimum=1)
    check_integer(cell.min_area, "min_area", minimum=1)
    if cell.max_area is not None:
        check_integer(cell.max_area, "max_area", minimum=cell.min_area)
    check_strict_bool(cell.exclude_border, "exclude_border")


def check_extraction_settings(extraction: object) -> None:
    """Refuse an ``ExtractionConfig`` with a non-integer output size or order, or a non-number fill."""
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


def check_whole_fov_settings(whole_fov: object) -> None:
    """Refuse a ``WholeFOVConfig`` whose ``square_cover`` is not literally ``True`` or ``False``."""
    check_strict_bool(whole_fov.square_cover, "square_cover")


def check_common_flags(common: object) -> None:
    """Refuse a ``CommonConfig`` whose ``extract_arrays`` is not literally ``True`` or ``False``."""
    check_strict_bool(common.extract_arrays, "extract_arrays")


def check_run_settings(config: object) -> None:
    """Re-check the scientific settings of a ``CellPatcherConfig`` before a run uses them.

    Called by ``run_patching`` and by the folder and tree routes (``open_file_run``) before any
    file is read, so a setting edited after construction is refused as at construction. Every
    section is checked, whatever the mode.
    """
    check_common_flags(config.common)
    check_whole_fov_settings(config.whole_fov)
    check_random_settings(config.random)
    check_grid_settings(config.grid)
    check_filter_settings(config.filters)
    check_cell_settings(config.cell)
    check_extraction_settings(config.extraction)
    check_normalization_ranges(config.normalization)
