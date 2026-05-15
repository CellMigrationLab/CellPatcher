# CellPatcher Configuration Reference

This document is the complete runtime configuration reference for `CellPatcherConfig`.

## `config.mode`

| Field | Type | Default | Allowed options |
|---|---|---|---|
| `mode` | `str` | `"cell"` | `"cell"`, `"random"`, `"grid"`, `"whole_fov"` |

## `config.common` (`CommonConfig`)

| Field | Type | Default | Allowed options / notes |
|---|---|---|---|
| `image_axes` | `str` | `"YX"` | Any valid axes string containing spatial axes used by your image array |
| `mask_axes` | `str` | `"YX"` | Any valid axes string containing spatial axes used by your mask array |
| `output_dir` | `str \| None` | `None` | Output folder path for exports; keep `None` if not exporting files |
| `extract_arrays` | `bool` | `False` | `True` returns extracted patch arrays in-memory in `PatchingResult` |

## `config.cell` (`CellConfig`, used for `mode="cell"`)

| Field | Type | Default | Allowed options / notes |
|---|---|---|---|
| `patch_size` | `int \| str` | `"auto"` | Positive `int` or `"auto"` |
| `mask_channel` | `int \| None` | `None` | Channel index for multi-channel masks |
| `center_on` | `str` | `"bbox_center"` | `"centroid"` or `"bbox_center"` |
| `exclude_border` | `bool` | `True` | If `True`, excludes objects touching image border |
| `min_area` | `int` | `1` | Must be `>= 1` |
| `max_area` | `int \| None` | `None` | If set, must be `>= min_area` |
| `binary_mask_policy` | `str` | `"connected_components"` | `"connected_components"` or `"single_object"` |

## `config.random` (`RandomConfig`, used for `mode="random"`)

| Field | Type | Default | Allowed options / notes |
|---|---|---|---|
| `patch_size` | `int` | `128` | Must be `> 0` |
| `n_patches` | `int` | `100` | Must be `> 0` |
| `seed` | `int \| None` | `0` | Integer seed or `None` |

## `config.grid` (`GridConfig`, used for `mode="grid"`)

| Field | Type | Default | Allowed options / notes |
|---|---|---|---|
| `tile_size` | `int` | `128` | Must be `> 0` |
| `overlap` | `float` | `0.0` | Must satisfy `0 <= overlap < 1` |
| `include_partial` | `bool` | `False` | Include edge tiles that are smaller than `tile_size` |

## `config.filters` (`FilterConfig`, optional post-generation filters)

| Field | Type | Default | Allowed options / notes |
|---|---|---|---|
| `min_foreground_fraction` | `float \| None` | `None` | If set, must be within `[0, 1]` |
| `foreground_threshold` | `float \| str \| None` | `None` | Numeric threshold or strategy string (e.g., `"otsu"`) |
| `min_mean_intensity` | `float \| None` | `None` | Scalar intensity threshold |

## `config.extraction` (`ExtractionConfig`, array extraction / resize)

| Field | Type | Default | Allowed options / notes |
|---|---|---|---|
| `pad_mode` | `str` | `"clip"` | `"clip"` or `"constant"` |
| `constant_values` | `float` | `0.0` | Fill value used with `pad_mode="constant"` |
| `output_size` | `int \| tuple[int,int] \| None` | `None` | `None`, positive `int`, or positive `(height, width)` |
| `resize_order` | `int` | `1` | Interpolation order, must be `>= 0` |
| `apply_mask` | `bool` | `False` | If `True`, applies selected-cell mask before resize/export |
| `mask_fill_value` | `float` | `0.0` | Fill value for pixels outside selected label when `apply_mask=True` |

## `config.export` (`ExportConfig`, saving options)

| Field | Type | Default | Allowed options / notes |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch for file export |
| `save_metadata_csv` | `bool` | `True` | If `True`, writes metadata CSV alongside patches |
| `image_format` | `str` | `"tif"` | `"tif"`, `"tiff"`, or `"npy"` |

> To save files you typically set all three pieces together: `config.export.enabled=True`, `config.common.output_dir="..."`, and optionally `config.export.image_format` / `config.export.save_metadata_csv`.

## `config.matching` (`MatchConfig`, image↔mask filename matching)

| Field | Type | Default | Allowed options / notes |
|---|---|---|---|
| `strategy` | `str` | `"exact_or_suffix"` | `"exact"`, `"exact_or_suffix"`, `"substring"` |
| `mask_suffixes` | `tuple[str, ...]` | `("_mask", "_labels", "_label", "_seg", "_segmentation")` | Candidate suffixes used by suffix-aware matching |
| `case_sensitive` | `bool` | `False` | Case-sensitive filename matching toggle |
