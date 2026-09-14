# CellPatcher Configuration Reference

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

This document is the complete runtime configuration reference for `CellPatcherConfig` and all nested config sections.

---

## `config.mode`

| Field | Type | Default | Allowed values |
|---|---|---|---|
| `mode` | `str` | `"cell"` | `"cell"`, `"random"`, `"grid"`, `"whole_fov"` |

---

## `config.common` (`CommonConfig`)

Shared settings that apply to every run, regardless of mode.

| Field | Type | Default | Notes |
|---|---|---|---|
| `image_axes` | `str` | `"YX"` | Axes string for image arrays (e.g. `"YX"`, `"CYX"`, `"ZYX"`). Must include `Y` and `X`. |
| `mask_axes` | `str` | `"YX"` | Axes string for mask arrays. Must include `Y` and `X`. |
| `output_dir` | `str \| None` | `None` | Output folder path. Required when `export.enabled=True`; keep `None` for in-memory-only runs. |
| `extract_arrays` | `bool` | `False` | When `True`, extracted patch arrays are returned in `PatchingResult.patch_arrays`. |

**Axes strings** may contain any combination of `T`, `C`, `Z`, `Y`, `X` (in any order). Examples: `"YX"`, `"CYX"`, `"TCYX"`, `"ZYX"`.

---

## `config.cell` (`CellConfig`, used for `mode="cell"`)

Settings for cell/object-centred patching. A mask is required in this mode.

| Field | Type | Default | Notes |
|---|---|---|---|
| `patch_size` | `int \| str` | `"auto"` | Patch side length in pixels (square). Use `"auto"` for per-cell adaptive sizing based on bounding box, or call `scan_cell_patch_size()` to determine a good fixed value first. |
| `mask_channel` | `int \| None` | `None` | Channel index to select from multi-channel masks (e.g. `0` for first channel). Leave `None` for 2-D masks. |
| `center_on` | `str` | `"bbox_center"` | Patch centre strategy: `"bbox_center"` (centre of bounding box) or `"centroid"` (centre of mass). |
| `exclude_border` | `bool` | `True` | When `True`, objects whose bounding box touches the image border are skipped. |
| `min_area` | `int` | `1` | Minimum object area in pixels. Objects smaller than this are skipped. Must be `>= 1`. |
| `max_area` | `int \| None` | `None` | Maximum object area in pixels. Objects larger than this are skipped. Must be `>= min_area` when set. |
| `binary_mask_policy` | `str` | `"connected_components"` | How to handle binary (0/1) masks: `"connected_components"` labels each contiguous region separately; `"single_object"` treats the entire foreground as one object. |

---

## `config.random` (`RandomConfig`, used for `mode="random"`)

Settings for randomly sampled patches of a fixed size.

| Field | Type | Default | Notes |
|---|---|---|---|
| `patch_size` | `int` | `128` | Side length of each square patch in pixels. Must be `> 0`. |
| `n_patches` | `int` | `100` | Number of patches to sample per image. Must be `> 0`. |
| `seed` | `int \| None` | `0` | Random seed for reproducibility. Set to `None` for non-deterministic sampling. When processing multiple images the seed is incremented per image (`seed + image_index`). |

---

## `config.grid` (`GridConfig`, used for `mode="grid"`)

Settings for regular tiled patches covering the full image.

| Field | Type | Default | Notes |
|---|---|---|---|
| `tile_size` | `int` | `128` | Side length of each square tile in pixels. Must be `> 0`. |
| `overlap` | `float` | `0.0` | Fractional overlap between adjacent tiles. Must satisfy `0 <= overlap < 1`. For example, `0.1` means 10% overlap. |
| `include_partial` | `bool` | `False` | When `True`, edge tiles that are smaller than `tile_size` are included. When `False`, only full-sized tiles are returned. |

---

## `config.filters` (`FilterConfig`, optional post-generation filters)

Intensity-based filters applied after patch regions are generated. All fields default to `None` (disabled). Filters are applied to patches from all modes.

| Field | Type | Default | Notes |
|---|---|---|---|
| `min_foreground_fraction` | `float \| None` | `None` | Minimum fraction of pixels in a patch that must be "foreground" (above threshold). Must be in `[0, 1]` when set. Requires `foreground_threshold` to determine what counts as foreground (falls back to image mean if `None`). |
| `foreground_threshold` | `float \| str \| None` | `None` | Threshold used to binarise pixel values for foreground fraction computation. Can be a numeric value or one of the strategy strings below. |
| `min_mean_intensity` | `float \| None` | `None` | Minimum mean pixel intensity for a patch to be kept. Patches with mean intensity below this value are discarded. |

**Supported `foreground_threshold` strategy strings:**

| String | Description |
|--------|-------------|
| `"mean"` | Mean intensity of the full image |
| `"median"` | Median intensity of the full image |
| `"otsu"` | Otsu's threshold (requires `scikit-image`) |
| `"percentile:<N>"` | N-th percentile of the full image (e.g. `"percentile:75"`) |

> Threshold statistics are computed over the entire image array (all channels/planes). Pass a pre-projected image if channel-specific filtering is needed.

---

## `config.extraction` (`ExtractionConfig`, array extraction and resize)

Controls how patch arrays are extracted from images and optionally resized.

| Field | Type | Default | Notes |
|---|---|---|---|
| `pad_mode` | `str` | `"clip"` | Border handling when a patch extends outside the image. `"clip"` clamps to image bounds; `"constant"` pads with `constant_values`. |
| `constant_values` | `float` | `0.0` | Fill value used when `pad_mode="constant"`. |
| `output_size` | `int \| tuple[int,int] \| None` | `None` | Target size after extraction. `None` keeps the natural patch size. A positive `int` produces a square output of that side length. A `(height, width)` tuple specifies the exact output shape. Requires `scipy` for interpolation orders `> 0`. |
| `resize_order` | `int` | `1` | Spline interpolation order for resizing (0 = nearest, 1 = bilinear, 3 = bicubic). Must be `>= 0`. |
| `apply_mask` | `bool` | `False` | **Cell mode only.** When `True`, pixels outside the selected cell label are set to `mask_fill_value` before resize/export. |
| `mask_fill_value` | `float` | `0.0` | Fill value for pixels outside the selected label when `apply_mask=True`. |

> `apply_mask=True` is only supported in `mode="cell"` and requires masks to be provided.

---

## `config.export` (`ExportConfig`, saving options)

Controls whether and how patch arrays are written to disk.

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch for file export. When `True`, patch arrays are written to `common.output_dir`. |
| `save_metadata_csv` | `bool` | `True` | When `True`, a `patches.csv` file with per-patch metadata is written alongside the patch files. |
| `image_format` | `str` | `"tif"` | Output file format. One of `"tif"`, `"tiff"` (both write TIFF), or `"npy"` (NumPy binary). |

> To export files, set `export.enabled=True` **and** `common.output_dir` to a valid path. Enabling export automatically triggers array extraction even if `common.extract_arrays=False`.

---

## `config.matching` (`MatchConfig`, image↔mask filename matching)

Controls how image files are paired with mask files when reading from folders.

`run_patching_pipeline` uses these settings automatically when a `mask_folder_path` is provided. The standalone helper `validate_image_and_mask_folders` accepts matching parameters explicitly (it does not read `config.matching` directly).

| Field | Type | Default | Notes |
|---|---|---|---|
| `strategy` | `str` | `"exact_or_suffix"` | Filename matching strategy (see table below). |
| `mask_suffixes` | `tuple[str, ...]` | `("_mask", "_labels", "_label", "_seg", "_segmentation")` | Candidate suffixes used by the `exact_or_suffix` strategy. |
| `case_sensitive` | `bool` | `False` | When `False`, filenames are compared case-insensitively. |

**Matching strategies:**

| Strategy | Behaviour |
|----------|-----------|
| `"exact_or_suffix"` | Exact stem match **or** `image_stem + suffix` match (default; recommended) |
| `"exact"` | Exact stem match only |
| `"substring"` | Legacy: mask stem contains image stem as a substring. May false-match names like `img1` and `img10_mask`. |

> `config.matching` is only relevant for `mode="cell"` and `mode="random"` (the only modes that load masks from disk).

---

## Mode-to-section reference

The table below shows which config sections are written by `save_config_for_mode(mode, path)`. The common sections (`common`, `filters`, `export`, `extraction`) are always included.

| Mode | Common sections | Mode-specific sections |
|------|----------------|------------------------|
| `"cell"` | `common`, `filters`, `export`, `extraction` | `cell`, `matching` |
| `"random"` | `common`, `filters`, `export`, `extraction` | `random`, `matching` |
| `"grid"` | `common`, `filters`, `export`, `extraction` | `grid` |
| `"whole_fov"` | `common`, `filters`, `export`, `extraction` | *(none)* |

`save_default_config(path)` writes all sections regardless of mode.

---

## `PatchRegion` fields

Every generated patch is a `PatchRegion` dataclass. The fields below are populated for all modes.

| Field | Type | Description |
|-------|------|-------------|
| `patch_id` | `str` | Unique identifier for the patch (used as the export filename stem). |
| `image_id` | `str \| None` | Identifier of the source image (e.g. `"image_0000"`). |
| `mode` | `str \| None` | Patching mode that generated this patch. |
| `y0`, `y1` | `int` | Vertical extent of the patch in pixels (`y0` inclusive, `y1` exclusive). |
| `x0`, `x1` | `int` | Horizontal extent of the patch in pixels (`x0` inclusive, `x1` exclusive). |
| `label_id` | `int \| None` | Cell label value from the mask (cell mode only). |
| `center_y`, `center_x` | `float \| None` | Patch centre coordinates in pixels. |
| `touches_border` | `bool` | Whether the patch touches the image border. |
| `area_px` | `int \| None` | Object area in pixels (cell mode only). |
| `height` | `int` | Patch height (`y1 - y0`). Computed property. |
| `width` | `int` | Patch width (`x1 - x0`). Computed property. |

Fields `z0`, `z1`, `t`, `c`, and `center_z` are reserved for future multidimensional support.
