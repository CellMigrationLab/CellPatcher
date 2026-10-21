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
| `condition` | `str \| None` | `"single"` | Experimental group label, stamped onto every `PatchRegion`/CSV row for this run. Part of the MorphoBricks shared CSV contract (see `AGENTS.md` §5). Set it explicitly for a single-folder `run_patching_pipeline` call, or let `run_patching_condition_repeat_tree` derive it per leaf from a `<condition>/<biological_repeat>/` folder tree (see "Condition/biological-repeat folder trees" below). |
| `biological_repeat` | `str \| None` | `"single"` | Biological replicate identifier (e.g. `"YYYY_MM_DD"`, or `cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL`/`"single"` for a condition folder with no biological-repeat subfolder), stamped onto every `PatchRegion`/CSV row for this run. Same status as `condition`. Formerly `repeat`, which still works as a deprecated alias (see "Renamed: `repeat` → `biological_repeat`" below). |
| `technical_repeat` | `str \| None` | `"single"` | Technical replicate within the biological repeat: a well, a dish or a coverslip, never a biological repeat. Stamped onto every `PatchRegion`/CSV row for this run; a per-image `image_metadata=[{"technical_repeat": ...}]` entry overrides it for that image. `run_patching_condition_repeat_tree` derives it per leaf from an optional `<condition>/<biological_repeat>/<technical_repeat>/` level, and stamps `cellpatcher.SINGLE_TECHNICAL_REPEAT_LABEL` (`"single"`) when a biological-repeat folder holds its images directly. |

| `z_policy` | `str` | `"keep"` | How a Z axis of the images becomes 2D planes before patching. `"keep"`: every patch keeps all Z planes (the behaviour before this option). `"all"`: one patch per Z plane, with `z0`/`z1` set and `_z<NNN>` appended to `patch_id` (e.g. `image_0000_grid_00003_z002`, before a split channel's `_c<n>`). `"first"`/`"select"` (with `z_index`): one plane, with `z0`/`z1` set. `"max"`/`"mean"`/`"sum"`: a projection (no `z0`). Images without a Z axis are not affected. |
| `z_index` | `int \| None` | `None` | Z plane for `z_policy="select"`. |
| `t_policy` | `str` | `"keep"` | As `z_policy` for a T axis: `"keep"`, `"all"` (sets `t`, appends `_t<NNN>`), `"first"` or `"select"`; no projections. With both split, planes come in T-then-Z order (`..._t001_z002`). |
| `t_index` | `int \| None` | `None` | Time point for `t_policy="select"`. |
| `mask_source` | `str` | `"tiff_tree"` | Where folder and tree runs read masks. `"tiff_tree"`: a mask folder (or tree) mirroring the images, matched by file name with `config.matching`. `"ngff_labels"`: the label image of each OME-Zarr image's own `labels` group (no mask folder; every image must be OME-Zarr). See "OME-Zarr images" below. |
| `mask_label_name` | `str \| None` | `None` | With `mask_source="ngff_labels"`, the label image to use (e.g. `"cells"`); may stay `None` when every image has exactly one. |

**Masks with split planes:** a mask's Z or T axis with the image's length follows the image's planes (one label plane per image plane); a mask without that axis, or with length 1, is shared by every plane. Any other length is an error, and so is a Z projection with masks that follow the Z planes. Image ids stay one per source image; with `extraction.split_channels`, every plane and channel patch of an image shares one `position_id`.

**Reading planes:** folder and tree runs that split or select planes (`"all"`, `"first"`, `"select"`) handle one image at a time. With the `preprocessing` extra, TIFF and OME-Zarr images are then read one 2D plane at a time with the preprocessing brick's reader, so only the chosen planes are decoded (channels and kept axes are read plane by plane and stacked back); a plane that cannot be read whole within the brick's memory limit (1 GiB) stops the run, because CellPatcher needs each plane whole to patch it. `"keep"`, projections and other formats read the whole image, as before. The patches are the same either way, and the same as `run_patching` on the whole arrays.

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
| `split_channels` | `bool` | `False` | When `True`, every region becomes one single-channel patch per channel of its image instead of one multichannel patch (requires a `C` axis in `common.image_axes`). Channel patches keep the region's coordinates, get `c` = channel index and a `_c<index>` suffix on `patch_id`, and share one `position_id`: the one supplied via `image_metadata`, or by default `<condition>/<biological_repeat>/<image file stem>` (parts that are not set are left out; `image_id` replaces the stem when no `image_file` is known). Filters are evaluated on the whole multichannel region before splitting; masking and resizing are applied before splitting too. Without `split_channels`, `position_id` is only set when supplied. |

> `apply_mask=True` is only supported in `mode="cell"` and requires masks to be provided.

---

## `config.export` (`ExportConfig`, saving options)

Controls whether and how patch arrays are written to disk.

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch for file export. When `True`, patch arrays are written to `common.output_dir`. |
| `save_metadata_csv` | `bool` | `True` | When `True`, a `patches.csv` file with per-patch metadata is written alongside the patch files. |
| `image_format` | `str` | `"tif"` | Output file format. One of `"tif"`, `"tiff"` (both write TIFF), or `"npy"` (NumPy binary). |
| `save_run_config` | `bool` | `True` | When `True`, `run_config.json` (the full serialized config) is written next to the patch files and named in every row's `config_file`. `run_patching_condition_repeat_tree(combine_metadata=True)` turns it off per folder and writes one for the whole tree instead. |

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

## `config.normalization` (`NormalizationConfig`)

Normalization with the statistics of the preprocessing brick, following the [normalization contract](normalization_contract.md) (decisions in the [plan](plan.md)). Needs the `preprocessing` extra (`pip install "cellpatcher[preprocessing]"`). Off by default: with `enabled=False`, patch arrays and every existing `patches.csv` column are exactly as before, and the `normalization_*` columns stay empty.

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `False` | Normalize every patch. |
| `plane_histograms` | `str \| None` | `None` | Path of the preprocessing brick's `plane_histograms.csv` (schema version 2); required for every scope except `"patch"`. Give CellPatcher the same experiment root as the preprocessing brick. |
| `manifest` | `str \| None` | `None` | Path of its `run_manifest.json` for extra checks (file sizes and times, CSV hash, axis roles); `None` uses the one next to the CSV when there is one. |
| `method` | `str` | `"percentile"` | `"minmax"`, `"percentile"`, `"zscore"` or `"robust_zscore"` (MAD × 1.4826). |
| `percentile` | `(float, float)` | `(1.0, 99.8)` | Low and high percentile for `"percentile"`, `0 ≤ low < high ≤ 100`. |
| `scope` | `str` | `"image"` | Pixels the parameters come from: `"condition"`, `"biological_repeat"`, `"technical_repeat"`, `"image"` (all Z and T planes of the source image) or `"patch"` (the patch's own pixels). Always per channel. |
| `per_time_point` | `bool` | `False` | Normalize each time point separately (`t` joins the group). |
| `patch_region` | `str \| None` | `None` | For `scope="patch"`: `"mask"` (the patch's own label), `"bbox"` (the label's whole bounding box, whatever the patch size) or `"patch"`; `None` means `"mask"` in `cell` mode and `"patch"` otherwise. Padding is never used. |

`export_values` and `memory_limit` (contract §12–§13) come with the delivery milestone (Milestone 4).

**What happens** (contract §6–§9):

- The pipeline becomes generate → filter → extract → **normalize** → apply mask → resize → export. Filters see raw values; `extraction.mask_fill_value` is in normalized units (`NaN` allowed); resizing works on the float32 values. Padding added by `pad_mode="constant"` is normalized like the raw value `constant_values`.
- Each patch becomes `(pixels - offset) / scale` per channel, float32, without clipping.
- Group and image scopes pool the CSV rows of the patch's group (same condition / biological repeat / technical repeat / image, same channel, and same `t` with `per_time_point`), including planes that were not patched.
- Before anything is normalized, every patched plane must have its CSV row with the same labels, dtype, size, exact finite extrema and finite pixel count; with the manifest, each image's size, modification time and axes must match too. Every image without rows is listed in one message.
- Normalization needs one plane per patch: `common.z_policy`/`t_policy` `"keep"` or a projection along an axis longer than 1 stops the run.
- A group or patch with no finite pixels or zero spread stops the run with the list of groups or patches (decision D8 is open).
- Warnings (`cellpatcher.normalization.NormalizationWarning`): approximate parameters (float images); group and image statistics include background; per-condition and per-repeat scopes remove real differences between groups.

**Recorded:** six `PatchRegion` fields, written after `source_relative_path` in `patches.csv`: `normalization_method` (e.g. `percentile 1-99.8`), `normalization_scope`, `normalization_group` (`<condition>`, `<condition>/<biological_repeat>`, `<condition>/<biological_repeat>/<technical_repeat>`, `<source_relative_path>`, or `<source_relative_path>::<patch_id>` for the patch scope; `::t=<t>` with `per_time_point`), `normalization_offset` and `normalization_scale` (one full-precision number per channel of the patch, `;`-separated; one value for a split channel) and `normalization_exact` (`true`/`false`). `run_config.json` gains the `normalization` section and a top-level `normalization_inputs` object (`plane_histograms_sha256`, `schema_version`, `manifest_used`, `manifest_version`, `preprocessing_version`); a `run_config.json` without `normalization` loads with normalization off.

**In-memory runs:** `run_patching` normalizes too; for the CSV scopes each image needs `image_metadata=[{"source_relative_path": ...}]` and the right `common` labels (folder and tree runs fill both). The manifest's file checks run in folder and tree runs.

The `common.mask_source` and `common.mask_label_name` fields for OME-Zarr label masks (contract §4) exist since Milestone 2, and `cellpatcher.statistics` (loading `plane_histograms.csv`/`run_manifest.json`, contract §5 checks) since Milestone 1.

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

Every generated patch is a `PatchRegion` dataclass. The fields below are populated for all modes; `to_dict()`/the exported `patches.csv` also include the `height`/`width` computed properties as extra convenience columns.

| Field | Type | Description |
|-------|------|-------------|
| `patch_id` | `str` | Unique identifier for the patch (used as the export filename stem). |
| `image_id` | `str \| None` | Identifier of the source image (e.g. `"image_0000"`). |
| `mode` | `str \| None` | Patching mode that generated this patch. |
| `condition` | `str \| None` | From `config.common.condition`. See the shared CSV contract below. |
| `biological_repeat` | `str \| None` | From `config.common.biological_repeat`. See the shared CSV contract below. Formerly `repeat` (deprecated alias). |
| `technical_repeat` | `str \| None` | From `config.common.technical_repeat`, or per image from `image_metadata`. See the shared CSV contract below. |
| `position_id` | `str \| None` | Optional stable spatial acquisition-position identifier. Use the same value for all Z/C/T planes from one XY position. Filled automatically for channel patches when `extraction.split_channels=True` (see above). |
| `image_file` | `str \| None` | Source image filename. Populated automatically by `run_patching_pipeline`; `None` for direct `run_patching()` calls unless supplied via `image_metadata`. |
| `mask_file` | `str \| None` | Source mask filename (cell/random mode only, when a mask was loaded). Same population rule as `image_file`. |
| `y0`, `y1` | `int` | Vertical extent of the patch in pixels (`y0` inclusive, `y1` exclusive). |
| `x0`, `x1` | `int` | Horizontal extent of the patch in pixels (`x0` inclusive, `x1` exclusive). |
| `z0`, `z1` | `int \| None` | Selected Z-slice range (`z0` inclusive, `z1` exclusive — one slice thick). Populated when the caller passes `image_metadata=[{"z": ...}, ...]` to `run_patching`; `None` otherwise. |
| `t` | `int \| None` | Selected time index. Populated the same way via `image_metadata=[{"t": ...}, ...]`. |
| `c` | `int \| None` | Selected channel index. Populated the same way via `image_metadata=[{"c": ...}, ...]`. |
| `label_id` | `int \| None` | Cell label value from the mask (cell mode only). |
| `center_y`, `center_x` | `float \| None` | Patch centre coordinates in pixels. |
| `center_z` | `float \| None` | Reserved for future multidimensional support; not yet populated. |
| `touches_border` | `bool` | Whether the patch touches the image border. |
| `area_px` | `int \| None` | Object area in pixels (cell mode only). |
| `patch_file` | `str \| None` | Exported patch filename, stamped by `export_patches`/`export.enabled=True`. `None` when the patch was not exported to disk. |
| `output_height`, `output_width` | `int \| None` | Final exported array shape. Differs from `y1 - y0`/`x1 - x0` when `extraction.output_size` resizes the patch. `None` unless arrays were extracted (`extract_arrays=True` or `export.enabled=True`). |
| `config_file` | `str \| None` | Filename of the full run-config JSON saved alongside `patches.csv` (see below). `None` unless `export.enabled=True`. |
| `valid`, `reason` | `bool`, `str \| None` | Reserved for future rejected-patch reporting; not yet populated (`valid` is always `True`). |
| `source_relative_path` | `str \| None` | The source image's path relative to the experiment root, with `/` separators, from its real location (`control/img.tif` for an image directly in `control/`, never the `single` output folders). Links the patch to the preprocessing brick's `plane_id` ([normalization contract](normalization_contract.md) §2). Tree runs make it relative to `image_root_path`; `run_patching_pipeline` to the image folder (the file name); direct `run_patching()` calls take it from `image_metadata`, else `None`. Last column of `patches.csv`. |

### Shared MorphoBricks CSV contract

`condition`, `biological_repeat`, `technical_repeat` (optional), `position_id` (when needed), `image_file`, `mask_file`, and `patch_file` are the columns of the MorphoBricks ecosystem's shared CSV contract (`AGENTS.md` §5), alongside the 0-based plane indices `z0`/`z1`, `c` and `t` (see the table above). CellPatcher populates them as follows:

- `condition`/`biological_repeat` — either set `config.common.condition`/`config.common.biological_repeat` explicitly and call `run_patching_pipeline` on a single folder, or call `run_patching_condition_repeat_tree(image_root_path, ...)` on a `raw_data/<condition>/<biological_repeat>/` root and let both be derived per leaf from the folder path (see below). A label that is not given, or set to `None`, is written as `"single"` (`SINGLE_CONDITION_LABEL`, `SINGLE_BIOLOGICAL_REPEAT_LABEL`, `SINGLE_TECHNICAL_REPEAT_LABEL`), as for a missing folder level, so the label columns are never empty.
- `technical_repeat` — the well, dish or coverslip within the biological repeat. Tree runs derive it from an optional `<condition>/<biological_repeat>/<technical_repeat>/` level and write `"single"` when a biological-repeat folder holds its images directly, so the column is never empty in a tree run. For a single folder set `config.common.technical_repeat`; when one folder holds several wells, pass `image_metadata=[{"technical_repeat": ...}, ...]` to `run_patching`.
- `image_file`/`mask_file`/`patch_file` are populated automatically by `run_patching_pipeline` (folder-based runs, including each leaf of `run_patching_condition_repeat_tree`) and `export_patches` respectively.

### OME-Zarr images

With the `preprocessing` extra (`pip install "cellpatcher[preprocessing]"`), folders whose names end in `.zarr` (usually `.ome.zarr`) are images, in folder runs, tree runs and `read_array`. CellPatcher reads level 0 of the store's only image with the preprocessing brick's reader; a store with several images (positions, series or plate wells) is an error. The axes come from the OME-NGFF metadata and are matched to `common.image_axes` (or `mask_axes`) by role: a length-1 axis that the configured axes lack is dropped, a configured axis the store lacks is added with length 1, and the dimensions are reordered; any other mismatch is an error. Discovery never enters a store, and `img.ome.zarr` matches masks named like `img.tif` or `img_mask.tif`. Without the extra, reading a store stops with the install command. OME-Zarr copies written by the preprocessing brick are not read: CellPatcher reads the source images.

**Masks from OME-NGFF labels:** with `common.mask_source="ngff_labels"`, each OME-Zarr image's mask is its label image `common.mask_label_name` (from its `labels` group, OME-NGFF 0.4 and 0.5; the name may stay `None` when there is only one). Level 0 is read; its values must be integers, each dimension must equal the image's or have length 1, and a channel dimension must have length 1. It is then put into `common.mask_axes` by role, so a label Z or T axis with the image's length follows the image's planes and one of length 1 is shared (`z_policy`/`t_policy`). `mask_file` is written as `<image>.ome.zarr/labels/<name>`. A mask folder, or an image that is not OME-Zarr, is an error with this source.

### Condition/biological-repeat folder trees

`run_patching_condition_repeat_tree(image_root_path, *, config=None, mask_root_path=None, ...)` (in `cellpatcher.orchestration`) walks a MorphoBricks-style root and calls `run_patching_pipeline` once per discovered leaf, using `discover_condition_repeat_folders` (in `cellpatcher.io`) to find them. Both functions keep their historical `condition_repeat` names; the level below each condition is the **biological repeat**. The MorphoBricks folder hierarchy is `experiment/condition/biological_repeat/technical_repeat/images`, with levels omitted only from the bottom up (a level's meaning comes from its position); CellPatcher reads all three levels:

- `<condition>/<biological_repeat>/<image files>` — one or more biological-repeat subfolders, each its own biological repeat.
- `<condition>/<image files>` — **no biological-repeat subfolder.** The whole condition folder is treated as a single implicit biological repeat, labeled `cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL` (`"single"`) — consistently in the discovered `biological_repeat` value, the `condition`/`biological_repeat` stamped on every patch from that folder, and (when exporting) its own `<output_dir>/<condition>/single/` output subfolder, same as an explicit biological repeat gets.
- `<image files>` directly in the root — **no condition subfolders.** The whole dataset is treated as one condition and one biological repeat, labeled `cellpatcher.SINGLE_CONDITION_LABEL`/`cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL` (both `"single"`), exported to `<output_dir>/single/single/`.
- `<condition>/<biological_repeat>/<technical_repeat>/<image files>` — **technical repeats.** A biological-repeat folder holding only subfolders has one subfolder per technical repeat (a well, a dish, a coverslip), each its own leaf, stamped with the subfolder name and exported to `<output_dir>/<condition>/<biological_repeat>/<technical_repeat>/`. A biological-repeat folder holding its images directly is one technical repeat, stamped `cellpatcher.SINGLE_TECHNICAL_REPEAT_LABEL` (`"single"`), and keeps the two-level output folder. The technical level needs the biological-repeat level above it: in `<condition>/<well>/<images>` the wells are read as biological repeats, so put them under a dated (or `single`) biological-repeat folder.
- A root, condition or biological-repeat folder mixing loose image files with subfolders is ambiguous and raises `ValueError`. Hidden subfolders (names starting with `.`, e.g. `.ipynb_checkpoints`) are ignored.

When `mask_root_path` is given, the same structure is mirrored under it per leaf. Mask folder requirements follow the same per-mode rules as `run_patching_pipeline` (required for `mode="cell"`, optional for `mode="random"`, not accepted for `mode="grid"`/`"whole_fov"`), applied independently to each leaf — a leaf missing a required mask folder raises immediately and aborts the whole tree run rather than being silently skipped.

Returns a list of `ConditionRepeatResult(condition, biological_repeat, image_folder, result, technical_repeat, relative_folder)`, one per leaf, in the same (condition, then biological repeat, then technical repeat) sorted order `discover_condition_repeat_folders` returns (whose `ConditionRepeatFolder` entries likewise carry `biological_repeat` and `technical_repeat`).

- `combine_metadata=True` (requires `export.enabled=True` and `common.output_dir`) writes **one** `patches.csv` and **one** `run_config.json` directly in `output_dir` instead of a pair per leaf folder. Patch files still go to each leaf's `<output_dir>/<condition>/<biological_repeat>[/<technical_repeat>]/`, and each row's `patch_file` is the path relative to the CSV, e.g. `treated/2026_01_01/image_0000_grid_00000.tif` or `treated/2026_01_01/B02/image_0000_grid_00000.tif` — the same "relative to the CSV" meaning as in the per-leaf layout, so `ExportedPatchTorchDataset` reads the combined CSV unchanged. `patch_id`/`image_id` are only unique within a leaf; identify rows across the combined CSV by `condition`/`biological_repeat`/`technical_repeat`/`image_file` (or `patch_file`).
- `progress_callback(completed, total, leaf)` is called before each leaf and once more at the end (`leaf=None`), e.g. to drive a progress bar.

### Renamed: `repeat` → `biological_repeat`

The biological-replicate field was called `repeat` in earlier releases. It is now `biological_repeat` everywhere — the `PatchRegion` field, the `CommonConfig` field, the `patches.csv` column and the `run_config.json` key, and the `ConditionRepeatFolder`/`ConditionRepeatResult` attribute — so it cannot be confused with the MorphoBricks `technical_repeat` column (a well, dish or coverslip within a biological repeat). Nothing else changed: the values, the folder level they come from, the `"single"` label and the output layout are the same.

For a transition period the old name keeps working, and each use emits one `DeprecationWarning` pointing at the calling line:

- `CommonConfig(repeat=...)`, `PatchRegion(repeat=...)`, `ConditionRepeatFolder(repeat=...)`, `ConditionRepeatResult(repeat=...)`: the keyword is mapped onto `biological_repeat`; passing both with different values raises `TypeError`.
- `config.common.repeat`, `patch.repeat`, `leaf.repeat`: read and write `biological_repeat`.
- `apply_overrides(..., {"common.repeat": ...})` (and `overrides=` on the run functions), and `load_config()` on an older `run_config.json` whose `common` section has a `repeat` key.
- `ExportedPatchTorchDataset` reading an older `patches.csv` that has a `repeat` column and no `biological_repeat` column: the column is read as `biological_repeat` (one warning per file).

Output is identical whichever name is used; files are always written with `biological_repeat`. The constant `SINGLE_REPEAT_LABEL` stays available (not deprecated) as the same value as the new `SINGLE_BIOLOGICAL_REPEAT_LABEL`, and the function names `run_patching_condition_repeat_tree`/`discover_condition_repeat_folders` are unchanged.

### Deterministic reconstruction

Each exported `patches.csv` row, together with the `run_config.json` file it references via `config_file` (written alongside it whenever `export.enabled=True`; see `cellpatcher.RUN_CONFIG_FILENAME`), is enough to reconstruct the patch without re-deriving anything from code:

- The row's own coordinates (`y0/y1/x0/x1`, `z0/z1`, `t`, `c`) and `output_height`/`output_width` fully describe the patch geometry and final exported array shape.
- `run_config.json` is the complete, serialized `CellPatcherConfig` for the run — strategy parameters (including `random.seed`), extraction/resize settings, matching strategy, and axes — everything needed to reproduce the same coordinates deterministically from the same inputs.
