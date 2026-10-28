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

| `z_policy` | `str` | `"keep"` | How a Z axis of the images becomes 2D planes before patching. `"keep"`: every patch keeps all Z planes (the behaviour before this option). `"all"`: one patch per Z plane, with `z0`/`z1` set and a `z<NNN>` level in `patch_id` (e.g. `<position_id>/z002/grid-y0-x8-h8-w8`; see "Patch identity" below). `"first"`/`"select"` (with `z_index`): one plane, with `z0`/`z1` set. `"max"`/`"mean"`/`"sum"`: a projection (no `z0`). Images without a Z axis are not affected. |
| `z_index` | `int \| None` | `None` | Z plane for `z_policy="select"`. |
| `t_policy` | `str` | `"keep"` | As `z_policy` for a T axis: `"keep"`, `"all"` (sets `t`, a `t<NNNN>` level of `patch_id`), `"first"` or `"select"`; no projections. With both split, planes come in T-then-Z order (`.../t0001/z002/...`). |
| `t_index` | `int \| None` | `None` | Time point for `t_policy="select"`. |
| `mask_source` | `str` | `"tiff_tree"` | Where folder and tree runs read masks. `"tiff_tree"`: a mask folder (or tree) mirroring the images, matched by file name with `config.matching`. `"ngff_labels"`: the label image of each OME-Zarr image's own `labels` group (no mask folder; every image must be OME-Zarr). See "OME-Zarr images" below. |
| `mask_label_name` | `str \| None` | `None` | With `mask_source="ngff_labels"`, the label image to use (e.g. `"cells"`); may stay `None` when every image has exactly one. |
| `object_set` | `str \| None` | `None` | Cell mode: the name of the object set (mask) written to the `object_set` column and to `patch_id`. `None`: named after the mask (see "Patch identity" below), `"mask"` when the mask gives no name. Letters, digits, `.`, `_`, `-` only. |

**Masks with split planes:** a mask's Z or T axis with the image's length follows the image's planes (one label plane per image plane); a mask without that axis, or with length 1, is shared by every plane. Any other length is an error, and so is a Z projection with masks that follow the Z planes. Image ids stay one per source image, and every plane and channel patch of an image shares one `position_id`.

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
| `binary_mask_policy` | `str` | `"single_object"` | Masks are instance labels (decision D47): an **integer** mask is used as given, whatever this setting, each non-zero value one object (a 0/1 or 0/255 mask is one object, label 1 or 255). Only a **boolean** mask (or a float mask of 0.0/1.0), which has no object ids, depends on it: `"single_object"` (default) refuses it with an error; `"connected_components"` is the explicit opt-in that labels its 4-connected components, emits a `UserWarning` and is recorded in `run_config.json` (`cell.binary_mask_policy`). It cannot be combined with `extraction.apply_mask` (the mask does not hold the computed labels). Labels above the int32 maximum (2147483647) are refused instead of wrapping. Before 2026-10-26 the default was `"connected_components"`, which also relabelled 0/1 **integer** masks. |

---

## `config.random` (`RandomConfig`, used for `mode="random"`)

Settings for randomly sampled patches of a fixed size.

| Field | Type | Default | Notes |
|---|---|---|---|
| `patch_size` | `int` | `128` | Side length of each square patch in pixels. Must be `> 0`. |
| `n_patches` | `int` | `100` | Number of patches to sample per image. Must be `> 0`. |
| `seed` | `int \| None` | `0` | Random seed for reproducibility. Set to `None` for non-deterministic sampling. Each image (each plane with split planes) is sampled with a seed derived from this seed and its identity — `position_id` and its `t`/`z`/`c` indices — never from its place in the run, so an image gets the same rectangles alone or among other files (see "Patch identity" below). Must be `>= 0`. |

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
| `split_channels` | `bool` | `False` | When `True`, every region becomes one single-channel patch per channel of its image instead of one multichannel patch (requires a `C` axis in `common.image_axes`). Channel patches keep the region's coordinates and `position_id`, and get `c` = channel index (a `c<index>` level of `patch_id`; see "Patch identity" below). Filters are evaluated on the whole multichannel region before splitting; masking and resizing are applied before splitting too. |

> `apply_mask=True` is only supported in `mode="cell"` and requires masks to be provided.

---

## `config.export` (`ExportConfig`, saving options)

Controls whether and how patch arrays are written to disk.

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch for file export. When `True`, patch arrays are written to `common.output_dir`. |
| `save_metadata_csv` | `bool` | `True` | When `True`, a `patches.csv` file with per-patch metadata is written alongside the patch files, together with the per-image table `images.csv` (see "Per-image table" below). |
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
| `ignore_unrequested_masks` | `bool` | `False` | When `True`, masks whose stem ends in none of `mask_suffixes` are left out **before** matching: they are neither candidates nor reported as "Masks with no matching image". For a mask folder holding several masks per image (`img_cells.tif`, `img_nuclei.tif`, `img_mito.tif`, ...), set `mask_suffixes=("_cells",)` and `ignore_unrequested_masks=True`. A mask named exactly like its image ends with no suffix and is left out too; an image without a requested mask, or a requested mask without an image, still raises. Needs `exact_or_suffix` or `substring` (`exact` uses no suffix). `validate_image_and_mask_folders(..., ignore_unrequested_masks=True)` is the same option for the standalone helper. |

With `ignore_unrequested_masks=False` (the default), every mask in the folder must match an image, so `mask_suffixes=("_cells",)` alone still refuses a folder that also holds `_nuclei` masks; the error then says how many of the unmatched masks end in none of `mask_suffixes` and names this option.

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
| `plane_histograms` | `str \| None` | `None` | Path of the preprocessing brick's `plane_histograms.csv` (schema version 4, preprocessing `v0.3.0`; earlier versions are refused with a message asking to run the brick again); required for every scope except `"patch"`. Give CellPatcher the same experiment root as the preprocessing brick. |
| `manifest` | `str \| None` | `None` | Path of its `run_manifest.json` for extra checks (file sizes and times, CSV hash, axis roles); `None` uses the one next to the CSV when there is one. |
| `method` | `str` | `"percentile"` | `"minmax"`, `"percentile"`, `"zscore"` or `"robust_zscore"` (MAD × 1.4826). |
| `percentile` | `(float, float)` | `(1.0, 99.8)` | Low and high percentile for `"percentile"`, `0 ≤ low < high ≤ 100`. |
| `scope` | `str` | `"image"` | Pixels the parameters come from: `"biological_repeat"` (one repeat group: the same biological repeat in every condition, technical repeats merged, as paired in the preprocessing brick's `experiment_design.csv`), `"image"` (all Z and T planes of the source image) or `"patch"` (the patch's own pixels). Per channel, except for `image_type="rgb"`. `"condition"` and `"technical_repeat"` were removed (October 2026) and are refused. |
| `per_time_point` | `bool` | `False` | Normalize each time point separately (`t` joins the group). |
| `experiment_design` | `str \| None` | `None` | For `scope="biological_repeat"`: path of the preprocessing brick's `experiment_design.csv`; `None` uses the one next to `plane_histograms`. Every repeat group must be confirmed, and the file must be the one the preprocessing manifest records. |
| `pooling` | `str` | `"pixels"` | For `scope="biological_repeat"`: `"pixels"` (each pixel counts once) or `"conditions"` (each condition of a repeat group weighs the same). |
| `image_type` | `str` | `"fluorescence"` | `"fluorescence"` (each channel normalized separately) or `"rgb"`: colour images of three 8- or 16-bit channels, normalized with StarDist's percentile method, one offset and scale from the R, G and B pixels together (written for `C0`, `C1` and `C2`). `"rgb"` needs `method="percentile"` and `extraction.split_channels=False`. When the preprocessing manifest records an image type, this must be the same. |
| `patch_region` | `str \| None` | `None` | For `scope="patch"`: `"mask"` (the patch's own label), `"bbox"` (the label's whole bounding box, whatever the patch size) or `"patch"`; `None` means `"mask"` in `cell` mode and `"patch"` otherwise. Padding is never used. |

| `export_values` | `str` | `"normalized"` | What exported files hold: `"normalized"` (float32) or `"raw"` (the source values; `patches.csv` holds the parameters and the loaders normalize on load). A raw patch that is resized or masked is stored as float32 raw values. |
| `memory_limit` | `int` | `4 * 1024**3` | Bytes allowed for the normalized patches of eager delivery (`common.extract_arrays=True`); above it the run stops before extracting and suggests `iter_patch_arrays`. |

**Delivery** (contract §12) — all modes give the same arrays for the same configuration (raw files of resized or masked patches up to float32 rounding):

| Mode | How | Values |
|---|---|---|
| Normalized files (default) | `export.enabled=True` | float32 `.tif`/`.npy` |
| Raw files plus parameters | `export.enabled=True`, `normalization.export_values="raw"`; read back with `cellpatcher.iter_exported_patches(csv)`/`load_exported_patch` or `ExportedPatchTorchDataset`, which normalize on load with each row's `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>` columns (never recomputed; the `;`-separated `normalization_offset`/`normalization_scale` of earlier tables are read too) | Source dtype when the patch is only cropped; float32 raw values when resized or masked |
| In memory, eager | `common.extract_arrays=True` | float32 arrays in `PatchingResult.patch_arrays`, within `memory_limit` |
| In memory, lazy | Run with `extract_arrays=False` (the patches still record their normalization), then `cellpatcher.iter_patch_arrays(patches, images, config, masks=...)` or `patch_array(...)`; `PatchTorchDataset(images, patches, config=config)` is built on them | float32, one patch at a time |

```python
from cellpatcher import iter_patch_arrays, list_supported_files, run_patching_condition_repeat_tree

for leaf in run_patching_condition_repeat_tree(root, config=config):  # extract_arrays=False
    images = list_supported_files(leaf.image_folder)  # paths: read one plane at a time
    for patch, array in iter_patch_arrays(leaf.result.patches, images, config):
        ...  # one normalized patch at a time
```

**What happens** (contract §6–§9):

- The pipeline becomes generate → filter → extract → **normalize** → apply mask → resize → export. Filters see raw values; `extraction.mask_fill_value` is in normalized units (`NaN` allowed); resizing works on the float32 values. Padding added by `pad_mode="constant"` is normalized like the raw value `constant_values`.
- Each patch becomes `(pixels - offset) / scale` per channel, float32, without clipping.
- The repeat and image scopes pool the CSV rows of the patch's group (the normalization brick's `scope_groups`: the repeat group across all conditions, or the image; per channel, and per `t` with `per_time_point`), including planes that were not patched.
- Before anything is normalized, every patched plane must have its CSV row with the same labels, dtype, size, exact finite extrema and finite pixel count; with the manifest, each image's size, modification time and axes must match too. Every image without rows is listed in one message.
- Normalization needs one plane per patch: `common.z_policy`/`t_policy` `"keep"` or a projection along an axis longer than 1 stops the run.
- A group with no finite pixels or zero spread stops the run with the list of groups. At the patch scope, a patch whose own pixels have zero spread or no finite pixels is discarded (decision D8): it is not exported or delivered, and one `NormalizationWarning` lists every discarded patch.
- Warnings (`cellpatcher.normalization.NormalizationWarning`): approximate parameters (float images); group and image statistics include background; the `biological_repeat` scope removes differences between repeats and keeps those between conditions; a repeat group without images from some condition.

**Recorded:** written after `source_relative_path` in `patches.csv`: `normalization_method` (e.g. `percentile 1-99.8`), `normalization_scope`, `normalization_group` (the repeat group label, `<source_relative_path>`, or `<source_relative_path>::<patch_id>` for the patch scope; `::t=<t>` with `per_time_point`) and `normalization_exact` (`true`/`false`), then one `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>` pair per channel, grouped by channel (full-precision numbers from the normalization brick's `Parameters.to_columns()`; a split-channel patch fills only its own channel's pair). In memory, `PatchRegion.normalization_parameters` is the brick's `Parameters` (`offset`, `scale`, `channels`); `normalization_offset`/`normalization_scale` are deprecated read-only properties giving the former `;`-separated text. `run_config.json` gains the `normalization` section, a top-level `normalization_inputs` object (`plane_histograms_sha256`, `schema_version`, `manifest_used`, `manifest_version`, `preprocessing_version`, and for `biological_repeat` `experiment_design_file`/`experiment_design_sha256`) and `normalization_settings` (the brick's `NormalizationSettings.to_dict()`); a `run_config.json` without `normalization` loads with normalization off. With normalization, the images of a run must have the same number of channels.

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

## Patch identity (`patch_id`, `position_id`, `object_set`)

Owner decisions D44, D45 and D47 (2026-10-26). A patch's identity comes only from its field of view and its own coordinates, never from the place of its image in the run, so the same file gives the same `patch_id` alone, in a folder of ten files, after a file that sorts first, or in a tree run beside other wells; a rerun writes a byte-identical `patches.csv`. The code is `cellpatcher.ids`.

**`position_id`** (always filled; D45). The one supplied with `image_metadata=[{"position_id": ...}]` wins. Otherwise one file is taken as one XY position: `<condition>/<biological_repeat>/<technical_repeat>/<file stem>` (the stem of `image_file`, `img.ome.zarr` giving `img`; labels not given are `single`). An in-memory image passed to `run_patching` with neither a `position_id` nor an `image_file` is **refused** with an error naming its place in the run: pass `image_metadata=[{"position_id": ...}, ...]`, one entry per image (an `{"image_file": ...}` entry also works: its stem then names the position). Folder and tree runs read one image per file: a file holding several images (an OME-Zarr plate or multi-position store, a TIFF with several series) is refused, whether its planes are read one at a time (`z_policy`/`t_policy` `"all"`, `"first"` or `"select"` with the `preprocessing` extra) or the file is read whole (`"keep"`, projections, or without the extra), with an error naming the file and its number of images/series. Give one file per position, or pass `position_id` (with the plane indices) through `run_patching`.

**`object_set`** (cell mode; `None` in other modes). Precedence: `image_metadata[i]["object_set"]`; then `common.object_set`; then, in folder and tree runs, the mask's own name — the OME-NGFF label image name with `mask_source="ngff_labels"` (e.g. `cells`), or the part of the mask file's stem after the image's stem without leading `_`, `-`, `.` (`img_cells.tif` → `cells`, `img_mask.tif` → `mask`); finally `"mask"` (the mask role) when nothing names it (in-memory runs, a mask named like its image, or a mask whose name does not start with the image's stem). Characters other than letters, digits, `.`, `_`, `-` in a derived name become `_`.

**`patch_id` grammar.** Levels are joined by `/`; absent levels are omitted:

```text
patch_id := position_id "/" [object_set "/"] ["t" T "/"] ["z" Z "/"] ["c" C "/"] token
token    := "label" LABEL                                (cell)
          | "grid-y" Y0 "-x" X0 "-h" H "-w" W            (grid)
          | "random-y" Y0 "-x" X0 "-h" H "-w" W ["-n" K] (random)
          | "whole_fov"                                  (whole_fov)
```

`T` is `t` with 4 digits, `Z` is `z0` with 3 digits, `C` is `c` (no padding); each is present when the patch has that index (a plane split or selected by `t_policy`/`z_policy`, an index from `image_metadata`, or a split channel). `LABEL` is the object's own `label_id`; `Y0`/`X0`/`H`/`W` the rectangle's top-left corner and size in pixels. `K` (2, 3, ...) numbers repeated draws of the same rectangle in one random plane, in sampling order. Examples:

| Run | `patch_id` | Exported file |
|---|---|---|
| Cell mode, `ctrl/2026_01_01/B02/img.tif` with mask `img_cells.tif` | `ctrl/2026_01_01/B02/img/cells/label42` | `ctrl_2026_01_01_B02_img_cells_label42.tif` |
| Grid mode, `img.tif` in a flat folder | `single/single/single/img/grid-y0-x128-h128-w128` | `single_single_single_img_grid-y0-x128-h128-w128.tif` |
| Cell mode, time series with `t_policy="all"`, `split_channels=True`, label image `nuclei` | `ctrl/2026_01_01/B02/fov1/nuclei/t0003/c1/label7` | `ctrl_2026_01_01_B02_fov1_nuclei_t0003_c1_label7.tif` |

Exported files are named `sanitize_filename(patch_id)` plus the extension: `/` and any other character outside letters, digits, `.`, `_`, `-` become `_`. The export refuses two ids that give the same file name, and every run refuses two patches with the same `patch_id` (other than repeated random draws): for example two images given the same `position_id` without different `t`/`z`/`c` indices.

**Random mode is seeded per FOV.** Each image, or each plane with split planes, is sampled with its own seed: `numpy.random.SeedSequence([random.seed, *w])` with `w` the eight 32-bit little-endian words of the SHA-256 of the canonical identity string `<position_id>|t=<t>|z=<z>|c=<c>` (`-` for an absent index), from which one 64-bit state word is the generator's seed (`cellpatcher.ids.random_seed`; Python's salted `hash()` is not used). The same image therefore gets the same rectangles, and ids, alone, in a folder of ten files or after a file that sorts first; another `random.seed`, position or plane gets others. Channels split with `split_channels` share their region's rectangles. `random.seed=None` stays unseeded (different on every run).

**`image_id`** (`image_0000`, ...) is still the image's place in the run (sorted file order in a folder run, restarting in every leaf of a tree run). It joins `patches.csv` to `images.csv` and indexes a run's image list for lazy delivery; it is not an identity across runs.

**Before 2026-10-26** (breaking change): `patch_id` was `<image_id>_<mode>_<index>` (`image_0000_cell_00042`, `image_0000_grid_00003`, `image_0000_random_00007`, `image_0000_whole_fov_0`) plus `_t<NNN>`/`_z<NNN>` for split planes and `_c<n>` for split channels, so it changed with the run; `position_id` was only filled with `split_channels`; `binary_mask_policy` defaulted to `"connected_components"`; random mode sampled image *k* of a run with `random.seed + k` (plane *k* with split planes); unnamed in-memory images were accepted; and a multi-series TIFF was read from its first series.

## `PatchRegion` fields

Every generated patch is a `PatchRegion` dataclass. The fields below are populated for all modes; `to_dict()`/the exported `patches.csv` also include the `height`/`width` computed properties as extra convenience columns.

| Field | Type | Description |
|-------|------|-------------|
| `patch_id` | `str` | Run-independent identifier of the patch: its `position_id`, object set, plane indices and label or rectangle (see "Patch identity" above). Its sanitized form is the export filename stem. |
| `image_id` | `str \| None` | The source image's place in the run (e.g. `"image_0000"`); joins `images.csv`, not an identity across runs. |
| `mode` | `str \| None` | Patching mode that generated this patch. |
| `condition` | `str \| None` | From `config.common.condition`. See the shared CSV contract below. |
| `biological_repeat` | `str \| None` | From `config.common.biological_repeat`. See the shared CSV contract below. Formerly `repeat` (deprecated alias). |
| `technical_repeat` | `str \| None` | From `config.common.technical_repeat`, or per image from `image_metadata`. See the shared CSV contract below. |
| `position_id` | `str \| None` | Stable spatial acquisition-position identifier, shared by all Z/C/T planes from one XY position. Always filled: supplied via `image_metadata`, else `<condition>/<biological_repeat>/<technical_repeat>/<file stem>` (see "Patch identity" above). |
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
| `source_relative_path` | `str \| None` | The source image's path relative to the experiment root, with `/` separators, from where it sits in the folder tree (`control/img.tif` for an image directly in `control/`, never the `single` output folders). A linked image or folder keeps the link's path under the root, not its target's, as the preprocessing brick records it (see "Trees of links" below). Links the patch to the preprocessing brick's `plane_id` ([normalization contract](normalization_contract.md) §2). Tree runs make it relative to `image_root_path`; `run_patching_pipeline` to the image folder (the file name); direct `run_patching()` calls take it from `image_metadata`, else `None`. Followed in `patches.csv` by the six `normalization_*` columns (see Normalization above). |
| `image_height`, `image_width` | `int \| None` | Y/X size of the source image (the plane, for split Z/T planes) that `y0/y1/x0/x1` refer to. Set by `run_patching` for every patch, with or without arrays. |
| `pad_top`, `pad_bottom`, `pad_left`, `pad_right` | `int \| None` | Rows/columns of `extraction.constant_values` added on each side where the rectangle extends past the image (`extraction.pad_mode="constant"`), counted before resizing; `0` when nothing was padded (`"clip"` never pads, and the built-in modes keep every rectangle inside the image). `None` unless arrays were extracted. |
| `resized` | `bool \| None` | Whether `extraction.output_size` changed the patch's Y/X shape: `False` without an output size, and for a patch that already had it (resizing then leaves the pixels unchanged). `None` unless arrays were extracted. |
| `output_axes` | `str \| None` | Axes of the delivered/exported array, in order: `common.image_axes` without the Z/T axes a plane policy resolved and, for split channels, without `C` (e.g. `"CYX"`, `"YX"`, `"ZCYX"` with `z_policy="keep"`). Exported TIFF/NPY files carry no axes of their own. `None` unless arrays were extracted. |
| `object_set` | `str \| None` | Cell mode: the object set (mask) the patch's `label_id` belongs to (see "Patch identity" above); `None` in other modes. Last column. |

The geometry columns come after every earlier one, and `object_set` after them, so the order of the existing columns does not change.

### Shared MorphoBricks CSV contract

`condition`, `biological_repeat`, `technical_repeat` (optional), `position_id` (when needed), `image_file`, `mask_file`, and `patch_file` are the columns of the MorphoBricks ecosystem's shared CSV contract (`AGENTS.md` §5), alongside the 0-based plane indices `z0`/`z1`, `c` and `t` (see the table above). CellPatcher populates them as follows:

- `condition`/`biological_repeat` — either set `config.common.condition`/`config.common.biological_repeat` explicitly and call `run_patching_pipeline` on a single folder, or call `run_patching_condition_repeat_tree(image_root_path, ...)` on a `raw_data/<condition>/<biological_repeat>/` root and let both be derived per leaf from the folder path (see below). A label that is not given, or set to `None`, is written as `"single"` (`SINGLE_CONDITION_LABEL`, `SINGLE_BIOLOGICAL_REPEAT_LABEL`, `SINGLE_TECHNICAL_REPEAT_LABEL`), as for a missing folder level, so the label columns are never empty.
- `technical_repeat` — the well, dish or coverslip within the biological repeat. Tree runs derive it from an optional `<condition>/<biological_repeat>/<technical_repeat>/` level and write `"single"` when a biological-repeat folder holds its images directly, so the column is never empty in a tree run. For a single folder set `config.common.technical_repeat`; when one folder holds several wells, pass `image_metadata=[{"technical_repeat": ...}, ...]` to `run_patching`.
- `position_id` — always filled (see "Patch identity" above): supplied per image via `image_metadata`, else `<condition>/<biological_repeat>/<technical_repeat>/<file stem>`. The canonical MorphoBricks FOV key is `condition` + `biological_repeat` + `technical_repeat` + `position_id`.
- `image_file`/`mask_file`/`patch_file` are populated automatically by `run_patching_pipeline` (folder-based runs, including each leaf of `run_patching_condition_repeat_tree`) and `export_patches` respectively.

### Per-image table (`images.csv`)

Wherever `patches.csv` is written (`export.enabled=True` with `save_metadata_csv=True`: `run_patching`, `run_patching_pipeline`, each leaf of `run_patching_condition_repeat_tree`, its combined output, and `export_patches`), `images.csv` is written beside it (`cellpatcher.export.IMAGE_TABLE_FILENAME`). It has **one row per input image the run considered**, including images that gave no patch, so a reader can tell an image without patches from one that was never patched. In memory, `PatchingResult.image_records` holds the same rows as `ImageRecord` objects (also without export), and `PatchingResult.image_table_file` names the file.

| Column | Notes |
|---|---|
| `condition`, `biological_repeat`, `technical_repeat` | The labels the image's patches get in `patches.csv` (`"single"` when not given). |
| `image_id` | `image_0000`, ... as in `patches.csv`: unique within one `patches.csv` of a folder or leaf, the only key of a direct `run_patching` call without file names. |
| `image_file`, `mask_file`, `source_relative_path` | As in `patches.csv`; empty when not known (direct `run_patching` without `image_metadata`, no mask). |
| `image_height`, `image_width` | The image's Y/X size, as in the patches' `image_height`/`image_width`. Empty for a skipped image. |
| `n_patches` | The image's number of rows in `patches.csv` (one per channel patch with `split_channels`). |
| `status` | `processed`: `n_patches` > 0. `no_patches`: the image was read and patched, but no patch was kept. `skipped`: the image had nothing to patch. |
| `reason` | Empty for `processed`. For `no_patches`, the step that left none: `the patch mode generated no region in this image` (e.g. no labeled object in its mask, every object excluded by border or area), `the filters removed all <n> region(s)`, or `normalization discarded all <n> patch(es): they cannot be normalized from their own pixels` (decision D8). For `skipped`: `the image has no plane to patch along the axes split by z_policy/t_policy` (a Z or T axis of length 0 with `"all"`). |
| `position_id` | The `position_id` every patch of the image carries in `patches.csv`, also for an image without patches: supplied via `image_metadata`, or the default of "Patch identity" above (one file is one position). Never empty since 2026-10-26. |
| `n_regions`, `n_kept`, `n_normalized` | Regions the patch mode generated, those left by the filters, and those left by normalization (equal to `n_kept` without it), counted before `split_channels`; they tell the three `no_patches` reasons apart without reading `reason`. `0` for a skipped image. |

`position_id`, `n_regions`, `n_kept` and `n_normalized` were added after the first release of the table, as its last columns, so earlier columns keep their positions.

There is no `failed` status: an image that cannot be read, matched to a mask or patched stops the whole run with an error before any table is written, as before, so no image is ever left out silently.

**Joining the two tables.** Every `patches.csv` row's image has exactly one `images.csv` row with the same `condition`/`biological_repeat`/`technical_repeat`/`image_id`/`image_file`, and that row has the patch's `position_id` (`image_id` alone is not unique across a combined tree CSV; `source_relative_path` is unique in folder and tree runs). An image's `n_patches` is its number of `patches.csv` rows. `export_patches` called directly writes the records it is given (`image_records=`), or else one `processed` row per image of the patches, since it cannot know about images without patches.

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
- **Trees of links.** Image files and folders may be symbolic links, for example a tree that gathers a subset of images stored elsewhere into the `<condition>/<biological_repeat>/` layout. Links are followed for reading; `image_file` is the link's name and `source_relative_path` the link's path under the root (as the preprocessing brick records it), so the patches of a linked tree are those of a copied tree. A link whose target does not exist raises (it is a missing image), in folder and tree runs alike. Two links to the same file are two images.

When `mask_root_path` is given, the same structure is mirrored under it per leaf. Mask folder requirements follow the same per-mode rules as `run_patching_pipeline` (required for `mode="cell"`, optional for `mode="random"`, not accepted for `mode="grid"`/`"whole_fov"`), applied independently to each leaf — a leaf missing a required mask folder raises immediately and aborts the whole tree run rather than being silently skipped.

Returns a list of `ConditionRepeatResult(condition, biological_repeat, image_folder, result, technical_repeat, relative_folder)`, one per leaf, in the same (condition, then biological repeat, then technical repeat) sorted order `discover_condition_repeat_folders` returns (whose `ConditionRepeatFolder` entries likewise carry `biological_repeat` and `technical_repeat`).

- `combine_metadata=True` (requires `export.enabled=True` and `common.output_dir`) writes **one** `patches.csv`, **one** `images.csv` (every image of every leaf) and **one** `run_config.json` directly in `output_dir` instead of a set per leaf folder. Patch files still go to each leaf's `<output_dir>/<condition>/<biological_repeat>[/<technical_repeat>]/`, and each row's `patch_file` is the path relative to the CSV, e.g. `treated/2026_01_01/image_0000_grid_00000.tif` or `treated/2026_01_01/B02/image_0000_grid_00000.tif` — the same "relative to the CSV" meaning as in the per-leaf layout, so `ExportedPatchTorchDataset` reads the combined CSV unchanged. `patch_id`/`image_id` are only unique within a leaf; identify rows across the combined CSV by `condition`/`biological_repeat`/`technical_repeat`/`image_file` (or `patch_file`).
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

- The row's own coordinates (`y0/y1/x0/x1`, `z0/z1`, `t`, `c`) and `output_height`/`output_width` fully describe the patch geometry and final exported array shape; `image_height`/`image_width`, the `pad_*` columns, `resized` and `output_axes` say, without the source image or the config, where the rectangle lies in its image, which pixels of the array are padding, whether the array was resampled and what its axes are.
- `run_config.json` is the complete, serialized `CellPatcherConfig` for the run — strategy parameters (including `random.seed`), extraction/resize settings, matching strategy, and axes — everything needed to reproduce the same coordinates deterministically from the same inputs.
