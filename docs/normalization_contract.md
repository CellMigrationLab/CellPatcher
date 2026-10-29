# Normalization Contract (Milestone 0)

> See also: [Docs map](README.md) · [Plan](plan.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

This document fixes the rules for normalizing patches with the statistics of the MorphoBricks
preprocessing brick. It is the reference for Milestones 1–5 of the [plan](plan.md), which
records why each rule was chosen (decisions D1–D16). Changing a rule after it is implemented
needs owner approval (AGENTS.md §8).

**Status: implemented (Milestones 1–5).** Nothing here is available in the
package until the milestone named in each section is merged; until then `CellPatcherConfig`
has no `normalization` section and `patches.csv` has no normalization columns. Milestone 1
added the `preprocessing` extra (§1), the `source_relative_path` field and column (§2, §10),
and `cellpatcher.statistics`, which loads the preprocessing outputs and runs the checks of
§5; the pipeline uses them from Milestone 3. Milestone 2 added OME-Zarr images (§2), the Z/T
plane policies with plane-by-plane reading (§3, except the blocks under normalization, which
need `NormalizationConfig` from Milestone 3), and OME-NGFF label masks (§4). Milestone 3 added `config.normalization` and the
normalization step (§6–§9, §14, the §3 blocks), and, by owner decision, the `patches.csv`
columns and `run_config.json` records of §10–§11. Milestone 4 added `export_values`,
`memory_limit` and the delivery modes of §12 (`cellpatcher.delivery`, the torch adapters).
Milestone 5 decided D8 (§8), added the normalization step to `CellPatcher.ipynb` and the user
guide [normalization.md](normalization.md).

| Section | Implemented in |
|---|---|
| 1. Dependency | Milestone 1 |
| 2. Images and discovery | Milestones 1–2 |
| 3. Planes | Milestone 2 |
| 4. Masks | Milestone 2 |
| 5. Preprocessing outputs and checks | Milestone 1 |
| 6. Methods | Milestone 3 |
| 7. Scopes | Milestone 3 |
| 8. Patch scope | Milestone 3 |
| 9. Pipeline order | Milestone 3 |
| 10. `patches.csv` columns | Milestone 3 (moved from 4) |
| 11. `run_config.json` | Milestone 3 (moved from 4) |
| 12. Delivery | Milestone 4 |
| 13. Configuration | Milestones 1–4 |
| 14. Validation | Milestones 1–4 |
| 15. Deprecations | Milestone 3 |

The statistics themselves (histograms, pooling, the four methods) follow the preprocessing
brick's
[contract](https://github.com/CellMigrationLab/morphobricks-preprocessing/blob/main/docs/contract.md)
§4, §6 and §7, and are computed by its package, never reimplemented here.

## 1. Dependency

| Rule | Detail |
|---|---|
| Extra | `pip install "cellpatcher[preprocessing]"` installs `morphobricks-preprocessing[zarr]` and the `morphobricks-normalization` it requires. It is never a core dependency. Histograms, the plane table, plane IDs, pooling and normalization parameters, which moved to the normalization brick, are imported from `morphobricks_normalization`. |
| Version | Installed from GitHub at a pinned commit or tag, recorded in `setup.py` and raised deliberately. |
| Formats | Reads `plane_histograms.csv` schema version `2` (every row has `z`, `c` and `t`) and `run_manifest.json` manifest version `1`; any other version is an error. |
| Needed for | Normalization (every scope), OME-Zarr images and labels, and plane-by-plane image reading. |
| Without it | Patching works exactly as today; enabling normalization or giving OME-Zarr input stops the run with an install instruction. |
| Imported | `read_plane_histograms`, `read_manifest`, `manifest.source_fingerprint`, `make_plane_id`, `group_records`, `compute_plane_histogram`, `PooledHistogram`, `min_max`, `percentile_range`, `z_score`, `robust_z_score`, `NormalizationParameters`, the OME-Zarr reader (`zarr_input`) and the plane readers (`planes`). |
| Missing pieces | Functionality CellPatcher needs that belongs with these (for example reading OME-Zarr label images) is added to the preprocessing package and imported. |

## 2. Images and discovery

| Rule | Detail |
|---|---|
| Discovery | CellPatcher always discovers images with its own folder rules (`discover_condition_repeat_folders`), whether or not preprocessing was run. The manifest never decides which images are patched. |
| Experiment root | The folder given to CellPatcher as image root must be the folder given to the preprocessing brick, so relative paths match. |
| Image types | TIFF files (`.tif`, `.tiff`, any case) and, with the extra, OME-Zarr images: folders whose names end in `.zarr`, read from level 0 of their first image. A store with several images is an error, as in preprocessing. |
| OME-Zarr copies | Copies written by the preprocessing brick (`<output>/ome_zarr/...`) are not read; CellPatcher reads the source images. |
| `source_relative_path` | The image's path relative to the experiment root, with `/` separators, from its real location in the tree (`control/img.tif` for an image directly in `control/`; a linked image or folder keeps the link's path under the root, as the brick records it when it walks the tree). Never built from `ConditionRepeatFolder.relative_folder`, which inserts `single` for missing levels. |
| Labels | `condition`, `biological_repeat` and `technical_repeat` come from CellPatcher's discovery as today. For images both bricks accept, they equal the CSV's labels; a difference is an error (§5). |

## 3. Planes

Patching is 2D. With normalization enabled, every axis other than Y and X must end as one
index per plane:

| Axis | Allowed | Blocked under normalization |
|---|---|---|
| Z | `z_policy="all"` (each slice its own plane), `"select"`, `"first"` | `"keep"` (all slices in one patch, the default) and `"max"`, `"mean"`, `"sum"` projections |
| T | `t_policy="all"`, `"select"`, `"first"` | `"keep"` (all time points in one patch, the default) |
| C | `split_channels=True` (one patch per channel) or a list of channel indices (one patch per selected channel), `channel_policy="select"`/`"first"`, or channels kept together in one patch | `channel_policy="max"`/`"mean"` (merged channels) and `"repeat_to_rgb"` in the preparation helpers |

Repeating a normalized grayscale patch to RGB stays available in the torch adapters
(`PatchTensorConfig.channel_policy="repeat_to_rgb"`), which run after normalization.

**Plane identity.** A row of `plane_histograms.csv` (schema version 4) is an XYC plane: every
channel of an image at one Z and T, each channel's histogram in its own columns. It is
identified by `make_plane_id(source_relative_path, z=..., t=...)`, with CellPatcher's `z0` as
`z`. Both bricks always write both coordinates: an axis the image does not have, or has with
length 1, has coordinate 0 (`control/img.tif::z=0,t=0` for a 2D image), so the same plane has
the same `plane_id` whether or not its file or CellPatcher's axes include the axis. A channel
`c` is found inside its row; a row without that channel is a missing row. A patch that keeps
several channels belongs to one plane per channel, `c = 0 … C−1`, all in one row; messages
name them `<plane_id> c=<c>`.

**Memory.** CellPatcher needs each plane whole in memory to patch it. A plane larger than the
memory limit is an error, even if the preprocessing brick read it in blocks.

## 4. Masks

`common.mask_source` selects where masks come from:

| Value | Masks | Matching |
|---|---|---|
| `"tiff_tree"` (default) | TIFF files in a separate tree mirroring the image tree (`masks/<condition>/<biological_repeat>[/<technical_repeat>]/`), as today | By file name with `config.matching`; works for TIFF and OME-Zarr images |
| `"ngff_labels"` | Label images inside the image's own `.ome.zarr`, in its `labels/` group (OME-NGFF 0.4 and 0.5) | By name: `common.mask_label_name` |

Rules for `"ngff_labels"`:

- The `labels` group's metadata lists the available label images; `mask_label_name` must be
  one of them. If the image has exactly one, it may be left `None`.
- Label values are integers (OME-NGFF requires it); level 0 is read.
- Each label dimension equals the image's or has length 1. The channel dimension, when
  present, must have length 1.
- A label dimension equal to the image's Z or T follows the image's planes (one label plane
  per image plane); a dimension of length 1 is shared by all planes along that axis.
- Only images given as OME-Zarr can use `"ngff_labels"`.

## 5. Preprocessing outputs and checks

| Input | Required | Detail |
|---|---|---|
| `plane_histograms.csv` | For the `biological_repeat` and `image` scopes | Path given in `normalization.plane_histograms`. Not needed for the `patch` scope. |
| `run_manifest.json` | Never | Given in `normalization.manifest`, or found next to the CSV. An explicitly given path that does not exist is an error. |
| `experiment_design.csv` | For the `biological_repeat` scope | Given in `normalization.experiment_design`, or found next to the CSV. It must exist, every repeat group must be confirmed with one folder per condition (the preprocessing brick's `design_issues`), its SHA-256 must equal the manifest's `experiment_design_sha256` when the manifest records one, and every image's condition and biological repeat must be in it. Checked before any image is read. |

Checks before any patch is normalized; every failure blocks the run:

**Always (with the CSV):** for every plane CellPatcher patches:

- a row with the same `plane_id` exists;
- the row's `condition`, `biological_repeat` and `technical_repeat` equal CellPatcher's;
- `dtype` and `total_pixel_count` (Y × X) are equal;
- the plane's exact finite minimum and maximum equal `pixel_min` and `pixel_max`, and its
  finite pixel count equals `valid_pixel_count`.

An image CellPatcher found without any row in the CSV (excluded or skipped in preprocessing,
or added later) is reported with all such images in one message.

**With the manifest:** in addition,

- `csv_sha256` equals the SHA-256 of the CSV given;
- each patched image's `manifest.source_fingerprint()` (size and modification time) equals its
  manifest entry;
- CellPatcher's image axes give the same role for every axis as the manifest's
  `axis_mapping`;
- the manifest's `image_type`, when it records one, equals `normalization.image_type`
  (otherwise `image_type_differs`, checked before any image is read). A manifest without an
  image type (written before the preprocessing brick asked for it), or no manifest, leaves
  the choice to the configuration (normalization brick decision N34).

## 6. Methods

| `normalization.method` | Preprocessing function | Name written to `patches.csv` |
|---|---|---|
| `"minmax"` | `min_max` | `min-max` |
| `"percentile"` (default) | `percentile_range(low, high)` with `normalization.percentile`, default `(1.0, 99.8)` | `percentile 1-99.8` (with the chosen values) |
| `"zscore"` | `z_score` | `z-score` |
| `"robust_zscore"` | `robust_z_score` (MAD × 1.4826) | `robust z-score` |

Applying parameters computes `(pixels − offset) / scale` with
`NormalizationParameters.apply(pixels, dtype=float32)`, per channel. There is no clipping.
Non-finite input pixels stay non-finite. Parameters are **exact** when every pooled histogram
has a zero representative error (`PooledHistogram.max_representative_error == 0`: integer
images with one bin per value), and histogram approximations otherwise (always for float
images).

**RGB images (follow-up C3).** With `normalization.image_type="rgb"` the images must have
exactly three channels of an 8- or 16-bit integer type, checked before any parameters are
fitted, and the only method is `"percentile"`: StarDist's
`csbdeep.utils.normalize(x, low, high, axis=(0, 1, 2))`. At every scope the R, G and B pixels of
the group are pooled together into one histogram, giving one shared `offset = P(low)` and
`scale = P(high) − P(low) + 1e−20` (linear percentiles), applied to the three channels with
CSBDeep's float32 arithmetic (normalization brick N33, N40, N48). The method is written as
`rgb percentile 1-99.8`. Because of the `1e−20` term, a group or patch without spread gives
zeros instead of an error. `normalization.clip` still applies afterwards.

## 7. Scopes

`normalization.scope` chooses which pixels the parameters come from. Each group uses its own
parameters, always per channel.

| Scope | Pooled planes, per channel | From | `normalization_group` |
|---|---|---|---|
| `"biological_repeat"` | One repeat group: the same biological repeat in every condition, technical repeats merged, as paired in `experiment_design.csv` | CSV rows, the normalization brick's `scope_groups(records, settings, repeat_groups)` | `<repeat group>` |
| `"image"` (default) | All Z and T planes of one image | `scope_groups(records, settings)` | `<source_relative_path>` |
| `"patch"` | The patch's selected pixels (§8) | the brick's `fit` on the pixels | `<source_relative_path>::<patch_id>` |

- Every channel has its own parameters (the brick's `fit`), so channels never share them.
- `normalization.pooling` (`biological_repeat` only): `"pixels"` (default) counts each pixel
  once; `"conditions"` weighs each condition of the repeat group equally.
- `biological_repeat` keeps the differences between conditions within a repeat and removes
  the differences between repeats; a repeat group without images from some condition is
  normalized with the conditions it has, with the brick's `repeat_missing_conditions` warning.
- The `condition` and `technical_repeat` scopes were removed on 2026-10-28, without a
  deprecation period (normalization brick N29): a configuration or `run_config.json` naming
  them is refused. Raw patches exported with them still normalize on load, from their
  `patches.csv` columns. Until then `biological_repeat` meant one group per condition and
  repeat, named `<condition>/<biological_repeat>`.
- `normalization.per_time_point=True` adds `t` to the key; `::t=<t>` is appended to
  `normalization_group`.
- All CSV rows of the group are pooled, including planes of the same images that were not
  patched (for example Z slices without cells).
- A group with no finite pixels, or whose parameters have zero or non-finite scale, is an
  error (preprocessing contract §7).

## 8. Patch scope

`normalization.patch_region` chooses the pixels of a patch-scope histogram, taken from the
plane before resizing:

| Value | Pixels | Modes |
|---|---|---|
| `"mask"` (default in `cell` mode) | The pixels of the patch's own label inside the patch | `cell` |
| `"bbox"` | All pixels of the label's bounding box in the plane, whatever the patch size | `cell` |
| `"patch"` (default in other modes) | All pixels of the patch | all |

- Padding pixels (`pad_mode="constant"`) are never used.
- The histogram uses `compute_plane_histogram` with its default bin count, so 8- and 16-bit
  images give exact parameters and float images approximations within half a bin.
- **Zero spread (decision D8):** a patch whose parameters have zero scale (for example an
  empty background tile, or for robust z-score a patch where more than half the pixels share
  one value), or whose region has no finite pixels, is discarded: it is dropped before
  extraction and never exported or delivered, and one warning lists every discarded patch and
  suggests another method, region or scope.
- With `image_type="rgb"`, the region selects the same pixels in R, G and B, pooled together
  (§6). An RGB patch without spread becomes zeros and is kept.

## 9. Pipeline order

generate regions → filter → extract → **normalize** → apply mask → resize → export.

- Filters (`config.filters`) are evaluated on raw values.
- `extraction.mask_fill_value` is in normalized units; `NaN` is allowed.
- Resizing works on the float32 normalized values.

## 10. `patches.csv` columns

Optional columns appended after all existing columns in this order: `source_relative_path`,
four `normalization_*` columns, then one `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>`
pair per channel, grouped by channel (normalization brick decisions N41, N42; CellPatcher
follow-up C1). Existing columns keep their names, order and values, so readers that select
columns by name are unaffected. With normalization off, the four `normalization_*` columns are
empty and there are no channel pairs.

| Column | Type | Meaning |
|---|---|---|
| `source_relative_path` | text | §2; filled whenever it is known, also without normalization |
| `normalization_method` | text | §6 name |
| `normalization_scope` | text | §7 scope value |
| `normalization_group` | text | §7 group key |
| `normalization_exact` | `true`/`false` | `true` when every channel's parameters are exact (§6) |
| `Normalization_Offset_C<i>` | number | Offset of channel `i` of the image |
| `Normalization_Scale_C<i>` | number | Scale of channel `i` of the image |

The pairs are written by the normalization brick's `Parameters.to_columns()`, with full
precision (Python `repr` of the float). A table has one pair for every channel any of its
patches holds. A patch that keeps several channels fills every pair, and its `c` stays empty;
a split-channel patch (`c` = `i`), including one of a channel selection
(`split_channels=[i, ...]`, follow-up C5), fills only its own pair, from the group's
`Parameters.select([i])` (normalization brick N46), and leaves the others blank; a table whose
patches all come from one channel has only that channel's pair.
An RGB patch (§6) fills `C0`, `C1` and `C2` with the same shared offset and scale.

Tables written before 2026-10-28 have `;`-separated `normalization_offset` and
`normalization_scale` columns instead (one value per channel of the patch, in channel order);
CellPatcher still reads them when normalizing raw files on load (normalization brick N47).
In memory, `PatchRegion.normalization_parameters` holds the brick's `Parameters`; the former
`normalization_offset`/`normalization_scale` attributes are deprecated read-only properties.

## 11. `run_config.json`

- The serialized config gains a `normalization` section (§13). Loading a `run_config.json`
  without it means normalization off.
- A top-level `normalization_inputs` object records: `plane_histograms_sha256`,
  `schema_version`, `manifest_used` (`true`/`false`), `manifest_version` (or `null`) and
  `preprocessing_version`. Paths are not recorded, as they are machine-specific.
- With the `biological_repeat` scope, `normalization_inputs` also records
  `experiment_design_file` (the file name) and `experiment_design_sha256`.
- A top-level `normalization_settings` object holds the normalization brick's
  `NormalizationSettings.to_dict()` (scope, method, its options and, for
  `biological_repeat`, the pooling weight).

## 12. Delivery

| Mode | How | Values |
|---|---|---|
| Normalized files (default) | `export.enabled=True`, `normalization.export_values="normalized"` | float32 `.tif`/`.npy` |
| Raw files plus parameters | `export.enabled=True`, `normalization.export_values="raw"` | Source dtype; the columns of §10 hold the parameters, and loaders normalize on load |
| In memory, eager | `common.extract_arrays=True` | float32 arrays in `PatchingResult.patch_arrays` |
| In memory, lazy | A NumPy iterator over patches (core) and `PatchTorchDataset` (torch extra) | float32, one patch at a time |

- All modes give identical arrays for the same configuration.
- Eager delivery estimates its memory first and stops above `normalization.memory_limit`,
  suggesting the lazy iterator.
- Loading raw files normalizes with the row's parameter columns (§10: the brick's
  `Parameters.from_columns`, or the `;`-separated columns of earlier tables) and never
  recomputes parameters.

## 13. Configuration

`config.normalization` (`NormalizationConfig`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | `bool` | `False` | Off keeps current behaviour exactly |
| `plane_histograms` | `str \| None` | `None` | Path of `plane_histograms.csv` (§5) |
| `manifest` | `str \| None` | `None` | Path of `run_manifest.json`; `None` looks next to the CSV |
| `method` | `str` | `"percentile"` | §6 |
| `percentile` | `tuple[float, float]` | `(1.0, 99.8)` | Low and high percentiles, `0 ≤ low < high ≤ 100` |
| `clip` | `tuple[float, float] \| None` | `None` | Limits applied right after normalizing (by the brick's `NormalizationParameters.apply`), before masking and resizing; the same in eager, lazy, exported and raw-on-load delivery |
| `scope` | `str` | `"image"` | §7: `"biological_repeat"`, `"image"` or `"patch"` |
| `per_time_point` | `bool` | `False` | §7 |
| `patch_region` | `str \| None` | `None` | §8; `None` means `"mask"` in `cell` mode and `"patch"` otherwise |
| `export_values` | `str` | `"normalized"` | §12: `"normalized"` or `"raw"` |
| `memory_limit` | `int` | `4 * 1024**3` | Bytes, for eager delivery (§12) |
| `experiment_design` | `str \| None` | `None` | Path of `experiment_design.csv` (§5); `None` looks next to the CSV |
| `pooling` | `str` | `"pixels"` | §7: `"pixels"` or `"conditions"` |
| `image_type` | `str` | `"fluorescence"` | §6: `"fluorescence"` (each channel separately) or `"rgb"` (R, G and B together, `"percentile"` only, `extraction.split_channels=False`); must equal the manifest's image type when it records one (§5) |

`config.common` gains `mask_source` (`"tiff_tree"` or `"ngff_labels"`, default `"tiff_tree"`)
and `mask_label_name` (`str | None`, default `None`) (§4), and `z_policy`/`t_policy`
(`"keep"` by default, or `"all"`, `"first"`, `"select"`; Z also `"max"`, `"mean"`, `"sum"`)
with `z_index`/`t_index` (`int | None`), which split Z and T in folder, tree and
`run_patching` runs (§3; Milestone 2, decided October 2026).

## 14. Validation

Every message states what happened in plain language, lists the affected paths or patches,
and suggests an action, in the preprocessing brick's style. Technical details go to the log.

| Severity | Situations |
|---|---|
| Error (run blocked) | Extra not installed; CSV missing for a group or image scope; unsupported CSV or manifest version; explicitly given manifest missing; any check of §5 failing; image without CSV rows at a group or image scope; experiment design missing, unreadable, not confirmed, changed since preprocessing, or without an image's repeat (§5); a removed scope (§7); an `image_type` other than the manifest's, `"rgb"` with a method other than `"percentile"` or with `split_channels`, or `"rgb"` images that are not three 8- or 16-bit channels (§5, §6); projection, merged channels or `repeat_to_rgb` under normalization (§3); plane over the memory limit; `"ngff_labels"` on a TIFF image, a missing or ambiguous label name, or a label image whose dimensions do not fit (§4); `"mask"`/`"bbox"` region outside `cell` mode; zero or non-finite scale, or no finite pixels, for a group; deprecated normalization arguments combined with normalization (§15); eager delivery over the memory limit; OME-Zarr input without `zarr` |
| Warning (run continues) | Patches discarded without spread or finite pixels (§8, D8); approximate parameters; group statistics include background pixels (once per run); the `biological_repeat` scope (removes differences between repeats, keeps those between conditions); a repeat group without images from some condition; use of a deprecated normalization argument |

## 15. Deprecations

Deprecated when Milestone 3 is merged; each use emits one `DeprecationWarning` naming the
replacement (`config.normalization`). Removal, and changing the `scale_policy` default, need
owner approval in a later version.

| Deprecated | Where |
|---|---|
| `normalize=`, `percentile=` arguments | `prepare_image_for_patching`, `iter_image_planes_for_patching`, `iter_cell_planes_for_patching` |
| `percentile_normalize`, `minmax_normalize` | `cellpatcher.preprocess` |
| `scale_policy="auto"` rescaling an integer patch by its dtype maximum (the default) | `PatchTensorConfig` in `cellpatcher.torch_dataset`; it never rescales normalized float patches |
