# CellPatcher — Normalization and Preprocessing Integration Plan

> See also: [Docs map](README.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md) · [Developer notes](developer_notes.md)

**Status:** design brief; decisions D1–D16 were reviewed in October 2026 and are recorded in
§13; **[D8]** (patches without intensity spread) was decided with Milestone 5. The exact rules are
fixed in the [normalization contract](normalization_contract.md) (Milestone 0); where this
plan and the contract differ in detail, the contract applies.
**Scope:** use the outputs of the MorphoBricks preprocessing brick
([morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing))
to normalize patches at a chosen scope with a chosen method; read TIFF and OME-Zarr images
and masks; deliver the normalized patches either as files or in memory.
**Out of scope for this milestone:** new patching strategies (the `cell`, `random`, `grid`
and `whole_fov` modes stay as they are), true 3D patches, spatial corrections (flat-field,
BaSiC, rolling-ball, CLAHE), and planes too large to hold in memory.

The statistics and normalization formulas are defined by the preprocessing brick's
[contract](https://github.com/CellMigrationLab/morphobricks-preprocessing/blob/main/docs/contract.md)
(§4 histograms, §6 group statistics, §7 normalization parameters) and implemented in its
package. This plan does not redefine or reimplement them; it decides how CellPatcher
selects, applies and records them.

## Instructions for implementation

Follow [AGENTS.md](../AGENTS.md). In particular: keep the pipeline coordinate-first, keep
model-specific code out of the core package, never rename or repurpose a `PatchRegion`
field, and keep every feature usable from the notebook, from scripts and as an imported
dependency. Normalization is off by default; with it off, patch arrays and the values of
every existing `patches.csv` column must stay the same (the CSV only gains the new columns of
§8).

Import functionality that `morphobricks_preprocessing` already provides (reading its CSV and
manifest, histograms, pooling, normalization parameters, OME-Zarr reading, plane reading)
instead of reimplementing it. If something CellPatcher needs is missing there and belongs
there, add it to the preprocessing package rather than to CellPatcher.

Never apply an intensity transformation the user did not choose. Never modify source images
or the preprocessing outputs. Present validation messages in plain language, with the
affected paths and a suggested action, following the preprocessing brick's style.

## 1. Goals

CellPatcher must:

1. Accept TIFF and OME-Zarr images, and masks either as a TIFF mask tree or as OME-Zarr
   labels inside the image.
2. Split any Z, C and T axes into 2D planes; never project a Z-stack.
3. Use `plane_histograms.csv` from the preprocessing brick for group statistics, with
   `run_manifest.json` as an optional extra for checks.
4. Link every patch to the histogram rows of the plane(s) it was cut from.
5. Normalize patches with min-max, percentile range, z-score or robust z-score at the
   condition, biological-repeat, technical-repeat, image (default) or patch scope.
6. Deliver normalized patches as files on disk or in memory (eagerly or lazily) for deep
   learning and feature extraction.
7. Record in `patches.csv` exactly which normalization was applied to each patch, so it can
   be reproduced and, if needed, undone.

## 2. Current state

Already implemented in this repository:

- The four strategies and the coordinate-first pipeline: generate regions → filter →
  extract → apply mask (cell mode) → resize → export (`pipeline.py`).
- `patches.csv` with the shared MorphoBricks columns and `run_config.json`
  (`export.py`, `orchestration.py`).
- Folder discovery for `<condition>/<biological_repeat>[/<technical_repeat>]/` trees
  (`discover_condition_repeat_folders`), with `single` for missing levels, and mask trees
  that mirror the same levels.
- The `technical_repeat` field and column (PR #9): derived from the folder level in tree
  runs (`SINGLE_TECHNICAL_REPEAT_LABEL` when a biological repeat holds its images
  directly), or set with `CommonConfig.technical_repeat` or per image with
  `image_metadata`.
- Splitting stacks into 2D planes: `z_policy="all"` and `t_policy="all"` in
  `iter_image_planes_for_patching`/`iter_cell_planes_for_patching`, and
  `ExtractionConfig.split_channels`. The same helpers can also project Z (`"max"`,
  `"mean"`, `"sum"`).
- Per-image normalization: `percentile_normalize` and `minmax_normalize`, used through the
  `normalize=`/`percentile=` arguments of the preparation helpers (`preprocess.py`), and
  the torch adapters' `PatchTensorConfig.scale_policy="auto"`, which divides integer patches
  by the dtype maximum (`torch_dataset.py`). All of these are deprecated by this plan (§6.8).
- Cell patch sizes: `CellConfig.patch_size="auto"` sizes each patch to its own object's
  bounding box; a fixed size can be recommended from all masks by `scan_cell_patch_size`
  (`max`, `median`, `mean`, `p90`, `p95`).

Already implemented in the preprocessing brick and imported here:

- `read_plane_histograms()` (rows as `PlaneRecord`) and `read_manifest()`.
- `group_records(records, level)` for the levels `condition`, `biological_repeat`,
  `technical_repeat` and `image`; `make_plane_id()`.
- `compute_plane_histogram()` for any set of pixels, `PooledHistogram` to pool histograms,
  and `min_max`, `percentile_range`, `z_score`, `robust_z_score`, each returning
  `NormalizationParameters(method, offset, scale)` with `apply()`.
- OME-Zarr reading (`zarr_input.open_ngff`, level 0) and plane-by-plane reading with an axis
  mapping and a memory limit (`planes`).

Gaps this plan closes:

- `read_array` only reads `.npy`, TIFF and imageio formats, and loads whole files with
  `tifffile.imread`; there is no OME-Zarr reader and no OME-Zarr label reader.
- `PatchRegion` has `image_file`, the three labels and `z0/z1`, but not
  `source_relative_path`, so a patch cannot be matched to its histogram row unambiguously:
  two technical repeats can both hold `field_001.tif`.
- There is no way to choose a normalization scope or record the parameters used.

## 3. Inputs

### 3.1 Images and discovery

CellPatcher always finds and reads the source images itself, with its existing folder
discovery, whether or not the preprocessing brick was run **[D2]**. The experiment root
given to CellPatcher must be the same folder that was given to the preprocessing brick, so
that relative paths match.

- TIFF images are read as today. With the preprocessing extra installed, they are read
  plane by plane with its readers instead of as whole files.
- OME-Zarr images (folders ending in `.zarr`) are read from level 0 with the preprocessing
  brick's reader, with the axes declared in their metadata. They need the preprocessing
  extra with `zarr` **[D1]**.
- OME-Zarr copies written by the preprocessing brick are not used in this milestone;
  CellPatcher reads the source images, which hold the same values.

### 3.2 Splitting into 2D planes

Patching is 2D. Every Z, C and T axis can be split until a 2D plane remains **[D4]**: Z with
`z_policy="all"`, T with `t_policy="all"`, C with `split_channels=True`, or a single index
with `"select"`/`"first"`. When normalization is enabled, Z and T must be split or selected:
Z projections (`"max"`, `"mean"`, `"sum"`) are blocked, because each slice of a Z-stack is
treated as an individual plane and the preprocessing statistics describe single planes
**[D11]**. Channels may be split, selected or kept together in one patch; their parameters
are always per channel (§6.1). For the same reason, merging channels
(`channel_policy="max"` or `"mean"`) and `channel_policy="repeat_to_rgb"` in the preparation
helpers are blocked under normalization: the resulting channels no longer match the CSV rows.
Repeating a normalized grayscale patch to RGB stays available in the torch adapters
(`PatchTensorConfig.channel_policy`), which run after normalization.

### 3.3 Masks

Two mask sources, chosen by the user **[D3]**:

- **TIFF mask tree** (existing): `masks/<condition>/<biological_repeat>[/<technical_repeat>]/`
  mirroring the image tree, matched by file name. Works for TIFF and OME-Zarr images.
- **OME-Zarr labels**: masks stored inside the image's own `.ome.zarr`, in its `labels/`
  group, as the OME-NGFF specification defines. The `labels` group lists the available
  label images (for example `cells`, `nuclei`); the user chooses one by name. Label images
  hold integer values, have the same dimensions as the image or length 1 where a dimension
  does not apply (for example the channel axis), and have the same number of resolution
  levels. CellPatcher reads level 0, and resolves each label image to one 2D plane per image
  plane with the same Z and T rules as images. Reading label images is added to the
  preprocessing package's OME-Zarr reader if it cannot already open them, and imported.

### 3.4 Preprocessing outputs

To normalize at a group or image scope, the user gives the path of `plane_histograms.csv`.
`run_manifest.json` is optional **[D2]**: it is used when it is next to the CSV or given by
the user, and everything still works without it.

Checks, which block the run when they fail **[D16]**:

- **Always:** every plane CellPatcher patches has a row in the CSV (same
  `source_relative_path`, `z`, `c`, `t`), with the same `dtype` and `total_pixel_count`.
  CellPatcher holds each plane in memory to patch it, so it also compares the plane's exact
  finite minimum and maximum and its finite pixel count with the row's `pixel_min`,
  `pixel_max` and `valid_pixel_count`. This detects changed images without the manifest.
- **With the manifest:** each image's size and modification time match its manifest entry,
  the CSV's SHA-256 matches `csv_sha256`, and CellPatcher's axes match the manifest's
  confirmed axis mapping.

An image found by CellPatcher but absent from the CSV (excluded or skipped in
preprocessing, or new) blocks group and image scopes, with the list of images; the user can
remove those images from the run or use the patch scope.

## 4. Linking patches to plane statistics

Each 2D plane is identified by the preprocessing `plane_id`, built with `make_plane_id()`:
`source_relative_path` plus `::z=…,c=…,t=…`, always with all three coordinates (0 for an
axis the image does not have; preprocessing schema version 2). CellPatcher adds one
optional `PatchRegion` field, `source_relative_path` (default `None`, so existing outputs are
unchanged): the image's path relative to the experiment root, `/`-separated, exactly as in
preprocessing. It is computed from the image's real location under the root. It must not be
built from `ConditionRepeatFolder.relative_folder`, which names the output folder and
inserts `single` for missing levels (`control/single/` for images directly in `control/`).

The preprocessing CSV writes `z`; CellPatcher writes `z0/z1` one plane thick, so the join uses
`z = z0`. A patch that keeps several channels links to one row per channel.

## 5. Normalization methods

The four methods of the preprocessing contract §7: min-max, percentile range (default
1–99.8), z-score and robust z-score (MAD × 1.4826). Each is an offset and a scale; applying
it computes `(pixels − offset) / scale` with `NormalizationParameters.apply()` and returns
float32, without clipping **[D10]**.

Normalization needs the preprocessing extra **[D1]**. All parameters come from histograms
pooled with `PooledHistogram`, so every scope uses the same definitions:

- **Group and image scopes:** the rows of `plane_histograms.csv`. Parameters are exact for
  exact integer histograms and approximate otherwise (always for float images); the output
  records which.
- **Patch scope:** a histogram of the patch's selected pixels (§6.3) from
  `compute_plane_histogram()`, with the same default bin counts. Exact for 8- and 16-bit
  images; within half a bin width for floats.

## 6. Normalization scopes and semantics

### 6.1 Channels are always separate

Parameters are always computed and applied per channel; channels are never pooled
together **[D4]**.

### 6.2 Scopes

Each group is normalized with its own parameters **[D5]**:

| Scope | Pixels pooled (per channel) | Source |
|---|---|---|
| `condition` | All planes with the same `condition` | CSV |
| `biological_repeat` | Same `(condition, biological_repeat)` | CSV |
| `technical_repeat` | Same `(condition, biological_repeat, technical_repeat)` | CSV |
| `image` (default) | All Z and T planes of one source image | CSV |
| `patch` | The patch's selected pixels (§6.3) | Patch pixels |

Groups come from `group_records()` with the channel added to the key, so `bio_rep1` under
two conditions is two groups. Scopes pool all time points by default; `per_time_point=True`
adds `t` to the key, so each time point is normalized separately (for example to compensate
photobleaching) **[D4]**.

Normalizing per condition or per repeat puts each group on the same scale, which removes
real intensity differences between groups. The notebook explains this next to the scope
choice.

### 6.3 Patch scope

Patch-scope parameters come from the patch's own pixels, before resizing, from one of three
regions **[D6]**:

| Region | Pixels used | Modes |
|---|---|---|
| `mask` (default in cell mode) | Only the pixels of the patch's own label | `cell` |
| `bbox` | All pixels of the label's bounding box in the image | `cell` |
| `patch` | All pixels of the patch | all modes |

- `bbox` uses the full bounding box from the image: if the patch is larger than the box
  (for example a fixed size from `scan_cell_patch_size(...).recommended_size("max")`), the
  extra pixels,
  which may belong to neighbouring cells, are not used; if the box is larger than the patch,
  the whole box is still used. A cell's parameters therefore do not depend on the patch
  size.
- Padding pixels (`pad_mode="constant"`) are never used.
- A patch with zero intensity spread cannot be normalized (for example an empty background
  tile, or for robust z-score any patch where more than half the pixels share one value,
  which makes the MAD zero). Such a patch, and one whose region has no finite pixels, is
  discarded and reported in one warning listing the patches and suggesting another method,
  region or scope **[D8]**.

### 6.4 Background pixels in group statistics

Group and image statistics come from whole planes, so in sparse images they are dominated by
background; percentile and z-score parameters then describe the background more than the
cells. Masks are not used in the preprocessing statistics **[D7]**; the notebook states this
once per run. The patch scope with the `mask` region is the alternative when cell-only
parameters are needed.

### 6.5 Order of operations

The pipeline becomes: generate regions → filter → extract → **normalize** → apply mask →
resize → export **[D9]**. AGENTS.md §2 is updated in the milestone that adds the step.

- Filters (`min_mean_intensity`, `foreground_threshold`) keep working on raw values, so
  existing thresholds keep their meaning **[D15]**.
- Normalizing before masking means `mask_fill_value` is in normalized units (default 0).
  For z-score, 0 is the mean intensity, so a `NaN` fill is offered as well.
- Normalizing before resizing avoids rounding resized integer values. Without clipping,
  the order does not change the result, because resizing is linear in the pixel values.

### 6.6 Output values

Normalized patches are float32 and never clipped **[D10]**: values outside 0–1 (min-max and
percentile range) or far from 0 (z-scores) are kept.

### 6.7 Without preprocessing

Patching without the preprocessing extra or the CSV works exactly as today. Without the CSV,
only the patch scope is available; choosing another scope stops the run with a message
suggesting to run the preprocessing brick or to choose the patch scope.

### 6.8 Deprecated normalizations

Everything related to normalization in the current version is deprecated in favour of this
design **[D12]**:

- The `normalize=` and `percentile=` arguments of `prepare_image_for_patching`,
  `iter_image_planes_for_patching` and `iter_cell_planes_for_patching`, and the
  `percentile_normalize` and `minmax_normalize` functions: each use emits a
  `DeprecationWarning` naming the replacement. Combining them with the new normalization is
  an error, to prevent normalizing twice.
- `PatchTensorConfig.scale_policy="auto"` (the default): a `DeprecationWarning` when it
  actually rescales an integer patch. It never rescales normalized float patches.
- Removal and the change of the `scale_policy` default happen in a later version, with
  owner approval (AGENTS.md §8).

## 7. Delivery

Three modes, all producing identical arrays for the same configuration **[D13]**:

1. **Normalized files** (default): patches exported as float32 `.tif` or `.npy` with
   `patches.csv` and `run_config.json`. Float32 needs twice the disk space of uint16; the
   notebook shows an estimate before running.
2. **Raw files plus parameters:** patches exported in their source dtype, with the
   parameters in `patches.csv`; normalization happens when loading (a NumPy loader in the
   core, and `ExportedPatchTorchDataset`). Smaller, reversible, and the method can change
   without re-patching.
3. **In memory:**
   - eager: `run_patching(...)` with `extract_arrays=True` returns normalized arrays. The
     memory needed is estimated first; above a configurable limit the run stops and
     suggests the lazy option, as in preprocessing.
   - lazy (recommended for large data): a NumPy iterator in the core that extracts and
     normalizes one patch at a time, and `PatchTorchDataset` built on it. Torch stays an
     optional extra.

## 8. Output contract changes

All additions are optional new `PatchRegion` fields appended after the existing columns;
the normalization columns are empty when normalization is off. The shared label columns,
including `technical_repeat`, are unchanged. Owner approval for these additions is recorded
by the decisions in §13 (AGENTS.md §8).

**New `PatchRegion` fields / `patches.csv` columns** **[D14]**:

| Column | Meaning |
|---|---|
| `source_relative_path` | Image path relative to the experiment root, `/`-separated (§4) |
| `normalization_method` | `min-max`, `percentile 1-99.8`, `z-score` or `robust z-score` (the preprocessing method names) |
| `normalization_scope` | `condition`, `biological_repeat`, `technical_repeat`, `image` or `patch` |
| `normalization_group` | Group key: the labels joined with `/` (e.g. `control/bio_rep1`), the `source_relative_path` for image scope, or `<source_relative_path>::<patch_id>` for patch scope (patch ids are only unique within a leaf); `::t=<t>` is appended when `per_time_point` is on |
| `normalization_offset`, `normalization_scale` | One number per channel; for a patch that keeps several channels, numbers separated by `;` in channel order |
| `normalization_exact` | `true` when every channel's parameters are exact, `false` when any is a histogram approximation |

**`run_config.json`** records the normalization settings and, in a `normalization_inputs`
object, the CSV's SHA-256 and `schema_version`, whether a manifest was used, and the
preprocessing package version (no machine-specific paths). Loading an older
`run_config.json` without these keys means normalization off.

**AGENTS.md §5** is updated: inputs (OME-Zarr images and labels, the preprocessing CSV and
optional manifest) and outputs (the new columns).

## 9. Configuration

A new `NormalizationConfig` in `CellPatcherConfig`; names fixed in Milestone 0 (contract §13):

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `False` | Normalization off keeps current behaviour |
| `plane_histograms` | `None` | Path of `plane_histograms.csv`; required except for patch scope |
| `manifest` | `None` | Path of `run_manifest.json`; found next to the CSV when not given; optional |
| `method` | `"percentile"` | `minmax`, `percentile`, `zscore`, `robust_zscore` |
| `percentile` | `(1.0, 99.8)` | Low and high percentiles |
| `scope` | `"image"` | One of §6.2 |
| `per_time_point` | `False` | Make `t` part of the group key |
| `patch_region` | `None` | One of §6.3; `None` means `"mask"` in cell mode, else `"patch"` |
| `export_values` | `"normalized"` | Exported files hold `"normalized"` or `"raw"` values (§7 modes 1 and 2) |
| `memory_limit` | 4 GiB | Limit for eager in-memory delivery |

The delivery modes of §7 reuse the existing switches: `export.enabled` for files (with
`export_values` choosing normalized or raw), `common.extract_arrays` for eager arrays, and a
lazy iterator in the API. Mask sources are set with `common.mask_source` (`"tiff_tree"`, the
default, or `"ngff_labels"`) and `common.mask_label_name` for the OME-Zarr label image.

## 10. Validation and messages

Block the run for:

- Normalization without the preprocessing extra installed.
- A group or image scope without `plane_histograms.csv`.
- A patched plane without a CSV row, or whose dtype, pixel count, extrema or finite pixel
  count differ from its row; with a manifest, a changed image, CSV or axis mapping (§3.4).
- An image not in the CSV at a group or image scope.
- A group with zero or non-finite spread, or with no finite pixels (contract §7). Patches
  without spread or finite pixels are discarded and reported instead **[D8]**.
- A Z projection, merged channels or `channel_policy="repeat_to_rgb"` in the preparation
  helpers while normalization is enabled (§3.2).
- A deprecated normalization argument combined with the new normalization.
- An OME-Zarr label name that the image does not have, or a label image whose dimensions do
  not fit the image.
- Eager in-memory delivery above `memory_limit`.
- OME-Zarr images or labels without the `zarr` dependency.

Warn and continue for:

- Approximate parameters (float images or reduced-bin histograms).
- Group statistics that include background (§6.4), once per run.
- Per-condition and per-repeat scopes, with the explanation of §6.2.
- Use of a deprecated normalization argument (§6.8).

## 11. Architecture and milestones

New modules, independent of widgets: `statistics.py` (load the CSV and optional manifest,
link planes, run the checks of §3.4) and `normalization.py` (scopes, groups, patch regions,
parameters, apply); OME-Zarr image and label reading in `io.py` through the preprocessing
package. Histograms, pooling, parameters and readers are imported from
`morphobricks_preprocessing`, installed with an optional `preprocessing` extra
(`pip install cellpatcher[preprocessing]`, which installs `morphobricks-preprocessing[zarr]`
from GitHub at a pinned version) **[D1]**.

- **Milestone 0 — Contract:** document scopes, patch regions, columns and checks; fix the
  configuration names. *Done (October 2026):*
  [normalization_contract.md](normalization_contract.md), with pointers in
  `configuration_reference.md` and AGENTS.md §4–§6. The configuration reference describes
  the new fields as planned until they are implemented.
- **Milestone 1 — Preprocessing link:** the optional extra; read the CSV and optional
  manifest; `source_relative_path` on every patch; the checks of §3.4. *Done (October
  2026):* `setup.py` extra, `PatchRegion.source_relative_path`, `cellpatcher.statistics`.
- **Milestone 2 — Images and masks:** OME-Zarr images; OME-Zarr labels (added to the
  preprocessing reader if needed); plane-by-plane TIFF reading; the splitting rules of §3.2
  with projections blocked under normalization. *Decided (October 2026):* folder, tree and
  `run_patching` runs split Z and T with new `common.z_policy`/`z_index`/`t_policy`/`t_index`
  (the preparation helpers' values plus `"keep"`, the default, which keeps today's behaviour);
  masks follow the image's planes when their Z/T length equals the image's and are shared
  when they lack the axis or have length 1 (the §4 rule, for every mask); a split axis keeps
  one `image_id` per source image and appends the plane to `patch_id`
  (`image_0000_grid_00003_z002`, before `_c<n>`). Blocking projections under normalization
  needs `NormalizationConfig`, so it lands with Milestone 3. Delivered in steps: plane
  splitting, OME-Zarr images, OME-NGFF labels (with `ngff_label_names`/`open_ngff_labels`
  added to the preprocessing package), plane-by-plane reading (one image at a time, and with
  the extra only the chosen TIFF/OME-Zarr planes, within the preprocessing memory limit).
- **Milestone 3 — Normalization engine:** the scopes of §6.2 per channel and optionally per
  time point; patch regions; the pipeline step and the AGENTS.md §2 update; deprecations.
  *Decided (October 2026):* the `patches.csv` columns and `run_config.json` records (§8)
  move into Milestone 3, so every normalized export is reproducible from the start;
  `export_values` and `memory_limit` arrive with the delivery modes in Milestone 4. For
  in-memory `run_patching`, the CSV scopes take each image's `source_relative_path` from
  `image_metadata`, and the manifest's file checks run in folder and tree runs. Padding
  added by `pad_mode="constant"` is normalized like the raw value `constant_values`.
- **Milestone 4 — Delivery:** the `patches.csv` columns and `run_config.json` keys; the three
  delivery modes; normalization on load for raw files; the NumPy iterator and torch
  adapters. *Decided (October 2026):* raw files keep the source dtype when a patch is only
  cropped and hold float32 raw values when it is resized or masked (a NaN mask fill
  survives), so loading always gives the delivered patch, up to float32 rounding;
  `iter_patch_arrays`, `patch_array`, `iter_exported_patches` and `load_exported_patch` are
  exported from the package. `PatchTorchDataset` accepts the run's config to cut each patch
  from its `z0`/`t` plane. The columns and `run_config.json` records were delivered with
  Milestone 3.
- **Milestone 5 — Notebook and docs:** a normalization step in `CellPatcher.ipynb` (choose the
  CSV and optional manifest, method, scope and patch region; table of parameters per group;
  before/after preview of a few patches with their histograms; warnings), plus user
  documentation. *Done (October 2026):* D8 decided (discard and report); the notebook (v0.0.6)
  gains a mask source choice (OME-Zarr labels), plane policies, the normalization section with
  `normalization.group_parameters`, readiness checks of the preprocessing inputs, and the
  before/after preview; user guide [normalization.md](normalization.md).
- **Later:** reading only patch windows for planes too large to hold in memory (the
  preprocessing brick now computes statistics for those planes in blocks, but patching still
  needs the whole plane); reading the preprocessing OME-Zarr copies; reference parameters
  (one group's parameters applied to all patches) if needed.

## 12. Tests and acceptance criteria

Use simulated images, masks and preprocessing outputs generated in the tests; the
`morphobricks-preprocessing` package produces real CSVs and manifests for fixtures.

- With normalization off, all existing tests pass and outputs are unchanged.
- A patch normalized at a group or image scope equals `NormalizationParameters.apply()` on
  the raw crop, with parameters from `PooledHistogram` over exactly the group's rows.
- With exact integer histograms, group parameters equal direct calculations on the pooled
  pixels.
- Patch-scope parameters use exactly the pixels of the chosen region: only the label for
  `mask`, the full bounding box for `bbox` (larger or smaller than the patch), the patch for
  `patch`, never padding.
- Channels never share parameters; `per_time_point` separates time points; each Z slice is a
  separate plane, and projections are blocked under normalization.
- OME-Zarr images and labels give the same patches as the same data stored as TIFF images
  and a TIFF mask tree.
- Each row's normalization columns, with `run_config.json`, reproduce the normalized patch,
  including `;`-separated parameters for multichannel patches.
- Normalized files, raw files normalized on load, eager and lazy delivery give identical
  arrays.
- The checks of §3.4 block changed images both with and without a manifest.
- Deprecated arguments warn, and combining them with the new normalization is an error.
- Labels such as `NA`, `001` and `single` survive the round trip through both CSVs.
- On a three-level tree that both bricks accept, the labels CellPatcher's discovery assigns
  equal the CSV's labels, and two technical repeats holding the same file name get different
  `source_relative_path` and `plane_id` values.

## 13. Decisions

Reviewed in October 2026.

1. **[D1] Dependency on the preprocessing package.** *Decided:* import
   `morphobricks_preprocessing` as an optional extra and reuse its functionality instead of
   reimplementing it. Normalization and OME-Zarr input need the extra.
2. **[D2] Role of the preprocessing outputs.** *Decided:* the preprocessing brick is
   optional, and CellPatcher discovers and reads the source images the same way whether or
   not it was run. It does not rely on the manifest: the manifest is used for extra checks
   when available, and `plane_histograms.csv` is needed only for group and image scopes.
3. **[D3] Masks for OME-Zarr images.** *Decided:* support both a TIFF mask tree and OME-NGFF
   labels inside the image.
4. **[D4] Splitting and channels.** *Decided:* any of Z, C and T can be split until a 2D plane
   remains; parameters are always per channel; time points are pooled by default, with an
   option to normalize each separately.
5. **[D5] Scopes.** *Decided:* condition, biological repeat, technical repeat, image and patch,
   from the preprocessing CSV (patch scope from the patch's pixels); image is the default.
   Each group uses its own parameters; reference parameters are not offered in this
   milestone.
6. **[D6] Patch-scope pixels.** *Decided:* three regions: the label's pixels (`mask`), its
   bounding box (`bbox`) and the whole patch (`patch`).
7. **[D7] Masked group statistics.** *Decided:* no; the preprocessing statistics stay
   unmasked, with a note about background.
8. **[D8] Patches without intensity spread.** *Decided (October 2026, Milestone 5):*
   discard them. A patch whose own pixels (patch scope) have zero spread, or no finite
   pixels, is dropped before extraction, never kept unnormalized, and every dropped patch is
   listed in one warning suggesting another method, region or scope. A group without spread
   still stops the run (contract §7).
9. **[D9] Pipeline order.** *Decided:* extract → normalize → mask → resize; update AGENTS.md.
10. **[D10] Clipping and dtype.** *Decided:* no clipping; float32 output.
11. **[D11] Z projections.** *Decided:* no projections under normalization; each Z slice is an
    individual plane.
12. **[D12] Existing normalizations.** *Decided:* deprecate everything related to
    normalization in the current version and replace it with this design.
13. **[D13] Delivery modes.** *Decided:* all three of §7, normalized files by default.
14. **[D14] Parameters storage.** *Decided:* columns in `patches.csv`, with `;`-separated
    values per channel for multichannel patches.
15. **[D15] Filters.** *Decided:* keep filters on raw values.
16. **[D16] Changed images.** *Decided:* block the run.
