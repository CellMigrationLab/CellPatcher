# CellPatcher — Normalization and Preprocessing Integration Plan

> See also: [Docs map](README.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md) · [Developer notes](developer_notes.md)

**Status:** design brief for review; open decisions are marked **[D1]**, **[D2]**, … and
collected in §13 with a recommended default for each.
**Scope:** read the outputs of the MorphoBricks preprocessing brick
([morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing)),
read TIFF and OME-Zarr images, normalize patches at a chosen scope with a chosen method,
and deliver the normalized patches either as files or in memory.
**Out of scope for this milestone:** new patching strategies (the `cell`, `random`, `grid`
and `whole_fov` modes stay as they are), true 3D patches, spatial corrections (flat-field,
BaSiC, rolling-ball, CLAHE), and planes larger than the memory limit.

The statistics and normalization formulas are defined by the preprocessing brick's
[contract](https://github.com/CellMigrationLab/morphobricks-preprocessing/blob/main/docs/contract.md)
(§4 histograms, §6 group statistics, §7 normalization parameters). This plan does not
redefine them; it decides how CellPatcher selects, applies and records them.

## Instructions for implementation

Follow [AGENTS.md](../AGENTS.md). In particular: keep the pipeline coordinate-first, keep
model-specific code out of the core package, never rename or repurpose a `PatchRegion`
field, and keep every feature usable from the notebook, from scripts and as an imported
dependency. Normalization is off by default; with it off, every existing output must stay
byte-for-byte the same.

Never apply an intensity transformation the user did not choose. Never modify source images
or the preprocessing outputs. Present validation messages in plain language, with the
affected paths and a suggested action, following the preprocessing brick's style.

## 1. Goals

CellPatcher must:

1. Accept TIFF images and OME-Zarr images, including the OME-Zarr copies written by the
   preprocessing brick.
2. Optionally take the preprocessing output folder (`plane_histograms.csv` and
   `run_manifest.json`) as the description of the experiment: files, labels, axis mapping
   and statistics.
3. Link every patch to the histogram rows of the plane(s) it was cut from.
4. Normalize patches with min-max, percentile range, z-score or robust z-score, at one of
   the scopes of §6.2, including a patch-level scope.
5. Deliver normalized patches in two ways: saved to disk for later use, or kept in memory
   (eagerly or lazily) for deep learning and feature extraction.
6. Record in its outputs exactly which normalization was applied, so every patch can be
   reproduced and, if needed, un-normalized.

## 2. Current state

Already implemented in this repository:

- The four strategies and the coordinate-first pipeline: generate regions → filter →
  extract → apply mask (cell mode) → resize → export (`pipeline.py`).
- `patches.csv` with the shared MorphoBricks columns and `run_config.json`
  (`export.py`, `orchestration.py`).
- Folder discovery for `<condition>/<biological_repeat>/` trees
  (`discover_condition_repeat_folders`), with `single` for missing levels.
- Per-image normalization helpers: `percentile_normalize` and `minmax_normalize`, used by
  `prepare_image_for_patching(normalize=...)` (`preprocess.py`).
- A lazy `PatchTorchDataset` and an `ExportedPatchTorchDataset`; their
  `PatchTensorConfig.scale_policy="auto"` divides integer patches by the dtype maximum
  (`torch_dataset.py`).

Already implemented in the preprocessing brick and reusable here:

- `read_plane_histograms()` and `read_manifest()` for its two output files.
- `PooledHistogram`, which pools any set of plane rows into one distribution.
- `min_max`, `percentile_range`, `z_score`, `robust_z_score`, each returning
  `NormalizationParameters(method, offset, scale)` with `apply(pixels, clip, dtype)`.
- Plane-by-plane readers for TIFF and OME-Zarr with the confirmed axis mapping and a
  memory limit (`planes.iter_planes`, `planes.read_plane`, `zarr_input`).

Gaps this plan closes:

- `read_array` only reads `.npy`, TIFF and imageio formats, and loads whole files with
  `tifffile.imread`; there is no OME-Zarr reader and no plane-by-plane reading.
- Discovery reads only the condition and biological-repeat levels, not technical repeat.
  A folder mixing images and subfolders is a `ValueError` here but a warning in
  preprocessing.
- `PatchRegion` has `image_file` and `z0/z1`, but not `source_relative_path` or
  `technical_repeat`, so a patch cannot be matched to its histogram row unambiguously.
- There is no way to choose a normalization scope or record the parameters used.

## 3. Inputs

### 3.1 Two ways to describe an experiment

- **Without preprocessing** (unchanged): the existing folder entry points, with `image_axes`
  given by the user. Only the image, plane and patch scopes are available, because there are
  no stored statistics; their parameters are computed directly from pixels.
- **With preprocessing:** the user points CellPatcher at the preprocessing output folder.
  The manifest provides the experiment root, the analysed files, their labels (including
  `technical_repeat`), the confirmed axis mapping, exclusions and the OME-Zarr copy paths.
  CellPatcher does not ask for axes again and does not walk the folders again
  **[D2]**. All scopes are available.

### 3.2 Images

- TIFF images are read plane by plane, never as a whole file, using the confirmed axis
  mapping.
- OME-Zarr images are read from level 0 only, as in the preprocessing contract §2.
- When the manifest records an OME-Zarr copy of a TIFF (`ome_zarr_path`), CellPatcher reads
  the copy by default (faster partial reads) and still keys its rows by the TIFF's
  `source_relative_path`, matching the histogram rows.
- OME-Zarr input needs the optional `zarr` dependency, as in preprocessing.

### 3.3 Masks

Cell mode keeps using a separate mask tree that mirrors the image tree
(`masks/<condition>/<biological_repeat>/<technical_repeat>/`), matched by file name with the
existing strategies. Masks for OME-Zarr images may be TIFF files matched to the image's name
without `.ome.zarr`, or the image's NGFF `labels/` group **[D3]**.

### 3.4 Staleness

Before patching with stored statistics, compare each image's size and modification time
with its manifest entry, and the CSV's SHA-256 with `csv_sha256`. If anything differs, the
statistics may not describe the images: block and ask the user to rerun preprocessing
**[D16]**. A file in the experiment but not in the manifest (excluded or new) cannot be
normalized at a group scope.

## 4. Linking patches to plane statistics

Each 2D plane CellPatcher patches is identified by the preprocessing `plane_id`:
`source_relative_path` plus `::z=…,c=…,t=…` for the coordinates present. To build it,
CellPatcher adds two optional `PatchRegion` fields (default `None`, so existing outputs are
unchanged):

| New field | Meaning |
|---|---|
| `source_relative_path` | Path relative to the experiment root, `/`-separated, as in preprocessing |
| `technical_repeat` | Folder label, `single` when absent; the shared contract column CellPatcher does not yet populate |

The preprocessing CSV writes `z`; CellPatcher writes `z0/z1` one plane thick, so the join uses
`z = z0`. A multichannel patch (`image_axes` with `C`) links to one row per channel. With
`split_channels=True` each patch links to exactly one row.

## 5. Normalization methods

The four methods of the preprocessing contract §7: min-max, percentile range (default
1–99.8), z-score and robust z-score (MAD × 1.4826). Each is an offset and a scale; applying
it computes `(pixels − offset) / scale` in float64 and returns float32.

- **Group scopes** take their parameters from pooled histograms (`PooledHistogram`), never
  from averaged per-plane statistics. They are exact for exact integer histograms and
  approximate otherwise (always for float images); the output records which.
- **Image, plane and patch scopes** without preprocessing, and the patch scope always, are
  computed directly from the pixels with the same definitions: finite pixels only,
  percentiles with NumPy's `inverted_cdf` method, population standard deviation. The existing
  `percentile_normalize` uses NumPy's default (linear) method and is not reused for this.

## 6. Normalization scopes and semantics

### 6.1 Channels are always separate

Parameters are always computed per channel and applied per channel. Pooling across channels
is not offered **[D4]**.

### 6.2 Scopes

| Scope | Planes pooled (per channel) | Needs preprocessing |
|---|---|---|
| `experiment` | All planes of the experiment | Yes |
| `condition` | All planes with the same `condition` | Yes |
| `biological_repeat` | Same `(condition, biological_repeat)` | Yes |
| `technical_repeat` | Same `(condition, biological_repeat, technical_repeat)` | Yes |
| `position` | Same `position_id`; only when every file has one | Yes |
| `image` | All Z and T planes of one file | No (direct otherwise) |
| `plane` | The one plane the patch was cut from | No (direct otherwise) |
| `patch` | The patch's own pixels (§6.4) | No |

Groups are always identified by the full label path, so `bio_rep1` under two conditions is
two groups. By default the `image` and coarser scopes pool all time points; an option makes
`t` part of the group key, so each time point is normalized separately (for example to
compensate photobleaching) **[D4]**.

### 6.3 Per-group or reference parameters

Two ways to use a scope **[D5]**:

- **Per group** (default): each group is normalized with its own parameters.
- **Reference:** one set of planes chosen by label (for example the `control` condition, or
  the training split) provides the parameters, which are applied to every patch.

The notebook must explain that normalizing per condition can remove real intensity
differences between conditions, and that for deep learning, parameters computed on all data
include the test data. Reference parameters address both.

### 6.4 Patch scope

Parameters come from the patch's own pixels, before resizing:

- Padding pixels (`pad_mode="constant"`) are excluded.
- With `apply_mask=True`, only the pixels of the patch's own label are used **[D6]**.
- A patch with zero intensity spread (for example an empty background tile in random or grid
  mode) cannot be normalized. Default: the patch is dropped and counted in the run report,
  with the reason **[D8]**.

### 6.5 Background pixels in group statistics

Group statistics come from whole planes, so in sparse images they are dominated by
background. Percentile and z-score parameters then describe the background more than the
cells. This is acceptable for this milestone with a note in the notebook; masked histograms
would need a change in the preprocessing brick **[D7]**.

### 6.6 Z projections

If a stack is projected (`z_policy="max"`, `"mean"` or `"sum"`), per-plane statistics do not
describe the projected image. Group, image and plane scopes are blocked for projected images
with a clear message; the patch scope remains available **[D11]**. Selecting single planes
(`"select"`, `"first"`, `"all"`) is unaffected.

### 6.7 Order of operations

The pipeline becomes: generate regions → filter → extract → **normalize** → apply mask →
resize → export **[D9]**.

- Filters (`min_mean_intensity`, `foreground_threshold`) keep working on raw values, so
  existing thresholds keep their meaning **[D15]**.
- Normalizing before masking means `mask_fill_value` is in normalized units (default 0).
  For z-score, 0 is the mean intensity, so a `NaN` fill is offered as well.
- Normalizing before resizing avoids rounding resized integer values. Resizing is linear in
  the pixel values, so the order does not change the result apart from clipping.
- Optional clipping after normalizing, for example `(0, 1)` for min-max or percentile
  range. Default: no clipping **[D10]**.

### 6.8 Existing normalizations

To prevent double normalization **[D12]**:

- When normalization is enabled, `prepare_image_for_patching(normalize=...)` must be `None`;
  otherwise the run stops with a message.
- The torch adapters' `scale_policy="auto"` must not rescale normalized (float) patches. It
  already leaves float data unchanged; this is made explicit and tested.

## 7. Delivery

Three modes, all producing identical arrays for the same configuration **[D13]**:

1. **Normalized files:** patches exported as float32 `.tif` or `.npy`, with `patches.csv`,
   the parameters table (§8) and `run_config.json`. Float32 needs twice the disk space of
   uint16; the notebook shows an estimate before running.
2. **Raw files plus parameters:** patches exported in their source dtype with the same
   parameters table; normalization happens when loading (a NumPy loader in the core, and
   `ExportedPatchTorchDataset`). Smaller, reversible, and the method can change without
   re-patching.
3. **In memory:**
   - eager: `run_patching(..., config with extract_arrays=True)` returns normalized arrays.
     The memory needed is estimated first; above a configurable limit the run stops and
     suggests the lazy option, as in preprocessing.
   - lazy (recommended for large data): a NumPy iterator in the core that extracts and
     normalizes one patch at a time, and `PatchTorchDataset` built on it. Torch stays an
     optional extra.

## 8. Output contract changes

All additions are optional and empty when normalization is off. Each needs owner approval
under AGENTS.md §8.

**New `PatchRegion` fields / `patches.csv` columns:** `source_relative_path`,
`technical_repeat`, `normalization_method`, `normalization_scope`, `normalization_group`.

**New `normalization_parameters.csv`** next to `patches.csv`, one row per group and channel
**[D14]**:

| Column | Meaning |
|---|---|
| `normalization_group` | Group key, e.g. `control/bio_rep1` or a `patch_id` for patch scope |
| `c` | Channel index, blank when there is no channel axis |
| `method`, `offset`, `scale` | As in `NormalizationParameters` |
| `clip_min`, `clip_max` | Clipping limits, blank when not clipped |
| `source` | `histogram` or `pixels` |
| `exact` | Whether the parameters are exact or histogram approximations |
| `plane_count`, `valid_pixel_count` | What was pooled |

For patch scope this table has one row per patch and channel, which can be large; it may
instead be written as Parquet or JSON per patch **[D14]**.

**`run_config.json`** records the normalization settings plus the preprocessing CSV's path,
`csv_sha256`, `schema_version` and the preprocessing package version. Loading an older
`run_config.json` without these keys means normalization off.

**AGENTS.md §5** is updated: inputs (preprocessing output, OME-Zarr), outputs (new columns
and table), and `technical_repeat` now populated.

## 9. Configuration

A new `NormalizationConfig` in `CellPatcherConfig` (proposal; names to confirm):

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `False` | Normalization off keeps current behaviour |
| `statistics_dir` | `None` | Preprocessing output folder; required for group scopes |
| `method` | `"percentile"` | `minmax`, `percentile`, `zscore`, `robust_zscore` |
| `percentile` | `(1.0, 99.8)` | Low and high percentiles |
| `scope` | `"image"` | One of §6.2 |
| `per_time_point` | `False` | Make `t` part of the group key |
| `reference` | `None` | Label selection for reference parameters (§6.3) |
| `clip` | `None` | `(low, high)` or `None` |
| `zero_spread` | `"drop"` | Patch scope: `drop` or `error` |
| `delivery` | `"normalized_files"` | `normalized_files`, `raw_files_with_parameters`, `memory` |
| `memory_limit` | 4 GiB | Limit for eager in-memory delivery |

## 10. Validation and messages

Block the run for:

- A group scope without `statistics_dir`, or with statistics that are stale (§3.4).
- A patched plane with no histogram row, or a file not in the manifest at a group scope.
- A group with zero or non-finite spread (contract §7), or with no finite pixels.
- Group, image or plane scope on a Z-projected image.
- Normalization combined with `prepare_image_for_patching(normalize=...)`.
- `position` scope when any file lacks a `position_id`.
- Eager in-memory delivery above `memory_limit`.
- OME-Zarr input without the `zarr` dependency.

Warn and continue for:

- Patches dropped for zero spread at patch scope (count and reason).
- Approximate parameters (float images or reduced-bin histograms).
- Group statistics dominated by background (§6.5), shown once per run in the notebook.
- Per-condition scopes, with the explanation of §6.3.

## 11. Architecture and milestones

New modules, independent of widgets: `statistics.py` (load and check preprocessing outputs,
link planes), `normalization.py` (scopes, groups, parameters, apply), and image reading for
OME-Zarr and plane-by-plane TIFF in `io.py`. The pooling and parameter code is imported from
`morphobricks_preprocessing`, not copied **[D1]**.

- **Milestone 0 — Contract:** decide §13, document scopes, columns and the parameters table
  in `configuration_reference.md` and AGENTS.md §5.
- **Milestone 1 — Preprocessing outputs:** read the manifest and CSV, staleness checks,
  manifest-driven discovery, `source_relative_path` and `technical_repeat` on every patch.
- **Milestone 2 — Image reading:** OME-Zarr input and plane-by-plane TIFF reading, reusing
  the preprocessing readers; OME-Zarr copies.
- **Milestone 3 — Normalization engine:** group, image, plane and patch parameters; per
  channel; reference parameters; the pipeline step of §6.7; projection guard; double
  normalization guard.
- **Milestone 4 — Delivery:** the parameters table and `run_config.json` keys; the three
  delivery modes; normalization on load for raw files; the NumPy iterator and torch
  adapters.
- **Milestone 5 — Notebook and docs:** a normalization step in `CellPatcher.ipynb` (choose
  the preprocessing folder, method, scope, reference and clipping; table of parameters per
  group; before/after preview of a few patches with their histograms; warnings), plus user
  documentation.
- **Later:** masked histograms for cell-only group statistics (with preprocessing), reading
  only patch windows from OME-Zarr for planes above the memory limit.

## 12. Tests and acceptance criteria

Use simulated images and preprocessing outputs generated in the tests; the
`morphobricks-preprocessing` package can produce real CSVs and manifests for fixtures.

- With normalization off, all existing tests pass and outputs are unchanged.
- A patch normalized at a group scope equals `NormalizationParameters.apply()` on the raw
  crop, with parameters from `PooledHistogram` over exactly the group's planes.
- With exact integer histograms, group parameters equal direct calculations on the pooled
  pixels.
- Patch-scope parameters equal direct NumPy calculations (`inverted_cdf`), excluding padding
  and, with `apply_mask`, pixels of other labels.
- Channels never share parameters; `per_time_point` separates time points.
- Every patch's `normalization_group` and channel resolve to one parameters row, and the
  row plus `run_config.json` reproduce the normalized patch.
- Normalized files, raw files normalized on load, eager and lazy delivery give identical
  arrays.
- Stale manifests, missing rows, projections and double normalization are blocked with
  the messages of §10.
- Labels such as `NA`, `001` and `single` survive the round trip through both CSVs.

## 13. Open decisions

Each has a recommended default; replace *Pending* with the decision.

1. **[D1] Dependency on the preprocessing package.** Import `morphobricks_preprocessing` as
   an optional `cellpatcher[normalization]` extra, installed from GitHub at a pinned tag,
   or move the shared code (pooling, normalization, readers) into a small shared package?
   *Recommended:* optional extra now; a shared package only if a third brick needs it.
   *Pending.*
2. **[D2] Discovery with preprocessing.** When `statistics_dir` is given, use the manifest
   as the only source of files, labels and axes, ignoring folder walking and `image_axes`?
   *Recommended:* yes; it removes the differences between the two bricks' folder rules.
   *Pending.*
3. **[D3] Masks for OME-Zarr images.** TIFF masks in a mirrored tree, NGFF `labels/`, or
   both? *Recommended:* mirrored TIFF tree only for this milestone. *Pending.*
4. **[D4] Channels and time.** Always per channel; time pooled by default with an option per
   time point? *Recommended:* yes to both. *Pending.*
5. **[D5] Reference parameters.** Offer a reference selection in this milestone, and
   select it by label only or also by an explicit list of files (for training splits)?
   *Recommended:* by label now, by file list later. *Pending.*
6. **[D6] Patch scope with masks.** Only the label's pixels, or the whole patch?
   *Recommended:* only the label's pixels. *Pending.*
7. **[D7] Background in group statistics.** Accept with a warning now, and propose masked
   histograms to the preprocessing brick? *Recommended:* yes. *Pending.*
8. **[D8] Zero spread at patch scope.** Drop the patch, stop the run, or leave it
   unnormalized? *Recommended:* drop and report. *Pending.*
9. **[D9] Normalization step position.** Extract → normalize → mask → resize. This adds a
   step to the pipeline order fixed in AGENTS.md §2. *Recommended:* yes, and update
   AGENTS.md. *Pending.*
10. **[D10] Clipping and output dtype.** No clipping by default; float32 output only?
    *Recommended:* yes. *Pending.*
11. **[D11] Projections.** Block stored-statistics scopes for projected images, or compute
    image-scope parameters from the projection itself? *Recommended:* block, and compute
    image scope from the projection only when preprocessing is not used. *Pending.*
12. **[D12] Existing normalizations.** Keep `prepare_image_for_patching(normalize=...)` as is
    and make the combination an error, or deprecate it in favour of the new image scope?
    *Recommended:* make the combination an error now; deprecate later. *Pending.*
13. **[D13] Delivery modes.** All three of §7, with normalized files as the default?
    *Recommended:* yes; raw files plus parameters as the recommended choice for deep
    learning. *Pending.*
14. **[D14] Parameters storage.** A `normalization_parameters.csv` for all scopes, or
    columns in `patches.csv`, or JSON for patch scope? *Recommended:* the CSV for all scopes.
    *Pending.*
15. **[D15] Filters.** Keep filters on raw values? *Recommended:* yes. *Pending.*
16. **[D16] Stale statistics.** Block, or warn and continue? *Recommended:* block, with an
    explicit override for advanced users. *Pending.*
