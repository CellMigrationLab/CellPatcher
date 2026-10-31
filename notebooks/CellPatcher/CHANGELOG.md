# Changelog - CellPatcher

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - Reading patches.csv back (D86, C2)

### Added
- `cellpatcher.read_patch_table(csv_path)` reads a `patches.csv` back as `PatchRegion` objects
  with its `run_config.json`: the typed inverse of `save_metadata_csv`, for the bricks that
  parsed the table themselves (MorphoRelate, fiberfeatures). Integers from integer text only,
  labels and ids as text, `mask_fraction_<name>` and normalization parameter columns read back;
  every problem is listed with its line (`cellpatcher.PatchTableError`). A column CellPatcher
  does not write and a repeated `patch_id` are refused.

## [Unreleased] - Object bounding boxes for other bricks (D86, C1)

### Added
- `cellpatcher.object_sides(labels)` (label -> bounding-box side of every object that does not
  touch the image border), `cellpatcher.border_labels(labels)` and `cellpatcher.object_boxes`
  (every object's area and box), computed in one pass over a 2D instance mask. MorphoFeature
  and MorphoEmbed kept copies of the first two; labels are checked with the preprocessing
  brick's `check_labels` (int32, float masks refused).

### Changed
- `scan_cell_objects` and `scan_cell_patch_size` use `object_boxes` instead of one pass over
  the mask per object: identical records, faster on masks with many objects (a 2048 x 2048
  mask with 1,269 objects: 13.9 s before, 0.8 s after, same records).

## [Unreleased] - Dependencies: preprocessing v0.3.3, normalization declared (D86)

### Changed
- CellPatcher requires `morphobricks-preprocessing` `v0.3.3` and declares
  `morphobricks-normalization` `v0.3.2`, the tag preprocessing `v0.3.3` pins (owner decision
  D-h; a test checks that the two pins agree). Preprocessing `v0.3.3` refuses folder and image
  names with leading or trailing spaces, or named `NA`, `nan`, `none`, `null`, `n/a` or `<NA>`
  (any case), and names of one level that differ only in case anywhere in the experiment.
- The `preprocessing` extra is now called `zarr`: install OME-Zarr support with
  `pip install "cellpatcher[zarr]"`. The preprocessing brick has been a core dependency since
  issue #45, so normalization needs no extra.

### Deprecated
- `cellpatcher.require_preprocessing()` warns and will be removed in v0.2.0: import
  `morphobricks_preprocessing` directly.

## [Unreleased] - Image/mask pairing of the preprocessing brick (issue #45)

### Changed
- Images and masks are paired with the MorphoBricks preprocessing brick's rule
  (`morphobricks_preprocessing.find_masks`, through `cellpatcher.io.pair_masks`), so every brick
  pairs the same files: a mask is a TIFF file `<image>_<mask type>.tif` or
  `<mask type>/<condition>/.../<image>.tif` in the masks folder. The whole tree is paired at
  once.
- Notebook, **Advanced: mask type** replaces *Advanced: image ↔ mask filename matching*: enter
  the mask type to use when the masks folder holds several (`cells`, `nuclei`, ...); leave it
  empty when it holds one. The match strategy, suffix list and case options are gone.
- Masks named exactly like their image (`masks/.../img1.tif`), masks that are not TIFF files,
  and mask names that could belong to two images are refused; masks of any type
  (`img1_cells.tif`) and the `<mask type>/` folder layout now pair.
- Images are found with the preprocessing brick's `discover_experiment`
  (`cellpatcher.io.discover_images`), so every brick reads the same images under the same labels:
  TIFF files and OME-Zarr folders only. PNG, JPEG, BMP, GIF, NumPy and microscope-format
  images (`.czi`, `.nd2`, `.lif`, ...) are refused with their names (convert them to TIFF or
  OME-Zarr), never skipped; before, a `.czi` file was silently left out. A folder holding both images and subfolders
  is refused in folder runs too; so are a fourth folder level, folder names that differ only in
  case and a misused `single` folder. Empty folders are left out with a warning.
- CellPatcher requires `morphobricks-preprocessing` `v0.3.2` (core dependency and
  `preprocessing` extra; it brings `morphobricks-normalization` `v0.3.2`). Its discovery
  reports a folder of images and subfolders as an error (`tiff_outside_leaf`). `find_masks`'
  header checks are refused for the mask type used (unreadable, not 2D apart from named Z/T
  axes, another Y/X size than the image, non-integer pixels, different planes between masks of
  that type) and ignored for the other types.
- Mask files with a channel (C) axis, or saved without axis metadata, are refused; no axis is
  guessed, and *Mask axes* with a `C` is refused for mask files (it used to read a mask's other
  axes as label channels, with *Mask channel*). Save each label channel as its own mask type,
  for example `<image>_cells.tif` and `<image>_nuclei.tif`, and save a Z or T stack with its
  Z/T axes named (ImageJ hyperstack or OME-TIFF).

## [Unreleased] - Channel selection for single-channel patches (follow-up C5)

### Added
- `extraction.split_channels` also accepts a list of channel indices (for example `[1]`): one
  single-channel patch and `patches.csv` row per selected channel of each region, in channel
  order. `True` (every channel) and `False` (one multichannel patch) are unchanged. An index
  the images do not have stops the run before any patch is made or written, naming the image's
  channel count. A selected channel keeps its normalization parameters (only its
  `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>` pair is filled).
- Notebook, **Channels**: with **Split channels**, one checkbox per channel (`C0`, `C1`, ...,
  with the channel names an OME-TIFF or OME-Zarr header records). All ticked gives every
  channel, as before; ticking some gives only those.
- `cellpatcher.io.channel_names(path)`: the channel names an OME-TIFF or OME-Zarr image
  records, read from its header (`()` when unknown).

## [Unreleased] - Notebook: image type and pooling (follow-up C4)

### Added
- Normalization step: **Pool conditions** ("Every pixel counts once" or "Every condition weighs
  the same", `normalization.pooling`), shown only for the biological-repeat scope.
- Normalization step: **Image type** (`normalization.image_type`, fluorescence or RGB colour).
  When the preprocessing `run_manifest.json` records the image type, it is set from it and
  locked; otherwise you choose it. For RGB only the percentile method is offered, and
  **Split channels** is refused.
- `cellpatcher.statistics.recorded_image_type(plane_histograms, manifest=None)`: the image type
  a preprocessing manifest records (None without one), read without loading the table.

## [Unreleased] - RGB images (follow-up C3)

### Added
- `normalization.image_type`: `"fluorescence"` (default, unchanged behaviour) or `"rgb"`
  (normalization brick decisions N33, N34). RGB images are normalized with StarDist's
  percentile method, one offset and scale from the R, G and B pixels together, at every scope;
  the result equals `csbdeep.utils.normalize(x, 1, 99.8, axis=(0, 1, 2))` at the patch scope.
  `patches.csv` writes the shared value for `C0`, `C1` and `C2`, and the method as
  `rgb percentile 1-99.8`. An RGB patch without spread becomes zeros and is kept.
- With a preprocessing manifest that records the image type, a different
  `normalization.image_type` stops the run before any image is read (`image_type_differs`).
- `"rgb"` with a method other than `"percentile"`, with `extraction.split_channels=True`, or
  with images that are not three 8- or 16-bit channels is refused.

## [Unreleased] - preprocessing v0.3.0: tables of schema version 4

### Changed
- The `preprocessing` extra installs `morphobricks-preprocessing` at its tag `v0.3.0`, which
  brings `morphobricks-normalization` `v0.3.1` (MorphoFeature decision D79: one preprocessing
  tag per brick).
- `plane_histograms.csv` schema version 4 is read: each channel also has its underflow and
  overflow, which the preprocessing brick writes with its robust histogram range (its
  follow-up P6). The normalization brick's pooling and fitting use them, so a hot pixel no
  longer squeezes a float plane into a few bins when the table was written with that range.
  Tables of schema version 3 or earlier are refused, asking to run the preprocessing brick
  again; normalization parameters from a version-3 table and a version-4 table of the default
  range are the same.

## [Unreleased] - BREAKING: three normalization scopes (follow-up C2)

### Changed (breaking)
- **Scopes.** `normalization.scope` is `"biological_repeat"`, `"image"` (default) or `"patch"`
  (normalization brick decisions N29, N30). `"condition"` and `"technical_repeat"` are refused,
  with a message naming the three scopes; raw patches exported with them still normalize on
  load.
- **`biological_repeat`** now pools one repeat group across all conditions, technical repeats
  merged, as paired in the preprocessing brick's `experiment_design.csv`, instead of one group
  per condition and repeat. It keeps the differences between conditions and removes those
  between repeats. `normalization_group` is the repeat group label (was
  `<condition>/<biological_repeat>`).

  | Images | Old groups | New group |
  |---|---|---|
  | `control/2026_01_01/*`, `treated/2026_01_01/*` | `control/2026_01_01`, `treated/2026_01_01` | `2026_01_01` |

### Added
- `normalization.experiment_design` (path of `experiment_design.csv`; default: next to
  `plane_histograms.csv`). With `biological_repeat`, a missing file, an unconfirmed repeat
  group, a file changed since preprocessing or an image not in the file stops the run before
  any image is read.
- `normalization.pooling`: `"pixels"` (default) or `"conditions"` (each condition weighs the
  same), for `biological_repeat` (N32).
- `run_config.json`: `experiment_design_file` and `experiment_design_sha256` in
  `normalization_inputs`, and `normalization_settings` for every scope.
- The normalization brick's warnings: what the `biological_repeat` scope keeps and removes,
  and repeat groups without images from some condition.

## [Unreleased] - BREAKING: per-channel normalization columns (follow-up C1)

### Changed (breaking)
- **`patches.csv` parameter columns.** `normalization_offset` and `normalization_scale`
  (`;`-separated, one value per channel) are replaced by one `Normalization_Offset_C<i>` and
  `Normalization_Scale_C<i>` pair per channel, right after `normalization_exact` and grouped
  by channel (normalization brick decisions N41, N42). A split-channel patch fills only its
  own channel's pair. Values are unchanged.

  | Old | New |
  |---|---|
  | `normalization_offset` = `1.5;2.0` | `Normalization_Offset_C0` = `1.5`, `Normalization_Offset_C1` = `2.0` |
  | `normalization_scale` = `3.0;4.0` | `Normalization_Scale_C0` = `3.0`, `Normalization_Scale_C1` = `4.0` |
  | split channel 1: `normalization_offset` = `2.0` | `Normalization_Offset_C1` = `2.0`, `C0` pair blank |

- **`PatchRegion.normalization_parameters`** holds the normalization brick's `Parameters`
  (`offset`, `scale`, `channels`). `PatchRegion.normalization_offset`/`normalization_scale`
  are deprecated read-only properties giving the old `;`-separated text.
- With normalization, images with different numbers of channels are refused (N43).

### Added
- `run_config.json` records the brick's `NormalizationSettings.to_dict()` as
  `normalization_settings` for the image and patch scopes.

### Unchanged
- Parameters and normalized arrays: fitting now goes through the brick's `fit` and
  `Parameters.apply`, with the same numbers. `normalization.clip` is still applied by
  CellPatcher before masking and resizing.
- Raw patches exported earlier still normalize on load: their `;`-separated columns are read
  (N47).

## [Unreleased] - BREAKING: run-independent patch identity and instance-label masks

### Changed: preprocessing tables of schema version 3
- The `preprocessing` extra installs `morphobricks-preprocessing` at its tag `v0.2.0`, which
  brings `morphobricks-normalization` `v0.2.0`; there is no separate normalization pin
  (MorphoFeature decision D79: every brick pins the same preprocessing tag).
- `plane_histograms.csv` schema version 3 is read; schema version 2 tables are refused
  ("run the preprocessing brick again"). A row is an XYC plane holding every channel, so
  `plane_id` is `<path>::z=<z>,t=<t>` (no `c`), and a channel is found inside its row.
- `check_planes` returns `cellpatcher.statistics.ChannelRecord` (one channel of a row: its
  `histogram`, labels, `z`, `t`, `c`, `group_key`) instead of a schema-2 `PlaneRecord`.
  Plane messages name `<plane_id> c=<c>`. Normalization parameters are unchanged: each
  channel's histogram is the one schema version 2 stored.

Owner decisions D44, D45 and D47 (2026-10-26), and the owner's follow-up answers of the same day
(unnamed images refused, random mode seeded per FOV, multi-series TIFFs refused). Details: "Patch identity" in
`docs/configuration_reference.md`; code in `src/cellpatcher/ids.py`.

### Changed (breaking)
- **`patch_id` format (D44).** Built from the FOV identity and the patch's own coordinates,
  never from run order, file-list position or the number of files:
  `position_id "/" [object_set "/"] ["t"NNNN "/"] ["z"NNN "/"] ["c"N "/"] token`.
  Exported file names follow (`sanitize_filename(patch_id)`, `/` becomes `_`).

  | Case | Old `patch_id` | New `patch_id` |
  |---|---|---|
  | cell | `image_0000_cell_00042` | `<position_id>/<object_set>/label42` |
  | grid | `image_0000_grid_00003` | `<position_id>/grid-y<y0>-x<x0>-h<h>-w<w>` |
  | random | `image_0000_random_00007` | `<position_id>/random-y<y0>-x<x0>-h<h>-w<w>` (`-n<k>` for a repeated draw) |
  | whole_fov | `image_0000_whole_fov_0` | `<position_id>/whole_fov` |
  | split planes | `..._t001_z002` suffix | `t0001/z002` levels before the token |
  | split channels | `..._c1` suffix | `c1` level before the token |
  | strategy classes alone | `<image_id>_cell_00042`, ... | the token only (`label42`, ...) |

  Example: `image_0002_cell_00007_t003_c1` (third file of a run) becomes
  `ctrl/2026_01_01/B02/fov1/nuclei/t0003/c1/label7` wherever `fov1.tif` is in the run.
- **`position_id` always filled (D45).** Supplied via `image_metadata`, else
  `<condition>/<biological_repeat>/<technical_repeat>/<file stem>` (one file is one position).
  Previously only filled with `split_channels` (where an image without file name got
  `image_<index>`), and empty otherwise, also in `images.csv`.
- **Unnamed in-memory images are refused.** `run_patching` on an image with neither
  `image_metadata[i]["position_id"]` nor `image_metadata[i]["image_file"]` raises `ValueError`
  naming the image's place in the run. Before, any array was accepted. Migration: pass
  `run_patching(images=..., image_metadata=[{"position_id": "<fov>"}, ...])`, one entry per image
  (plane metadata from the preprocessing helpers can be merged: `{**p.metadata, "position_id": ...}`).
- **Random mode is seeded per FOV.** Image *k* of a run (plane *k* with split planes) used to be
  sampled with `random.seed + k`, so its rectangles changed with the other files in the run. Each
  image (plane) is now sampled with a seed derived from `random.seed` and its identity: SHA-256 of
  `<position_id>|t=<t>|z=<z>|c=<c>` combined with the seed in `numpy.random.SeedSequence`. Same
  seed, same rectangles for the same image wherever it is in the run; the rectangles drawn for a
  given seed differ from those of earlier versions.
- **`CellConfig.binary_mask_policy` default `"connected_components"` → `"single_object"` (D47).**
  Integer masks are always instance labels used as given (a 0/1 integer mask is now one object,
  label 1, also with `"connected_components"`). A boolean (or 0.0/1.0 float) mask is refused
  unless `"connected_components"` is chosen, which warns and is recorded in `run_config.json`.
  Labels above the int32 maximum are refused instead of silently wrapping.
- Runs refuse duplicate `patch_id` values (e.g. two in-memory images with the same pixels and no
  `position_id`, or two images given one `position_id` without plane indices).

### Added

- `NormalizationConfig.clip` (default `None`): limits applied right after normalizing, before masking and resizing, in every delivery path (requested by MorphoEmbed so per-patch inputs are clipped before resizing, as its other scopes are).
- `object_set` `PatchRegion` field / last `patches.csv` column (cell mode), and
  `CommonConfig.object_set`. Precedence: `image_metadata["object_set"]`, `common.object_set`,
  the mask's own name in folder/tree runs (OME-NGFF label name, or the mask stem after the image
  stem, `img_cells.tif` → `cells`), else `"mask"`.

### Fixed
- A multi-image store (OME-Zarr plate or multi-position store, multi-series TIFF) read plane by
  plane was silently patched from its first image only; it is now refused, naming the store and
  its image count. A multi-series TIFF read whole (`z_policy`/`t_policy` `"keep"`, projections,
  or without the `preprocessing` extra; also any TIFF read by `read_array`, masks included) is
  refused the same way, naming the file and its series count.
- `extraction.apply_mask` with connected-component labels of a binary mask blanked every object
  but label 1; that combination is now refused.

---
## [0.0.6] - 2026-10-21

### Changed
- Version updated from 0.0.5 to 0.0.6

**Commit Details:**
- Hash: [`968c9670`](https://github.com/CellMigrationLab/CellPatcher/commit/968c96705949e3c8f753c4338693391ecb734a5f)
- Author: Iván Hidalgo
- Date: 2026-10-21 15:14:47 +0300
- Message: Merge pull request #24 from CellMigrationLab/feature/m5-notebook-normalization

Normalization, plane and OME-Zarr options in the notebook; user guide (Milestone 5)

---
## [0.0.5] - 2026-05-17

### Changed
- Version updated from 0.0.0 to 0.0.5

**Commit Details:**
- Hash: [`a28ef863`](https://github.com/guijacquemet/CellPatcher/commit/a28ef8637c516902a05d35d4ae2cbfea04e2ac00)
- Author: guijacquemet
- Date: 2026-05-17 20:16:11 +0300
- Message: Merge pull request #24 from guijacquemet/codex/fix-torch-testing-notebook-issues

Improve Colab Torch validation notebook: token installs, PYTHONPATH for tests, and robust skip handling

---
## [0.0.5] - 2026-05-17

### Changed
- Version updated from 0.0.0 to 0.0.5

**Commit Details:**
- Hash: [`a28ef863`](https://github.com/guijacquemet/CellPatcher/commit/a28ef8637c516902a05d35d4ae2cbfea04e2ac00)
- Author: guijacquemet
- Date: 2026-05-17 20:16:11 +0300
- Message: Merge pull request #24 from guijacquemet/codex/fix-torch-testing-notebook-issues

Improve Colab Torch validation notebook: token installs, PYTHONPATH for tests, and robust skip handling

---
## [0.0.4] - 2026-05-17

### Changed
- Version updated from 0.0.0 to 0.0.4

**Commit Details:**
- Hash: [`a28ef863`](https://github.com/guijacquemet/CellPatcher/commit/a28ef8637c516902a05d35d4ae2cbfea04e2ac00)
- Author: guijacquemet
- Date: 2026-05-17 20:16:11 +0300
- Message: Merge pull request #24 from guijacquemet/codex/fix-torch-testing-notebook-issues

Improve Colab Torch validation notebook: token installs, PYTHONPATH for tests, and robust skip handling

---
## [0.0.5] - 2026-05-17

### Changed
- Version updated from 0.0.0 to 0.0.5

**Commit Details:**
- Hash: [`a28ef863`](https://github.com/guijacquemet/CellPatcher/commit/a28ef8637c516902a05d35d4ae2cbfea04e2ac00)
- Author: guijacquemet
- Date: 2026-05-17 20:16:11 +0300
- Message: Merge pull request #24 from guijacquemet/codex/fix-torch-testing-notebook-issues

Improve Colab Torch validation notebook: token installs, PYTHONPATH for tests, and robust skip handling

---
## [0.0.1] - 2026-04-09

### Changed
- Version updated from 0.0.0 to 0.0.1

**Commit Details:**
- Hash: [`bd1c5f8f`](https://github.com/CellMigrationLab/CellPatcher/commit/bd1c5f8f8b82daf4c1218ab7c0242decd0e0b278)
- Author: Iván Hidalgo
- Date: 2026-04-09 19:21:59 +0300
- Message: Merge pull request #2 from CellMigrationLab/submission/IvanHCenalmor-202604091621

Upload/Update notebook CellPatcher.ipynb

---
