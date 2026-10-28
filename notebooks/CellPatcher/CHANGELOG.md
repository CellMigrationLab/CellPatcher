# Changelog - CellPatcher

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
