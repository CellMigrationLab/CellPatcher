# Changelog - CellPatcher

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-11-10

### Changed
- Version updated from 0.2.0 to 0.3.0

**Commit Details:**
- Hash: [`d490fd22`](https://github.com/CellMigrationLab/CellPatcher/commit/d490fd2228b39f0d1bc60bb388e38468ff29853b)
- Author: guijacquemet
- Date: 2026-11-10 17:39:55 +0200
- Message: Merge pull request #98 from CellMigrationLab/claude/release-0.3.0

Release v0.3.0

---
## [0.2.0rc2] - 2026-11-03

### Changed
- Version updated from 0.2.0rc1 to 0.2.0rc2

**Commit Details:**
- Hash: [`af70da9c`](https://github.com/CellMigrationLab/CellPatcher/commit/af70da9c99f5019bc258045969610e52f05ac3e7)
- Author: guijacquemet
- Date: 2026-11-03 07:29:01 +0200
- Message: Merge pull request #75 from CellMigrationLab/claude/pin-preprocessing-rc4

Pin preprocessing v0.4.0rc4; prepare v0.2.0rc2

---
## [0.4.0] - Preprocessing v0.5.1, the resolver, one shared mask labeller (Stage 2 W2: CP-4, CP-3, CP-2)

The v0.4.0 release. It pins preprocessing `v0.5.1` (core and `zarr` extra; was `v0.5.0`) and
normalization `v0.4.0` (was `v0.3.4`), the tag preprocessing `v0.5.1` pins (owner decision D-h; pip
refuses two different direct URLs for one package). `__version__`, `setup.py` and every notebook
(`RELEASE_TAG`, `current_version`, `notebook_latest_versions.yaml`) are `0.4.0`. It carries #99
(CP-4), #100 (CP-3) and #101 (CP-2). Owner policy of 2026-11-10 (Spec ADR 0002 AR10): no backward
compatibility for the removed behaviour below; old settings are refused, not converted. Instance
masks give the same patches as v0.3.0: on the Spec fixtures `v03 v13 v30 v33` (grid, no/image/
biological-repeat normalization) and `v07` (cell mode, instance masks) every exported patch file,
`images.csv` and `positions.csv` is byte-identical to v0.3.0, and `patches.csv` only gains the
`channel_id` column.

### Breaking
- **A binary mask is an object set only with an explicit `cell.binary_mask_policy`** (Spec ADR 0002
  AR5, AR11 (4); P27). There is no default: `binary_mask_policy` is `None` unless set to
  `"connected_components"`. In a file run, cell mode asks the run's resolver for its object set by
  mask ID (`MaskRequest(<mask ID>, "objects", cell.binary_mask_policy)`); a mask the description
  declares `binary` is refused without the policy (`A_MASK_NOT_INSTANCES`) before any pixel is read.
  Before, such a 0/1 mask was silently one object. A policy on an `instances` mask is refused
  (`ValueError` naming `cell.binary_mask_policy`); an `instances` mask is used as stored, never
  relabelled (D47: an `instances` mask holding 0 and 1 is one object). In memory, a boolean mask is
  the binary one and follows the same rule.
- **In memory, an integer mask with `cell.binary_mask_policy` is refused** (#103). Only a boolean
  mask is binary in memory; an integer mask is instances, used as stored (D47: a 0/1 integer mask is
  one object), so the policy has nothing to convert. Before, `run_patching` (`mode="cell"`), the
  object scan (`scan_cell_objects`, `scan_cell_patch_size`) and `CellSegmentationStrategy` called
  directly silently ignored the policy and gave one object; they now raise a `ValueError` naming the
  image and the fix (pass `mask != 0` to label it, or leave the policy `None`), before any output is
  written. File runs are unchanged (the resolver labels a `binary` mask and refuses the policy on an
  `instances` one).
- **`binary_mask_policy="single_object"` is removed.** It was the implicit default and only meant
  "refuse a boolean mask", which `None` now does. It is refused by name, in code and in a saved
  `run_config.json`.
- **8-connected labelling by default** (owner decision of 2026-11-11). The one shared labeller,
  `cellpatcher.label_objects(mask, policy, connectivity)`, labels a binary mask's connected
  components; `cell.binary_mask_connectivity` is `2` (8-connected) by default and `1`
  (4-connected) on request; `cellpatcher.BINARY_MASK_CONNECTIVITY` is `2`. A boolean mask labelled
  in v0.3.0 (4-connected) can give fewer objects now where objects touch only at a corner.
- **Cell-mode resolver refusals** stop a file run before any read, each with its code:
  `A_MASK_NOT_INSTANCES`, `A_MASK_UNCHECKED` (masks the mask step has not checked, "k of n files")
  and `A_SCOPE_INCOMPLETE` (a mask some image lacks, "present in k of n"; an undeclared mask ID is
  0 of n). An image without its label image is now `A_SCOPE_INCOMPLETE` (was CellPatcher's own
  "recorded no mask").
- **The declared image type wins.** Each position's image type is the one its unit declares in the
  description (`Fov.image_type`: `rgb`, or `channels`, normalized per channel as `"fluorescence"`).
  A `normalization.image_type` that contradicts it is refused naming both (`image_type_differs`), at
  every scope of a file run (before: the CSV scopes only, and a run that recorded no type let the
  setting decide). A run that declares mixed types is refused (`image_types_mixed`).
  `RunSummary.image_type` is a `str`, never `None`.
- **Preprocessing v0.5.1 reads manifest version 4 only.** An older run is refused
  (`preprocessing_run_refused`, no longer a traceback); a run without a published description is
  refused by `RunInputs.open` itself (`A_DESCRIPTION_REQUIRED`, as before).
- **`patches.csv`** has a new last column, `channel_id` (after `square_index`; every other column
  keeps its place). In the cell runs, the dynamic `mask_fraction_*` columns come after it.

### Added
- **Channels by canonical ID** (CP-3, G2-id): `extraction.split_channels` may name channel IDs of
  the run's description (for example `["DNA"]`). A file run resolves them per image through the
  resolver, so one ID may be a different C index in another image; an unknown ID is refused
  (`A_SCOPE_INCOMPLETE`, nothing selected). A list mixing IDs and indices, an empty or repeated ID,
  and an ID list for in-memory images are refused. `patch_id` is unchanged (it keeps `c` and is
  never parsed).
- `PatchRegion.channel_id` and the `channel_id` column: the canonical ID of the patch's `c` in its
  own image (`Fov.channel_ids`), for index selections and `True` too; empty for a patch without `c`.
- `matching.mask_id`: the description's mask ID of the cell-mode object set. When `None`, the
  description's one mask of `matching.mask_type` (or of the run's only type) is used; given
  together, the two must name the same mask.
- `run_config.json` records the mask ID read (`matching.mask_id`), `cell.binary_mask_policy` and
  `cell.binary_mask_connectivity`. Nothing is written to `morphobricks.toml`.
- `cellpatcher.label_objects`, `CONNECTIVITIES` and `BINARY_MASK_CONNECTIVITY`: the one labeller
  every route calls for a recorded `binary_mask_policy` (MorphoFeature MF-3, fiberfeatures FF-1,
  MorphoRelate MR-2). Its hand-written goldens are `tests/fixtures/shared_labeller/golden.json`.

### Changed
- Dependencies: preprocessing `v0.5.1` (tag `8726cb4`; was `v0.5.0`), normalization `v0.4.0` (was
  `v0.3.4`), both checked by `tests/test_dependency_pins.py` (release tags only).
- MorphoBricks-Spec `b4b1008` is vendored in `tests/spec_fixtures` (fixtures unchanged).
- The test helpers describe each test's images before the run, from preprocessing's preflight.
- Docs: RGB support stated (P24), the image-type rule, the mask rule, channel IDs.

### Migration
- Re-run preprocessing `v0.5.1` on the experiment (describe and publish it in the preprocessing
  notebook first) and give CellPatcher that run's output folder; a run of preprocessing `v0.5.0` or
  earlier is refused (`preprocessing_run_refused`).
- Remove `binary_mask_policy="single_object"` from code and from saved `run_config.json` files
  (`None` is its meaning). For a mask declared `binary` that should be split into objects, set
  `cell.binary_mask_policy="connected_components"` (the notebook's "The mask is binary: label its
  connected components (8-connected)"); set `cell.binary_mask_connectivity=1` to keep v0.3.0's
  4-connected labels. For an `instances` mask, leave the policy `None`.
- Drop any `normalization.image_type` that contradicts the description's declared unit type, or fix
  the description.
- Code that reads `patches.csv` by position (not by column name) must allow the new last column,
  `channel_id`.
- Bricks that pin CellPatcher pin `v0.4.0` together with preprocessing `v0.5.1` and normalization
  `v0.4.0`; they call `cellpatcher.label_objects(mask, selection.conversion,
  BINARY_MASK_CONNECTIVITY)` for a `binary` `MaskSelection` with a conversion, record `(mask_id,
  conversion, connectivity)`, and compare with the golden.

---
## [0.3.0] - Preprocessing v0.5.0, every run described (Stage 2 CP-1)

The v0.3.0 release. It pins preprocessing `v0.5.0` (core and `zarr` extra; was `v0.4.0`) and keeps
normalization at `v0.3.4`, the tag preprocessing `v0.5.0` pins (owner decision D-h; pip refuses two
different direct URLs for one package). `__version__`, `setup.py` and every notebook (`RELEASE_TAG`,
`current_version`, `notebook_latest_versions.yaml`) are `0.3.0`. Owner policy of 2026-11-10: no
backward compatibility for the removed behaviour below. Everything under `0.2.0` applies as before.

### Breaking
- **Every run needs a published description.** A preprocessing run whose manifest records no
  published experiment description is refused when it is opened, with code `A_DESCRIPTION_REQUIRED`
  (Spec ADR 0002 AR10), on every route: tree, folder and in-memory runs, the CSV normalization
  scopes, `validate_image_and_mask_folders`, the notebooks' check. Nothing is written. Preprocessing
  `v0.5.0` still writes an undescribed run (with a warning); CellPatcher refuses it.
- A described run's image without a `repeat_group`, `condition_id` or `technical_repeat_id` is refused
  (`described_run_missing_id`); `positions.csv` carries the three IDs as the description gives them.
- **`NormalizationConfig.experiment_design` is removed**, with `experiment_design_path` and
  `load_experiment_design`. The `biological_repeat` scope pools by each image's `Fov.repeat_group`,
  read through `RunInputs`; the derived `experiment_design.csv` is never read back. A saved
  `run_config.json` that names `normalization.experiment_design` is refused (a `null` value loads).
- Refusals renamed or added: `image_not_in_design` is now `image_not_in_description`;
  `missing_repeat_group` and `repeat_group_conflict` are new; `missing_experiment_design` and the
  design checks no longer exist.
- **`run_config.json`**: `normalization_inputs` records `description_sha256` (the SHA-256 of the
  published description) instead of `experiment_design_file` and `experiment_design_sha256`, and
  `manifest_version` is `"4"` for a run of preprocessing `v0.5.0`.
- The CLI-only route (`check`, `design`, `run`) is no longer offered in the notebooks: the
  description is published in the preprocessing notebook.

### Changed
- Dependencies: preprocessing `v0.5.0` (tag `3f10545`; was `v0.4.0`), normalization `v0.3.4`
  (unchanged, checked by `tests/test_dependency_pins.py`, which now also requires the preprocessing
  pins to be a release tag, not a commit).
- The notebooks' step 1 and the troubleshooting name preprocessing notebook version 0.5, the
  described run and `A_DESCRIPTION_REQUIRED`.
- The docs treat T and Z separately (#94): `z_policy="all"` no longer reads as patching every Z
  plane; Z>1 is refused under every `z_policy`. No behaviour changed.

### Tests
- MorphoBricks-Spec `6d22761` is vendored in `tests/spec_fixtures` (`tests/test_spec_pin.py`, the
  weekly spec-freshness workflow and a `vendor.py --check` CI job). The test helpers describe their
  own images (`describe_experiment`; `preprocess`, `run_described` and `preprocessed_tree` take
  `representations`, for example `{"nuclei": "binary"}`; a binary mask is no longer described as
  instances by default).

### Migration
- Describe the experiment and publish its description in the preprocessing notebook (v0.5) first,
  run preprocessing again, and give CellPatcher that run's output folder. A run made before
  (undescribed, or by preprocessing `v0.4`) is refused: `A_DESCRIPTION_REQUIRED`, or
  `preprocessing_run_refused` for a manifest `RunInputs` no longer reads.
- Remove `normalization.experiment_design` from code and from saved `run_config.json` files; the
  `biological_repeat` scope needs no file, only the description's repeat groups.
- Code that matched `image_not_in_design` or `missing_experiment_design` matches the new codes.
- Bricks that pin CellPatcher pin `v0.3.0` together with preprocessing `v0.5.0` and normalization
  `v0.3.4`.

---
## [0.2.0] - Preprocessing v0.4.0, normalization v0.3.4, strict settings

The v0.2.0 release. It pins preprocessing `v0.4.0` and normalization `v0.3.4`, the tag preprocessing
`v0.4.0` pins (owner decision D-h), and carries the changes since `v0.2.0rc3` described below.
Everything described under `0.2.0rc1` to `0.2.0rc3` applies to this release.

### Summary
- Dependencies: preprocessing `v0.4.0` (core and `zarr` extra; was `v0.4.0rc5`), normalization
  `v0.3.4` (was `v0.3.3`). `__version__`, `setup.py` and every notebook (`RELEASE_TAG`,
  `current_version`, `notebook_latest_versions.yaml`) are `0.2.0`.
- **2-D planes only.** Preprocessing `v0.4.0` refuses images with Z > 1 (Z = 1 is allowed; T > 1
  needs masks with the same T). CellPatcher reads runs only through that brick, so a Z-stack must be
  projected by preprocessing first. The tests that built fresh Z > 1 runs now use Z = 1 or T > 1
  equivalents, or assert the refusal (#83; removing the dead Z code is post-release work).
- **Strict settings (#82, #84).** No scientific setting is coerced any more: `True`/`False`, text,
  NaN and infinity are refused where a number or a flag is expected, for every section of the
  configuration (`mode`, common, cell, random, whole_fov, grid, filters, export, extraction,
  matching and normalization), at construction and again where a run uses the setting
  (`run_patching`, `iter_patch_arrays`, `scan_cell_objects`, and the folder and tree routes before
  any file is read). Integer settings accept Python and NumPy integers, never floats such as `16.0`.
  `min_mask_fraction` follows the same rule (#80). `cell.mask_channel` is `None` or an integer >= 0
  (`True`, `"1"` and `1.9` used to select channel 1).
- `NormalizationConfig.percentile` and `clip` keep the numbers as given; they are no longer cast to
  `float`. Some error messages changed.
- The fitted cell `side` is an integer (checked; preprocessing and the strict settings refuse `64.0`).

### Migration
- Code or a saved `run_config.json` that passed a float (`patch_size=64.0`), a bool or text for a
  numeric setting now raises `ValueError` naming the setting; pass an integer or a number.
- Experiments with Z > 1 must be projected (or reduced to Z = 1) by preprocessing `v0.4.0` before
  CellPatcher reads them.
- Bricks that pin CellPatcher pin `v0.2.0` together with preprocessing `v0.4.0` and normalization
  `v0.3.4`: pip refuses two different direct URLs for one package.

---
## [0.3.0] - The docs treat T and Z separately: z_policy="all" no longer reads as patching every Z plane (#94)

#### Fixed
- `docs/preprocessing.md`, the configuration reference, the normalization docs, the README, the notebook
  and the `prepare_labels_for_cell_patching` docstring still said `z_policy="all"` patches every plane,
  and the docstring recommended `select`, `first` or `all` for a Z stack. Every route refuses Z>1 since
  #83. The text now separates T (`all`, `first`, `select` work) from Z (Z=1 passes; Z>1 is refused under
  every `z_policy`; project or select first with `convert --project` in morphobricks-preprocessing). The
  cell-mode mask message adds the same note for a Z mask axis. No behaviour changed. Test:
  `tests/test_docs_z_policy.py` fails on a paragraph that pairs `z_policy` with `"all"` without saying
  that Z>1 is refused.

---
## [Unreleased] - A sampling mask filters random draws only when it excludes them (matrix finding F8, for 0.2.0)

The cross-brick matrix run of 2026-11-03 (finding F8) found that a random-mode run given a mask
with `min_mask_fraction=0` cut other rectangles than the same run without one: at 0 every draw
without a mask pixel was rejected (the "any overlap" rule), and each rejected draw shifted the
draws after it. Owner decision: a mask with no fraction keeps the any-overlap rule (today's
results); an explicit 0 means the mask removes nothing. MorphoFeature cuts random sets without a mask, so its `patch_id`s matched a
CellPatcher run only when that run had no mask either.

### Changed
- **`RandomConfig.min_mask_fraction` defaults to `None` (was `0.0`).** `None` is the any-overlap
  rule (a draw needs one mask pixel); an explicit `0` is a mask that excludes nothing.
- **Random rectangles depend only on the seed, the FOV key and plane (`random.seed_scope`), the
  plane's shape, the patch size and a mask filter that excludes draws.** The draw sequence is
  documented in `cellpatcher.strategies.random`: one stream, `numpy.random.default_rng(seed)` with
  the image's seed (`ids.random_seed`, unchanged); every draw takes `y0` and then `x0`, whatever
  the mask. At an explicit `min_mask_fraction=0` a mask excludes nothing: the first `n_patches` draws are
  the patches, the same with and without a mask (the mask is still checked: 2-D, the plane's
  shape). At a fraction above 0 the patches are the stream's draws whose share of mask pixels
  reaches it, in order; a rejected draw is still consumed. Only then can a plane run out of
  attempts (`SamplingExhaustedError`, as before).
- A mask given with an explicit `min_mask_fraction=0` gives a `UserWarning` that it excludes no
  rectangle.

### Unchanged (checked bit for bit against `main` 40a68b1)
- Every run without a mask: the same `patches.csv`, `images.csv`, `run_config.json` and patch
  files, in memory and in file runs, for every seed scope and plane policy.
- **Every run with a mask and no `min_mask_fraction` (the default, the notebooks' "Sample only
  inside the masks" included)**: the same rectangles and `patch_id`s as before.
- Every run with a mask and `min_mask_fraction > 0`: a draw reaching a positive fraction always
  had a mask pixel, so the former rule rejected nothing more.

### Migration notes
- **`patch_id`s and rectangles change for random runs given a mask and an explicit
  `min_mask_fraction=0`** (`run_patching(..., masks=...)`, a random file run with a masks folder or
  `matching.mask_type`, a MorphoFeature random set with a `sampling_mask` whose fraction is passed
  as 0). Their rectangles are now
  those of the run without the mask, and may lie outside it. Re-run them; patches exported before
  keep their ids, and code that joins patches across the two versions must not mix them.
- **To keep the former rule**, leave `min_mask_fraction` unset (or set `1 / patch_size**2`): it keeps
  exactly the draws with at least one mask pixel and gives the former rectangles and ids bit for
  bit (`tests/test_random_masks.py`). Any other positive fraction filters as before.
- An empty mask at an explicit 0 no longer raises `SamplingExhaustedError`; it filters nothing.
- A random file run given a masks folder with two mask types still needs `matching.mask_type`:
  the mask is read (its `mask_file` is recorded) even when it excludes nothing.

---
## [0.2.0rc3] - Preprocessing v0.4.0rc5, normalization v0.3.3

This release candidate pins preprocessing `v0.4.0rc5` and normalization `v0.3.3` (#78) and carries
the audit round 2 fixes (#77) and the cross-brick test's Z/T mask message (#76), described below.
As for the earlier release candidates, no installer is built for it: install it with pip or on
Google Colab (`docs/developer_notes.md`).

### Summary
- **`patch_id` changes** for 2-D images in mixed runs (a `YX` image beside `ZYX` images loses
  `z000`) and for OME-Zarr images with T or Z of length 1 under `t_policy`/`z_policy` `"all"`,
  `"first"` and `"select"` (they gain `t0000`/`z000`; under `"keep"` they stay out). Ids of runs
  whose images share their roles are unchanged (#77, A-M1).
- `run_config.json`'s `preprocessing.layout` records each image's own layout (`images`) when the
  layouts of a run differ (#77, A-M1).
- `discover_condition_repeat_folders` is deprecated (#77, A-m4).
- The design of an unfinished but valid preprocessing publication is read (#77, A-m2); a cell-mode
  run with Z or T masks names `z_policy`/`t_policy` (#76, F7).
- Dependencies: preprocessing `v0.4.0rc5`, normalization `v0.3.3` (#78).

### Changed (#78)
- CellPatcher requires `morphobricks-preprocessing` `v0.4.0rc5` (core and `zarr` extra; was
  `v0.4.0rc4`) and `morphobricks-normalization` `v0.3.3` (was `v0.3.2`), the tag preprocessing
  `v0.4.0rc5` pins (owner decision D-h). `__version__` and `setup.py` are `0.2.0rc3` (`main` after
  rc2 still reported `0.2.0rc2`), and every notebook installs `v0.2.0rc3` (`RELEASE_TAG`,
  `current_version`, `notebook_latest_versions.yaml`).
- **A `patches.csv` row is read at the scope its run recorded.** Normalization `v0.3.3`
  (its #37) refuses parameter columns whose `normalization_scope` differs from the settings'
  scope; CellPatcher read every row with the default `"image"` scope, so rows of the
  `biological_repeat` and `patch` scopes could no longer be read back (`read_patch_table`,
  `iter_exported_patches`, raw files normalized on load). It now passes the scope of the row's
  `run_config.json` (`"image"` when that names none). Rows of earlier tables, with `;`-separated
  `normalization_offset`/`normalization_scale`, which CellPatcher converts itself, are checked the same way.
- **Raw patches of the removed `condition` and `technical_repeat` scopes still normalize on
  load.** Normalization `v0.3.3` refuses these scopes as unknown. As CellPatcher's policy has
  been since they were removed (contract §7, normalization brick N29), their rows are read from
  their own offset and scale columns: with the brick's column reader, as image-scope columns,
  keeping their recorded scope, which must be the run's. Nothing is refitted or substituted.
- `run_config.json` records `normalization_inputs.manifest_version` `"3"` for a preprocessing
  run with `--verify-content` (preprocessing `v0.4.0rc5`), `"2"` otherwise; both are read through
  `RunInputs` and give the same patches (`tests/test_run_inputs.py`).

### Migration notes (from rc2)
- **Re-run mixed file runs** (2-D beside 3-D images) and runs of OME-Zarr images with T or Z of
  length 1 under `"all"`, `"first"` or `"select"` to get the new ids; patches exported before keep
  theirs. Code that joins patches across the two versions by `patch_id` must not mix them.
- **Read back a run with the `run_config.json` it wrote.** A row whose `normalization_scope`
  differs from that file's scope is now refused (`ValueError` naming both), never read as the
  run's; a table whose rows or `run_config.json` were edited by hand may be refused.
- **Bricks that pin CellPatcher** pin `v0.2.0rc3` together with preprocessing `v0.4.0rc5` and
  normalization `v0.3.3`: pip refuses two different direct URLs for one package, so `v0.2.0rc2`
  (preprocessing `v0.4.0rc4`, normalization `v0.3.2`) cannot be installed beside them.
- Replace `discover_condition_repeat_folders` with a preprocessing run's `RunInputs` (`fovs()`,
  `read_mask`), or `run_patching_condition_repeat_tree` with `common.preprocessing_run`.

### What preprocessing rc5 and normalization v0.3.3 change for CellPatcher users
CellPatcher reads every file run through `RunInputs` and discovers images with the brick's
`discover_experiment`, so the brick's new refusals reach CellPatcher runs with the brick's reason
(preprocessing `CHANGELOG.md` at `79dd68e`, normalization `README.md` "Migrating to v0.3.3" at
`320f304`):
- **Mount points** below the experiment or masks folder (`mounted_folder`) are refused by
  discovery and by `RunInputs` (`preprocessing_run_refused`). Copy the files into the experiment,
  or select the mounted folder itself.
- **`position_id`** is required in a version 2 manifest, equal to the image's name without its
  extension (D85), the `position_id` CellPatcher gives every patch; runs written by rc1 to rc4
  open unchanged, a hand-edited manifest may not.
- **Strict runs** (`--verify-content`) have manifest version 3, read as above; rc4 and earlier
  refuse them, so a strict run needs CellPatcher `v0.2.0rc3`.
- CellPatcher reads neither the mask report (`mask_relationships.json`, whose tie count
  `ambiguous_parent` is new in report version 3) nor the intensity warnings (the new
  `channel_without_finite_pixels`): the preprocessing run shows them. The warning
  `hard_linked_file` is not repeated by CellPatcher either. Cell mode reads the masks the run
  paired; nothing changes for its patches.
- Normalization `v0.3.3` gives bit-identical parameters for valid statistics and refuses corrupt
  or ambiguous ones (a `plane_histograms.csv` with no rows or a repeated `plane_id`, plane `z` or
  `t` that are not whole numbers 0 or above, a non-finite offset or a scale not above 0).

### Each image of a file run is read in its own roles (#77, audit round 2, A-M1)

#### Changed
- **`patch_id` changes for 2-D images in mixed runs.** A folder or tree run read every image in
  one layout per run: Z (or T) when *any* image had more than one plane along it. A `YX` image
  patched beside a `ZYX` image therefore got `z0=0` and `z000` in its `patch_id`, other random
  rectangles with `seed_scope="plane"` and `(1, h, w)` arrays under `z_policy="keep"`, and
  `z_policy="select"` refused the run. Owner decision of 2026-11-03: an axis is in an image's plane
  choices and ids only if *that image* has the role, whatever else is in the run (the rule
  MorphoFeature's run planes follow). Such an image now gets the ids, rectangles and arrays it gets
  alone (`ctrl/rep1/single/a/grid-...`, `z0` empty), equal to MorphoFeature's `cut_patch_set` ids,
  and `z_policy="select"` applies only to the images with Z. Ids of runs whose images share their
  roles do not change. Re-run a mixed run to get the new ids; patches exported before keep theirs.
- An OME-Zarr image records T and Z of length 1: a `t_policy`/`z_policy` that chooses planes
  (`"all"`, `"first"`, `"select"`) now names them (`t0000`, `z000`), as MorphoFeature does; before,
  they were left out because no image had more than one plane along them. Under `"keep"` nothing
  changes (a kept axis of length 1 stays out of the arrays and ids).
- `run_config.json`'s `preprocessing.layout` maps each image's path to its own `image` and `mask`
  layout (`images`) when the images' layouts differ, and exported patches are normalized on load in
  their own image's axes; a row whose `source_relative_path` names none of those images is refused
  (its axes are unknown), never given the run's union of axes. A run of one layout is recorded as before.
- Every image of a file run is patched one at a time (the path plane policies already took), so
  `z_policy="keep"` reads one image at a time too. Tests in `tests/test_mixed_run_layouts.py`
  (from the audit probes p1 and p3, with a parity check against MorphoFeature's `cut_patch_set`
  as it calls CellPatcher at MorphoFeature 4f25a79, run in every job: MorphoFeature is private, so CI
  cannot install it).

### The design of an unfinished publication is read (#77, audit round 2, A-m2)

#### Fixed
- `normalization.scope="biological_repeat"` read `<output>/experiment_design.csv` itself, so a
  preprocessing run whose publication was unfinished but valid (the manifest's design still in
  `<output>/.publish`, which `RunInputs.open` accepts) was refused with `changed_experiment_design`.
  The design is now found with the preprocessing brick's `published_path`, as `RunInputs` finds it,
  and still read with `read_confirmed_design` against the manifest's SHA-256. Test in
  `tests/test_design_publication.py`.

### `discover_condition_repeat_folders` is deprecated (#77, audit round 2, A-m4)

#### Deprecated
- `cellpatcher.discover_condition_repeat_folders` emits a `DeprecationWarning` and will be removed.
  Its `mask_folder` mirrors each leaf under `mask_root_path` (`mask_root/<leaf>`), a pairing the
  preprocessing run does not use, and no CellPatcher run calls it since CP-2a. It is a listed public
  interface (AGENTS.md), so it keeps its results until removal. Read a preprocessing run's images,
  FOV keys and masks instead (`RunInputs.open(run).fovs()`, `read_mask`), or patch them with
  `run_patching_condition_repeat_tree` and `common.preprocessing_run`. Test in
  `tests/test_deprecations.py`.

### The Z/T mask message names the real options (#76, cross-brick test F7)

#### Fixed
- A cell-mode run whose masks have a Z or T axis (e.g. a `ZCYX` file run with `ZYX` masks) and
  the default `common.z_policy="keep"` failed with "mask_channel is required ... Set
  config.mask_channel and config.common.mask_axes", but a file run refuses `common.mask_axes`
  and `mask_channel` selects a label channel, not a plane. The message now names the mask axes
  and shape and the options that work: `common.z_policy`/`common.t_policy` (`"all"`, `"first"`,
  `"select"`), or 2-D (`YX`) masks (in a file run, the preprocessing run's masks). Behaviour is
  unchanged: the same inputs are refused, before any patch. Tests in
  `tests/test_plane_policies.py`.

## [0.2.0rc2] - Preprocessing v0.4.0rc4

Owner decision of 2026-11-03: every brick is aligned on preprocessing `v0.4.0rc4` through
CellPatcher `v0.2.0rc2`. This release candidate also carries CP-4 and the LabConstrictor template
sync below. As for `v0.2.0rc1`, no installer is built for it: install it with pip or on Google
Colab (`docs/developer_notes.md`).

### Changed
- CellPatcher requires `morphobricks-preprocessing` `v0.4.0rc4` (core and `zarr` extra; was
  `v0.4.0rc2`). Its normalization pin is unchanged (`v0.3.2`). Every notebook installs
  `v0.2.0rc2` (`RELEASE_TAG`, `current_version`, `notebook_latest_versions.yaml`).
- Bricks that pin CellPatcher pin `v0.2.0rc2` together with preprocessing `v0.4.0rc4`: pip
  refuses two different direct URLs for one package, so `v0.2.0rc1` (preprocessing `v0.4.0rc2`)
  cannot be installed beside preprocessing `v0.4.0rc4`.

### What preprocessing rc3 and rc4 change for CellPatcher users
CellPatcher reads every file run through `RunInputs`, so the brick's new refusals reach
CellPatcher runs as `preprocessing_run_refused`, with the brick's reason; no CellPatcher code
changed for them (preprocessing `CHANGELOG.md` and `docs/migration-0.3-to-0.4.md` at `c6d8aff`):
- **Links** in the experiment and masks folders are followed only inside them: a link out of the
  folder is `link_outside_folder`, one file reached by two paths `linked_twice`, a loop
  `link_loop`. A tree of links that gathers images stored elsewhere must now hold its targets
  inside the experiment, for example in a hidden `.raw/` folder, which discovery skips
  (`docs/configuration_reference.md`, "Trees of links"). A link out added after the run is
  refused by `RunInputs` too.
- **OME-Zarr** images recorded by preprocessing `v0.4.0rc3` or earlier have no file inventory in
  their fingerprint: `RunInputs` refuses such runs. Run preprocessing again after upgrading.
  OME-NGFF 0.1 and 0.2 stores, and stores without `axes`, are `unreadable_ome_zarr`.
- **Named-axis copies**: conversion records of version 1 (`v0.4.0rc3` or earlier) are refused;
  convert again into a new folder.
- **Masks folder**: an experiment, named-axis copies or the output folder inside the masks folder
  is refused, also by `RunInputs.open` for a run recorded that way.
- `RunInputs.read_plane`/`read_mask` refuse a `z` or `t` outside the image (`IndexError`);
  CellPatcher only reads the planes the run recorded, so nothing changes for its runs.
- Pixel sizes: a header size without a length unit is unknown. CellPatcher reads no pixel size,
  so nothing changes for its patches.

### Tests
- The link-tree tests store their targets in the experiment's `.raw/` folder, and two new tests
  check that a link out of the experiment stops preprocessing and, added after the run, the
  CellPatcher file run (`tests/test_source_relative_path.py`).

### No silent rescale or channel repetition in the torch adapters (CP-4)

Owner decision of 2026-11-02 (the CP-3b review's question): the torch adapters' defaults change
before the final `v0.2.0`; `v0.2.0rc1` ships with the old ones, `v0.2.0rc2` with the new ones.

#### Breaking
- **`PatchTensorConfig.scale_policy` has no default.** `PatchTensorConfig()`, a config without
  `scale_policy`, `patch_array_to_tensor(..., config=None)` and `PatchTorchDataset`/
  `ExportedPatchTorchDataset` without `tensor_config` raise a `TypeError` naming the choices:
  `"none"` keeps the values; `"auto"` (deprecated) divides an integer patch by its dtype maximum.
  Before, the default `"auto"` turned a uint16 patch of 1000 into 0.0153. A required choice, not
  a `"none"` default: code written for the old default stops with an error instead of receiving
  values up to 65535 times larger without a word (contract §15).
- **`PatchTensorConfig.channel_policy` defaults to `"keep"`** (the channels as exported). Before,
  the default `"repeat_to_rgb"` repeated a single channel to three. A one-channel patch is now a
  `(1, H, W)` tensor; patches with 2 or more than 3 channels, refused by the old default, are kept.
- Each dataset item records the repetition beside the divisor: `channel_repeat` (3 when a single
  channel was repeated to three, else 1; `cellpatcher.torch_dataset.channel_repeat`), as
  `scale_divisor` (1.0 when unscaled) records a rescale.

**Migration.**
- Keep the values (recommended; normalize with `config.normalization`):
  `PatchTensorConfig(scale_policy="none")`. The PyTorch notebook (`cellpatcher.torch_session`)
  already uses it with `channel_policy="keep"`; nothing changes there.
- The old behaviour, exactly: `PatchTensorConfig(channel_policy="repeat_to_rgb",
  scale_policy="auto")`. `"auto"` still warns when it rescales (`ScalePolicyDeprecationWarning`,
  kept: it now marks the named, deprecated option, no longer a default) and is recorded per item.
- An RGB model on normalized patches: `PatchTensorConfig(channel_policy="repeat_to_rgb",
  scale_policy="none")`.

**Downstream calls (no other repository is changed here).** MorphoEmbed `main` `a8e88df`:
`src/morphoembed/provider_crops.py` already passes `channel_policy="keep"`, `scale_policy="none"`
(`TENSOR_CONFIG`), unaffected; its `ROADMAP.md` (lines 126-127) describes the old defaults.
MorphoFeature `main` `4e6970f`: documentation only (`AGENTS.md` line 247,
`docs/audits/pytorch-encoder-phenome-audit.md`), which already names `scale_policy="none"`; the
audit's PT08 row (line 121) still describes the old default RGB repetition.
MorphoRelate `ea19bb0`, fiberfeatures `362f1ba`, MorphoObjects `2080ff5`: no use.

#### Changed
- `docs/api_examples.md`, `docs/quick_start.md`: every `PatchTensorConfig` names its
  `scale_policy` (`"none"`), and the RGB examples their `channel_policy`; contract §3 and §15 and
  `docs/plan.md` §6.8 record the decision.

### LabConstrictor template sync (0.1.19)

No change to the package or the notebooks. Before the final `v0.2.0` (owner decision of
2026-11-02):
- The installers, release and notebook workflows and `.tools/` follow LabConstrictor template
  version 0.1.19, synced with the template's `.template_sync` migrations (later syncs come as the
  template's pull requests; `docs/developer_notes.md`, "LabConstrictor template"). For the
  installers built at `v0.2.0` this brings the template's TLS-resilient JupyterLab launcher
  (`launch_jupyter.py`), install checks that stop on a failed step, an NVIDIA-GPU requirements file
  (`requirements_gpu.txt`, the same packages as `requirements.txt` here), the inert LabConstrictor
  tools-bridge hooks and actions pinned by commit SHA in the template workflows.
- The installer copies the package to `<app>/src/cellpatcher/` (the template's fixed
  `bump_constructor.py`); `setup.py` reads `src/` only again (CP-3b's two-layout workaround is
  removed). The app's package is still installed with its dependencies and the `zarr` extra (and
  with build isolation, so pip can build the bricks from GitHub), and release candidates still get
  no installer.
- The template's weekly sync schedule is off: syncs are run by hand. The final `v0.2.0` installers
  get a manual install test on Windows, macOS and Linux before the release is announced
  (`docs/developer_notes.md`).
- CI: the workflow is `.github/workflows/cellpatcher_tests.yml` (was `tests.yml`; same jobs), with
  a new job running every test with `.[test,zarr,notebook]` and CPU `torch` 2.15.0: the torch tests
  and the PyTorch notebook's kernel run now run in CI, and the job fails if any test is skipped.
- `pyyaml` is added to the `test` extra (the installer test runs the template's
  `bump_constructor.py`).
- `docs/labconstrictor-upstream-proposals.md`: what CellPatcher had to change in template files,
  proposed for the template (not sent).

---
## [0.2.0rc1] - 2026-11-02

### Changed
- Version updated from 0.0.6 to 0.2.0rc1

**Commit Details:**
- Hash: [`3b420930`](https://github.com/CellMigrationLab/CellPatcher/commit/3b420930fccf9731836659ebbaf62b29f235146b)
- Author: guijacquemet
- Date: 2026-11-02 12:42:52 +0200
- Message: Merge pull request #71 from CellMigrationLab/claude/cp-3-notebook-docs-installers

CP-3: notebook, docs, installers and code standards for v0.2.0 (preprocessing v0.4.0rc2), with the independent review's fixes. v0.2.0rc1 is tagged after CP-3b.

---
## [0.2.0rc1] - The other notebooks, the normalization views, the installer layout (CP-3b)

The follow-ups the owner decided on 2026-11-02, before `v0.2.0rc1` is tagged.

### Changed
- **`CellPatcher_Strategy_Examples.ipynb` and `CellPatcher_Pytorch.ipynb` are rewritten for v0.2**
  with the main notebook's LabConstrictor treatment. They installed `main` with a `GITHUB_TOKEN`
  secret, worked on synthetic in-memory arrays rather than a preprocessing run, and computed in
  their cells (synthetic data, galleries, `pytest` and parity checks). Now:
  `# @title` cells only calling the package; the main notebook's Step 0 (Drive, the widget manager
  and the zarr extra only under `if "google.colab" in sys.modules:`, no magics, no `/content`
  outside it, the install pinned to `RELEASE_TAG` `v0.2.0rc1`); `current_version` `0.2.0rc1` and
  `notebook_name` as in `notebooks/notebook_latest_versions.yaml`; Colab and JupyterLab callouts;
  data on Google Drive on Colab.
  - **Strategy examples** (`cellpatcher.notebook_examples`): the main notebook's step 1 (the
    preprocessing run, read through `RunInputs`), then one page per strategy (cell, grid, random,
    whole FOV) with that strategy's settings and a button that previews the chosen image with it.
    It is the main notebook's preview, so it shows the patches the export gives; nothing is written.
  - **PyTorch** (`cellpatcher.notebook_torch`, `cellpatcher.torch_session.load_exported_batch`):
    the output folder of the main notebook's export is read with `ExportedPatchTorchDataset`, with
    the arrays' axes from `patches.csv` (`output_axes`), every channel and the values as exported
    (`channel_policy="keep"`, `scale_policy="none"`), and its first patches are shown as one batch
    (shape, type, range, images). Patches of different sizes are refused, never resized. The old
    notebook was a developer check that ran `pytest` from a clone; the torch tests run in the test
    suite instead.
- **The main notebook's normalization views are back**, computed in the package:
  - **Show parameters per group** (step 2, Normalization card): the offset and scale of every group
    and channel (and time point) of the run's table, before any patch, with the warnings of the
    settings; a group that cannot be normalized is shown in red
    (`cellpatcher.notebook_session.group_parameters_of_run`, which opens and checks the run as an
    export does and calls `cellpatcher.normalization.group_parameters`). They are the parameters
    the patches get (tested against an export).
  - **Before and after normalization** (step 3): with normalization, the preview also patches the
    image without it and shows the first four patches both ways, each beside its intensity
    histogram (`Preview.before_after`, `cellpatcher.normalization_review`: paired by `patch_id`,
    binned by the normalization brick's `compute_plane_histogram`; nothing is re-implemented).
- **Installer layout fixed.** This repository's copy of LabConstrictor's `bump_constructor.py`
  (older than the template's fix `a76be61`) copies the package to `src/cellpatcher/cellpatcher/`,
  where `setup.py` found no package: the installed app had no `cellpatcher`. `setup.py` now reads
  the package from there when it is laid out so, and from `src/` otherwise (the current template's
  layout); a test installs both layouts and imports `cellpatcher`. No template file is changed.
- A missing file a table names (`FileNotFoundError`) is shown as a red settings card, not a
  traceback.
- All three notebooks share one Step 0, checked by a test; each runs headless in a kernel in the
  tests (the PyTorch notebook when `torch` is installed). **No CI job has `torch` yet**: the
  PyTorch notebook's kernel run and the torch tests are skipped in CI; they were run locally with
  CPU `torch` 2.15.0 (see "Not yet").

### Fixed (independent review of CP-3b)
- **The torch adapters' default rescaling was silent.** `PatchTensorConfig(scale_policy="auto")`
  (the default) divides an integer patch by its dtype maximum. Its `DeprecationWarning` was raised
  inside the library, where Python's default filters hide it, so `ExportedPatchTorchDataset` and
  `PatchTorchDataset` rescaled without a word. The warning is now a
  `cellpatcher.torch_dataset.ScalePolicyDeprecationWarning` (a `DeprecationWarning`, so existing
  filters still match it) shown once per place under the default filters, naming the divisor, and
  every dataset item records it as `scale_divisor` (1.0 when the values are not rescaled). The
  default itself is unchanged (changing it needs an owner decision, contract §15).
- **The PyTorch notebook's batch** was converted to float32 without a check (an int32/uint32 value
  above 2^24 or a float64 value would have been rounded). The patches are now read in their
  exported type and converted only when float32 holds every value exactly, otherwise refused; the
  exported type is recorded (`TorchBatch.exported_dtype`) and shown.
- The before/after figure titled a patch without a label with the end of its `patch_id` (cut at a
  `/`); an identity is never parsed: it is now numbered.
- The test extra installs `pip` (`tests/test_installers.py` pip-installs both installer layouts;
  a venv made by `uv` has no pip), and the kernel tests put their IPC sockets in a short temporary
  folder (pytest's `tmp_path` can exceed macOS's 104-byte socket path limit).

---
## [0.2.0rc1] - Preprocessing v0.4.0rc2, unrecorded files refused, the v0.2 notebook (CP-3)

The first release candidate of v0.2.0, after CP-1 to CP-3 (see the sections below for CP-1,
CP-2a and CP-2b, all part of it).

### Changed
- CellPatcher requires `morphobricks-preprocessing` `v0.4.0rc2` (core and `zarr` extra; was
  `v0.4.0rc1`). Its normalization pin is unchanged (`v0.3.2`).
- **Files the preprocessing run did not record are refused, never left out.** Since rc2
  `RunInputs.open` refuses an experiment holding images or masks the run did not record (for
  example added after the run); CellPatcher reports it as `unrecorded_files`, naming the files.
  This closes the CP-2a known limitation (an added image was not patched and nothing said so).
  Run preprocessing again on the experiment as it is, or exclude the images there.
- A run with OME-Zarr images opened without the zarr package is refused (`zarr_extra_missing`,
  naming `cellpatcher[zarr]`): their label images cannot be checked, and none is skipped.
- A mask whose header puts X before Y is refused by the preprocessing run itself (rc2), and a
  run written by rc1 that recorded such a mask without axes is refused when it is opened
  (`preprocessing_run_refused`). CellPatcher's own refusal of such a record stays as a safety
  net.
- The preprocessing command CellPatcher gives (`no_preprocessing_run`) names rc2's `design`
  step: `check` now writes `experiment_design.csv`, and `run` stops until it is confirmed.
- **The notebook is rewritten for v0.2** (it was broken since CP-2b). It asks for the
  preprocessing run's output folder and, only for an experiment moved or copied since, where the
  experiment and masks folder are now; no axes, labels or mask source. Four steps: the run, the
  settings, a preview of one image (the file-run code, so the patches the export gives), and the
  export (one `patches.csv`, `images.csv`, `run_config.json`). It follows LabConstrictor's
  portability guide: `# @title` cells, Colab code (Drive, the widget manager, the zarr extra)
  only under `if "google.colab" in sys.modules:`, the install pinned to `v0.2.0rc1` (no
  `GITHUB_TOKEN`), Colab and JupyterLab callouts. Its code is in the package:
  `cellpatcher.notebook_session` (no widgets) and `cellpatcher.notebook_ui`/`notebook_form`.
- **Breaking:** in-memory `common.image_axes` with X before Y (`"XY"`, `"CXY"`, ...) is refused by
  `run_patching` and lazy delivery, as masks already were (design v0.4 §6: a canonical-only
  layout for in-memory arrays). Before, such an image was patched in the transposed frame: a
  12 x 20 array gave the whole-FOV patch `y1=20`, `x1=12`. Swap the spatial axes
  (`numpy.swapaxes`) and declare Y before X.
- `refuse_transposed_mask_axes` moved from `cellpatcher.io` to `cellpatcher.axes`;
  `file_runs._read_mask` is public as `read_run_mask`.
- Version 0.2.0rc1 (`setup.py`, `cellpatcher.__version__`, the notebook's `current_version`).
- Installer: `construct.yaml` lists every module again (it lacked the CP-1/CP-2 modules); a
  test keeps it complete. The release script refuses a pre-release version: release candidates
  are installed with pip or on Colab only, and the installers are built at the final release.
- Code standards (design v0.4 §9): `tools/check.py`, the shared `tools/code_rules.py` (SHA-256
  checked) and `tools/code_rules_baseline.json` (today's debt, which may only shrink, with a
  `# noqa` ratchet); Ruff 0.17.3 in the `test` extra, a CI rules job.

- In-memory X-before-Y axes are also refused by `scan_cell_patch_size`/`scan_cell_objects`
  (before, a square mask declared `XY` gave the transposed mask's sizes), by
  `CellSegmentationStrategy(mask_axes=...)`, and by `PatchTorchDataset` when it is built (before,
  only when an item was loaded, in a DataLoader worker).
- `tools/check.py` runs Ruff with `--ignore-noqa`: a `# noqa` comment silences no finding past
  the baseline.

### Not yet (follow-ups)
- **CP-3b, before the `v0.2.0rc1` tag** (owner decisions of 2026-11-02): done, see the CP-3b
  section above (the other two notebooks, the normalization views, the installer layout).
- **Before the final `v0.2.0` (not before `v0.2.0rc1`): one LabConstrictor template-sync PR**
  (owner decision of 2026-11-02): sync `.github`/`.tools` with the current template (fixing
  `bump_constructor.py`'s package layout upstream-style; `setup.py` reads both layouts until then),
  re-apply CellPatcher's CI rules job, extras matrix and the release-candidate refusal of
  `bump_version.py` (and propose them upstream), and **add a CPU-torch CI job**. Until then the
  PyTorch notebook's kernel run and the torch tests are skipped in CI and run locally only.
- **After preprocessing v0.4.0 is final:** the CellPatcher notebook will include the
  preprocessing steps through preprocessing's declared embedding API (`PreprocessingNotebook`,
  `notebook_style`), never a copy of its code. Until then, run the preprocessing notebook or
  command line first.

### Migration notes for the bricks (pin CellPatcher `v0.2.0rc1` with preprocessing `v0.4.0rc2`)
- **MorphoRelate** (`io/cellpatcher.py`): `check_image_axes` reads `common.image_axes` from
  `run_config.json`; a folder or tree run's version 2 file has none (its arrays' order is
  `preprocessing.layout.image`, and the run is `preprocessing.run`), so it reports "not checked".
  Take the axes from the preprocessing run (`RunInputs`) or `preprocessing.layout`.
  `read_patch_table` reads `run_config_version` 1 and 2 and refuses any other.
- **MorphoEmbed** (`provider_crops.py`): `PatchTorchDataset` takes arrays only; a path is a
  `TypeError` when the dataset is built (use `iter_run_patch_arrays` for a file run's patches).
  In-memory arrays must be declared Y before X.
- **fiberfeatures**: `data/experiment_001/reference/patch_config.json` holds
  `common.mask_source`/`mask_label_name` (at their defaults): `load_config` still reads it, but
  `CommonConfig(**common)` is a `TypeError`. Regenerate the reference with v0.2.0rc1 (both keys
  gone) or drop them before building a `CommonConfig`; the masks are the preprocessing run's,
  chosen with `matching.mask_type`.
- **Every brick:** run preprocessing v0.4 on the experiment; give CellPatcher
  `common.preprocessing_run`; files added after the run are refused (`unrecorded_files`).

## [Unreleased] - File-run removals and run_config.json version 2 (CP-2b)

### Changed
- `run_config.json` is version 2 (`run_config_version`). When a preprocessing run was read
  it has a `preprocessing` section: the run's folder (`run`), the SHA-256 of its
  `run_manifest.json` taken when the call opened it (`manifest_sha256`), and for a folder or
  tree run the array `layout` (`image`, `mask`) built from the run's recorded roles. A manifest
  that changes during the call (while it is opened, between two opens, or before
  `run_config.json` is written) is refused, and no `run_config.json` is left. A folder
  or tree run no longer records `common.image_axes`, `mask_axes` or its labels there (they are
  the run's, not settings), so its `run_config.json` loads back as a config a file run accepts.
  `load_config`, `read_patch_table` and the exported-patch readers refuse any version but 1 (no
  version: written before v0.2.0, or a plain config file) and 2.
- `validate_image_and_mask_folders(image_folder, mask_folder, *, preprocessing_run=, mask_type=)`
  returns the pairing the preprocessing run recorded (opened where the folders are now); it
  pairs nothing itself, and refuses without a run (`no_preprocessing_run`). It moved to
  `cellpatcher.file_runs` (still `cellpatcher.validate_image_and_mask_folders`).
- New `iter_run_patch_arrays(patches, config, *, experiment_root=, masks_root=)`: lazy
  delivery of a folder or tree run's patches, each plane read with `RunInputs.read_plane`
  (masks with `read_mask`) from the image the run recorded at the patch's
  `source_relative_path`, refusing another image or FOV key.
- `run_patching_pipeline` (and `run_patching_from_folders`) accept `mask_folder_path` in `grid`
  and `whole_fov` mode (owner decision of 2026-11-02): it says where the run's masks folder is
  now; those masks are checked there, never read. Before, it was refused.
- `normalization.experiment_root`/`masks_root` (owner decision of 2026-11-02): where the
  experiment and masks folder of a moved experiment are now, for `run_patching` at the `image`
  or `biological_repeat` scope. Folder and tree runs refuse them (their arguments say it).

### Removed (breaking)
- `patch_array`, `iter_patch_arrays` and `PatchTorchDataset` take arrays only: a file path is a
  `TypeError` naming `iter_run_patch_arrays` (`PatchTorchDataset` refuses it when it is built;
  before, CellPatcher read it with `common.image_axes` and its own header check).
- `images.csv` loses its last column `image_read_transposed` (always `false` or empty since
  preprocessing v0.4), and `ImageRecord` its field.
- `common.mask_source` and `common.mask_label_name`: `CommonConfig(mask_source=...)` is a
  `TypeError`; a config or `run_config.json` with their defaults still loads, one with another
  value is refused naming `matching.mask_type`.
- `cellpatcher.io`: `pair_masks`, `MaskPairing`, `read_image`, `conform_axes`,
  `read_ngff_mask`, `refuse_image_header_axes`, `refuse_mask_header_axes`,
  `refuse_header_axes`, `header_axes_disagreement`, `refuse_mask_channel_axis`,
  `source_relative_path`, `discover_folder_images` and `validate_image_and_mask_folders` (moved,
  above); `cellpatcher.planes`: `iter_file_planes`, `read_file_plane`; `cellpatcher.ids`:
  `object_set_from_label_path`; `cellpatcher.pipeline`: `check_source_identities`,
  `run_config_data` (now `cellpatcher.run_config_file`). `cached_statistics` moved from
  `cellpatcher.normalization` to `cellpatcher.run_open`.

### Known breakage until CP-3
- **CellPatcher's own notebook (`notebooks/CellPatcher/CellPatcher.ipynb`) does not work after
  this change.** Its import cell fails: it still imports `pair_masks`, `read_ngff_mask` and
  `cellpatcher.normalization.cached_statistics`, and it offers the removed settings. It is
  rewritten in CP-3, the next PR; no compatibility shim is added meanwhile (owner decision of
  2026-11-02). Use the Python API until then.
- Not in this change: CellPatcher does not yet call preprocessing's check for images added to
  the experiment after its run. It is wired in once preprocessing v0.4.0rc2 provides it.

### Downstream: for the brick migrations
- **MorphoRelate** (`io/cellpatcher.py`): `check_image_axes` reads `common.image_axes` from
  `run_config.json`; a folder or tree run's version 2 file has none, so it reports "not
  checked" (no error). Its migration should take the axes from the preprocessing run (or
  `preprocessing.layout`). `read_patch_table` now refuses an unknown `run_config_version`.
- **MorphoEmbed** (`provider_crops.py`): `PatchTorchDataset([image], ...)` is fine with an
  array; a path is now a `TypeError` when the dataset is built.
- **fiberfeatures**: `data/experiment_001/reference/patch_config.json` still holds
  `mask_source`/`mask_label_name` at their defaults. It loads with `load_config`, but
  `CommonConfig(**common)` on it is a `TypeError`; a regenerated reference loses both keys.

## [Unreleased] - File runs read through the preprocessing run (CP-2a)

### Changed
- **Breaking.** `run_patching_pipeline`, `run_patching_condition_repeat_tree` and
  `run_patching_from_folders` read files only through a clean preprocessing run (v0.4):
  `common.preprocessing_run`, its output folder, is required in every mode, normalization on or
  off (`no_preprocessing_run`, with the preprocessing command). The images patched are the
  run's, with their labels and `position_id` as it recorded them; images and masks are read
  with `RunInputs.read_plane`/`read_mask` in the roles it recorded, in one canonical layout per
  run (T, Z, C, then Y, X), recorded in `run_config.json` as `common.image_axes`/`mask_axes`
  (in its `preprocessing.layout` since CP-2b).
  (An image added to the experiment after the run was then not patched, and nothing said so; since
  preprocessing v0.4.0rc2 it is refused: see v0.2.0rc1 above.)
- The run is opened with the image folder/root and the masks folder given, so an experiment
  moved or copied since its run (times kept) is checked and read where it is now. A mask
  changed in a copy is refused (before, only the original was checked).
- For file runs, `common.image_axes`/`mask_axes` other than `"YX"`, `mask_source="ngff_labels"`,
  `mask_label_name`, and `condition`/`biological_repeat`/`technical_repeat` other than
  `"single"` are refused with a migration message. `matching.mask_type` chooses among the run's
  mask types (a label image's type is its name); `random` mode reads masks when a masks folder
  or `matching.mask_type` is given. `run_patching_pipeline` patches a run of one condition and
  biological repeat.
- The statistics cache is gone: a run is opened once per CellPatcher call and kept for that
  call only, so a mask or an unpatched image changed between two calls is refused.
- `load_plane_statistics`, `recorded_image_type` and `cached_statistics` take
  `experiment_root=`/`masks_root=`; they and `open_run` moved to `cellpatcher.run_open`
  (still importable from `cellpatcher.statistics`).

## [Unreleased] - The preprocessing run is read through RunInputs (CP-1, preprocessing v0.4.0rc1)

### Changed
- CellPatcher requires `morphobricks-preprocessing` `v0.4.0rc1` (also in the `zarr` extra; the
  tag reports version 0.3.4) and opens the run that wrote `plane_histograms.csv` with its
  `RunInputs` (design v0.4 §6). Refused now (`preprocessing_run_refused`), where before they
  warned or were read: a table without `run_manifest.json` beside it, any run before
  preprocessing v0.4 (manifest version 1, which was the only one read before), a table or
  design changed since the run, an image or mask of the run changed or missing, and an
  experiment moved since the run (until file runs, CP-2a).
- `normalization.plane_histograms`, `manifest` and `experiment_design` may only name the run's
  own files, unrenamed in its output folder (`not_the_run_output`); the `biological_repeat`
  scope refuses a run that recorded no design (`missing_experiment_design`). The warnings for a
  missing manifest and a missing design hash are gone.
- `common.image_axes`/`mask_axes` are checked with the brick's `review_axes`
  (`check_declared_axes` was removed in v0.4): a Y/X swap is refused (`axes_yx_swapped`) for
  images too, so nothing is read transposed and `image_read_transposed` is always `false`.
- Mask pairing: an image named like a mask of another image is refused
  (`image_named_as_mask`). In a mask folder run the types are the folder's: an OME-Zarr image's
  own label image is read only with `mask_source="ngff_labels"`, and a label image and a folder
  mask of one type for one image are `duplicate_mask`.
- `run_config.json`: `normalization_inputs` keeps its keys; `manifest_version` is `"2"`.
- Removed: `cellpatcher.normalization.run_inputs` (now `normalization_inputs`) and
  `cellpatcher.statistics.SUPPORTED_SCHEMA_VERSION`/`SUPPORTED_MANIFEST_VERSION`.

## [Unreleased] - Declared axes checked with the shared rule; Y and X swapped recorded

### Changed
- `common.image_axes` and `common.mask_axes` are checked against each TIFF file's header with the
  preprocessing brick's `check_declared_axes` (its contract §16, owner decision of 31 October
  2026); CellPatcher's own comparison is gone. In folder runs, tree runs and lazy delivery:
  - Y and X declared swapped relative to an image's header (`ZYX` declared `ZXY`; also a plain 2D
    TIFF declared `XY`) are still accepted, and the image is read transposed, as before; now it
    warns, naming the images, and `images.csv` records it in the new last column
    `image_read_transposed` (`true`/`false`; empty for in-memory arrays and OME-Zarr images).
  - Y and X declared swapped for a **mask** file are now refused (they were accepted, but the
    mask was read as stored: CellPatcher takes a label plane's dimensions as Y and X in their
    stored order, so on a square image the mask was silently misaligned). The refusal says how
    to fix the masks: save each one with Y before X (its last two axes swapped, e.g.
    `numpy.swapaxes(mask, -1, -2)`), then declare those axes. Reading masks transposed is planned
    for a later version.
  - For the same reason, any `common.mask_axes` with X before Y (`XY`, `ZXY`) is refused, with
    the same advice, also for in-memory masks (`run_patching(masks=...)`, lazy delivery) and
    OME-Zarr labels, and for a mask file whose header itself names X before Y. Before, an
    in-memory mask declared `XY` was read as stored (Y, X): with `image_axes='XY'` the patches
    missed their objects without a word.
  - With a CSV normalization scope (`image`, `biological_repeat`), a Y/X swap the preprocessing
    run did not declare too is refused by the preprocessing brick's `check_sources`
    (`axis_mapping_mismatch`), as before: the statistics were computed with the other roles.
  - A plain TIFF without axis metadata (a 2D page or a plain stack of pages) is now checked too:
    a declaration with another number of axes (`ZYX` for a 2D page, `TZYX` for a 3-page stack)
    is refused before any pixel is read; any order of its open axes is accepted (a plain `QYX`
    stack declared `YXC` or `CYX`), as before.
  - A file whose header cannot be read is refused at this check, naming it, where before it
    failed later when read.
  - Refusals carry the preprocessing brick's message (`The axes declared for a.tif ('TYX') do not
    fit the axes in its header ('ZYX'): ...`) after `common.image_axes='TYX' does not fit the header
    of N image file(s):`.
- `cellpatcher.io.header_axes_disagreement` returns `(message or None, read transposed)`;
  `refuse_image_header_axes` returns the images read transposed.

## [Unreleased] - The experiment design is read with the shared reader

### Changed
- With `normalization.scope='biological_repeat'`, `experiment_design.csv` is read with the
  preprocessing brick's `read_confirmed_design` (its contract §16), the one reader every brick
  uses; CellPatcher's own reading is gone. The file must be as the preprocessing brick writes
  it. Refused now (`unreadable_experiment_design`, naming the line and column), where before
  they were accepted: a byte-order mark (a file saved by a spreadsheet program), columns in
  another order, a blank line, a row with an extra value, a value with leading or trailing
  spaces, `control`/`confirmed` other than `true`/`false` (`yes`, `1`, `TRUE`), a table without
  rows, and condition or biological-repeat labels that are NA-like or differ only in case.
  Missing, unconfirmed, conflicting and changed designs are
  refused as before (same issue codes); the missing-design message comes from the
  preprocessing brick and still says to give the file in `normalization.experiment_design`.
- `cellpatcher.normalization.load_experiment_design` returns the preprocessing brick's
  `ConfirmedDesign` (same `path`, `sha256` and `repeat_groups`, now read-only);
  `cellpatcher.normalization.ExperimentDesignInput` is removed.

## [Unreleased] - read_patch_table checks the table's structure

### Changed
- `cellpatcher.read_patch_table` checks a `patches.csv`'s CSV structure before reading any value,
  and refuses (`PatchTableError`), naming the line or column:
  - a row with fewer cells than the header has columns (before, the missing cells were read as
    empty, so an optional field became None);
  - an unclosed quote: at the end of the table (before, the cell was read with a line break in
    it), or one that swallows the following lines into one cell (before, those rows were lost
    without a message); CellPatcher never writes a cell holding a line break;
  - a column name with leading or trailing whitespace (before, refused as a column CellPatcher
    does not write), and a repeated column name, now with its column positions.
  Blank lines are still skipped.

## [Unreleased] - Dependencies: preprocessing v0.3.4

### Changed
- CellPatcher requires `morphobricks-preprocessing` `v0.3.4` (also in the `zarr` extra) and still
  declares `morphobricks-normalization` `v0.3.2`, the tag preprocessing `v0.3.4` pins (owner
  decision D-h; `tests/test_dependency_pins.py` checks that the two pins agree).
- With `normalization.scope='biological_repeat'`, an `experiment_design.csv` whose
  `repeat_group` is empty, `NA`-like (`NA`, `nan`, `none`, `null`, `n/a`, `<NA>`, any case),
  has leading or trailing spaces or is spelled with different case in different rows, or whose
  `confirmed_on` is not empty or a `YYYY-MM-DD` date, is now refused naming the line and column
  (`unreadable_experiment_design`; preprocessing v0.3.4, owner decision of 31 October 2026).

## [Unreleased] - Legacy functions and options removed (D86, CP-2/CP-5/CP-6)

### Removed (breaking)
- The former image/mask matching: `find_candidate_matches`, `filter_shared_matches`,
  `select_requested_masks`, `LEGACY_MATCHING_DEFAULTS` and `cellpatcher.io.image_stem`. Images
  and masks are paired by `cellpatcher.io.pair_masks` (the preprocessing brick's `find_masks`);
  the image name is `morphobricks_preprocessing.discovery.position_id`.
- The matching options `strategy`, `mask_suffixes`, `case_sensitive` and
  `ignore_unrequested_masks`, until now accepted at their old defaults with a warning: in
  `MatchConfig`, a `run_config.json` and as arguments of `validate_image_and_mask_folders`,
  `run_patching_pipeline`, `run_patching_condition_repeat_tree` and `run_patching_from_folders`
  they are refused by name, saying to use `matching.mask_type`. An old `run_config.json` holding
  them fails until they are deleted from its `matching` section.
- Per-image normalization: `percentile_normalize`, `minmax_normalize` and the
  `normalize=`/`percentile=` arguments of `prepare_image_for_patching`,
  `iter_image_planes_for_patching` and `iter_cell_planes_for_patching`. Use
  `config.normalization`.
- `cellpatcher.require_preprocessing()`: import `morphobricks_preprocessing` directly (a core
  dependency).
- `cellpatcher.ids.object_set_from_mask_file`, which nothing called (the object set of a mask
  file is its mask type from the preprocessing brick's pairing).

## [Unreleased] - Shared MorphoBricks rules for labels, identity and sources (D86)

### Changed
- Labels, `position_id` and the condition/repeat labels follow the preprocessing brick's
  identity rules (owner decision D-e), as every brick does: text with leading or trailing spaces
  and names tables read as missing (`NA`, `nan`, `none`, `null`, `n/a`, `<NA>`, empty; any case)
  are refused, where before only empty text was. Names of one level that differ only in case
  are refused in the whole run (`A` and `a` as technical repeats; `Img` and `img` in one
  technical repeat, even as different time points).
- Plane indices in `image_metadata` (`t`, `z`, `c`) must be integers: text such as `"1"` is
  refused (it was read as 1).
- Masks are checked with the preprocessing brick's `check_labels` (owner decision D-b): labels
  from 0 to 2,147,483,647, as before; a float mask is now refused even when it holds only 0.0
  and 1.0 (it was treated as a binary mask): save the mask as integers. Messages changed.
- With a `run_manifest.json`, `statistics.check_images` uses the preprocessing brick's
  `check_sources`: an image that no longer exists is reported (`images_missing`), a recorded
  `position_id` that its name does not give is refused (`identity_differs`), an OME-Zarr image
  is compared by its files' total size and newest time, and its axes are now checked against
  the run's roles too (they were skipped).
- `SINGLE_CONDITION_LABEL`, `SINGLE_BIOLOGICAL_REPEAT_LABEL`, `SINGLE_REPEAT_LABEL` and
  `SINGLE_TECHNICAL_REPEAT_LABEL` are the normalization brick's `SINGLE` (same value).

## [Unreleased] - Patch rectangle and region-mask checks for other bricks (D86, C3)

### Added
- `cellpatcher.check_patch_bounds(bounds, shape)` and `cellpatcher.check_region_mask(mask, shape)`:
  the checks the feature bricks made on the patches MorphoFeature hands them, in one place.
  Bounds `(y0, y1, x0, x1)` must be four integers (floats such as `4.0` are refused, never
  converted), a non-empty rectangle, inside the plane; a region mask must be an integer or
  boolean array of the plane's shape (`cellpatcher.PatchBoundsError`).

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
- `cellpatcher.require_preprocessing()` warns: import `morphobricks_preprocessing` directly
  (removed by the v0.1.1 entry above).

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
