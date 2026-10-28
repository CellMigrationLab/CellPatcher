# AGENTS.md — CellPatcher

> This file is the authoritative context for any AI agent working in this
> repository. Read it before taking action. Prefer repository evidence over
> assumptions, and inspect the relevant code, tests, and documentation before
> making non-trivial changes.

---

## 1. Project identity

| Field       | Value                                                                                                                                                    |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repo        | `CellMigrationLab/CellPatcher`                                                                                                                            |
| Purpose     | Turns microscopy images (and, for cell mode, segmentation masks) into reproducible patch datasets — coordinates, optionally extracted/resized arrays, optionally exported files — using `cell`, `random`, `grid`, and `whole_fov` patch modes. |
| Ecosystem   | MorphoBricks — one of several interconnected repos                                                                                                        |
| Position    | Step 1 of 3 in the MorphoBricks pipeline (image patching)                                                                                                 |
| Language    | Python 3.11+                                                                                                                                              |
| Tests       | `PYTHONPATH=src python -m pytest -q` (equivalent to `pytest tests/` once the package is installed editable)                                              |
| Environment | `python -m pip install -e .` (add extras as needed: `.[test]`, `.[notebook]`, `.[torch]`) or `conda env create -f environment.yaml`                       |

Key dependencies: `numpy`, `scipy`, `tifffile`, `imageio` (core, required). Optional extras: `ipywidgets`, `matplotlib`, `pillow`, `tqdm`, `joblib`, `jl-hidecode` (notebook extra); `torch` (optional `cellpatcher.torch_dataset` adapter only — never a core dependency); `morphobricks-preprocessing[zarr]` from GitHub at a pinned tag, which brings `morphobricks-normalization` (no separate pin; MorphoFeature decision D79) (`preprocessing` extra, for `cellpatcher.statistics` and normalization — never a core dependency; histograms, the plane table, plane IDs, pooling and normalization parameters are imported from `morphobricks_normalization`, not from their deprecated preprocessing paths).

Current health snapshot (2026-10-28, after reading `plane_histograms.csv` schema version 4 with preprocessing `v0.3.0`): 728 tests passing on Python 3.12 (Windows) with `torch` installed, plus `test_torch_dataset.py::test_exported_patch_torch_dataset_filename_column_and_relative_path`, which fails on Windows only (it compares a `/` path with a `\` one). Earlier (schema version 3, preprocessing `v0.2.0`): 717 passing on Python 3.11 with `torch`. Earlier (2026-10-26): 685 passing, 4 skipped, on Python 3.11 and 3.13.

---

## 2. Non-negotiable rules

These rules override other implementation preferences.

* **Keep the core package coordinate-first.** The pipeline is: (1) generate
  patch regions (`PatchRegion` coordinates + metadata) via a strategy, (2)
  optionally filter regions (on raw values), (3) optionally extract arrays,
  (4) optionally normalize them (`config.normalization`, decision D9 of
  `docs/plan.md`), (5) optionally apply the label mask (cell mode only), (6)
  optionally resize, (7) optionally export. Strategies (`src/cellpatcher/strategies/`) only
  generate coordinates; extraction/export is handled by the pipeline.
* **Keep Torch/DINO/GPU/model-specific logic out of the core package.** The
  optional `cellpatcher.torch_dataset` adapter is the one sanctioned
  exception and must stay optional and model-agnostic.
* **Never rename or repurpose an existing `PatchRegion` field**
  (`patch_id`, `image_id`, `mode`, `condition`, `biological_repeat`, `image_file`,
  `mask_file`, `y0/y1/x0/x1`, `z0/z1`, `t`, `c`, `label_id`,
  `center_y`/`center_x`/`center_z`, `touches_border`, `area_px`,
  `patch_file`, `output_height`/`output_width`, `config_file`, `valid`,
  `reason`, `source_relative_path`, `object_set`). These are exactly the columns written to the exported metadata
  CSV (`save_metadata_csv` in `src/cellpatcher/export.py`, plus the
  computed `height`/`width` convenience columns from `PatchRegion.to_dict()`),
  and downstream repos may already depend on them. New fields may be
  *added* (as optional, default-`None` fields) without violating this rule;
  existing ones may not be renamed or have their meaning changed.
  (`biological_repeat` was renamed from `repeat` with explicit owner
  approval; `repeat` remains a deprecated alias for a transition — see §5.
  The *format* of `patch_id` was changed with explicit owner approval,
  decision D44 of 2026-10-26: it is now built by `cellpatcher.ids` from the
  FOV identity and the patch's own coordinates, never from run order — see
  §5. Its meaning, the patch's unique identifier, is unchanged.)
* **The MorphoBricks-wide `condition`/`biological_repeat`/`technical_repeat`/
  `image_file`/`mask_file`/`patch_file` CSV contract in §5 is now
  implemented** — `run_patching`/`run_patching_pipeline` populate these
  columns (see §5 for exactly how
  and what still requires explicit caller input). Any further change to
  these columns' meaning, or to how they're populated, is still a
  deliberate, user-approved change (see §8), same as any other shared
  column.
* **Never hard-code machine-specific paths.** Use config dataclasses
  (`CellPatcherConfig` and friends), function arguments, or environment
  variables.
* **Never break public interfaces** without explicit user approval:
  `run_patching`, `run_patching_pipeline`, `run_patching_condition_repeat_tree`,
  config dataclasses, `read_array`, `validate_image_and_mask_folders`,
  `discover_condition_repeat_folders`, `scan_cell_patch_size`,
  `prepare_image_for_patching`, `iter_image_planes_for_patching`. Other
  repos (and notebooks) import these.
* **Keep public pipeline functionality usable from three contexts:** Jupyter
  notebooks, CLI/script workflows, and as an imported pipeline dependency.
* **Do not copy backend logic into notebooks.** Notebooks
  (`notebooks/CellPatcher/*.ipynb`) must call package APIs (`run_patching`,
  config dataclasses, utility helpers) rather than duplicating patching
  logic inline.
* **Do not import from `references/legacy_patch_generation/`.** It is
  reference/provenance only, not production code.
* **Make the smallest coherent change needed for the task.** Do not perform
  unrelated refactors, dependency upgrades, renames, or formatting changes.
* **Run relevant tests after code changes.** For substantive changes, run
  `PYTHONPATH=src python -m pytest -q` before considering the task complete
  when the environment permits it.
* **Never claim a test or validation command passed unless it was actually
  run successfully.** If something could not be run, say so explicitly.
* **Don't hand-edit `requirements.txt`.** It is generated by
  `.tools/python/merge_requirements.py` from
  `notebooks/CellPatcher/requirements.yaml`; edit the source YAML and
  regenerate instead.

---

## 3. Coding standards

Follow PEP 8 and any formatter/linter configuration present in the repo.

Key rules:

* 4-space indentation
* Descriptive variable names; avoid single-letter names outside obvious loop
  counters
* Blank lines between functions and classes
* Docstrings on public modules, classes, and functions
* Follow the existing quote and formatting style of the repo
* Prefer simple, readable implementations over clever ones
* Comments should explain non-obvious reasoning, constraints, or scientific
  assumptions — do not narrate obvious code

Error handling:

* Raise exceptions with descriptive messages; never fail silently
* Prefer specific exception types over bare `Exception`
* When catching errors, explain what went wrong and, where useful, what the
  caller should do

Design principles:

* Functions should be modular and focused
* Minimise unnecessary coupling and unintended side effects between modules
* Reuse existing utilities and patterns before introducing new ones
* Minimise function complexity; if a function is difficult to reason about,
  split independent responsibilities
* Do not add abstractions for hypothetical future requirements

---

## 4. Repository structure

```text
src/
  cellpatcher/
    __init__.py
    module.py
    submodule/
      __init__.py
      xxx.py

tests/
  test_*.py             # mirrors src/cellpatcher modules
  fixtures/              # small, deterministic TIFF fixtures (see tests/fixtures/readme.md)

docs/
  README.md               # documentation map / entry point
  normalization.md        # user guide: normalizing patches with the preprocessing statistics
  quick_start.md, patching_modes.md, api_examples.md,
  preprocessing.md, configuration_reference.md, developer_notes.md
  plan.md                  # normalization design brief and decisions D1–D16
  normalization_contract.md  # normalization rules (specified; implemented over Milestones 1–5)
  assets/                  # doc images

notebooks/
  README.md
  notebook_latest_versions.yaml
  CellPatcher/
    CellPatcher.ipynb         
    requirements.yaml
    CHANGELOG.md

references/
  legacy_patch_generation/   # reference-only; DO NOT import from here

app/                # Colab / conda-constructor packaging (menuinst, installer scripts) — not pipeline code
.tools/               # CI/maintenance tooling (requirements merge, version bump, notebook sync, docs figures) — not pipeline code
  python/generate_docs_figures.py   # regenerates docs/assets/*.png; run automatically by .github/workflows/update_docs_figures.yml

README.md
setup.py
requirements.txt       # GENERATED — do not hand-edit, see .tools/python/merge_requirements.py
environment.yaml
```

CellPatcher-specific conventions:

* `image_axes`/`mask_axes` are explicit and axis-aware (see `axes.py`).
  Image/mask matching supports `exact`, `exact_or_suffix` (default), and
  `substring` (legacy, permissive).
* `scan_cell_patch_size` computes dataset-level patch-size recommendations.
* `ExtractionConfig.apply_mask` is cell-mode oriented and clears pixels
  outside selected labels before resize/export.
* Patch identity lives in `src/cellpatcher/ids.py` (decisions D44/D45,
  2026-10-26): `patch_id` = `position_id` + object set + plane indices +
  label or rectangle; `position_id` is always filled. Never derive an
  identity from `image_id` (the image's place in the run) or from list
  order; every run refuses duplicate `patch_id` values.
* Cell-mode masks are instance labels used as given (decision D47):
  boolean masks are refused unless `CellConfig.binary_mask_policy=
  "connected_components"` (warns; recorded in `run_config.json`); labels
  above the int32 maximum are refused. `cell_scan` shares the rule
  (`CellSegmentationStrategy.label_objects`).
* Strategies generate strictly 2D Y/X regions. Z/T stacks become planes
  either with `common.z_policy`/`t_policy` in the pipeline
  (`cellpatcher.planes`; default `"keep"` keeps all planes in each patch)
  or, for in-memory workflows, with the preprocessing helpers; C/Z/T
  handling is otherwise a separate
  preprocessing step (`preprocess.py`, `docs/preprocessing.md`), not a
  patch mode.
* Whenever `export.enabled=True`, `pipeline.py` also writes
  `run_config.json` (filename: `cellpatcher.RUN_CONFIG_FILENAME`) next to
  `patches.csv` — the full serialized `CellPatcherConfig` for the run, so
  every exported patch is deterministically reproducible from its CSV row
  (`config_file` column) plus that file (see §5).
* Every `patches.csv` has the per-image table `images.csv`
  (`cellpatcher.export.IMAGE_TABLE_FILENAME`) beside it, written by
  `export_patches` and by the combined tree output; it lists every input
  image, including those without patches (see §5).

Reusable pipeline logic should live in importable modules rather than only
in notebooks where practical.

---

## 5. Inter-repo interfaces

> This section defines the contract between this repo and the rest of the
> MorphoBricks ecosystem. Keep it accurate — other agents and developers
> rely on it.

### Inputs this repo consumes

CellPatcher reads image files, and (for `cell` mode) matching label/mask
files, directly from user-provided folders. Images are TIFF/PNG/JPEG/BMP/GIF/
NPY files and, with the `preprocessing` extra, OME-Zarr images: folders ending
in `.zarr`, read from level 0 of their only image, with their axes from the
OME-NGFF metadata put into `common.image_axes` by role (`io.read_image`);
discovery never enters them, and `img.ome.zarr` matches masks named like
`img.tif`. With `common.mask_source="ngff_labels"`, the masks of OME-Zarr
images are their own OME-NGFF label images (`common.mask_label_name`,
`io.read_ngff_mask`, contract §4) instead of a mask folder. Matching is
validated via
`validate_image_and_mask_folders`/`io.py`, using an `exact`,
`exact_or_suffix` (default), or `substring` strategy.

Two ways to point CellPatcher at input data:

- `run_patching_pipeline(image_folder_path, ...)` — a single flat folder of
  images (optionally with a matching mask folder). `condition`/
  `biological_repeat` must be set explicitly via
  `CellPatcherConfig.common.condition`/`.biological_repeat` when using this
  entry point; they are not parsed from the folder path.
- `run_patching_condition_repeat_tree(image_root_path, ...)` — the
  `<condition>/<biological_repeat_YYYY_MM_DD>[/<technical_repeat>]/` layout
  from §6, walked
  automatically via `discover_condition_repeat_folders`/`io.py` (both
  functions keep their historical `condition_repeat` names; the level they
  read below each condition is the biological repeat). Also accepts a
  condition folder holding images directly with **no biological-repeat
  subfolder**; that case is treated as a single implicit biological repeat,
  labeled `cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL` (`"single"`; the
  older name `SINGLE_REPEAT_LABEL` is the same constant), consistently
  through input discovery, processing (stamped as `common.biological_repeat`
  on every patch from that condition), and output (its own
  `<output_dir>/<condition>/single/` subfolder, same as an explicit
  biological repeat gets). A root folder holding images directly with
  **no condition subfolders** is likewise one implicit condition and
  biological repeat, labeled `cellpatcher.SINGLE_CONDITION_LABEL`/
  `SINGLE_BIOLOGICAL_REPEAT_LABEL` (both `"single"`), exported to
  `<output_dir>/single/single/`. A root or condition folder that mixes
  loose image files with subfolders is ambiguous and raises `ValueError`
  rather than guessing, and so does a biological-repeat folder (whose
  subfolders are technical repeats). Hidden subfolders (e.g. `.ipynb_checkpoints`) are
  ignored.

### Outputs this repo produces

- Patch coordinate/region metadata (`PatchRegion` fields, including the
  shared CSV contract columns below) as a CSV (`save_metadata_csv`, default
  filename `patches.csv`) via `export_patches` / `run_patching_pipeline`.
- Whenever export is enabled, a companion `run_config.json`
  (`cellpatcher.RUN_CONFIG_FILENAME`) with the full run config — every CSV
  row's `config_file` column names it — so a row plus that file fully
  determine how to reproduce the patch (coordinates, strategy parameters
  including `random.seed`, extraction/resize settings, axes).
- For `run_patching_condition_repeat_tree`, one `patches.csv` +
  `run_config.json` pair per
  `<condition>/<biological_repeat>[/<technical_repeat>]/` output folder by
  default, or with `combine_metadata=True` a single pair directly in
  `output_dir` (used by the CellPatcher notebook). In the combined CSV,
  `patch_file` is the path relative to the CSV
  (`<condition>/<biological_repeat>[/<technical_repeat>]/<file>`); in both
  layouts it is relative to the CSV's folder.
- Beside every `patches.csv` (same folder, same `save_metadata_csv`
  switch; per leaf, or once in `output_dir` with `combine_metadata=True`),
  a per-image table `images.csv` with **one row per input image the run
  considered, including images that gave no patch**: `condition`,
  `biological_repeat`, `technical_repeat`, `image_id`, `image_file`,
  `mask_file`, `source_relative_path` (the `patches.csv` names and values),
  `image_height`/`image_width`, `n_patches` (its rows in `patches.csv`),
  `status` (`processed` when `n_patches` > 0; `no_patches` when it was
  patched but none kept; `skipped` when it had no plane to patch),
  `reason` (empty for `processed`; otherwise the step that left no patch:
  the patch mode, the filters or normalization discards), then, appended
  later, `position_id` (the one its patches carry, also for an image
  without patches; always filled since decision D45) and
  `n_regions`/`n_kept`/`n_normalized` (counts after the patch mode, the
  filters and normalization). There is no
  `failed` status: a failing image stops the run with an error before any
  table is written. Every `patches.csv` row joins exactly one `images.csv`
  row on `condition`/`biological_repeat`/`technical_repeat`/`image_id`/
  `image_file`, with the same `position_id`. In memory: `PatchingResult.image_records`. Documented in
  `docs/configuration_reference.md` ("Per-image table").
- Per patch, the geometry a reader of exported arrays needs (appended after
  every earlier `patches.csv` column): `image_height`/`image_width` (the
  source image's Y/X size), `pad_top`/`pad_bottom`/`pad_left`/`pad_right`
  (constant padding per side), `resized` (whether `output_size` changed the
  shape) and `output_axes` (the array's axes; the files carry none).
  Documented in `docs/configuration_reference.md`.
- Optionally, exported patch arrays as `.npy` or `.tif`/`.tiff` files
  (`save_patch_array`), named by each row's `patch_file` column. With
  normalization they are float32 normalized values, or with
  `normalization.export_values="raw"` the raw values (source dtype, float32
  when resized or masked) to be normalized on load with the row's
  `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>` columns (or the
  `;`-separated `normalization_offset`/`normalization_scale` of tables
  written before 2026-10-28)
  (`cellpatcher.iter_exported_patches`, `ExportedPatchTorchDataset`).
- In memory: eager arrays (`common.extract_arrays=True`, within
  `normalization.memory_limit`) or one patch at a time
  (`cellpatcher.iter_patch_arrays`, `PatchTorchDataset`); every delivery mode
  gives the same arrays.
- Consumed downstream by feature-extraction repos — currently
  [fiberfeatures](https://github.com/CellMigrationLab/fiberfeatures/tree/main)
  (an early starting point, expected to change), and in future by other
  MorphoBricks feature-extraction repos.

### Planned: normalization with the preprocessing brick

**Status: implemented (Milestones 1–5; user guide `docs/normalization.md`).** The rules are fixed in
`docs/normalization_contract.md` (decisions in `docs/plan.md`) and land over
Milestones 1–5. Implemented so far (Milestone 1): the optional
`preprocessing` extra (pinned commit in `setup.py`), the
`source_relative_path` `PatchRegion` field/`patches.csv` column (last
column; filled by folder and tree runs from each image's path under the
root, a linked image or folder keeping the link's path as in the brick), and `cellpatcher.statistics`, which loads `plane_histograms.csv` and
the optional `run_manifest.json` and runs the checks of contract §5 — not
yet called by the pipeline. Milestone 2 adds plane splitting
(`common.z_policy`/`t_policy`), OME-Zarr images and OME-NGFF label masks
(`common.mask_source`) (see "Inputs" above). Milestone 3 adds
`config.normalization` (`cellpatcher.normalization`): the pipeline normalizes
patches per channel at a biological-repeat (across conditions, from the
preprocessing brick's `experiment_design.csv`), image or patch scope
(follow-up C2 removed the condition and technical-repeat scopes; follow-up C3
added `normalization.image_type="rgb"`, R, G and B sharing one offset and
scale), and records
the parameters in six `normalization_*` columns after `source_relative_path`
and in `run_config.json` (`normalization` section, `normalization_inputs`).
Until each later part is merged, the inputs and
outputs above are the whole interface. In summary:

- **Inputs:** with the optional `preprocessing` extra, CellPatcher reads
  `plane_histograms.csv` from
  [morphobricks-preprocessing](https://github.com/CellMigrationLab/morphobricks-preprocessing)
  (needed for the condition, biological-repeat, technical-repeat and image
  scopes) and, optionally, its `run_manifest.json` for extra checks. It still
  discovers images with its own folder rules. OME-Zarr images and their
  OME-NGFF `labels/` masks become readable.
- **Outputs:** optional columns appended to `patches.csv` —
  `source_relative_path`, four `normalization_*` columns and one
  `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>` pair per channel
  (CellPatcher follow-up C1, normalization brick N41/N42; in memory
  `PatchRegion.normalization_parameters`, the brick's `Parameters`) — plus a
  `normalization` config section, a `normalization_inputs` object (with the
  experiment design's name and SHA-256 at the `biological_repeat` scope) and
  `normalization_settings` (the brick's `NormalizationSettings.to_dict()`) in
  `run_config.json`.
- **Reuse:** histograms, pooling, fitting (`fit`, `Parameters.apply`,
  `to_columns`/`from_columns`) and settings are imported
  from `morphobricks_normalization` (`normalization.clip` stays in
  CellPatcher, because the brick never clips), image readers, manifests and messages
  from `morphobricks_preprocessing`; never reimplemented here.

### Shared CSV column contract

Any CSV produced or consumed by this repo that connects to another MorphoBricks
repo **must** include these columns with exactly these names:

| Column       | Type   | Required | Notes                                         |
| ------------ | ------ | -------- | --------------------------------------------- |
| `condition`  | string | Always   | Experimental group label (`cellpatcher.SINGLE_CONDITION_LABEL`/`"single"` when the root folder holds the images directly — see "Inputs" above) |
| `biological_repeat` | string | Always | Biological replicate ID (format: `YYYY_MM_DD`, or `cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL`/`"single"` for a condition with no biological-repeat subfolder — see "Inputs" above). Formerly `repeat` (see "Renamed column" below) |
| `technical_repeat` | string | Optional | Technical repeat within a biological repeat (a well, dish or coverslip); never a biological repeat, which stays in `biological_repeat`. `SINGLE_TECHNICAL_REPEAT_LABEL`/`"single"` when a biological repeat has no technical-repeat subfolders. Constant for every row of one source image |
| `position_id` | string | If needed | Stable spatial acquisition-position ID; all Z/C/T planes from one XY position share it |
| `image_file` | string | Always   | Source image filename; storage provenance, not necessarily the spatial sampling unit |
| `z` or `z0`/`z1` | int | If needed | 0-based Z plane (CellPatcher writes the half-open range `z0`/`z1`, one plane thick) |
| `c`          | int    | If needed | 0-based channel index |
| `t`          | int    | If needed | 0-based time index |
| `mask_file`  | string | If used  | Mask filename; omit column if not applicable  |
| `patch_file` | string | If used  | Patch filename; omit column if not applicable |

Do not silently change the meaning of an existing shared column.

**Folder hierarchy:** `experiment/condition/biological_repeat/technical_repeat/images`.
Levels may be omitted only from the bottom up, and a level's meaning comes
from its position, not its name; a missing level is the value `"single"`
(`SINGLE_CONDITION_LABEL`/`SINGLE_BIOLOGICAL_REPEAT_LABEL`/
`SINGLE_TECHNICAL_REPEAT_LABEL`). CellPatcher reads the `condition`,
`biological_repeat` and `technical_repeat` levels.

**Renamed column (`repeat` → `biological_repeat`):** owner-approved rename
of the `PatchRegion` field, the `CommonConfig` field, the `patches.csv`
column and the `run_config.json` key (also the `ConditionRepeatFolder`/
`ConditionRepeatResult` attribute); values and folder behaviour are
unchanged. For a transition, the old name still works with one
`DeprecationWarning` per use: `CommonConfig(repeat=...)`,
`PatchRegion(repeat=...)`, `config.common.repeat`/`patch.repeat`/
`leaf.repeat` reads and writes, `apply_overrides({"common.repeat": ...})`,
`load_config` of an older `run_config.json`, and `ExportedPatchTorchDataset`
on an older `patches.csv` with a `repeat` column. Output is always written
as `biological_repeat`. Downstream readers of CellPatcher CSVs should read
`biological_repeat` (and may fall back to `repeat` for older files).

**Spatial sampling rule:** one FOV means one independently selected XY acquisition position. `position_id` is the shared identifier when one file does not map 1:1 to one position (for example one TIFF per z-plane or a multi-position container). `image_file` remains file provenance. Z/C/T indices describe observations within a position and never create new spatial FOVs.

**Status for this repo: implemented.** All of these columns are real
`PatchRegion` fields (not just present when exporting) and are written to
`patches.csv` alongside CellPatcher's own richer, reconstruction-oriented
columns: `patch_id`, `image_id`, `mode`, `y0`/`y1`/`x0`/`x1`, `height`/
`width`, `z0`/`z1`, `t`, `c`, `label_id`, `center_y`/`center_x`/`center_z`,
`touches_border`, `area_px`, `output_height`/`output_width`, `config_file`,
`valid`, `reason` (documented in `src/cellpatcher/types.py` and
`docs/configuration_reference.md`). How each contract column is populated:

- `condition`/`biological_repeat` — either set
  `CellPatcherConfig.common.condition`/`.biological_repeat` explicitly and
  call `run_patching_pipeline`, or call `run_patching_condition_repeat_tree`
  on a `<condition>/<biological_repeat>/` root and let it derive both from
  the folder path per leaf (condition folders without a biological-repeat
  subfolder get `SINGLE_BIOLOGICAL_REPEAT_LABEL`; a flat root gets
  `SINGLE_CONDITION_LABEL` too). Never inferred from paths by
  `run_patching()` itself; a label not given (or `None`) is `"single"`
  in every entry point, as for a missing folder level, so these columns
  are never empty.
- `technical_repeat` — the well, dish or coverslip within the biological
  repeat. `run_patching_condition_repeat_tree` derives it from an optional
  `<condition>/<biological_repeat>/<technical_repeat>/` level (a
  biological-repeat folder that holds only subfolders) and stamps
  `SINGLE_TECHNICAL_REPEAT_LABEL` (`"single"`) when a biological-repeat
  folder holds its images directly, so it is
  never empty in a tree run; each technical repeat gets its own output
  folder. Otherwise set `CellPatcherConfig.common.technical_repeat`, or per
  image `image_metadata=[{"technical_repeat": ...}, ...]`. `"single"`
  when neither is set. MorphoCoverage treats the
  technical repeat as a sampling unit (it resamples wells before fields),
  so a well must never be written into `biological_repeat`.
- `image_file`/`mask_file` — populated automatically by
  `run_patching_pipeline` (and therefore `run_patching_condition_repeat_tree`,
  which calls it per leaf) from the folders it read. `None` for direct
  `run_patching()` calls unless the caller supplies them via
  `image_metadata=[{"image_file": ..., "mask_file": ...}, ...]`.
- `patch_file` — populated automatically by `export_patches`
  (`export.enabled=True`); `None` when a patch isn't exported to a file.
- `position_id` — always filled (owner decision D45, 2026-10-26), in every
  run type: taken from `image_metadata` when supplied (a user-given value
  wins); otherwise one file is one position, as
  `<condition>/<biological_repeat>/<technical_repeat>/<image file stem>`
  (`single/single/single/<stem>` when no labels are given; `io.image_stem`,
  so `img.ome.zarr` gives `img`), so every plane and channel patch of one
  image shares it and equal file names in two wells stay distinct. An
  in-memory image given to `run_patching` with neither a `position_id`
  nor an `image_file` is refused (owner decision, 2026-10-26): never a
  run-order value. A file holding several images or series (OME-Zarr plate,
  multi-series TIFF) is refused however it is read. The
  canonical MorphoBricks FOV key is `condition` + `biological_repeat` +
  `technical_repeat` + `position_id` (MorphoBricks decision D02, 2026-10-26).
- `patch_id` — run-independent (owner decision D44, 2026-10-26), built by
  `cellpatcher.ids`:
  `position_id "/" [object_set "/"] ["t"NNNN "/"] ["z"NNN "/"] ["c"N "/"] token`,
  token `label<label_id>` (cell), `grid-y<y0>-x<x0>-h<h>-w<w>` (grid),
  `random-y<y0>-x<x0>-h<h>-w<w>[-n<k>]` (random; `-n` numbers repeated
  draws of one rectangle) or `whole_fov`. Exported file names are
  `sanitize_filename(patch_id)` (`/` → `_`). Formerly
  `<image_id>_<mode>_<index>[_t<NNN>][_z<NNN>][_c<n>]`, which changed with
  the run. Random mode samples each image (plane) with
  `ids.random_seed(random.seed, identity)`: SHA-256 of
  `<position_id>|t=..|z=..|c=..` combined with the seed in a numpy
  `SeedSequence` (never Python's `hash()`, never the run index), so the
  rectangles do not depend on run order either.
- `object_set` — cell mode only: `image_metadata["object_set"]`, else
  `common.object_set`, else the mask's own name in folder/tree runs (the
  OME-NGFF label name, or the mask file stem's part after the image stem:
  `img_cells.tif` → `cells`), else `"mask"`. Appended as the last
  `patches.csv` column.

Mask/patch columns are always present (never omitted) but hold `None`/empty
when not applicable, rather than the column itself being dropped — simpler
for CSV consumers than a variable-width schema, and not a violation of "must
include these columns."

---

## 6. MorphoBricks pipeline (bigger picture)

MorphoBricks is a multi-repo image analysis system for microscopy data.

The full pipeline across all repos is:

```text
[Step 1] raw_data/ + masks/  →  **[THIS REPO] CellPatcher**  →  patches/
[Step 2] patches/            →  Feature extraction            →  features.csv
[Step 3] features.csv        →  Analysis & applications        →  results/
```

**Global input structure (raw data, produced outside MorphoBricks):**

```text
raw_data/
  <condition>/
    <biological_repeat_YYYY_MM_DD>/    # omit this level entirely for a
      <image_file>.tif                 # condition with only one biological
                                       # repeat -- CellPatcher then treats the
                                       # whole <condition>/ folder as that one
                                       # biological repeat ("single").
                                       # Omit <condition>/ too for a single-
                                       # condition experiment: images
                                       # directly in raw_data/.
      <technical_repeat>/              # optional: one folder per well/dish/
        <image_file>.tif               # coverslip of this biological repeat,
                                       # instead of images directly in the
                                       # biological-repeat folder (then
                                       # "single").

masks/                                 # optional
  <condition>/
    <biological_repeat_YYYY_MM_DD>/
      [<technical_repeat>/]
        <mask_file>.tif
```

CellPatcher's `run_patching_condition_repeat_tree` (§5) walks exactly this
layout, including the no-biological-repeat-subfolder, no-condition-subfolder
and technical-repeat-subfolder cases.

**Known repos in the ecosystem:**

| Step | Repo          | Description                                                          | Link                                                        |
| ---- | ------------- | --------------------------------------------------------------------- | ------------------------------------------------------------ |
| 0 (optional) | morphobricks-preprocessing | Folder and metadata validation, per-plane intensity histograms (`plane_histograms.csv`) and normalization methods; used by CellPatcher's planned normalization (§5) | https://github.com/CellMigrationLab/morphobricks-preprocessing |
| 1    | CellPatcher   | Image patching                                                       | https://github.com/CellMigrationLab/CellPatcher              |
| 2    | fiberfeatures | Feature extraction (starting point; many more feature-extraction repos expected over time) | https://github.com/CellMigrationLab/fiberfeatures/tree/main  |
| 3    | BioExplainer  | Downstream analysis / XAI (initial version; not yet updated to follow the standards in this document) | https://github.com/CellMigrationLab/BioExplainer             |

Note: `fiberfeatures` and `BioExplainer` are early/initial versions and are
not yet aligned to the shared CSV contract in §5 — do not assume interface
stability with them without checking their current state first. Add rows
here as new feature-extraction/analysis repos are created.

---

## 7. Glossary

| Term            | Definition                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------- |
| Condition       | An experimental group (e.g. `treated`, `control`). A `PatchRegion`/CSV column in CellPatcher, set via `CellPatcherConfig.common.condition` or discovered per folder by `run_patching_condition_repeat_tree` (§5) |
| Biological repeat | A biological replicate, identified by date: the shared column `biological_repeat` (formerly `repeat`, still accepted as a deprecated alias). Same status as Condition above (`common.biological_repeat`); see also Single repeat |
| Technical repeat | A well, dish or coverslip within a biological repeat: the optional shared column `technical_repeat` of the MorphoBricks contract (§5), derived from a `<condition>/<biological_repeat>/<technical_repeat>/` folder level or set via `common.technical_repeat`/`image_metadata`; `"single"` when a biological repeat has no such level |
| Single repeat    | A condition with no biological-repeat subfolder — `<condition>/<image files>` instead of `<condition>/<biological_repeat>/<image files>`. `run_patching_condition_repeat_tree` treats it as one implicit biological repeat, labeled `cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL` (`"single"`; also available as `SINGLE_REPEAT_LABEL`) (§5, §6). Likewise, images directly in the root folder are one condition labeled `cellpatcher.SINGLE_CONDITION_LABEL` (`"single"`) with a single biological repeat |
| Patch           | A sub-region cropped from a microscopy image                                                      |
| Mask            | Instance-label image marking objects (each non-zero integer value one object), used in `cell` mode; boolean masks need the `binary_mask_policy="connected_components"` opt-in |
| Feature         | A numeric descriptor extracted from a patch or image (produced downstream, e.g. by fiberfeatures)  |
| Patch mode      | One of `cell`, `random`, `grid`, `whole_fov` — determines how `PatchRegion` coordinates are generated |
| `PatchRegion`   | Coordinate-first descriptor of a single patch and its metadata (`src/cellpatcher/types.py`)        |
| `apply_mask`    | `ExtractionConfig` option that clears pixels outside the selected label before resize/export (cell mode) |
| `image_axes` / `mask_axes` | Explicit axis labels for images/masks, used by axis-aware matching and preprocessing (`axes.py`) |

---

## 8. When to ask before acting

Ask the user before:

* Renaming or restructuring any public function or class
* Adding a new required dependency
* Changing any output format, file structure, or `PatchRegion`/metadata CSV
  columns
* Changing the shared MorphoBricks CSV column contract in §5 (e.g. how
  `condition`/`biological_repeat`/`image_file`/`mask_file`/`patch_file` are populated
  or what they mean) — the contract itself is now implemented, but further
  changes to it still need approval like any other shared column
* Changing which notebook version (`CellPatcher.ipynb` / `_v2` / `_v3`) is
  presented as the preferred path
* Modifying anything in `__init__.py` that affects the public API
* Making changes that would require updates in other MorphoBricks repos
  (e.g. fiberfeatures, BioExplainer)
* Making a change that could materially alter scientific interpretation
* Performing destructive or difficult-to-reverse operations

Do not ask for permission to:

* Write or update tests
* Add appropriate docstrings or comments
* Fix clear bugs in private functions
* Improve error messages
* Run tests, linting, or other validation
* Make small internal changes clearly required by the requested task

For minor ambiguity, first inspect the repository, tests, documentation, and
similar existing implementations. Prefer the most conservative reversible
interpretation rather than stopping unnecessarily.

When finishing a substantive task, briefly report:

* what changed
* what tests or validation were actually run
* any important assumptions, limitations, or unresolved issues
