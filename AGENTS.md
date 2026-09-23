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

Key dependencies: `numpy`, `scipy`, `tifffile`, `imageio` (core, required). Optional extras: `ipywidgets`, `matplotlib`, `pillow`, `tqdm`, `joblib`, `jl-hidecode` (notebook extra); `torch` (optional `cellpatcher.torch_dataset` adapter only — never a core dependency).

Current health snapshot (latest audit): 467 tests passing.

---

## 2. Non-negotiable rules

These rules override other implementation preferences.

* **Keep the core package coordinate-first.** The pipeline is: (1) generate
  patch regions (`PatchRegion` coordinates + metadata) via a strategy, (2)
  optionally filter regions, (3) optionally extract arrays, (4) optionally
  apply the label mask (cell mode only), (5) optionally resize, (6)
  optionally export. Strategies (`src/cellpatcher/strategies/`) only
  generate coordinates; extraction/export is handled by the pipeline.
* **Keep Torch/DINO/GPU/model-specific logic out of the core package.** The
  optional `cellpatcher.torch_dataset` adapter is the one sanctioned
  exception and must stay optional and model-agnostic.
* **Never rename or repurpose an existing `PatchRegion` field**
  (`patch_id`, `image_id`, `mode`, `condition`, `repeat`, `image_file`,
  `mask_file`, `y0/y1/x0/x1`, `z0/z1`, `t`, `c`, `label_id`,
  `center_y`/`center_x`/`center_z`, `touches_border`, `area_px`,
  `patch_file`, `output_height`/`output_width`, `config_file`, `valid`,
  `reason`). These are exactly the columns written to the exported metadata
  CSV (`save_metadata_csv` in `src/cellpatcher/export.py`, plus the
  computed `height`/`width` convenience columns from `PatchRegion.to_dict()`),
  and downstream repos may already depend on them. New fields may be
  *added* (as optional, default-`None` fields) without violating this rule;
  existing ones may not be renamed or have their meaning changed.
* **The MorphoBricks-wide `condition`/`repeat`/`image_file`/`mask_file`/
  `patch_file` CSV contract in §5 is now implemented** — `run_patching`/
  `run_patching_pipeline` populate all five columns (see §5 for exactly how
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
  quick_start.md, patching_modes.md, api_examples.md,
  preprocessing.md, configuration_reference.md, developer_notes.md
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
* Patching itself is strictly 2D Y/X; C/Z/T handling is a separate
  preprocessing step (`preprocess.py`, `docs/preprocessing.md`), not a
  patch mode.
* Whenever `export.enabled=True`, `pipeline.py` also writes
  `run_config.json` (filename: `cellpatcher.RUN_CONFIG_FILENAME`) next to
  `patches.csv` — the full serialized `CellPatcherConfig` for the run, so
  every exported patch is deterministically reproducible from its CSV row
  (`config_file` column) plus that file (see §5).

Reusable pipeline logic should live in importable modules rather than only
in notebooks where practical.

---

## 5. Inter-repo interfaces

> This section defines the contract between this repo and the rest of the
> MorphoBricks ecosystem. Keep it accurate — other agents and developers
> rely on it.

### Inputs this repo consumes

CellPatcher reads image files, and (for `cell` mode) matching label/mask
files, directly from user-provided folders. Matching is validated via
`validate_image_and_mask_folders`/`io.py`, using an `exact`,
`exact_or_suffix` (default), or `substring` strategy.

Two ways to point CellPatcher at input data:

- `run_patching_pipeline(image_folder_path, ...)` — a single flat folder of
  images (optionally with a matching mask folder). `condition`/`repeat`
  must be set explicitly via `CellPatcherConfig.common.condition`/`.repeat`
  when using this entry point; they are not parsed from the folder path.
- `run_patching_condition_repeat_tree(image_root_path, ...)` — the
  `<condition>/<repeat_YYYY_MM_DD>/` layout from §6, walked automatically
  via `discover_condition_repeat_folders`/`io.py`. Also accepts a condition
  folder holding images directly with **no repeat subfolder**; that case is
  treated as a single implicit repeat, labeled
  `cellpatcher.SINGLE_REPEAT_LABEL` (`"single"`), consistently through
  input discovery, processing (stamped as `common.repeat` on every patch
  from that condition), and output (its own `<output_dir>/<condition>/single/`
  subfolder, same as an explicit repeat gets). A condition folder that mixes
  loose image files with repeat subfolders is ambiguous and raises
  `ValueError` rather than guessing.

### Outputs this repo produces

- Patch coordinate/region metadata (`PatchRegion` fields, including the
  shared CSV contract columns below) as a CSV (`save_metadata_csv`, default
  filename `patches.csv`) via `export_patches` / `run_patching_pipeline`.
- Whenever export is enabled, a companion `run_config.json`
  (`cellpatcher.RUN_CONFIG_FILENAME`) with the full run config — every CSV
  row's `config_file` column names it — so a row plus that file fully
  determine how to reproduce the patch (coordinates, strategy parameters
  including `random.seed`, extraction/resize settings, axes).
- Optionally, exported patch arrays as `.npy` or `.tif`/`.tiff` files
  (`save_patch_array`), named by each row's `patch_file` column.
- Consumed downstream by feature-extraction repos — currently
  [fiberfeatures](https://github.com/CellMigrationLab/fiberfeatures/tree/main)
  (an early starting point, expected to change), and in future by other
  MorphoBricks feature-extraction repos.

### Shared CSV column contract

Any CSV produced or consumed by this repo that connects to another MorphoBricks
repo **must** include these columns with exactly these names:

| Column       | Type   | Required | Notes                                         |
| ------------ | ------ | -------- | --------------------------------------------- |
| `condition`  | string | Always   | Experimental group label                      |
| `repeat`     | string | Always   | Replicate ID (format: `YYYY_MM_DD`, or `cellpatcher.SINGLE_REPEAT_LABEL`/`"single"` for a condition with no repeat subfolder — see "Inputs" above) |
| `image_file` | string | Always   | Source image filename                         |
| `mask_file`  | string | If used  | Mask filename; omit column if not applicable  |
| `patch_file` | string | If used  | Patch filename; omit column if not applicable |

Do not silently change the meaning of an existing shared column.

**Status for this repo: implemented.** All five columns are real
`PatchRegion` fields (not just present when exporting) and are written to
`patches.csv` alongside CellPatcher's own richer, reconstruction-oriented
columns: `patch_id`, `image_id`, `mode`, `y0`/`y1`/`x0`/`x1`, `height`/
`width`, `z0`/`z1`, `t`, `c`, `label_id`, `center_y`/`center_x`/`center_z`,
`touches_border`, `area_px`, `output_height`/`output_width`, `config_file`,
`valid`, `reason` (documented in `src/cellpatcher/types.py` and
`docs/configuration_reference.md`). How each contract column is populated:

- `condition`/`repeat` — either set `CellPatcherConfig.common.condition`/
  `.repeat` explicitly and call `run_patching_pipeline`, or call
  `run_patching_condition_repeat_tree` on a `<condition>/<repeat>/` root and
  let it derive both from the folder path per leaf (single-repeat condition
  folders get `SINGLE_REPEAT_LABEL`). Never inferred by `run_patching()`
  itself.
- `image_file`/`mask_file` — populated automatically by
  `run_patching_pipeline` (and therefore `run_patching_condition_repeat_tree`,
  which calls it per leaf) from the folders it read. `None` for direct
  `run_patching()` calls unless the caller supplies them via
  `image_metadata=[{"image_file": ..., "mask_file": ...}, ...]`.
- `patch_file` — populated automatically by `export_patches`
  (`export.enabled=True`); `None` when a patch isn't exported to a file.

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
    <repeat_YYYY_MM_DD>/        # omit this level entirely for a
      <image_file>.tif          # condition with only one repeat --
                                 # CellPatcher then treats the whole
                                 # <condition>/ folder as that one repeat.

masks/                          # optional
  <condition>/
    <repeat_YYYY_MM_DD>/
      <mask_file>.tif
```

CellPatcher's `run_patching_condition_repeat_tree` (§5) walks exactly this
layout, including the no-repeat-subfolder case.

**Known repos in the ecosystem:**

| Step | Repo          | Description                                                          | Link                                                        |
| ---- | ------------- | --------------------------------------------------------------------- | ------------------------------------------------------------ |
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
| Repeat          | A biological or technical replicate, identified by date. Same status as Condition above (`common.repeat`); see also Single repeat |
| Single repeat    | A condition with no repeat subfolder — `<condition>/<image files>` instead of `<condition>/<repeat>/<image files>`. `run_patching_condition_repeat_tree` treats it as one implicit repeat, labeled `cellpatcher.SINGLE_REPEAT_LABEL` (`"single"`) (§5, §6) |
| Patch           | A sub-region cropped from a microscopy image                                                      |
| Mask            | Binary/label image marking regions of interest (e.g. cell outlines), used in `cell` mode           |
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
  `condition`/`repeat`/`image_file`/`mask_file`/`patch_file` are populated
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
