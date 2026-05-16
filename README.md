# CellPatcher

CellPatcher is a microscopy patch-generation project centered on a reusable Python backend package (`src/cellpatcher`) and notebook workflows.

## Table of contents

- [Project overview](#project-overview)
- [Installation](#installation)
- [Supported patching modes](#supported-patching-modes)
- [Two entry points](#two-entry-points)
- [Quickstart: folder-based pipeline](#quickstart-folder-based-pipeline)
- [Using a config file](#using-a-config-file)
- [Parameter sweeps with `apply_overrides`](#parameter-sweeps-with-apply_overrides)
- [In-memory entry point: `run_patching`](#in-memory-entry-point-run_patching)
- [Cell mode — additional notes](#cell-mode--additional-notes)
- [Foreground filtering](#foreground-filtering)
- [Extracting and resizing patch arrays](#extracting-and-resizing-patch-arrays)
- [Saving and export examples](#saving-and-export-examples)
- [Image/mask filename matching](#imagemask-filename-matching)
- [Understanding `PatchingResult`](#understanding-patchingresult)
- [Supported file formats](#supported-file-formats)
- [Configuration reference](#configuration-reference)
- [Notebooks](#notebooks)
- [Testing](#testing)

---

## Project overview

CellPatcher generates patch coordinates and optional patch arrays for multiple sampling modes while keeping core logic in the package (not duplicated in notebooks).

## What CellPatcher does

- Generates patch regions (`PatchRegion`) from images and optional masks.
- Supports cell/object-centered, random, grid, and whole-field patching.
- Applies optional region filtering (intensity, foreground fraction).
- Optionally extracts patch arrays, applies label masking, resizes, and exports patches + metadata.

---

## Installation

```bash
pip install -e .   # editable install from the repository root
```

Optional dependencies (install as needed):

| Dependency | When needed |
|------------|-------------|
| `tifffile` | Reading/writing TIFF files |
| `imageio` | Reading/writing PNG, JPEG, BMP, GIF |
| `scipy` | Resize with interpolation order > 0 |
| `scikit-image` | `foreground_threshold="otsu"` filter |

---

## Supported patching modes

| Mode | Description | Masks |
|------|-------------|-------|
| `cell` | One patch per segmented object, centred on each cell | Required |
| `random` | Randomly sampled patches of a fixed size | Optional (restricts sample region) |
| `grid` | Regular non-overlapping (or overlapping) tiles | Not used |
| `whole_fov` | Single patch covering the entire field of view | Not used |

---

## Two entry points

CellPatcher exposes two complementary run functions:

| Function | When to use |
|----------|-------------|
| `run_patching_pipeline(folder, …)` | You have images (and masks) on disk in folders — the recommended starting point for scripts and simple one-off runs |
| `run_patching(images=[…], masks=[…], …)` | You already have arrays in memory — use inside interactive apps, custom loaders, and widget-driven notebooks |

Both return the same `PatchingResult` (regions, optional extracted arrays, exported file paths, metadata CSV path).

`run_patching_from_folders` is an alias for `run_patching_pipeline` provided for backwards compatibility.

---

## Quickstart: folder-based pipeline

Build a config, set attributes directly, then call `run_patching_pipeline`.

### Grid patches

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

cfg = CellPatcherConfig(mode="grid")
cfg.grid.tile_size = 128
cfg.grid.overlap = 0.1
cfg.common.image_axes = "YX"
cfg.common.output_dir = "./patches_out"
cfg.export.enabled = True

result = run_patching_pipeline("./images", config=cfg)
print(f"Regions:        {len(result.patches)}")
print(f"Exported files: {len(result.exported_files)}")
print(f"Metadata CSV:   {result.metadata_file}")
```

### Cell patches (requires masks)

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

cfg = CellPatcherConfig(mode="cell")
cfg.cell.patch_size = 128          # or "auto" for per-cell adaptive size
cfg.common.image_axes = "YX"
cfg.common.mask_axes  = "YX"
cfg.common.output_dir = "./cell_patches"
cfg.export.enabled = True

result = run_patching_pipeline(
    "./images",
    mask_folder_path="./masks",
    config=cfg,
)
```

### Random patches

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

cfg = CellPatcherConfig(mode="random")
cfg.random.patch_size = 96
cfg.random.n_patches  = 200
cfg.random.seed       = 42

result = run_patching_pipeline("./images", config=cfg)
```

### Whole field-of-view patches

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

cfg = CellPatcherConfig(mode="whole_fov")
result = run_patching_pipeline("./images", config=cfg)
```

---

## Using a config file

Generate a JSON template, edit it in any text editor, then load and run:

```python
from cellpatcher import save_config_for_mode, load_config, run_patching_pipeline

# Write a clean template for the chosen mode
save_config_for_mode("grid", "./grid_config.json")

# Edit grid_config.json, then load it back
cfg = load_config("./grid_config.json")
result = run_patching_pipeline("./images", config=cfg)
```

`save_default_config(path)` writes a template with every section; `save_config_for_mode(mode, path)` writes only the sections relevant to the requested mode.

You can also start from a fresh default config using `default_config()`:

```python
from cellpatcher import default_config, run_patching_pipeline

cfg = default_config()   # equivalent to CellPatcherConfig()
cfg.mode = "grid"
cfg.grid.tile_size = 64
cfg.common.output_dir = "./patches"
cfg.export.enabled = True

result = run_patching_pipeline("./images", config=cfg)
```

---

## Parameter sweeps with `apply_overrides`

For sweeps where a shared base config must stay immutable, use `apply_overrides` to derive independent copies:

```python
from cellpatcher import CellPatcherConfig, apply_overrides, run_patching_pipeline

base = CellPatcherConfig(mode="grid")
for tile_size in [64, 128, 256]:
    cfg = apply_overrides(base, {
        "grid.tile_size":       tile_size,
        "common.output_dir":    f"./patches_{tile_size}",
    })
    run_patching_pipeline("./images", config=cfg)
```

The `overrides` keyword on `run_patching_pipeline` provides the same inline shorthand:

```python
run_patching_pipeline(
    "./images",
    config=base,
    overrides={"grid.tile_size": 64, "common.output_dir": "./patches_64"},
)
```

---

## In-memory entry point: `run_patching`

Use `run_patching` when you already have arrays (e.g. inside an interactive app or custom loader):

```python
import numpy as np
from cellpatcher import CellPatcherConfig, run_patching

image = np.random.randint(0, 256, (512, 512), dtype=np.uint16)

cfg = CellPatcherConfig(mode="grid")
cfg.grid.tile_size = 64
cfg.common.extract_arrays = True   # return patch arrays in result

result = run_patching(images=[image], config=cfg)
print(f"Patches: {len(result.patches)}, arrays: {len(result.patch_arrays)}")
```

---

## Cell mode — additional notes

- Set both `config.common.image_axes` and `config.common.mask_axes` (e.g. `"CYX"`, `"YX"`).
- If patch size is unknown, call `scan_cell_patch_size()` first to get dataset-level recommendations, then set `config.cell.patch_size`.
- Enable `config.extraction.apply_mask = True` to zero-out pixels outside the selected cell label before resize/export.

### Determining patch size with `scan_cell_patch_size`

```python
import numpy as np
from cellpatcher import CellPatcherConfig, run_patching_pipeline, scan_cell_patch_size

# Gather your images and label masks (as arrays)
images = [...]   # list of image arrays
masks  = [...]   # list of corresponding label arrays

# Scan all objects to get bounding-box statistics
summary = scan_cell_patch_size(
    image_shapes=[img.shape for img in images],
    labels_list=masks,
    mask_axes="YX",
)
print(f"Objects included: {summary.n_included}")
print(f"Max bounding box: {summary.max_bbox_height} x {summary.max_bbox_width}")
print(f"Recommended size (95th percentile): {summary.recommended_size('p95')}")

# Use the recommended size
cfg = CellPatcherConfig(mode="cell")
cfg.cell.patch_size = summary.recommended_size("p95")
cfg.common.image_axes = "YX"
cfg.common.mask_axes  = "YX"

result = run_patching(images=images, masks=masks, config=cfg)
```

`recommended_size(method)` accepts: `"max"`, `"median"`, `"mean"`, `"p90"`, `"p95"`.

### Masking out background pixels (`apply_mask`)

```python
from cellpatcher import CellPatcherConfig, run_patching

cfg = CellPatcherConfig(mode="cell")
cfg.cell.patch_size = 128
cfg.common.extract_arrays = True
cfg.extraction.apply_mask = True       # zero out pixels outside the cell
cfg.extraction.mask_fill_value = 0.0   # fill value for background pixels

result = run_patching(images=images, masks=masks, config=cfg)
# Each array in result.patch_arrays has background pixels zeroed out
```

---

## Foreground filtering

Post-generation filters discard low-quality patches based on intensity statistics. All filter fields default to `None` (disabled).

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

cfg = CellPatcherConfig(mode="grid")
cfg.grid.tile_size = 128

# Keep only patches where at least 20 % of pixels are above the Otsu threshold
cfg.filters.foreground_threshold  = "otsu"   # also: "mean", "median", "percentile:75"
cfg.filters.min_foreground_fraction = 0.2

# Optionally also filter by mean intensity
cfg.filters.min_mean_intensity = 50.0

result = run_patching_pipeline("./images", config=cfg)
```

Supported `foreground_threshold` strings: `"mean"`, `"median"`, `"otsu"` (requires `scikit-image`), `"percentile:<N>"` (e.g. `"percentile:75"`). A numeric value is also accepted.

---

## Extracting and resizing patch arrays

By default, `run_patching` / `run_patching_pipeline` only returns coordinate regions. Enable array extraction and optional resizing:

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

cfg = CellPatcherConfig(mode="grid")
cfg.grid.tile_size = 128
cfg.common.extract_arrays = True      # extract arrays into result.patch_arrays

# Resize every patch to 64 × 64 (requires scipy for order > 0)
cfg.extraction.output_size  = 64      # or (64, 96) for non-square
cfg.extraction.resize_order = 1       # bilinear interpolation (0 = nearest)

# Pad patches near the border with zeros instead of clipping
cfg.extraction.pad_mode        = "constant"
cfg.extraction.constant_values = 0.0

result = run_patching_pipeline("./images", config=cfg)
# result.patch_arrays[i].shape → (64, 64) for a YX image
```

---

## Saving and export examples

### Save extracted patches to TIFF + metadata CSV

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

cfg = CellPatcherConfig(mode="grid")
cfg.grid.tile_size = 128
cfg.common.image_axes = "YX"
cfg.common.output_dir = "./patches_tiff"
cfg.common.extract_arrays = True

cfg.export.enabled = True
cfg.export.image_format = "tiff"          # "tif" and "tiff" are both supported
cfg.export.save_metadata_csv = True

result = run_patching_pipeline("./images", config=cfg)
print(f"Exported files: {len(result.exported_files)}")
print(f"Metadata CSV:   {result.metadata_file}")
```

### Save patches as NumPy arrays

```python
cfg.export.image_format = "npy"   # writes .npy files instead of TIFF
```

### Keep results in memory only (no file export)

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

cfg = CellPatcherConfig(mode="grid")
cfg.grid.tile_size = 128
cfg.common.image_axes = "YX"
cfg.common.extract_arrays = True

cfg.export.enabled = False                 # no files written
cfg.common.output_dir = None               # optional, keeps run in-memory

result = run_patching_pipeline("./images", config=cfg)
print(f"Patches: {len(result.patches)}")
print(f"Arrays:  {len(result.patch_arrays)}")
print(f"Files:   {len(result.exported_files)}")
```

---

## Image/mask filename matching

`run_patching_pipeline` automatically matches image and mask filenames. Three strategies are available:

| Strategy | Behaviour |
|----------|-----------|
| `exact_or_suffix` | Exact stem match **or** stem + suffix (default; recommended) |
| `exact` | Exact stem match only |
| `substring` | Legacy; may false-match names like `img1` and `img10_mask` |

The active strategy and suffix list are controlled by `config.matching`:

```python
cfg.matching.strategy       = "exact_or_suffix"
cfg.matching.mask_suffixes  = ("_mask", "_labels", "_seg")
cfg.matching.case_sensitive = False
```

> `config.matching` applies to `mode="cell"` and `mode="random"` only (the only modes that load masks from disk). It is ignored for `grid` and `whole_fov`.

---

## Understanding `PatchingResult`

Both `run_patching` and `run_patching_pipeline` return a `PatchingResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `patches` | `list[PatchRegion]` | All accepted patch regions. Always populated. |
| `patch_arrays` | `list \| None` | Extracted arrays when `common.extract_arrays=True`; otherwise `None`. |
| `exported_files` | `list[str]` | Paths to written patch files when `export.enabled=True`; otherwise an empty list. |
| `metadata_file` | `str \| None` | Path to the metadata CSV, or `None`. |

Each `PatchRegion` carries: `patch_id`, `image_id`, `mode`, `y0`, `y1`, `x0`, `x1`, `label_id` (cell mode), `center_y`, `center_x`, `touches_border`, `area_px`, and computed properties `height` / `width`. See the [configuration reference](docs/configuration_reference.md#patchregion-fields) for the full field list.

---

## Supported file formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| TIFF | `.tif`, `.tiff` | Requires `tifffile` (preferred) or `imageio` |
| NumPy | `.npy` | No extra dependencies |
| PNG | `.png` | Requires `imageio` |
| JPEG | `.jpg`, `.jpeg` | Requires `imageio` |
| BMP | `.bmp` | Requires `imageio` |
| GIF | `.gif` | Requires `imageio` |

---

## Configuration reference

For the complete settings/options reference see:

- [`docs/configuration_reference.md`](docs/configuration_reference.md)

It documents all fields, defaults, allowed values, supported threshold strings, mode-to-section mapping, and the `PatchRegion` field list.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/CellPatcher/CellPatcher.ipynb` | Legacy notebook — preserved as reference |
| `notebooks/CellPatcher/CellPatcher_v2.ipynb` | Step-by-step widget notebook backed by the `cellpatcher` package |
| `notebooks/CellPatcher/CellPatcher_v3.ipynb` | App-style widget notebook (`CellPatcherApp`) with a separate explicit run/QC cell |

Both v2 and v3 call `run_patching` directly with pre-loaded arrays; folder scanning, mask matching, and file loading are handled by the notebook UI before the run call.

## Testing

```bash
PYTHONPATH=src python -m pytest -q
```

## Real TIFF smoke-test fixture

CellPatcher includes a bundled real TIFF smoke-test fixture:
- `tests/fixtures/cellpatcher_demo_image.tif`
- `tests/fixtures/cellpatcher_demo_labels.tif`

Use this fixture for deterministic pytest smoke tests and quick notebook smoke testing only. It is **not** intended for benchmarking, model comparison, or biological conclusions.

`notebooks/CellPatcher/CellPatcher_strategy_examples.ipynb` can optionally load this fixture (`USE_REAL_TIFF_FIXTURE=True`).

`notebooks/CellPatcher/CellPatcher_v3.ipynb` includes a **Load bundled TIFF demo fixture** button for fast manual app testing.
