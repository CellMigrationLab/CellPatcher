# CellPatcher

CellPatcher is a microscopy patch-generation project centered on a reusable Python backend package (`src/cellpatcher`) and notebook workflows.

## Project overview

CellPatcher generates patch coordinates and optional patch arrays for multiple sampling modes while keeping core logic in the package (not duplicated in notebooks).

## What CellPatcher does

- Generates patch regions (`PatchRegion`) from images and optional masks.
- Supports cell/object-centered, random, grid, and whole-field patching.
- Applies optional region filtering.
- Optionally extracts patch arrays, applies label masking, resizes, and exports patches + metadata.

## Supported patching modes

| Mode | Description | Masks |
|------|-------------|-------|
| `cell` | One patch per segmented object, centred on each cell | Required |
| `random` | Randomly sampled patches of a fixed size | Optional (restricts sample region) |
| `grid` | Regular non-overlapping (or overlapping) tiles | Not used |
| `whole_fov` | Single patch covering the entire field of view | Not used |

## Two entry points

CellPatcher exposes two complementary run functions:

| Function | When to use |
|----------|-------------|
| `run_patching_pipeline(folder, …)` | You have images (and masks) on disk in folders — the recommended starting point for scripts and simple one-off runs |
| `run_patching(images=[…], masks=[…], …)` | You already have arrays in memory — use inside interactive apps, custom loaders, and widget-driven notebooks |

Both return the same `PatchingResult` (regions, optional extracted arrays, exported file paths, metadata CSV path).

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
- If patch size is unknown, call `scan_cell_patch_size(image_shapes, labels_list, mask_axes=…)` first to get dataset-level recommendations, then set `config.cell.patch_size`.
- Enable `config.extraction.apply_mask = True` to zero-out pixels outside the selected cell label before resize/export.

## Image/mask filename matching

`run_patching_pipeline` automatically matches image and mask filenames. Three strategies are available:

| Strategy | Behaviour |
|----------|-----------|
| `exact_or_suffix` | Exact stem match **or** stem + suffix (default; recommended) |
| `exact` | Exact stem match only |
| `substring` | Legacy; may false-match names like `img1` and `img10_mask` |

The active strategy is controlled by `config.matching.strategy`.

---

## Configuration reference

For readability, the complete settings/options reference now lives in:

- [`docs/configuration_reference.md`](docs/configuration_reference.md)

It includes all fields, defaults, and allowed values for `CellPatcherConfig` and nested sections.

---

## Saving/export examples

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
