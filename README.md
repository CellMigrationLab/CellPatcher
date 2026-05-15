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

- `cell`
- `random`
- `grid`
- `whole_fov`

## API pipeline (minimal, folder-based, orchestrated)

Use this sequence for a simple and correct end-to-end run:

1. Build a `CellPatcherConfig` and set `mode` + axes.
2. Enable export and set `output_dir` (optional).
3. Call `run_patching_pipeline(...)`.
4. Use `PatchingResult` for regions and exported file paths.

### Example: create grid patches from an images folder

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

config = CellPatcherConfig(mode="grid")
config.common.image_axes = "YX"
config.grid.tile_size = 128
config.grid.overlap = 0.0

config.export.enabled = True
config.common.output_dir = "./patches_out"

result = run_patching_pipeline("./images", config=config)
print(f"Generated regions: {len(result.patches)}")
print(f"Exported files: {len(result.exported_files)}")
print(f"Metadata CSV: {result.metadata_file}")
```

### Built-in orchestration helper

Use `run_patching_pipeline(...)` when you want a minimal folder-based entry point. It reads arrays, validates image↔mask matching for `cell` mode, and calls `run_patching` for you.

```python
from cellpatcher import CellPatcherConfig, run_patching_pipeline

# grid/random/whole_fov mode: images folder only
grid_config = CellPatcherConfig(mode="grid")
grid_config.grid.tile_size = 96
grid_result = run_patching_pipeline("./images", config=grid_config)

# cell mode: images + masks folders
cell_config = CellPatcherConfig(mode="cell")
cell_config.cell.patch_size = 128
cell_result = run_patching_pipeline(
    "./images",
    mask_folder_path="./masks",
    config=cell_config,
)
```



### Editable default config file

You can generate a default JSON config, edit it, then load and run:

```python
from cellpatcher import load_config, run_patching_pipeline, save_default_config

save_default_config("./cellpatcher_config.json")
config = load_config("./cellpatcher_config.json")
result = run_patching_pipeline("./images", config=config)
```

## Cell mode pipeline notes (images + masks)

For `mode="cell"`, provide both images and label masks in the same order:

- `run_patching(images=[...], masks=[...], config=...)`
- or use `run_patching_pipeline("./images", mask_folder_path="./masks", config=...)`
- set both `config.common.image_axes` and `config.common.mask_axes`
- optionally enable `config.extraction.apply_mask = True` to keep only selected label pixels within each extracted patch

If patch size is unknown, use `scan_cell_patch_size(...)` first, then set `config.cell.patch_size`.

## Image/mask matching notes

Matching helpers support:
- `exact`
- `exact_or_suffix` (default; recommended)
- `substring` (legacy; can false-match names like `img1` and `img10_mask`)

## `apply_mask` notes

`ExtractionConfig.apply_mask` is intended for cell-label patches and clears pixels outside the selected object label before resize/export.

## Notebook status

- Legacy notebook: `notebooks/CellPatcher/CellPatcher.ipynb`
- Planned package-backed notebook v2: `notebooks/CellPatcher/CellPatcher_v2.ipynb` (not yet created)

## Testing

```bash
PYTHONPATH=src python -m pytest -q
```
