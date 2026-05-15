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

## Core workflow

1. Validate/read inputs.
2. Configure explicit axes (`image_axes`, `mask_axes`).
3. Generate patch regions via strategy.
4. Optionally filter candidate regions.
5. Optionally extract arrays.
6. Optionally apply label mask (`apply_mask`).
7. Optionally resize extracted arrays.
8. Optionally export arrays and metadata CSV.

## Minimal example (cell mode)

```python
import numpy as np
from cellpatcher import CellPatcherConfig, run_patching, scan_cell_patch_size

# Example arrays
image = np.random.rand(256, 256).astype(np.float32)
labels = np.zeros((256, 256), dtype=np.int32)
labels[80:140, 100:170] = 1

config = CellPatcherConfig(mode="cell")
config.common.image_axes = "YX"
config.common.mask_axes = "YX"

summary = scan_cell_patch_size(
    image_shapes=[image.shape],
    labels_list=[labels],
    config=config.cell,
    image_axes=config.common.image_axes,
    mask_axes=config.common.mask_axes,
)
config.cell.patch_size = summary.recommended_size("p95") or summary.recommended_size("max")

config.extraction.apply_mask = True
config.extraction.output_size = (128, 128)

config.export.enabled = True
config.common.output_dir = "./patches_out"

result = run_patching(images=image, masks=labels, config=config)
print(len(result.patches), len(result.exported_files))
```

## Other mode examples (minimal)

- **Grid mode**: `CellPatcherConfig(mode="grid")`, set `config.grid.tile_size` and optional `config.grid.overlap`, then call `run_patching(images=...)`.
- **Random mode**: `CellPatcherConfig(mode="random")`, set `config.random.patch_size`, `config.random.n_patches`, optional seed/mask.
- **Whole FOV mode**: `CellPatcherConfig(mode="whole_fov")`, call `run_patching(images=...)` to generate one full-frame region per image.

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
