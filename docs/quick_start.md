# Quick start

This page is the shortest path to running CellPatcher from Python, with practical examples you can copy/paste.

## Install / dev setup

```bash
pip install -e .
# or run with local source
PYTHONPATH=src python -m pytest -q
```

## Quick start examples

### Example A — cell mode with scan + `apply_mask`

```python
import numpy as np
from cellpatcher import (
    CellPatcherConfig,
    CommonConfig,
    CellConfig,
    ExtractionConfig,
    scan_cell_patch_size,
    run_patching,
)

# Example in-memory image + labels
image = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
labels = np.zeros((256, 256), dtype=np.uint8)
labels[60:140, 80:170] = 1
labels[150:220, 30:110] = 2

summary = scan_cell_patch_size(
    image_shapes=image.shape,
    labels_list=labels,
    image_axes="YX",
    mask_axes="YX",
)
patch_size = summary.recommended_size("p95") or summary.recommended_size("max")

cfg = CellPatcherConfig(
    mode="cell",
    common=CommonConfig(image_axes="YX", mask_axes="YX", extract_arrays=True),
    cell=CellConfig(patch_size=patch_size),
    extraction=ExtractionConfig(output_size=128, apply_mask=True),
)

result = run_patching(images=image, masks=labels, config=cfg)
print(len(result.patches), len(result.patch_arrays or []))
```

### Example B — grid mode

```python
import numpy as np
from cellpatcher import CellPatcherConfig, CommonConfig, GridConfig, ExtractionConfig, run_patching

image = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)

cfg = CellPatcherConfig(
    mode="grid",
    common=CommonConfig(image_axes="YX", extract_arrays=True),
    grid=GridConfig(tile_size=128, overlap=0.0, include_partial=False),
    extraction=ExtractionConfig(output_size=64),
)

result = run_patching(images=image, config=cfg)
print(len(result.patches), result.patch_arrays[0].shape)
```

### Example C — dataset-wide max cell box -> masked patches -> DINO size

If you want one patch per segmented cell, sized from the largest cell in the dataset, then resized for DINO-style inputs:

```python
from cellpatcher import (
    CellPatcherConfig,
    CellConfig,
    CommonConfig,
    ExtractionConfig,
    run_patching,
    scan_cell_patch_size,
)

# images: list[np.ndarray]
# labels: list[np.ndarray] with one label map per image

summary = scan_cell_patch_size(
    image_shapes=[img.shape for img in images],
    labels_list=labels,
    image_axes="YX",
    mask_axes="YX",
)

# Use largest included bbox dimensions (convert to square for cell patch mode)
patch_size = summary.recommended_size("max")

cfg = CellPatcherConfig(
    mode="cell",
    common=CommonConfig(image_axes="YX", mask_axes="YX", extract_arrays=True),
    cell=CellConfig(patch_size=patch_size, center_on="centroid"),
    extraction=ExtractionConfig(apply_mask=True, output_size=224),
)

result = run_patching(images=images, masks=labels, config=cfg)
# result.patch_arrays -> masked, resized arrays suitable for DINO ingestion
```

Notes:

- `scan_cell_patch_size` reports dataset-wide `max_bbox_height` and `max_bbox_width`.
- `CellConfig(patch_size="auto")` is per-object/per-image (each cell gets its own square size), not dataset-wide fixed sizing.
- For one fixed size across the whole dataset, use `summary.recommended_size("max")`.
- Cell mode uses square patches, so dataset-wide fixed sizing should be a single int.
- `apply_mask=True` clears pixels outside each selected label before resize/export.

## Next steps

- Patching mode guide: `docs/patching_modes.md`
- API examples: `docs/api_examples.md`
- Multidimensional preprocessing: `docs/preprocessing.md`
