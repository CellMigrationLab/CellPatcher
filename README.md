# CellPatcher

CellPatcher is a package-backed microscopy patch generation project for scientists, notebook users, and Python users who want reproducible patch extraction workflows. It creates patch coordinates (and optional extracted arrays) from microscopy images using multiple strategies, including segmentation-based cell patches, random patches, grid tiles, and whole-field patches. You can run it through interactive notebooks or directly through the Python API, depending on your workflow.

## What CellPatcher does

- Matches image files and mask/label files with configurable matching strategies.
- Generates patch regions (`PatchRegion`) for `cell`, `random`, `grid`, and `whole_fov` modes.
- Optionally filters regions, extracts arrays, resizes outputs, and exports patch files.
- Optionally clears pixels outside the selected segmented object in cell mode (`apply_mask=True`).
- Writes patch metadata (CSV) alongside exported patch files when enabled.

## Which patching mode should I use?

| Mode | Use it when... | Needs masks? | Typical output |
|---|---|---|---|
| `cell` | You have segmentation labels and want one patch per object/cell. | Yes | One patch per labeled object/cell |
| `random` | You want random fixed-size sampling for controls or coverage checks. | Optional | Random fixed-size patches |
| `grid` | You want deterministic tiling over the image. | No | Regular tiles (with optional overlap/partials) |
| `whole_fov` | You want one patch representing the entire field of view. | No | One patch covering the full image |

## Workflow at a glance

```text
images/masks → CellPatcherConfig → run_patching → PatchRegion metadata + patch arrays and/or exported files
```

All four modes use the same orchestration function: `run_patching`. The notebooks build `CellPatcherConfig` for you and call the same backend API.


## Patching modes at a glance

![CellPatcher patching modes](docs/assets/patching_modes_overview.png)

## Quick start: notebooks

| Notebook | Best for | Status |
|---|---|---|
| `notebooks/CellPatcher/CellPatcher_v3.ipynb` | app-style UI, manual testing, most users | recommended |
| `notebooks/CellPatcher/CellPatcher_strategy_examples.ipynb` | learning the Python API and modes | recommended examples |
| `notebooks/CellPatcher/CellPatcher_v2.ipynb` | step-by-step guided workflow | fallback/reference |
| `notebooks/CellPatcher/CellPatcher.ipynb` | historical notebook | legacy/reference only |

V3 and strategy examples are the best starting points for most users. The legacy notebook is preserved for provenance and is not recommended for new work.

## Quick start: Python API

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
from cellpatcher import CellPatcherConfig, CellConfig, CommonConfig, ExtractionConfig, run_patching, scan_cell_patch_size

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
- `CellConfig(patch_size="auto")` is per-object/per-image (each cell gets its own square size), **not** dataset-wide fixed sizing.
- For one fixed size across the whole dataset, explicitly use `summary.recommended_size("max")` (largest included square size).
- Cell mode uses square patches, so dataset-wide fixed sizing should be a single int.
- `apply_mask=True` clears pixels outside each selected label before resize/export.

## Real TIFF demo fixture

A small real TIFF fixture is bundled for smoke tests and notebook demos:

- `tests/fixtures/cellpatcher_demo_image.tif`
- `tests/fixtures/cellpatcher_demo_labels.tif`

Use it for validation and examples, not for benchmarking or biological conclusions.

## Installation / local development

- From the repository root, use `PYTHONPATH=src` for development commands.
- If needed, install editable from source: `pip install -e .`.
- Notebook/runtime dependencies are pinned in `requirements.txt` and notebook-specific requirements files.

## Running tests

```bash
PYTHONPATH=src python -m pytest -q
```

Run the command above to verify the current test suite in your environment.

## Limitations / non-goals

- CellPatcher does **not** segment cells; you must provide masks/labels for cell mode.
- CellPatcher does **not** run DINO/Torch inference in the core package.
- Demo fixtures are for smoke testing, not biological benchmarking.
- Users must provide valid images and, for cell mode, masks/labels.

## Development notes

- Package code lives in `src/cellpatcher`.
- Notebooks should call package APIs and avoid duplicating backend patching logic.
- See `AGENTS.md` for contributor/agent guidance.
- See `notebooks/README.md` for notebook-specific guidance.


## More documentation

- [Patching mode guide](docs/patching_modes.md)
- [API examples](docs/api_examples.md)
- [Notebook guide](notebooks/README.md)

