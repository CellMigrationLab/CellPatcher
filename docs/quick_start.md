# Quick start

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

This page is the shortest path to running CellPatcher from Python, with practical examples you can copy/paste.

## Install / dev setup

```bash
pip install -e .
# or run with local source
PYTHONPATH=src python -m pytest -q
```

## Data-shape guide (what happens to my data?)

CellPatcher processes data in four stages. Use the tables below as a quick decision matrix.

---

### Stage 1: Preprocessing — getting to 2D

Patching modes work on 2D Y/X arrays. If your data has Z, T, or C axes, resolve them explicitly first.

| Your input | Goal | Helper to use | Result |
|---|---|---|---|
| `YX` image (grayscale / single-channel) | Already 2D | Use directly — no helper needed | `YX` array |
| `CYX` or `YXC` image (multichannel, including RGB) | Keep all channels | Set `image_axes="CYX"` or `"YXC"` directly | `CYX`/`YXC` array (channels preserved through extraction/masking/resize) |
| `ZYX` stack | Single projected plane | `prepare_image_for_patching(image, "ZYX", z_policy="max")` | `YX` array (max-projection) |
| `ZYX` stack | One run per Z slice | `iter_image_planes_for_patching(image, "ZYX", z_policy="all")` | Sequence of `YX` arrays |
| `ZYX` image + `ZYX` labels (cell mode) | Patch every Z slice with matched labels | `iter_cell_planes_for_patching(image, "ZYX", labels, "ZYX", z_policy="all")` | Sequence of matched 2D pairs |
| `ZYX` image + `YX` labels (cell mode) | Reduce image Z, keep 2D labels | `iter_cell_planes_for_patching(image, "ZYX", labels, "YX", z_policy="select"/"first"/"max")` | Single 2D pair |
| `CZYX` image + `ZYX` labels (cell mode) | Expand Z, preserve channels | `iter_cell_planes_for_patching(image, "CZYX", labels, "ZYX", z_policy="all")` | Sequence of `CYX`/`YX` pairs |
| `TCZYX` image + `TZYX` labels (cell mode) | Expand T and Z | `iter_cell_planes_for_patching(image, "TCZYX", labels, "TZYX", t_policy="all", z_policy="all")` | Sequence of matched pairs |
| Any layout without labels (`random`, `grid`, `whole_fov`) | Projection or per-slice expansion | `prepare_image_for_patching(...)` for projection; `iter_image_planes_for_patching(...)` for per-slice | `YX` array or sequence of `YX` arrays |

> **Principle:** CellPatcher never silently projects Z/T axes, merges channels, or normalizes intensities. All preprocessing is explicit.

See [Preprocessing](preprocessing.md) for full details, policy options, and examples.

---

### Stage 2: Patching — how regions are generated

Once you have a 2D image (and labels for cell mode), choose a patching mode.

| You want | Mode | Key config |
|---|---|---|
| One patch per segmented cell/object | `cell` | `CellConfig(patch_size=...)`, optionally `ExtractionConfig(apply_mask=True)` |
| Fixed-size random patches (controls, random coverage) | `random` | `RandomConfig(patch_size=64, n_patches=100)` |
| Deterministic full-image tiling | `grid` | `GridConfig(tile_size=128, overlap=0.0)` |
| One patch per full field of view | `whole_fov` | `ExtractionConfig(output_size=...)` for a fixed resize |

**Cell mode sizing tip:** use `scan_cell_patch_size(...)` to get a dataset-wide recommended fixed patch size. `patch_size="auto"` gives each cell its own adaptive size based on its bounding box — not a single shared size.

See [Patching modes](patching_modes.md) for mode details, figures, and common pitfalls.

---

### Stage 3: Patch processing and filtering

After regions are generated, configure extraction-time processing (mask clearing + resize) and optional quality/control filters.

| You want | Config | Behavior |
|---|---|---|
| Keep native extracted patch size | `ExtractionConfig(output_size=None)` | Cropped patch arrays are returned/exported without resize |
| Resize patches to a fixed model-ready size | `ExtractionConfig(output_size=224)` | Each extracted patch is resized to the requested square size |
| Clear background pixels outside each selected cell label (cell mode) | `ExtractionConfig(apply_mask=True)` | Non-cell pixels are zeroed before resize/export |
| Keep only cells meeting area constraints (cell mode) | `CellConfig(min_area=..., max_area=...)` | Objects outside the area range are excluded before patching |
| Keep only patches meeting intensity/content constraints | `FilterConfig(min_mean_intensity=..., min_foreground_fraction=...)` | Exclude low-information patches using intensity-based rules |
| Combine area constraints + intensity filters | `CellConfig(min_area=..., max_area=...)` + `FilterConfig(...)` | Apply both object-level and patch-level filtering before final outputs |

> **Tip:** apply filtering before export/training to reduce storage and avoid downstream sampling from low-quality patches.

See [Config reference](configuration_reference.md) for full filtering options and thresholds.

---

### Stage 4: Export and consumption

After patching, choose how to use the results.

| Goal | Config | Output / API |
|---|---|---|
| In-memory arrays only | `common.extract_arrays=True` | `result.patch_arrays` |
| Patch files + CSV metadata on disk (`.npy`, recommended for Torch) | `ExportConfig(enabled=True, image_format="npy", save_metadata_csv=True)` + `common.output_dir=...` | `.npy` files + `patches.csv` in `output_dir` |
| Patch files + CSV metadata on disk (TIFF) | `ExportConfig(enabled=True, image_format="tif")` + `common.output_dir=...` | `.tif` files + `patches.csv` |
| Live Torch dataset (no disk round-trip) | `common.extract_arrays=False`, no export | `PatchTorchDataset(images, result.patches, ...)` |
| Torch dataset from exported files (decoupled export + training) | Export to disk first, then load | `ExportedPatchTorchDataset(result.metadata_file, axes=..., tensor_config=...)` |

> **DINO/model recommendation:** keep CellPatcher responsible for coordinates, masking, resize, and export; apply channel merge and model normalization in your Torch transform. This decouples patch generation from model experiments so you can iterate channel and normalization strategies without regenerating patches.

See [API examples](api_examples.md) for end-to-end patterns and Torch dataset usage.

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

### Example C — dataset-wide max cell box -> masked patches -> DINO size (with preprocessing + Torch-compatible export)

If you want one patch per segmented cell, sized from the largest cell in the dataset, then resized for DINO-style inputs with explicit preprocessing and export that can be loaded by Torch:

```python
from cellpatcher import (
    CellPatcherConfig,
    CellConfig,
    CommonConfig,
    ExportConfig,
    ExtractionConfig,
    iter_cell_planes_for_patching,
    run_patching,
    scan_cell_patch_size,
)
from cellpatcher.torch_dataset import ExportedPatchTorchDataset, PatchTensorConfig

# raw_images/raw_labels can be C/Z/T-aware arrays loaded from your pipeline.
# Use explicit preprocessing to iterate 2D YX planes for cell patching.
prepared_images = []
prepared_labels = []
for raw_image, raw_label in zip(raw_images, raw_labels):
    for item in iter_cell_planes_for_patching(
        raw_image,
        "CZYX",   # image_axes (adjust to your real source axes)
        raw_label,
        "ZYX",    # label_axes (adjust to your real label axes)
        z_policy="all",
    ):
        prepared_images.append(item.image)
        prepared_labels.append(item.labels)

summary = scan_cell_patch_size(
    image_shapes=[img.shape for img in prepared_images],
    labels_list=prepared_labels,
    image_axes="YX",
    mask_axes="YX",
)

# Use largest included bbox dimensions (convert to square for cell patch mode)
patch_size = summary.recommended_size("max")

cfg = CellPatcherConfig(
    mode="cell",
    common=CommonConfig(
        image_axes="YX",
        mask_axes="YX",
        extract_arrays=True,
        output_dir="outputs/cell_dino_patches",
    ),
    cell=CellConfig(patch_size=patch_size, center_on="centroid"),
    extraction=ExtractionConfig(apply_mask=True, output_size=224),
    export=ExportConfig(enabled=True, image_format="npy", save_metadata_csv=True),
)

result = run_patching(images=prepared_images, masks=prepared_labels, config=cfg)
# result.patch_arrays  -> masked, resized arrays in memory.
# result.exported_files -> .npy patch files written to output_dir.
# result.metadata_file  -> path to patches.csv in output_dir.

torch_ds = ExportedPatchTorchDataset(
    result.metadata_file,   # path to patches.csv; root_dir defaults to metadata_file.parent
    axes="YX",
    tensor_config=PatchTensorConfig(channel_policy="repeat_to_rgb"),
)
sample = torch_ds[0]  # CHW tensor compatible with common DINO pipelines
```

Notes:

- `scan_cell_patch_size` reports dataset-wide `max_bbox_height` and `max_bbox_width`.
- `CellConfig(patch_size="auto")` is per-object/per-image (each cell gets its own square size), not dataset-wide fixed sizing.
- For one fixed size across the whole dataset, use `summary.recommended_size("max")`.
- Cell mode uses square patches, so dataset-wide fixed sizing should be a single int.
- `apply_mask=True` clears pixels outside each selected label before resize/export.
- Patching remains 2D `Y/X`; for C/Z/T data, use explicit preprocessing helpers first.
- `ExportConfig(... image_format="npy")` works well with `ExportedPatchTorchDataset` for Torch ingestion.

#### DINO preprocessing policy (explicit recommendation)

For model-facing pipelines (for example DINO/DINOv2), we recommend splitting responsibilities:

- **CellPatcher stage (patch generation):** coordinate generation, optional cell masking, resize, and optional `.npy` export.
- **Torch stage (model input):** channel merge strategy (for example 6-channel Cell Painting -> 3-channel RGB) and model normalization.

This keeps core patch generation model-agnostic and lets you iterate on channel/normalization strategy without regenerating patches.

Practical guidance:

- `PatchTensorConfig(channel_policy="repeat_to_rgb")` supports only 1- or 3-channel patches.
- For >3 channels (for example Cell Painting), implement an explicit 6->3 mapping in your Torch transform.
- Apply DINO/ImageNet-style normalization in that same transform stage.
- `scale_policy="auto"` in `PatchTensorConfig` is only dtype-based integer scaling, not model normalization.

Visual QA tip (recommended):

- Keep an intermediate exported/inspectable artifact for validation (for example masked/resized `.npy` patches from CellPatcher).
- Add a small debug utility in your training/inference code that saves a few post-transform RGB tensors to PNG so you can visually verify channel merge + normalization outputs.

## Next steps

- Patching mode guide: `patching_modes.md`
- API examples: `api_examples.md`
- Multidimensional preprocessing: `preprocessing.md`
