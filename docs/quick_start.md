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

Use this as a quick decision matrix before choosing an example.

| Your input data | Typical settings / helper | What happens in CellPatcher | Typical downstream step |
|---|---|---|---|
| `YX` image (single-channel IF or grayscale) + `YX` labels | `common.image_axes="YX"`, `mask_axes="YX"` (cell mode), optional `apply_mask=True`, optional `output_size=...` | Direct 2D patching. In cell mode, one patch per selected label; optional masking clears outside-label pixels; optional resize sets final spatial size. | Optional tensor conversion and model-specific normalization/augmentation. |
| `YXC` or `CYX` image (multichannel; includes RGB) + `YX` labels | `common.image_axes="YXC"` or `"CYX"`, `mask_axes="YX"` (cell mode), optional `apply_mask=True` | Channels are preserved during extraction/masking/resize. Label mask is applied on Y/X and broadcast across channels as needed. | Keep channels, or apply explicit channel selection/projection/mapping before model input. |
| `ZYX` image stack + `ZYX` labels (cell) | `iter_cell_planes_for_patching(..., z_policy="all")` | Expands to one 2D plane pair per Z slice; each plane then follows normal 2D patch flow. | Aggregate plane-level outputs or feed per-plane patches to downstream pipeline. |
| `ZYX` image stack + `YX` labels (cell) | `iter_cell_planes_for_patching(..., z_policy="select"/"first"/"max")` | Resolves image Z according to policy; labels remain a single `YX` plane; then normal 2D cell patching runs. | Use chosen projected/selected image representation consistently in analysis/modeling. |
| `CZYX`/`ZCYX` image + `ZYX` labels (cell) | `iter_cell_planes_for_patching(...)` + explicit `z_policy`, optional `image_channel_policy` (`keep`, `select`, `first`, `max`, `mean`, `repeat_to_rgb`) | Explicitly resolves Z/T as requested and returns 2D planes with chosen channel handling, then standard 2D patching. | Optional model-facing channel mapping and normalization; keep preprocessing parameters in metadata for reproducibility. |
| `TCZYX`/`TZCYX` image + `TZYX` labels (cell) | `iter_cell_planes_for_patching(..., t_policy="all", z_policy="all")` | Expands over matching T and Z planes; each emitted 2D plane pair is patched independently. | Group outputs by `plane_id`/metadata for time- or depth-aware downstream analysis. |
| `TCZYX`/`TZCYX` image + `YX` labels (cell) | `prepare_image_for_patching(...)` with explicit `t_policy` + `z_policy`, then `run_patching(...)` | Reduces image to a single 2D representation (select/projection), then runs standard 2D patching against `YX` labels. | Useful when you want one canonical representation per field of view for training/export. |
| Any image layout without labels (`random`, `grid`, `whole_fov`) | `run_patching(...)` directly for `YX`/`CYX`; use `iter_image_planes_for_patching(...)` for Z/T expansion or `prepare_image_for_patching(...)` for projection/selection | Region generation is label-free; extraction/resize/export behave the same once a 2D plane is provided. | Same downstream tensor/model pipeline as above; choose channel/normalization strategy explicitly. |

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
# result.patch_arrays -> masked, resized arrays suitable for DINO ingestion.
# result.exported_files -> .npy patches + metadata CSV on disk.

torch_ds = ExportedPatchTorchDataset(
    patches=result.patches,
    root_dir=cfg.common.output_dir,
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
