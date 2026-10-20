# API examples

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

These examples are designed for quick copy/paste experimentation.
Assume `image` and `labels` are already loaded NumPy arrays.
If you're reading from disk, use `read_array(...)` for TIFF/NPY inputs.
All examples call `run_patching(...)`.

These examples assume:

- arrays are already loaded unless the example is about folder matching
- axis strings describe the array layout
- export examples require `common.output_dir`
- CellPatcher does not run segmentation or DINO/Torch inference

## Imports

```python
from cellpatcher import (
    CellPatcherConfig,
    CommonConfig,
    CellConfig,
    RandomConfig,
    GridConfig,
    FilterConfig,
    ExtractionConfig,
    ExportConfig,
    MatchConfig,
    validate_image_and_mask_folders,
    read_array,
    scan_cell_patch_size,
    run_patching,
)
```


## Everyday APIs vs advanced helpers

Start with these for almost everything:

- `CellPatcherConfig` and config dataclasses
- `run_patching`
- `run_patching_pipeline`
- `read_array` / `validate_image_and_mask_folders`
- `scan_cell_patch_size`
- `prepare_image_for_patching` / `iter_image_planes_for_patching`

Reach for these when you want lower-level control:

- `strategies`
- low-level `extract` / `resize` / `export` helpers
- axis utilities

## Example 1 — Cell mode with scan + `apply_mask`

A strong default when you want one patch per labeled object, dataset-guided sizing, and masked backgrounds.

```python
summary = scan_cell_patch_size(
    image_shapes=image.shape,
    labels_list=labels,
    image_axes="YX",
    mask_axes="YX",
)
patch_size = summary.recommended_size("p95") or summary.recommended_size("max")
# Note: if the dataset has no included objects, both methods return 0.
# Guard against this before passing patch_size to CellConfig.

config = CellPatcherConfig(
    mode="cell",
    common=CommonConfig(image_axes="YX", mask_axes="YX", extract_arrays=True),
    cell=CellConfig(patch_size=patch_size, exclude_border=True, min_area=5),
    extraction=ExtractionConfig(apply_mask=True, output_size=224),
)

result = run_patching(images=image, masks=labels, config=config)
```

## Example 2 — Grid mode export to `.npy`

Use this when you want deterministic tiling plus exported patch files and CSV metadata.

```python
config = CellPatcherConfig(
    mode="grid",
    common=CommonConfig(image_axes="YX", output_dir="./patches_out"),
    grid=GridConfig(tile_size=128, overlap=0.25, include_partial=True),
    export=ExportConfig(enabled=True, image_format="npy", save_metadata_csv=True),
)

result = run_patching(images=image, config=config)
print(result.metadata_file)
```

## Example 3 — Random mode with sampling mask

Use this for random control patches constrained to allowed regions.

```python
sampling_mask = labels > 0

config = CellPatcherConfig(
    mode="random",
    common=CommonConfig(image_axes="YX", mask_axes="YX", extract_arrays=True),
    random=RandomConfig(patch_size=64, n_patches=16, seed=42),
)

result = run_patching(images=image, masks=sampling_mask, config=config)
```

Random sampling is with replacement, so duplicate regions are possible.

## Example 4 — Whole-FOV resize

Use this to produce one resized patch per full field of view.

```python
config = CellPatcherConfig(
    mode="whole_fov",
    common=CommonConfig(image_axes="YX", extract_arrays=True),
    extraction=ExtractionConfig(output_size=224),
)

result = run_patching(images=image, config=config)
print(len(result.patches))  # 1
```

## Example 5 — Folder matching and loading

Use this pattern when images and masks live in separate folders and need reliable pairing.

```python
matcher = MatchConfig(strategy="exact_or_suffix", mask_suffixes=("_mask", "_labels"), case_sensitive=False)
# pairs is a dict[Path, Path]: {image_path: mask_path} for every matched pair
pairs = validate_image_and_mask_folders(
    image_folder_path="./images",
    mask_folder_path="./masks",
    strategy=matcher.strategy,
    mask_suffixes=matcher.mask_suffixes,
    case_sensitive=matcher.case_sensitive,
)

first_image_path, first_mask_path = next(iter(pairs.items()))
image = read_array(first_image_path)
labels = read_array(first_mask_path)
```

When using `run_patching_pipeline`, the `config.matching` field is used automatically — no need to pass matching parameters separately.

## Example 6 — Filtering background patches

Use filtering to suppress low-signal or background-heavy patches.

```python
config = CellPatcherConfig(
    mode="grid",
    common=CommonConfig(image_axes="YX"),
    grid=GridConfig(tile_size=128, overlap=0.0, include_partial=True),
    filters=FilterConfig(
        min_foreground_fraction=0.05,
        foreground_threshold="mean",
    ),
)

result = run_patching(images=image, config=config)
```

Here, `foreground_threshold="mean"` means foreground is defined relative to each patch's mean intensity.

`FilterConfig` evaluates patch image-intensity/foreground-threshold criteria; it does not use a label mask for grid-mode filtering.

## Preparing multidimensional images and labels

For arrays with C/Z/T axes, prepare planes explicitly before patching.

Patching modes are 2D Y/X; for C/Z/T data, see `preprocessing.md`.

```python
from cellpatcher import prepare_image_for_patching, prepare_labels_for_cell_patching

image_yx, image_axes = prepare_image_for_patching(
    image_zstack,
    "ZYX",
    z_policy="max",
)

labels_yx, mask_axes = prepare_labels_for_cell_patching(
    labels_zstack,
    "ZYX",
    z_policy="select",
    z_index=0,
)
```

```python
image_rgb, image_axes = prepare_image_for_patching(
    image,
    "YX",
    channel_policy="repeat_to_rgb",
    normalize="percentile",
)
```

This pattern is handy for DINO-style model inputs, while CellPatcher itself still does not run DINO/Torch inference.

## Patching every Z slice or T frame

Use this when you want to run standard 2D patching on every Z slice or T frame.

`cell`, `grid`, `random`, and `whole_fov` are still Y/X patching modes. If you want every Z slice or T frame, first expand data into prepared 2D planes with iterators.

Use:
- `iter_image_planes_for_patching`
- `iter_cell_planes_for_patching`

This is explicit preprocessing/iteration—not `mode="all"` and not true volumetric 3D patching.

Note: iterator `metadata` (`t`/`z` indices) is written into `PatchRegion` rows and export CSV metadata when passed through as `run_patching(..., image_metadata=[p.metadata for p in pairs])`; see `docs/preprocessing.md`. The `plane_id` string itself is not mirrored onto `PatchRegion` and still needs an explicit join.

```python
from cellpatcher import (
    CellPatcherConfig,
    CommonConfig,
    GridConfig,
    iter_image_planes_for_patching,
    run_patching,
)

planes = list(iter_image_planes_for_patching(image, "ZYX", z_policy="all"))
images = [p.array for p in planes]

config = CellPatcherConfig(mode="grid")
config.common = CommonConfig(image_axes=planes[0].axes, extract_arrays=True)
config.grid = GridConfig(tile_size=128)

result = run_patching(images=images, config=config)
```

```python
from cellpatcher import (
    CellConfig,
    CellPatcherConfig,
    CommonConfig,
    iter_cell_planes_for_patching,
    run_patching,
)

pairs = list(iter_cell_planes_for_patching(
    image,
    "ZYX",
    labels,
    "ZYX",
    z_policy="all",
))

images = [p.image for p in pairs]
labels_2d = [p.labels for p in pairs]

config = CellPatcherConfig(mode="cell")
config.common = CommonConfig(image_axes=pairs[0].image_axes, mask_axes="YX", extract_arrays=True)
config.cell = CellConfig(patch_size="auto")

result = run_patching(images=images, masks=labels_2d, config=config)
```


## Optional Torch dataset

Torch integration is optional. CellPatcher does not run DINO/Torch inference in core; this adapter only makes patch tensors easier to feed into user models.

- Import from `cellpatcher.torch_dataset`
- Dataset returns `C,H,W` tensors
- Use your own transform for ImageNet/DINO normalization
- `scale_policy="auto"` scales integer arrays by dtype max; it does not percentile-normalize or ImageNet-normalize data
- For non-singleton Z/T, preprocess first with `prepare_image_for_patching(...)`, `iter_image_planes_for_patching(...)`, or `iter_cell_planes_for_patching(...)`
- Keep `extra_metadata_by_image_id` values DataLoader-collate-friendly (for example strings, numbers, lists, or small dictionaries)

```python
from torch.utils.data import DataLoader
from cellpatcher.torch_dataset import PatchTorchDataset, PatchTensorConfig

config.common.extract_arrays = False
config.export.enabled = False
result = run_patching(images=image, masks=labels, config=config)

dataset = PatchTorchDataset(
    images=image,
    patches=result.patches,
    image_axes=config.common.image_axes,
    masks=labels,
    mask_axes=config.common.mask_axes,
    extraction=config.extraction,
    tensor_config=PatchTensorConfig(channel_policy="repeat_to_rgb"),
    transform=dinov2_transform,
)

loader = DataLoader(dataset, batch_size=64)
```

```python
from cellpatcher import iter_image_planes_for_patching

planes = list(iter_image_planes_for_patching(image, "ZYX", z_policy="all"))
images = [p.array for p in planes]
plane_lookup = {
    f"image_{i:04d}": {"plane_id": p.plane_id, **p.metadata}
    for i, p in enumerate(planes)
}

dataset = PatchTorchDataset(
    images=images,
    patches=result.patches,
    image_axes=planes[0].axes,
    extra_metadata_by_image_id=plane_lookup,
)
```


## Loading exported `.npy` patches with Torch

Use this pattern when patch extraction/export and model training/inference happen at different times or on different machines.

- Export with `image_format="npy"`
- Load with `ExportedPatchTorchDataset`
- `.npy` is preferred over TIFF for Torch loading
- Apply model-specific normalization in your transform
- DINO/Torch inference is not included in CellPatcher core

```python
from cellpatcher import CellPatcherConfig, CommonConfig, ExportConfig, GridConfig, run_patching
from cellpatcher.torch_dataset import ExportedPatchTorchDataset, PatchTensorConfig

config = CellPatcherConfig(mode="grid")
config.common = CommonConfig(
    image_axes="YX",
    output_dir="./patches_out",
    extract_arrays=False,
)
config.grid = GridConfig(tile_size=128)
config.export = ExportConfig(
    enabled=True,
    image_format="npy",
    save_metadata_csv=True,
)

result = run_patching(images=image, config=config)

dataset = ExportedPatchTorchDataset(
    result.metadata_file,
    axes=config.common.image_axes,
    tensor_config=PatchTensorConfig(channel_policy="repeat_to_rgb"),
    transform=dinov2_transform,
)
```

`patches.csv` includes a `patch_file` column with the exact exported filename for every row (written by `export_patches`), and `ExportedPatchTorchDataset` uses it by default. Older CSVs without that column, or rows where it's empty, fall back to reconstructing filenames from sanitized `patch_id` values. If your workflow uses custom filenames in a different column, pass `filename_column` to use that instead. When multiple formats exist for a patch and neither `patch_file` nor `filename_column` resolves it, pass `prefer_format="npy"` or `prefer_format="tif"`; if no preference is given and multiple formats exist, a `ValueError` is raised. Rows loaded from `patches.csv` are returned as strings in `sample["patch"]`; convert fields to numbers if your downstream code needs typed values. A `patches.csv` written before the `repeat` → `biological_repeat` rename (a `repeat` column, no `biological_repeat` column) is still accepted: its `repeat` values are returned under `biological_repeat`, with one `DeprecationWarning` per file (see `configuration_reference.md`, "Renamed: `repeat` → `biological_repeat`").
