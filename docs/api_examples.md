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

Note: iterator `metadata` / `plane_id` values are available on yielded objects, but they are not automatically written into `PatchRegion` rows or export CSV metadata.

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
