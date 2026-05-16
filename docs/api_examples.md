# API examples

These examples assume `image` and `labels` are already loaded NumPy arrays.
Use `read_array(...)` for TIFF/NPY inputs when loading from disk.
All examples use `run_patching(...)`.

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

## Example 1 — Cell mode with scan + `apply_mask`

```python
summary = scan_cell_patch_size(
    image_shapes=image.shape,
    labels_list=labels,
    image_axes="YX",
    mask_axes="YX",
)
patch_size = summary.recommended_size("p95") or summary.recommended_size("max")

config = CellPatcherConfig(
    mode="cell",
    common=CommonConfig(image_axes="YX", mask_axes="YX", extract_arrays=True),
    cell=CellConfig(patch_size=patch_size, exclude_border=True, min_area=5),
    extraction=ExtractionConfig(apply_mask=True, output_size=224),
)

result = run_patching(images=image, masks=labels, config=config)
```

## Example 2 — Grid mode export to `.npy`

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

## Example 6 — Filtering background patches

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

result = run_patching(images=image, masks=(labels > 0), config=config)
```

Here, `foreground_threshold="mean"` means foreground is defined relative to each patch's mean intensity.
