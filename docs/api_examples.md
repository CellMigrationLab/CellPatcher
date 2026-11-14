# API examples

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

These examples are designed for quick copy/paste experimentation.
Assume `image` and `labels` are already loaded NumPy arrays.
If you're reading from disk, use `read_array(...)` for TIFF/NPY inputs.
All examples call `run_patching(...)`. An in-memory image needs an identity: pass `image_metadata=[{"position_id": ...}]`, one entry per image (an image without a `position_id` or `image_file` is refused; see "Patch identity" in `configuration_reference.md`).

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

result = run_patching(images=image, masks=labels, config=config, image_metadata=[{"position_id": "fov1"}])
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

result = run_patching(images=image, config=config, image_metadata=[{"position_id": "fov1"}])
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

result = run_patching(images=image, masks=sampling_mask, config=config, image_metadata=[{"position_id": "fov1"}])
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

result = run_patching(images=image, config=config, image_metadata=[{"position_id": "fov1"}])
print(len(result.patches))  # 1
```

## Example 5 — Folder matching and loading

Use this pattern to list the image/mask pairs of a preprocessing run. The pairing is the preprocessing run's (its `find_masks`; since v0.2.0 CellPatcher pairs nothing itself): masks are TIFF files named `<image name>_<mask type>.tif` or `<mask type>/<image name>.tif` (see [`config.matching`](configuration_reference.md)).

```python
# ./masks holds img_01_cells.tif, img_01_nuclei.tif, ...: choose the type.
# pairs is a dict[Path, Path]: {image_path: mask_path} for every image of the run
pairs = validate_image_and_mask_folders(
    image_folder_path="./images",
    mask_folder_path="./masks",
    preprocessing_run="./images_MorphoBricks_Preprocessing",
    mask_type="cells",
)

first_image_path, first_mask_path = next(iter(pairs.items()))
image = read_array(first_image_path)
labels = read_array(first_mask_path)
```

When using `run_patching_pipeline`, set `config.matching.mask_type` instead (`None` when the mask folder holds a single type).

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

result = run_patching(images=image, config=config, image_metadata=[{"position_id": "fov1"}])
```

Here, `foreground_threshold="mean"` means foreground is defined relative to each patch's mean intensity.

`FilterConfig` evaluates patch image-intensity/foreground-threshold criteria; it does not use a label mask for grid-mode filtering.

## Preparing multidimensional images and labels

For arrays with C/Z/T axes, prepare planes explicitly before patching.

Patching modes are 2D Y/X; for C/Z/T data, see `preprocessing.md`.

```python
from cellpatcher import prepare_image_for_patching, prepare_labels_for_cell_patching

# An image with more than one Z plane is refused (issue #83): a Z projection is a recorded step of
# morphobricks-preprocessing. Here one frame of a time series is chosen.
image_yx, image_axes = prepare_image_for_patching(
    image_series,
    "TYX",
    t_policy="select",
    t_index=0,
)

labels_yx, mask_axes = prepare_labels_for_cell_patching(  # one label layer: a mask is never picked by C index
    labels_series,
    "TYX",
    t_policy="select",
    t_index=0,
)
```

```python
image_rgb, image_axes = prepare_image_for_patching(
    image,
    "YX",
    channel_policy="repeat_to_rgb",
)
```

(`normalize="percentile"` used to normalize here too; it was removed in v0.1.1: use `config.normalization`, which normalizes the patches with the preprocessing brick's statistics and records the parameters.)

This pattern is handy for DINO-style model inputs, while CellPatcher itself still does not run DINO/Torch inference.

## Patching every T frame

Use this when you want to run standard 2D patching on every T frame. An image with more than one Z plane is refused (issue #83).

`cell`, `grid`, `random`, and `whole_fov` are still Y/X patching modes. If you want every T frame, first expand data into prepared 2D planes with iterators.

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

planes = list(iter_image_planes_for_patching(image, "TYX", t_policy="all"))
images = [p.array for p in planes]

config = CellPatcherConfig(mode="grid")
config.common = CommonConfig(image_axes=planes[0].axes, extract_arrays=True)
config.grid = GridConfig(tile_size=128)

result = run_patching(images=images, config=config, image_metadata=[{**p.metadata, "position_id": "fov1"} for p in planes])
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
    "TYX",
    labels,
    "TYX",
    t_policy="all",
))

images = [p.image for p in pairs]
labels_2d = [p.labels for p in pairs]

config = CellPatcherConfig(mode="cell")
config.common = CommonConfig(image_axes=pairs[0].image_axes, mask_axes="YX", extract_arrays=True)
config.cell = CellConfig(patch_size="auto")

result = run_patching(images=images, masks=labels_2d, config=config, image_metadata=[{**p.metadata, "position_id": "fov1"} for p in pairs])
```


## Optional Torch dataset

Torch integration is optional. CellPatcher does not run DINO/Torch inference in core; this adapter only makes patch tensors easier to feed into user models.

- Import from `cellpatcher.torch_dataset`
- Dataset returns `C,H,W` tensors
- `tensor_config=PatchTensorConfig(scale_policy=...)` is required (since v0.2.0): `scale_policy` has no default, and an omitted one is a `TypeError` naming the choices
- `scale_policy="none"`, the only choice, keeps the values; normalize patches with `config.normalization`. The former `scale_policy="auto"` (divide an integer patch by its dtype maximum) was removed and is refused
- `channel_policy` defaults to `"keep"`, the channels as exported; `"repeat_to_rgb"` (the default before v0.2.0) repeats a single channel to three for RGB models, and each item records it (`channel_repeat`: 3 when repeated, else 1)
- Use your own transform for ImageNet/DINO normalization
- For a non-singleton T (a Z stack is refused, issue #83), preprocess first with `prepare_image_for_patching(...)`, `iter_image_planes_for_patching(...)`, or `iter_cell_planes_for_patching(...)`
- Keep `extra_metadata_by_image_id` values DataLoader-collate-friendly (for example strings, numbers, lists, or small dictionaries)

```python
from torch.utils.data import DataLoader
from cellpatcher.torch_dataset import PatchTorchDataset, PatchTensorConfig

config.common.extract_arrays = False
config.export.enabled = False
result = run_patching(images=image, masks=labels, config=config, image_metadata=[{"position_id": "fov1"}])

dataset = PatchTorchDataset(
    images=image,
    patches=result.patches,
    image_axes=config.common.image_axes,
    masks=labels,
    mask_axes=config.common.mask_axes,
    extraction=config.extraction,
    # RGB model: repeat a single channel to three; values as normalized by config.normalization.
    tensor_config=PatchTensorConfig(channel_policy="repeat_to_rgb", scale_policy="none"),
    transform=dinov2_transform,
)

loader = DataLoader(dataset, batch_size=64)
```

```python
from cellpatcher import iter_image_planes_for_patching

planes = list(iter_image_planes_for_patching(image, "TYX", t_policy="all"))
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
    tensor_config=PatchTensorConfig(scale_policy="none"),
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

result = run_patching(images=image, config=config, image_metadata=[{"position_id": "fov1"}])

dataset = ExportedPatchTorchDataset(
    result.metadata_file,
    axes=config.common.image_axes,
    tensor_config=PatchTensorConfig(channel_policy="repeat_to_rgb", scale_policy="none"),
    transform=dinov2_transform,
)
```

`patches.csv` includes a `patch_file` column with the exact exported filename for every row (written by `export_patches`), and `ExportedPatchTorchDataset` uses it by default. The dataset refuses a `patches.csv` missing any column CellPatcher writes (as `read_patch_table` does); rows whose `patch_file` is empty fall back to reconstructing filenames from sanitized `patch_id` values. If your workflow uses custom filenames in a different column, pass `filename_column` to use that instead. When multiple formats exist for a patch and neither `patch_file` nor `filename_column` resolves it, pass `prefer_format="npy"` or `prefer_format="tif"`; if no preference is given and multiple formats exist, a `ValueError` is raised. Rows loaded from `patches.csv` are returned as strings in `sample["patch"]`; convert fields to numbers if your downstream code needs typed values. A `patches.csv` written before the `repeat` → `biological_repeat` rename (a `repeat` column) is refused: export the patches again.


## Patch geometry for other bricks: checks and execution windows

MorphoFeature hands CellPatcher's patches to the feature bricks as `bounds = (y0, y1, x0, x1)`:
half-open (`pixel_edge` in MorphoBricks-Spec ADR 0004's coordinate types; pixel `i` spans
`[i, i+1)`) in the frame of the 2D plane they lie in, with the plane's `(H, W)` shape. These pure
helpers check and grow such rectangles; they read no pixel and never convert a value.

- `check_patch_bounds(bounds, shape)` returns the rectangle as plain ints when it is four integers,
  non-empty and inside the plane; `check_region_mask(mask, shape)` checks a region mask of the plane.
  Both raise `PatchBoundsError` (a `ValueError`).
- `context_bounds(bounds, margin, shape)` gives the **execution window** of a patch (ADR 0004,
  decisions EW4/EW5): the patch grown by `margin = (my, mx)` pixels per axis, clipped at the plane
  edge (nothing is padded), as a frozen `WindowRecord`:

  | Field | Value |
  | --- | --- |
  | `context_bounds` | `(max(0, y0 - my), min(H, y1 + my), max(0, x0 - mx), min(W, x1 + mx))`, half-open, plane frame |
  | `origin` | `(y, x)` of the window in the plane: `(context_bounds[0], context_bounds[2])` |
  | `margin_requested` | `(my, my, mx, mx)` (top, bottom, left, right) |
  | `margin_clipped` | the margin kept per side; equal to the requested one except where the plane edge cuts it |
  | `local_interior` | the patch in window-local coordinates: `bounds` minus `origin` per axis |

  Every argument holds Python or NumPy integers: a float (even `1.0`) or boolean is a `TypeError`;
  a negative margin, a shape that is not positive, or bounds that are not a non-empty rectangle inside
  the plane are a `PatchBoundsError`. The patch keeps its global `bounds` and `patch_id`; a provider
  converts a window-local coordinate `l` back to the plane with `origin + l`.

```python
from cellpatcher import context_bounds

window = context_bounds((0, 256, 900, 1000), (67, 30), (300, 1000))
window.context_bounds    # (0, 300, 870, 1000)
window.origin            # (0, 870)
window.margin_requested  # (67, 67, 30, 30)
window.margin_clipped    # (0, 44, 30, 0): the top and right sides reach the plane edge
window.local_interior    # (0, 256, 30, 130)
# plane[0:300, 870:1000] is the window; window_array[0:256, 30:130] is the patch itself.
```
