# Multidimensional preprocessing (optional)

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

CellPatcher generates 2D Y/X patch regions. If your image has Z or T axes, choose your lane up front: select/project a single 2D plane before patching, or expand the stack into multiple 2D planes and run the same patching mode on each plane.

**In folder and tree runs** (and `run_patching`), set `config.common.z_policy`/`t_policy` instead of preparing the arrays yourself: `"all"` patches every plane, `"first"`/`"select"` one plane, and for Z `"max"`/`"mean"`/`"sum"` a projection; the default `"keep"` keeps every plane in each patch. Masks follow the image's planes or are shared (see the [config reference](configuration_reference.md#commonconfig)). The helpers below stay available for in-memory workflows.

## Core idea

- Core patching modes (`cell`, `random`, `grid`, `whole_fov`) are 2D Y/X region generators.
- Preprocessing is explicit and optional.
- CellPatcher does **not** silently project Z/T, merge channels, or normalize intensities. To normalize
  patches, enable `config.normalization` (see [Normalization](normalization.md)).

## Quick policy matrix

| Data situation | Recommended helper | Why |
|---|---|---|
| 2D `YX` image | Use directly | Already compatible |
| Z stack, one projected image wanted | `prepare_image_for_patching(..., z_policy="max")` | Explicit projection |
| Z stack, patch every slice | `iter_image_planes_for_patching(..., z_policy="all")` | One 2D workflow per slice |
| Multichannel image, keep all channels | Use `image_axes="CYX"` or `"YXC"` | Preserves channel data |
| Label stack | `prepare_labels_for_cell_patching(..., z_policy="select")` | Labels must resolve to one `YX` plane |
| DINO-like RGB input | `prepare_image_for_patching(..., channel_policy="repeat_to_rgb")` | Explicit model-facing prep |

## Image preparation helpers

Use:

- `prepare_image_for_patching(...)`
- `iter_image_planes_for_patching(...)`

Typical tasks:

- select a single Z/T plane
- apply max/mean/sum projection on image Z
- keep/select/project channels
- repeat grayscale to RGB for model-facing workflows
- no normalization: the former per-image percentile/min-max normalization (`normalize=`/`percentile=`, `percentile_normalize`/`minmax_normalize`), deprecated since Milestone 3 of the [plan](plan.md), was removed in v0.1.1 (D86): use `config.normalization`, which uses the preprocessing brick's statistics at a chosen scope and records the parameters

## Label preparation helpers (cell mode)

Use:

- `prepare_labels_for_cell_patching(...)`
- `iter_cell_planes_for_patching(...)`

Rules:

- labels must resolve to one 2D `YX` plane for cell mode
- no silent label averaging/summing
- strict matching for T/Z when iterating `all` planes

Note: `iter_cell_planes_for_patching` supports `z_policy` values `{error, first, select, max, all}`.
The `"mean"` and `"sum"` Z projection methods are **not** available through this iterator, even for
the image component. If you need a mean- or sum-projected image alongside per-plane labels, call
`iter_image_planes_for_patching` (which supports `"mean"` and `"sum"`) and
`prepare_labels_for_cell_patching` separately on each Z slice.

## Common mistakes

- Expecting automatic Z projection inside `run_patching(...)`.
- Expecting true 3D patch generation from core modes.
- Passing multichannel labels without selecting a label channel/plane first.
- Assuming iterator `metadata`/`plane_id` appear automatically in patch/export metadata.
- Using percentile normalization outputs for biological quantification without caution.

## Important limitation: no true 3D patching

CellPatcher currently does **not** generate volumetric 3D patches.
If you process stacks, helpers produce a sequence of 2D planes, and each plane goes through the normal 2D patching workflow.

## Example: patch every Z slice (grid mode)

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

result = run_patching(images=images, config=config, image_metadata=[{**p.metadata, "position_id": "fov1"} for p in planes])
```

## Example: cell mode over matched Z planes

```python
from cellpatcher import (
    CellPatcherConfig,
    CommonConfig,
    CellConfig,
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

result = run_patching(images=images, masks=labels_2d, config=config, image_metadata=[{**p.metadata, "position_id": "fov1"} for p in pairs])
```


## Keeping Z/T plane metadata with patch results

`iter_image_planes_for_patching` and `iter_cell_planes_for_patching` return `metadata` and `plane_id` on each yielded item.
When you pass a list of prepared planes into `run_patching`, image IDs are assigned as `image_0000`, `image_0001`, ... (the place in the run).
Pass each plane's `metadata` too: its `t`/`z` indices become levels of `patch_id`, which is built from the
position and the plane, not from the image ID (`configuration_reference.md`, "Patch identity"). Without them,
planes of one image with the same pixels or the same `position_id` give the same `patch_id` and the run is refused.

Pass that per-plane `metadata` straight through as `run_patching(..., image_metadata=...)` and it is stamped onto the
matching `PatchRegion`'s `t`/`z0`/`z1`/`c` fields automatically — and therefore into the exported `patches.csv` too,
since `patches.csv` is written from those same `PatchRegion` objects.

```python
pairs = list(iter_cell_planes_for_patching(image, "ZYX", labels, "ZYX", z_policy="all"))

images = [p.image for p in pairs]
labels_2d = [p.labels for p in pairs]
image_metadata = [p.metadata for p in pairs]  # each dict has "t"/"z" keys as applicable

result = run_patching(images=images, masks=labels_2d, config=config, image_metadata=image_metadata)

# result.patches[i].t / .z0 / .z1 now reflect pairs[i].metadata, and the same
# values are already in patches.csv if config.export.enabled=True.
```

If you need the exact `plane_id` string (not just the `t`/`z` indices it's built from) on each row, join it in
separately the same way — that one field is not currently mirrored onto `PatchRegion`:

```python
plane_id_lookup = {f"image_{i:04d}": p.plane_id for i, p in enumerate(pairs)}
rows = [{**patch.to_dict(), "plane_id": plane_id_lookup.get(patch.image_id)} for patch in result.patches]
```


## Spatial position identity for downstream MorphoBricks analysis

When several prepared planes come from the same microscope XY position, they are
**one spatial FOV**, not one FOV per plane. Add the same `position_id` to every
plane's metadata before calling `run_patching`:

```python
image_metadata = [
    {**p.metadata, "position_id": "P01", "image_file": f"P01_{p.plane_id}.tif"}
    for p in planes
]
result = run_patching(images=images, config=config, image_metadata=image_metadata)
```

The exported rows then keep both:
- `position_id`: spatial sampling identity;
- `z0/z1`, `t`, `c`: within-position acquisition coordinates;
- `image_file`: source-file provenance.

This distinction is required by MorphoCoverage when asking how many spatial
positions were sampled.
