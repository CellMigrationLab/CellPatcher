# Multidimensional preprocessing (optional)

CellPatcher generates 2D Y/X patch regions. If your image has Z or T axes, you can either select/project a single 2D plane before patching, or expand the stack into multiple 2D planes and run the same patching mode on each plane.

## Core idea

- Core patching modes (`cell`, `random`, `grid`, `whole_fov`) are 2D Y/X region generators.
- Preprocessing is explicit and optional.
- CellPatcher does **not** silently project Z/T, merge channels, or normalize intensities.

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
- optional percentile/min-max normalization

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
If you process stacks, helpers produce a sequence of 2D planes and each plane runs through normal 2D patching.

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

result = run_patching(images=images, config=config)
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

result = run_patching(images=images, masks=labels_2d, config=config)
```


## Keeping Z/T plane metadata with patch results

`iter_image_planes_for_patching` and `iter_cell_planes_for_patching` return `metadata` and `plane_id` on each yielded item.
When you pass a list of prepared planes into `run_patching`, image IDs are assigned as `image_0000`, `image_0001`, ... .
That plane metadata is not automatically copied into `PatchRegion` fields or export CSV rows, so preserve it with an explicit lookup.

```python
pairs = list(iter_cell_planes_for_patching(image, "ZYX", labels, "ZYX", z_policy="all"))

images = [p.image for p in pairs]
labels_2d = [p.labels for p in pairs]

plane_lookup = {
    f"image_{i:04d}": {
        "plane_id": p.plane_id,
        **p.metadata,
    }
    for i, p in enumerate(pairs)
}

result = run_patching(images=images, masks=labels_2d, config=config)

rows = []
for patch in result.patches:
    rows.append({
        **patch.to_dict(),
        **plane_lookup.get(patch.image_id, {}),
    })
```

## Metadata caveat

Iterator helpers return per-plane `metadata` and `plane_id`, but this information is **not automatically copied** into `PatchRegion` fields or export CSV metadata. If you need that mapping downstream, store and join it explicitly in your workflow.
