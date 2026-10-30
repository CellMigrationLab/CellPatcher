# Normalization with the preprocessing statistics

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md) · [Normalization contract](normalization_contract.md)

Microscopy intensities differ between images, days and microscopes for technical reasons
(illumination, exposure, detector gain, bleaching). Normalization puts patches on comparable
intensity scales before feature extraction or deep learning. CellPatcher normalizes patches
with the per-plane intensity statistics computed once by the MorphoBricks
[preprocessing brick](https://github.com/CellMigrationLab/morphobricks-preprocessing), so
parameters can be pooled over any group of images without reading the images again.

Normalization is **optional and off by default**. With it off, CellPatcher patches exactly as
before.

## The workflow

1. **Install** the optional extra: `pip install "cellpatcher[preprocessing]"` (the notebook's
   first cell does this on Google Colab). It also enables OME-Zarr images.
2. **Run the preprocessing notebook** (or `morphobricks-preprocessing run`) on your experiment
   folder. It writes `plane_histograms.csv` and `run_manifest.json` to an output folder next to
   the experiment.
3. **Run CellPatcher on the same experiment folder** with normalization enabled, giving it that
   `plane_histograms.csv`. Before any patch is normalized, CellPatcher checks that every image
   still matches the statistics (see [Checks](#checks-and-messages)).

In the CellPatcher notebook, normalization is the **Normalization (optional)** section of
step 4; **Show parameters per group** lists the parameters before you run, and the preview in
step 5 shows a few patches before and after normalization with their histograms. The
**Pool conditions** choice appears only for the biological-repeat scope. **Image type** is taken
from the preprocessing manifest, and locked, when the manifest records one; otherwise you choose
it. For RGB only the percentile method is offered.

In Python:

```python
from cellpatcher import CellPatcherConfig, NormalizationConfig, run_patching_condition_repeat_tree

config = CellPatcherConfig(mode="grid")
config.grid.tile_size = 128
config.common.image_axes = "ZCYX"
config.common.z_policy = "all"          # one patch per Z plane (needed for stacks)
config.common.output_dir = "./patches"
config.export.enabled = True
config.normalization = NormalizationConfig(
    enabled=True,
    plane_histograms="./experiment_MorphoBricks_Preprocessing/plane_histograms.csv",
    method="percentile",                 # 1st-99.8th percentile range (default)
    scope="image",                       # default
)

results = run_patching_condition_repeat_tree("./experiment", config=config, combine_metadata=True)
```

## What normalization does

Each patch is turned into `(pixels - offset) / scale`, **per channel**, as 32-bit floats,
**without clipping** (values outside 0–1, or far from 0 for z-scores, are kept). The offset and
scale come from one of four methods, applied to the pooled intensities of the chosen scope:

| `method` | Offset | Scale | Use it for |
|---|---|---|---|
| `"percentile"` (default) | low percentile (default 1) | high (default 99.8) − low | Robust 0–1 scaling, ignoring a few saturated or dead pixels |
| `"minmax"` | minimum | maximum − minimum | Exact 0–1 scaling of the full range (sensitive to outliers) |
| `"zscore"` | mean | standard deviation | Centring and scaling for statistics or models expecting z-scores |
| `"robust_zscore"` | median | MAD × 1.4826 | Z-scores that resist outliers |

**RGB images** (`image_type="rgb"`, for example brightfield histology): the three colour
channels are normalized together with StarDist's percentile method, which is the only method
offered. One offset and scale come from the R, G and B pixels pooled together, so the colours
keep their balance; the result equals `csbdeep.utils.normalize(x, 1, 99.8, axis=(0, 1, 2))`.
The images must have three 8- or 16-bit channels, kept in one patch
(`extraction.split_channels=False`). When the preprocessing brick was told the image type, its
manifest records it and CellPatcher's `image_type` must match.

The pipeline order is generate → filter → extract → **normalize** → mask → resize → export:
filters (`min_mean_intensity`, foreground thresholds) still see raw values, `mask_fill_value`
is in normalized units (`NaN` works), and resizing works on the normalized values.

## Choosing a scope

`scope` decides which pixels the parameters come from. Every group gets its own parameters, and
channels never share parameters (except the three channels of an RGB image).

| `scope` | Parameters from | Typical use |
|---|---|---|
| `"image"` (default) | All Z and T planes of the patch's source image | Correct per-image differences (exposure, illumination) |
| `"biological_repeat"` | One biological repeat in every condition, all its wells or dishes together | Correct day-to-day differences while keeping the differences between conditions |
| `"patch"` | The patch's own pixels | Cell-level normalization; no preprocessing table needed |

- **`biological_repeat` keeps the differences between conditions and removes the differences
  between repeats.** The repeats are paired across conditions in the preprocessing brick's
  step 1 (`experiment_design.csv`, next to `plane_histograms.csv` or given in
  `experiment_design`), where you also confirm that each repeat was prepared and acquired
  under the same circumstances; without that confirmation the run stops before reading any
  image. `pooling="conditions"` weighs every condition of a repeat equally instead of every
  pixel (`"pixels"`, the default).
- The former `"condition"` and `"technical_repeat"` scopes were removed (October 2026): they
  put conditions, or wells, on the same scale and so removed real differences. Patches
  exported with them still load.
- **Background:** the repeat and image scopes use whole planes, so in sparse images
  the parameters describe the background more than the cells. For parameters from the cells
  only, use `scope="patch"` with `patch_region="mask"`.
- **Time series:** `per_time_point=True` gives every time point its own parameters (for
  example to compensate photobleaching).
- **Patch regions** (`scope="patch"`): `"mask"` uses the pixels of the patch's own object
  (Cell mode), `"bbox"` all pixels of the object's bounding box whatever the patch size, and
  `"patch"` all pixels of the patch. Padding added by `pad_mode="constant"` is never used.
  `"mask"` and `"bbox"` need the mask to hold the patch labels: with a binary mask labelled by
  `cell.binary_mask_policy="connected_components"` they are refused (use integer instance labels).
  A patch without any intensity spread (for example an empty background tile, or for robust
  z-score a patch where more than half the pixels share one value) cannot be normalized from
  its own pixels: it is **discarded** and listed in a warning. An RGB patch without spread is
  kept and becomes zeros, as in StarDist.

## Stacks, channels and OME-Zarr images

- **Z-stacks and time series:** normalization needs one plane per patch. Set
  `common.z_policy`/`t_policy` to `"all"` (one patch per plane), `"first"` or `"select"`;
  `"keep"` and Z projections stop the run. Each Z slice is its own plane.
- **Channels** may be kept together in one patch (each channel gets its own parameters, listed
  in channel order) or split with `extraction.split_channels=True`, or `[i, ...]` for only the
  selected channels. A single-channel patch keeps the parameters of its channel in the image's
  group, so selecting channels never changes their values.
- **OME-Zarr images** (folders ending in `.zarr`) are read like TIFF files; their masks can be a
  mask folder or the label images inside them (`common.mask_source="ngff_labels"`, with
  `common.mask_label_name`).

## What is recorded

Every exported patch records its normalization in `patches.csv`, so it can be reproduced and
undone:

| Column | Meaning |
|---|---|
| `normalization_method` | e.g. `percentile 1-99.8`, `min-max`, `z-score`, `robust z-score` |
| `normalization_scope` | the scope used |
| `normalization_group` | the repeat group (e.g. `2026_01_01`), the image (e.g. `control/2026_01_01/a.tif`), or `<image>::<patch_id>` for the patch scope (`::t=<t>` with `per_time_point`) |
| `normalization_exact` | `true` for exact parameters (8- and 16-bit images), `false` for histogram approximations (float images) |
| `Normalization_Offset_C<i>`, `Normalization_Scale_C<i>` | the offset and scale of channel `i`, one pair per channel; a split-channel patch fills only its own channel's pair |

`run_config.json` records the normalization settings and which statistics were used
(`normalization_inputs`: the CSV's SHA-256, its version, whether a manifest was used, and the
preprocessing package version), and for the image and patch scopes the normalization brick's
own settings (`normalization_settings`).

## Getting the patches

| You want | Do |
|---|---|
| Normalized files (default) | `export.enabled=True`: float32 `.tif`/`.npy` files |
| Raw files plus parameters (smaller, reversible) | `normalization.export_values="raw"`; read them back normalized with `cellpatcher.iter_exported_patches("patches.csv")` or `ExportedPatchTorchDataset`, which apply each row's offset and scale |
| Arrays in memory | `common.extract_arrays=True` (stops above `normalization.memory_limit`, 4 GiB by default) |
| One patch at a time (large datasets) | Run with `extract_arrays=False`, then `cellpatcher.iter_patch_arrays(patches, images, config)`; `PatchTorchDataset(images, patches, config=config)` for PyTorch |

All of these give the same arrays for the same settings.

```python
from cellpatcher import iter_exported_patches

for row, patch in iter_exported_patches("./patches/patches.csv"):
    ...  # normalized float32 array, whatever export_values was
```

## Checks and messages

Every problem stops the run with a plain-language message listing the affected images or
planes and what to do:

| Message | Meaning and fix |
|---|---|
| *... image(s) found by CellPatcher have no rows in plane_histograms.csv* | Images were excluded or skipped by the preprocessing brick, or added after it ran. Run the preprocessing again on the same folder, remove those images, or use the patch scope. |
| *... changed after the preprocessing brick read them* / *... the images changed after the preprocessing run* | An image's size, time or pixel values differ from the statistics. Run the preprocessing again. |
| *... other condition, biological repeat or technical repeat labels* | CellPatcher and the preprocessing read different folder levels: give both the same experiment folder. |
| *... another role than the axis mapping confirmed in the preprocessing brick* | `common.image_axes` differs from the axes confirmed in the preprocessing notebook. |
| *Normalization needs one plane per patch* | Set `z_policy`/`t_policy` to `"all"`, `"first"` or `"select"`. |
| *... cannot be computed for N group(s)* | A group has no finite pixels or no intensity spread: choose a larger scope or another method. |
| *Keeping the normalized patches in memory needs about ...* | Use `iter_patch_arrays`, export to files, or raise `normalization.memory_limit`. |
| *... needs the optional 'preprocessing' extra* | `pip install "cellpatcher[preprocessing]"`. |

Warnings (the run continues): parameters that are approximations, background in group
statistics, the `biological_repeat` scope (what it keeps and removes), repeat groups without
images from some condition, and patches discarded for having no intensity spread.

The exact rules are in the [normalization contract](normalization_contract.md); the reasons
behind them are in the [plan](plan.md).
