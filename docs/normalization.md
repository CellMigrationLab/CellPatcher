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

1. **Install** CellPatcher: the preprocessing (v0.5.5) and normalization (v0.6.1) bricks are core
   dependencies. OME-Zarr images need the optional extra `pip install "cellpatcher[zarr]"` (the
   notebook's first cell installs it on Google Colab).
2. **Describe the experiment and run the preprocessing notebook** on your experiment folder: publish
   its experiment description (`morphobricks.toml`: conditions, biological and technical repeats,
   channels), then run. It writes `plane_histograms.csv` and `run_manifest.json` to an output folder
   next to the experiment. Every run needs a published description: CellPatcher refuses a run
   without one (`A_DESCRIPTION_REQUIRED`, Spec ADR 0002 AR10).
3. **Run CellPatcher on that preprocessing run** (`common.preprocessing_run`, its output folder)
   with normalization enabled. Since v0.2.0 a folder or tree run reads every image, and the CSV
   scopes read the run's own table, through the run (`RunInputs`): `plane_histograms` may be left
   unset. Before any patch is normalized, CellPatcher checks that every image still matches the
   statistics (see [Checks](#checks-and-messages)).

In the CellPatcher notebook (v0.2), normalization is the **Normalization** card of step 2:
**Normalize the patches**, **Statistics from** (image, biological repeat or patch), the
method and percentiles, **Normalize each time point separately**, and whether the files hold
normalized or raw values. The image type is the one the experiment description declares for
the run's images (`Fov.image_type`: RGB, or ordinary channels, normalized per channel as
fluorescence). **Show parameters per group** (image and biological-repeat scopes)
lists the offset and scale of every group and channel of the run's table before anything is
patched, the ones the patches get (`cellpatcher.notebook_session.group_parameters_of_run`, which
calls `cellpatcher.normalization.group_parameters`); a group that cannot be normalized (no finite
pixels, zero spread) is shown in red, as it would stop the run. The preview of step 3 shows the
normalized patches of one image and, for the first four, the same patch without and with
normalization beside its intensity histogram (`Preview.before_after`: the image is patched with
normalization off and on, the patches paired by `patch_id`, and each histogram binned by the
normalization brick's `compute_plane_histogram`).

In Python:

```python
from cellpatcher import CellPatcherConfig, NormalizationConfig, run_patching_condition_repeat_tree

config = CellPatcherConfig(mode="grid")
config.grid.tile_size = 128
# The preprocessing run: every image is read through it, in the roles it recorded (no axes here).
config.common.preprocessing_run = "./experiment_MorphoBricks_Preprocessing"
config.common.t_policy = "all"          # one patch per T frame (needed for time series)
config.common.output_dir = "./patches"
config.export.enabled = True
config.normalization = NormalizationConfig(
    enabled=True,                        # the CSV scopes read the run's own plane_histograms.csv
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
(`extraction.split_channels=False`). **CellPatcher supports RGB images** (experiment-v1 §4.4b,
P24): a unit the experiment description declares `image_type = "rgb"` is patched with its three
channels in one patch and normalized as above. The image type is the one the description declares
for each image (`Fov.image_type` through `RunInputs`; an ordinary channel unit is
`"fluorescence"`), at every normalization scope. The declared type wins: a
`normalization.image_type` that contradicts it is refused (`image_type_differs`, naming both),
never used in its place.

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
  between repeats.** The repeats are paired across conditions by the run's published experiment
  description (each image's `repeat_group`, read through `RunInputs`; the derived
  `experiment_design.csv` is an output and never read back, owner decision of 2026-11-10), where
  you also confirm that each repeat was prepared and acquired under the same circumstances; an
  image without a repeat group stops the run before reading any image. `pooling="conditions"` weighs every condition of a repeat equally instead of every
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
  `"mask"` and `"bbox"` need the mask to hold the patch labels: with a boolean in-memory mask labelled by
  `cell.binary_mask_policy="connected_components"` they are refused (use integer instance labels); a file
  run's `binary` mask is replaced by its computed labels before patching, so they apply.
  A patch without any intensity spread (for example an empty background tile, or for robust
  z-score a patch where more than half the pixels share one value) cannot be normalized from
  its own pixels: it is **discarded** and listed in a warning. An RGB patch without spread is
  kept and becomes zeros, as in StarDist.

## Stacks, channels and OME-Zarr images

- **Z-stacks and time series:** normalization needs one plane per patch. A file run never has Z>1
  (preprocessing v0.4.0 refuses it, `docs/preprocessing.md`); an in-memory array with Z>1 is refused too. Set
  `common.t_policy` to `"all"` (one patch per frame), `"first"` or `"select"`; `"keep"` along T stops
  the run. Z is separate: Z=1 passes (`z_policy` names or returns that one plane), Z>1 is refused whatever
  `z_policy` says (issue #83); project or select first with `convert --project` in morphobricks-preprocessing.
- **Channels** may be kept together in one patch (each channel gets its own parameters, listed
  in channel order) or split with `extraction.split_channels=True`, or a list of canonical channel IDs
  (`["DNA", ...]`, never positions) for only those channels. A single-channel patch keeps the parameters of its channel in the image's
  group, so selecting channels never changes their values.
- **OME-Zarr images** (folders ending in `.zarr`) are read like TIFF files; their masks can be a
  mask folder or the label images inside them, as the preprocessing run paired them (a label
  image's type is its name: `matching.mask_type`).

## What is recorded

Every exported patch records its normalization in `patches.csv`, so it can be reproduced and
undone:

| Column | Meaning |
|---|---|
| `normalization_method` | e.g. `percentile 1-99.8`, `min-max`, `z-score`, `robust z-score` |
| `normalization_scope` | the scope used |
| `normalization_group` | the repeat group (e.g. `2026_01_01`), the image (e.g. `control/2026_01_01/a.tif`), or `<image>::<patch_id>` for the patch scope (`::t=<t>` with `per_time_point`) |
| `normalization_exact` | `true` for exact parameters (8- and 16-bit images), `false` for histogram approximations (float images) |
| `normalization_parameters_version` | `2`: the parameter columns of normalization v0.5.0 (rows without it, written before, are refused on load: run CellPatcher again) |
| `normalization_channels` | the patch's channel IDs in its own C order, `;`-separated (`ACTIN;DNA`); each channel is normalized with the pair of its ID, never by position |
| `Normalization_Offset_<ID>`, `Normalization_Scale_<ID>` | the offset and scale of channel ID `<ID>` (the run's description's ID; for in-memory images without IDs, `C<i>` by position, or `R`, `G`, `B` for RGB), one pair per channel ID in sorted order; a split-channel patch fills only its own channel's pair |

`run_config.json` records the normalization settings and which statistics were used
(`normalization_inputs`: the CSV's SHA-256 and version as the run's manifest records them,
the manifest version, and the preprocessing package version), and for the image and patch scopes the normalization brick's
own settings (`normalization_settings`).

## Getting the patches

| You want | Do |
|---|---|
| Normalized files (default) | `export.enabled=True`: float32 `.tif`/`.npy` files |
| Raw files plus parameters (smaller, reversible) | `normalization.export_values="raw"`; read them back normalized with `cellpatcher.iter_exported_patches("patches.csv")` or `ExportedPatchTorchDataset`, which apply each row's offset and scale |
| Arrays in memory | `common.extract_arrays=True` (stops above `normalization.memory_limit`, 4 GiB by default) |
| One patch at a time (large datasets) | Run with `extract_arrays=False`, then `cellpatcher.iter_run_patch_arrays(patches, config)` for a folder or tree run (read through its preprocessing run), or `cellpatcher.iter_patch_arrays(patches, images, config)` for `run_patching`'s arrays; `PatchTorchDataset(images, patches, config=config)` for PyTorch (arrays) |

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
| *The preprocessing run in ... cannot be used: The experiment holds ... that the preprocessing run did not record* (`unrecorded_files`) | Since preprocessing v0.4.0rc2: images or masks were added to the experiment (or masks folder) after the run. None is left out: run preprocessing again on the experiment as it is, or exclude the images there. |
| *... the zarr package is not installed* (`zarr_extra_missing`) | The run holds OME-Zarr images: `pip install "cellpatcher[zarr]"`. |
| *The preprocessing run in ... cannot be used* (`preprocessing_run_refused`) | The preprocessing brick's `RunInputs` refuses the run (since v0.2.0): no `run_manifest.json` next to the table, a run written before preprocessing v0.4.0, a table or description changed since the run, or an image or mask changed or missing since the run where CellPatcher reads it. Run preprocessing v0.5 on the experiment as it is now; an experiment only moved or copied (times kept) since its run is read where it is now when CellPatcher is given that folder and its masks folder (since CP-2a). |
| *The preprocessing run in ... has no published experiment description* (`A_DESCRIPTION_REQUIRED`) | Every run needs one (Spec ADR 0002 AR10): describe the experiment and publish its description in the preprocessing notebook, run preprocessing again and give CellPatcher that run. |
| *normalization.plane_histograms/manifest names ...* (`not_the_run_output`) | Only the run's own files, unrenamed in its output folder, are read. Give that folder's `plane_histograms.csv` (and leave `manifest` empty). |
| *... these images of the run have no repeat group* (`missing_repeat_group`) / *... several repeat groups to the images of one condition and biological-repeat folder* (`repeat_group_conflict`) | The `biological_repeat` scope pools the repeat groups of the run's description: run preprocessing again with its published description. `normalization.experiment_design` was removed (the derived `experiment_design.csv` is never read). |
| *... image(s) found by CellPatcher have no rows in plane_histograms.csv* | In-memory runs: images whose `source_relative_path` the run has no rows for. Folder and tree runs patch only the run's images (since CP-2a); an image added after the run is refused as `unrecorded_files` (since preprocessing v0.4.0rc2). Run the preprocessing again, or use the patch scope. |
| *... changed after the preprocessing brick read them* / *... the images changed after the preprocessing run* | An image's size, time or pixel values differ from the statistics. Run the preprocessing again. |
| *... other condition, biological repeat or technical repeat labels* | CellPatcher and the preprocessing read different folder levels: give both the same experiment folder. |
| *... another position_id than their rows* | A `position_id` given in `image_metadata` is not the one the preprocessing recorded (the image name). Leave it to the image name, or give the recorded one. |
| *... another role than the axis mapping confirmed in the preprocessing brick* | In-memory runs: `common.image_axes` differs from the axes confirmed in the preprocessing notebook. (Folder and tree runs take the run's roles and declare no axes.) |
| *Normalization needs one plane per patch* | Set `t_policy` to `"all"`, `"first"` or `"select"`. A Z axis longer than 1 is refused (Z>1 is never patched, issue #83): project or select it first with `convert --project` in morphobricks-preprocessing. |
| *... cannot be computed for N group(s)* | A group has no finite pixels or no intensity spread: choose a larger scope or another method. Or `A_SCOPE_INCOMPLETE`: a channel only some planes of the group have (Spec fixture `v17`), never fitted over part of the group nor filled in: choose groups whose planes all have it (the image scope) or leave the channel out. Folder and tree runs check every group of the planes they patch for it from `plane_histograms.csv` when the run is opened, so this one stops them before any image is read or any file written (issue #107); no finite pixels and zero spread are found by the patching. |
| *Keeping the normalized patches in memory needs about ...* | Use `iter_patch_arrays`, export to files, or raise `normalization.memory_limit`. |
| *Reading the OME-Zarr image ... needs the zarr package* | `pip install "cellpatcher[zarr]"`. |

Warnings (the run continues): parameters that are approximations, background in group
statistics, the `biological_repeat` scope (what it keeps and removes), repeat groups without
images from some condition, and patches discarded for having no intensity spread. Before
v0.2.0 a CSV scope without `run_manifest.json` only warned; it is refused now (above).

The exact rules are in the [normalization contract](normalization_contract.md); the reasons
behind them are in the [plan](plan.md).
