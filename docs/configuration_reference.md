# CellPatcher Configuration Reference

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

This document is the complete runtime configuration reference for `CellPatcherConfig` and all nested config sections.

---

## `config.mode`

| Field | Type | Default | Allowed values |
|---|---|---|---|
| `mode` | `str` | `"cell"` | `"cell"`, `"random"`, `"grid"`, `"whole_fov"` |

---

## File runs read through the preprocessing run (v0.2.0, CP-2a and CP-2b)

`run_patching_pipeline`, `run_patching_condition_repeat_tree` and `run_patching_from_folders` read files only through a clean preprocessing run (v0.4), `common.preprocessing_run`, with the preprocessing brick's `RunInputs` (design v0.4 §6). Without it they stop with `no_preprocessing_run` and the preprocessing command (`morphobricks-preprocessing check <experiment> --masks <masks folder>`, then `morphobricks-preprocessing run <experiment> --axes <roles> --masks <masks folder>`).

- **Where the experiment is now.** The run is opened with `RunInputs.open(run, experiment_root=<image folder or root>, masks_root=<mask_folder_path or mask_root_path>)`; a masks folder left out is where the run recorded it. An experiment moved or copied since its run is checked and read where CellPatcher is given it: every image and mask must have the size and modification time the run recorded (copy with the times kept, e.g. `cp -p` or `shutil.copytree`), or the run is refused (`preprocessing_run_refused`, naming the files).
- **Images.** The run's images (`RunInputs.fovs()`), sorted by path, read with `RunInputs.read_plane` in the roles the run recorded. Each image's array is in its own canonical layout, from the roles the run recorded for *that image*: its T, Z and C roles, in that order, then Y and X (`ZCYX`, `TYX`, `CYX`, `YX`, ...). A T or Z role of length 1 that `t_policy`/`z_policy` keeps (`"keep"`) is left out (its one plane is the image); under any other policy it stays, so an OME-Zarr image's recorded length-1 T and Z give `t0000`/`z000` in its ids, as in MorphoFeature. An image is patched as it would be alone: a 2-D image beside a stack has no Z, no `z000` in its ids and the same random rectangles (owner decision of 2026-11-03, audit A-M1; before, one layout per run gave every image the Z or T any image had more than one plane along), and `z_policy="select"` applies only to the images with Z. Nothing is read transposed. With plane policies only the chosen planes are read. An image or mask in the experiment that the run did not record (for example added after the run) is never left out without a word: since preprocessing v0.4.0rc2 `RunInputs.open` refuses the run, and CellPatcher reports `unrecorded_files` naming the files (a brick calls no `discover_experiment` for files). Run preprocessing again on the experiment as it is, or exclude the images there. A run with OME-Zarr images opened without the `zarr` extra is refused too (`zarr_extra_missing`): their label images cannot be checked.
- **Labels and `position_id`.** Each image's `condition`, `biological_repeat`, `technical_repeat` and `position_id` are the ones the run recorded, never parsed. `run_patching_pipeline` patches a run of one condition and biological repeat (any technical repeats); use the tree run for more.
- **Masks folder in `grid`/`whole_fov` mode.** These modes read no masks; `mask_folder_path` (or `mask_root_path`) given with them only says where the run's masks folder is now, and its masks are checked there (owner decision of 2026-11-02; before CP-2b the folder run refused it).
- **Masks.** Those the run paired (masks-folder TIFFs and the label images of OME-Zarr images; a label image's type is its name), read with `RunInputs.read_mask` (the brick checks their labels). `matching.mask_type` chooses the type; `None` takes the run's only type. `cell` mode always reads masks; `random` mode when a masks folder or `matching.mask_type` is given. Every image needs a mask of the type. Each mask is read in its own layout too (its T and Z roles, then Y and X; `YX` for a stated shared 2-D mask), shared by every plane of its image along an axis it lacks or has length 1. When the images' layouts differ, `run_config.json`'s `preprocessing.layout` also maps each image's path to its own `image` and `mask` layout (`images`), which delivers its exported patches. A mask whose header puts X before Y is refused by the preprocessing run (since v0.4.0rc2); a run written by rc1 that recorded such a mask without axes is refused by `RunInputs.open` (`preprocessing_run_refused`), and CellPatcher keeps its own refusal of such a record as a safety net. A folder mask and a label image of the same type for one image are refused by the preprocessing run (`duplicate_mask`), so no clean run has them.
- **Normalization.** The CSV scopes read the run's own table (`normalization.plane_histograms` may be left unset, or must name it: `not_the_run_output`), design and image type.
- **`run_config.json`** (version 2, CP-2b; see "Deterministic reconstruction" below) records `run_config_version: 2` and a `preprocessing` section: the run (`run`), the SHA-256 of its `run_manifest.json` when the call opened it (`manifest_sha256`), and the `layout` (`image`, `mask`) the patches were read in. It does not record `common.image_axes`, `mask_axes` or the labels (the run's, not settings), so it loads back with `load_config` as a config a file run accepts.
- **Migration.** For file runs, `common.image_axes`/`mask_axes` other than `"YX"`, `common.condition`/`biological_repeat`/`technical_repeat` other than `"single"` and `normalization.experiment_root`/`masks_root` are refused with a message saying what replaces them. The axes and labels still describe in-memory arrays given to `run_patching`. `common.mask_source` and `mask_label_name` were removed in CP-2b: `CommonConfig(mask_source=...)` is a `TypeError`; a config file holding their defaults still loads, one with another value is refused.
- **Within one call** (`run_patching`, a folder or tree run) the run is opened once; nothing is kept beyond the call, so the next call checks every image and mask again (`cellpatcher.run_session`).

**No file is read or paired by CellPatcher itself** (CP-2b). `validate_image_and_mask_folders(image_folder, mask_folder, *, preprocessing_run=, mask_type=)` returns the pairing the run recorded, opened where the folders are now. Lazy delivery of a file run's patches is `iter_run_patch_arrays(patches, config, *, experiment_root=, masks_root=)`, which reads each patch's plane with `RunInputs.read_plane` (its mask with `read_mask`) from the run's image at the patch's `source_relative_path`, and refuses a patch whose image or FOV key is not the run's; `patch_array`, `iter_patch_arrays` and `PatchTorchDataset` take arrays only (a path is a `TypeError`).

---

## `config.common` (`CommonConfig`)

Shared settings that apply to every run, regardless of mode.

| Field | Type | Default | Notes |
|---|---|---|---|
| `image_axes` | `str` | `"YX"` | Axes string for image arrays given to `run_patching` (e.g. `"YX"`, `"CYX"`, `"ZYX"`). Must include `Y` and `X`, **Y before X**: in-memory arrays are taken only in the canonical layout (design v0.4 §6), so X before Y (`"XY"`, `"CXY"`) is refused since v0.2.0 (CP-3), as for masks; swap the array's spatial axes (`numpy.swapaxes`) instead. Before, such an image was patched in the transposed frame. Since v0.2.0 (CP-2a) folder and tree runs read the roles the preprocessing run recorded, refuse any other value than `"YX"`, and record the layout they read in `run_config.json`'s `preprocessing.layout` (CP-2b). Since CP-2b no file is read with these axes (lazy delivery takes arrays; see "File runs read through the preprocessing run"), so CellPatcher checks no file header against them: the preprocessing run checks each file's axes (design v0.4 §3; nothing is ever read transposed). |
| `mask_axes` | `str` | `"YX"` | Axes string for mask arrays given to `run_patching`. Must include `Y` and `X`. Since v0.2.0 (CP-2a) folder and tree runs read masks in the roles the preprocessing run recorded and refuse any other value than `"YX"`. CellPatcher takes a label plane's two dimensions as Y and X in their stored order, so any `mask_axes` with X before Y (`"XY"`, `"ZXY"`) is refused (before v0.1.2 it was ignored for in-memory masks). The refusal says how to fix the masks: give each one with Y before X (its last two axes swapped, e.g. `numpy.swapaxes(mask, -1, -2)`) and declare those axes. Nothing is ever read transposed. |
| `preprocessing_run` | `str \| None` | `None` | The output folder of the clean preprocessing run (v0.4) that folder and tree runs read their images, masks and labels through (see "File runs read through the preprocessing run"). Required for them; not used by in-memory `run_patching` (its CSV normalization reads `normalization.plane_histograms`). Since v0.2.0. |
| `output_dir` | `str \| None` | `None` | Output folder path. Required when `export.enabled=True`; keep `None` for in-memory-only runs. |
| `extract_arrays` | `bool` | `False` | When `True`, extracted patch arrays are returned in `PatchingResult.patch_arrays`. |
| `condition` | `str \| None` | `"single"` | In-memory `run_patching` only: since v0.2.0 (CP-2a) folder and tree runs take each image's labels from the preprocessing run and refuse any other value than `"single"`. Experimental group label, stamped onto every `PatchRegion`/CSV row for this run. Part of the MorphoBricks shared CSV contract (see `AGENTS.md` §5). Set it explicitly for a single-folder `run_patching_pipeline` call, or let `run_patching_condition_repeat_tree` derive it per leaf from a `<condition>/<biological_repeat>/` folder tree (see "Condition/biological-repeat folder trees" below). |
| `biological_repeat` | `str \| None` | `"single"` | In-memory `run_patching` only: since v0.2.0 (CP-2a) folder and tree runs take each image's labels from the preprocessing run and refuse any other value than `"single"`. Biological replicate identifier (e.g. `"YYYY_MM_DD"`, or `cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL`/`"single"` for a condition folder with no biological-repeat subfolder), stamped onto every `PatchRegion`/CSV row for this run. Same status as `condition`. Formerly `repeat`, which still works as a deprecated alias (see "Renamed: `repeat` → `biological_repeat`" below). |
| `technical_repeat` | `str \| None` | `"single"` | In-memory `run_patching` only: since v0.2.0 (CP-2a) folder and tree runs take each image's labels from the preprocessing run and refuse any other value than `"single"`. Technical replicate within the biological repeat: a well, a dish or a coverslip, never a biological repeat. Stamped onto every `PatchRegion`/CSV row for this run; a per-image `image_metadata=[{"technical_repeat": ...}]` entry overrides it for that image. `run_patching_condition_repeat_tree` derives it per leaf from an optional `<condition>/<biological_repeat>/<technical_repeat>/` level, and stamps `cellpatcher.SINGLE_TECHNICAL_REPEAT_LABEL` (`"single"`) when a biological-repeat folder holds its images directly. |

| `z_policy` | `str` | `"keep"` | How a Z axis of the images becomes 2D planes before patching. **Z>1 is refused under every value** (issue #83): a file run's images have one Z plane at most (preprocessing v0.4.0 refuses depth stacks), and an image or mask with more raises `ZStackNotSupportedError` in memory and in a file run, before `z_policy` applies. Z=1 (or no Z axis) passes: `"keep"` keeps that plane, `"all"` or `"first"`/`"select"` (with `z_index`) name it (`z0`/`z1` set, a `z000` level in `patch_id`, e.g. `<position_id>/z000/grid-y0-x8-h8-w8`; see "Patch identity" below), `"max"`/`"mean"`/`"sum"` return it unchanged. Images without a Z axis are not affected. To patch a Z stack, project or select first with `convert --project` in morphobricks-preprocessing (it must be a recorded step). |
| `z_index` | `int \| None` | `None` | Z plane for `z_policy="select"`. |
| `t_policy` | `str` | `"keep"` | T is not limited to one frame (unlike Z): for a T axis `"keep"`, `"all"` (sets `t`, a `t<NNNN>` level of `patch_id`), `"first"` or `"select"`; no projections. With both split, planes come in T-then-Z order (`.../t0001/z002/...`). |
| `t_index` | `int \| None` | `None` | Time point for `t_policy="select"`. |
| `object_set` | `str \| None` | `None` | Cell mode: the name of the object set (mask) written to the `object_set` column and to `patch_id`. `None`: named after the mask (see "Patch identity" below), `"mask"` when the mask gives no name. Letters, digits, `.`, `_`, `-` only. |

**Masks with split planes:** a mask's Z or T axis with the image's length follows the image's planes (one label plane per image plane); a mask without that axis, or with length 1, is shared by every plane. Any other length is an error. Image ids stay one per source image, and every plane and channel patch of an image shares one `position_id`.

**Reading planes:** folder and tree runs that split or select planes (`"all"`, `"first"`, `"select"`) handle one image at a time. TIFF and OME-Zarr images are then read one 2D plane at a time with the preprocessing brick's reader, so only the chosen planes are decoded (channels and kept axes are read plane by plane and stacked back); a plane that cannot be read whole within the brick's memory limit (1 GiB) stops the run, because CellPatcher needs each plane whole to patch it. `"keep"`, projections and other formats read the whole image, as before. The patches are the same either way, and the same as `run_patching` on the whole arrays.

**Axes strings** may contain any combination of `T`, `C`, `Z`, `Y`, `X` (in any order). Examples: `"YX"`, `"CYX"`, `"TCYX"`, `"ZYX"`.

---

## `config.cell` (`CellConfig`, used for `mode="cell"`)

Settings for cell/object-centred patching. A mask is required in this mode.

| Field | Type | Default | Notes |
|---|---|---|---|
| `patch_size` | `int \| str` | `"auto"` | Patch side length in pixels (square). Use `"auto"` for per-cell adaptive sizing based on bounding box, or call `scan_cell_patch_size()` to determine a good fixed value first. |
| `mask_channel` | `int \| None` | `None` | Channel index to select from multi-channel label **arrays** passed in memory (e.g. `0` for first channel). Leave `None` for 2-D masks. Mask files from a mask folder cannot have a channel axis (see `common.mask_axes`). |
| `center_on` | `str` | `"bbox_center"` | Patch centre strategy: `"bbox_center"` (centre of bounding box) or `"centroid"` (centre of mass). |
| `exclude_border` | `bool` | `True` | When `True`, objects whose bounding box touches the image border are skipped. |
| `min_area` | `int` | `1` | Minimum object area in pixels. Objects smaller than this are skipped. Must be `>= 1`. |
| `max_area` | `int \| None` | `None` | Maximum object area in pixels. Objects larger than this are skipped. Must be `>= min_area` when set. |
| `binary_mask_policy` | `str \| None` | `None` | The explicit per-request conversion of a **`binary`** mask to objects (Spec P27, ADR 0002 AR5; there is no default). An `instances` mask is used as given whatever this setting (decision D47: each non-zero value one object; an `instances` mask holding 0 and 1 is one object, label 1). A `binary` mask (in a file run: declared `binary` by the run's description; in memory: a **boolean** array) has no object ids: `None` refuses it (a file run: the resolver's `A_MASK_NOT_INSTANCES`, before any pixel is read); `"connected_components"` labels it with the one shared labeller, `cellpatcher.label_objects(mask, policy, connectivity)`, at `binary_mask_connectivity`, emits a `UserWarning` and is recorded in `run_config.json` (`cell.binary_mask_policy`, `cell.binary_mask_connectivity`, `matching.mask_id`), never in `morphobricks.toml`. Set on a mask declared `instances`, it is refused (nothing to convert). The former `"single_object"` (the default until CP-2, which refused a boolean mask) is refused: leave it `None`. With a **boolean in-memory** mask it cannot be combined with `extraction.apply_mask`, nor with `normalization.scope="patch"` over `patch_region="mask"`/`"bbox"` (the mask does not hold the computed labels); both are refused. A file run's `binary` mask is replaced by its computed labels before patching, so both apply there. Labels above the int32 maximum (2147483647), or negative labels, are refused instead of wrapping, with `cellpatcher.LabelRangeError` (a `CellPatcherRefusal`, see "Typed refusals"); the rule is the preprocessing brick's `check_labels`, shared by every brick (D86, owner decision D-b). A **float** mask is refused, even when its values are whole numbers or 0.0/1.0 (since D86; before, a 0.0/1.0 float mask was treated like a boolean one): save the mask as integers. Before 2026-10-26 the default was `"connected_components"`, which also relabelled 0/1 **integer** masks. |
| `binary_mask_connectivity` | `int` | `2` | The connectivity `binary_mask_policy="connected_components"` labels with: `2` (the default and the shared 2-D routes' rule, owner decision 11 November 2026) joins a pixel to its 4 edge and 4 corner neighbours (8-connected: two blobs touching only at a corner are one object); `1` only to its 4 edge neighbours (4-connected). Unused without a policy. |

---

## `config.random` (`RandomConfig`, used for `mode="random"`)

Settings for randomly sampled patches of a fixed size.

| Field | Type | Default | Notes |
|---|---|---|---|
| `patch_size` | `int` | `128` | Side length of each square patch in pixels. Must be `> 0`. |
| `n_patches` | `int` | `100` | Number of patches to sample per image. Must be `> 0`. |
| `seed_scope` | `str` | `"plane_channel"` | Which identity the seed is derived from (issue #33): `"plane_channel"` (the default, as before: `position_id`, `t`, `z`, `c`), `"plane"` (`position_id`, `t`, `z`: every channel given as its own image gets the same rectangles) or `"position"` (`position_id` only: every plane of a position gets the same rectangles). `c` enters the seed only when an image's metadata gives a channel; channels split with `split_channels` always share their region's rectangles. |
| `min_mask_fraction` | `float \| None` | `None` | With a sampling mask, the least share of a draw's pixels in the mask's foreground. `None` (unset) accepts any overlap (at least one mask pixel). An explicit `0` means the mask excludes nothing: the rectangles are those of a run without it, and a `UserWarning` says so (since v0.2.0, matrix finding F8; before, `0` kept every draw with at least one mask pixel, which unset still does). Needs a mask when `> 0`. Draw sequence: one stream from the image's seed, `y0` then `x0` per draw, independent of the mask; with a fraction `> 0` the patches are the stream's draws that reach it, in order (rejected draws are consumed). When a mask is too sparse to give `n_patches` draws within the attempt cap (`max(1000, 50 × n_patches)`), the image is refused with `cellpatcher.SamplingExhaustedError` (a `CellPatcherRefusal`, see "Typed refusals"). |
| `seed` | `int \| None` | `0` | Random seed for reproducibility. Set to `None` for non-deterministic sampling. Each image (each plane with split planes) is sampled with a seed derived from this seed and its identity — `position_id` and its `t`/`z`/`c` indices — never from its place in the run, so an image gets the same rectangles alone or among other files (see "Patch identity" below). Must be `>= 0`. |

---

## `config.whole_fov` (`WholeFOVConfig`, used for `mode="whole_fov"`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `square_cover` | `bool` | `False` | Also return, after each whole-FOV region, the ⌈long ÷ short⌉ squares of the short side that cover it, evenly spaced (mode `whole_fov_square`, `patch_id` `<parent>/sq<N>`, `parent_id` set; issue #33). |

---

## `config.grid` (`GridConfig`, used for `mode="grid"`)

Settings for regular tiled patches covering the full image.

| Field | Type | Default | Notes |
|---|---|---|---|
| `tile_size` | `int` | `128` | Side length of each square tile in pixels. Must be `> 0`. |
| `overlap` | `float` | `0.0` | Fractional overlap between adjacent tiles. Must satisfy `0 <= overlap < 1`. For example, `0.1` means 10% overlap. |
| `include_partial` | `bool` | `False` | When `True`, edge tiles that are smaller than `tile_size` are included. When `False`, only full-sized tiles are returned. |
| `partial_mode` | `str` | `"flush"` | Which edge tiles `include_partial=True` adds. `"flush"` (default, the behaviour before this option existed): the tiles clipped by the far edge **and**, when the stride does not already end there, one full tile flush with the far edge, which overlaps its neighbours. `"clipped"`: only the clipped edge tiles (each records its full `requested_y1`/`requested_x1`); with `overlap=0` no two tiles overlap and every pixel is in exactly one tile. `"clipped"` requires `include_partial=True`. |

---

## `config.filters` (`FilterConfig`, optional post-generation filters)

Intensity-based filters applied after patch regions are generated. All fields default to `None` (disabled). Filters are applied to patches from all modes.

| Field | Type | Default | Notes |
|---|---|---|---|
| `min_foreground_fraction` | `float \| None` | `None` | Minimum fraction of pixels in a patch that must be "foreground" (above threshold). Must be in `[0, 1]` when set. Requires `foreground_threshold` to determine what counts as foreground (falls back to image mean if `None`). |
| `foreground_threshold` | `float \| str \| None` | `None` | Threshold used to binarise pixel values for foreground fraction computation. Can be a numeric value or one of the strategy strings below. |
| `min_mean_intensity` | `float \| None` | `None` | Minimum mean pixel intensity for a patch to be kept. Patches with mean intensity below this value are discarded. |

**Supported `foreground_threshold` strategy strings:**

| String | Description |
|--------|-------------|
| `"mean"` | Mean intensity of the full image |
| `"median"` | Median intensity of the full image |
| `"otsu"` | Otsu's threshold (requires `scikit-image`) |
| `"percentile:<N>"` | N-th percentile of the full image (e.g. `"percentile:75"`) |

> Threshold statistics are computed over the entire image array (all channels/planes). Pass a pre-projected image if channel-specific filtering is needed.

---

## `config.extraction` (`ExtractionConfig`, array extraction and resize)

Controls how patch arrays are extracted from images and optionally resized.

| Field | Type | Default | Notes |
|---|---|---|---|
| `pad_mode` | `str` | `"clip"` | Border handling when a patch extends outside the image. `"clip"` clamps to image bounds; `"constant"` pads with `constant_values`. |
| `constant_values` | `float` | `0.0` | Fill value used when `pad_mode="constant"`. |
| `output_size` | `int \| tuple[int,int] \| None` | `None` | Target size after extraction. `None` keeps the natural patch size. A positive `int` produces a square output of that side length. A `(height, width)` tuple specifies the exact output shape. Requires `scipy` for interpolation orders `> 0`. |
| `resize_order` | `int` | `1` | Spline interpolation order for resizing (0 = nearest, 1 = bilinear, 3 = bicubic). Must be `>= 0`. |
| `apply_mask` | `bool` | `False` | **Cell mode only.** When `True`, pixels outside the selected cell label are set to `mask_fill_value` before resize/export. |
| `mask_fill_value` | `float` | `0.0` | Fill value for pixels outside the selected label when `apply_mask=True`. |
| `split_channels` | `bool`, `list[int]` or `list[str]` | `False` | When `True`, every region becomes one single-channel patch per channel of its image instead of one multichannel patch (requires a `C` axis in `common.image_axes`). A list of channel indices (for example `[1]` or `[0, 2]`) gives one patch per selected channel only, in channel order; an index the images do not have stops the run before any patch is made or written, naming the image's channel count (follow-up C5). In a folder or tree run, a list of the **channel IDs** of the run's description (for example `["DNA"]`; CP-3) is resolved per image by the run's resolver (`RunInputs.resolve_request`), so each ID selects its own C index in each image; an ID some image lacks, or no image has, is refused before anything is read (`A_SCOPE_INCOMPLETE`, "present in k of n acquisitions"; no subset is taken). In-memory images have no channel IDs (refused). Every split patch's `channel_id` column names its channel. Channel patches keep the region's coordinates and `position_id`, and get `c` = channel index (a `c<index>` level of `patch_id`; see "Patch identity" below). Filters are evaluated on the whole multichannel region before splitting; masking and resizing are applied before splitting too. |

> `apply_mask=True` is only supported in `mode="cell"` and requires masks to be provided.

---

## `config.export` (`ExportConfig`, saving options)

Controls whether and how patch arrays are written to disk.

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch for file export. When `True`, patch arrays are written to `common.output_dir`. |
| `save_metadata_csv` | `bool` | `True` | When `True`, a `patches.csv` file with per-patch metadata is written alongside the patch files, together with the per-image table `images.csv` (see "Per-image table" below) and the per-position table `positions.csv` (see "Per-position table"). |
| `image_format` | `str` | `"tif"` | Output file format. One of `"tif"`, `"tiff"` (both write TIFF), or `"npy"` (NumPy binary). |
| `save_run_config` | `bool` | `True` | When `True`, `run_config.json` (the full serialized config) is written next to the patch files and named in every row's `config_file`. `run_patching_condition_repeat_tree(combine_metadata=True)` turns it off per folder and writes one for the whole tree instead. |

> To export files, set `export.enabled=True` **and** `common.output_dir` to a valid path. Enabling export automatically triggers array extraction even if `common.extract_arrays=False`.

---

## Finding images

Images are found with the **MorphoBricks preprocessing brick's `discover_experiment`** (through `cellpatcher.io.discover_images`; issue #45, owner decision of 2026-10-30) in `list_supported_files` and the deprecated `discover_condition_repeat_folders` (audit A-m4: it warns, and its `mask_folder`, the leaf mirrored under `mask_root_path`, is a pairing the preprocessing run does not use). Since v0.2.0 folder and tree runs patch the images their preprocessing run recorded and discover nothing (design v0.4 §6); the rules below are the brick's, which that run applied.

- **Images are TIFF files** (`.tif`, `.tiff`, any case, optional `.ome`) **and OME-Zarr folders** (`.zarr`, usually `.ome.zarr`). Nothing else is an image.
- **PNG, JPEG, BMP, GIF and NumPy (`.npy`) images are no longer read**, and microscope formats (`.czi`, `.nd2`, `.lif`, `.lsm`, `.ims`, `.vsi`, ...; `cellpatcher.io.UNSUPPORTED_IMAGE_EXTENSIONS`) never were. Such a file where images are found is refused with an error naming it, never skipped: convert it to TIFF or OME-Zarr, or move it out of the image folder. Other files (`.txt`, `.csv`, ...) are ignored, as before. Exported patches may still be written as `.npy` (`export.image_format`) and are read back by `read_array` and the Torch dataset.
- **A folder holding both images and subfolders is refused**, in folder runs as in tree runs (the brick's error `tiff_outside_leaf` since preprocessing v0.3.2; every brick refuses it). A folder run reads the images directly in one folder: a folder of image subfolders is a tree (`run_patching_condition_repeat_tree`).
- Every error the brick reports is a refusal (`duplicate_position_id`, `case_conflict`, `folder_too_deep`, `reserved_folder_name`, `unreadable_folder`). A folder with no images and no subfolders is left out with a `UserWarning` naming it, as the brick warns. A link to a missing image is refused.
- A mask folder that is the image folder or inside it is refused before discovery (`[masks_inside_experiment]`): its folders would be read as image folders.

**Behaviour change (2026-10-30).** Formerly, PNG/JPEG/BMP/GIF/NumPy images were patched; a folder run ignored the subfolders of its image folder; folders below the technical-repeat level were ignored; `Ctrl` and `ctrl` were two conditions; a `Single` folder or a `single` folder with siblings was a label like any other; and an empty condition folder was skipped without a message (an empty repeat folder stopped the run; it is now left out with a warning, as the brick does). `discover_condition_repeat_folders(supported_extensions=...)` is refused; `list_supported_files` lists TIFF/OME-Zarr images and refuses the former formats unless given `supported_extensions`. `cellpatcher.SUPPORTED_EXTENSIONS` is now `{".tif", ".tiff", ".zarr"}`.

---

## `config.matching` (`MatchConfig`, which masks to pair)

Images and masks are paired by the **preprocessing run** (its `find_masks`; issue #45, owner decision of 2026-10-30). Since v0.2.0 CellPatcher reads the pairing the run recorded (`RunInputs.read_mask`; `validate_image_and_mask_folders` returns it) and has no pairing of its own (`pair_masks` was removed in CP-2b), so every brick reads the same pairs. The layouts below are the brick's.

A mask is a **TIFF file** (`.tif`, `.tiff`, any case, optional `.ome`) in the mask folder, in either layout (they may be mixed):

| Layout | Example for image `ctrl/rep1/img_01.tif` |
|---|---|
| `<mask type>/<image's folders>/<image name>.tif` | `masks/cells/ctrl/rep1/img_01.tif` |
| `<image's folders>/<image name>_<mask type>.tif` | `masks/ctrl/rep1/img_01_cells.tif` |

The image's folders are its path below the image folder (folder run: none) or the image root (tree run); the image name is its `position_id` (the file name without its extension, D85). Names are compared ignoring case. The mask folder may not be inside the image folder/root.

| Field | Type | Default | Notes |
|---|---|---|---|
| `mask_type` | `str \| None` | `None` | The mask type to use (`"cells"` for `img_cells.tif` or `cells/img.tif`, or a label image's name). Folder and tree runs (CP-2a): one of the types the preprocessing run recorded; `None` takes the run's only type and refuses a run with several. `validate_image_and_mask_folders(..., mask_type=)` chooses the same way. A label image and a folder mask of the same type for one image are refused by the preprocessing run (`duplicate_mask`). |
| `mask_id` | `str \| None` | `None` | `mode="cell"`, file runs (Stage 2 CP-2): the object set by the **mask ID** of the run's description (`masks.<ID>`). The run's resolver is asked for `MaskRequest(mask_id, "objects", cell.binary_mask_policy)` and its refusals stop the run with their codes before any read (`A_SCOPE_INCOMPLETE` "present in k of n" for an undeclared ID or a mask some image lacks, `A_MASK_UNCHECKED`, `A_MASK_NOT_INSTANCES`). `None`: the description's one mask whose source type is `mask_type` (or the run's only type). Given with `mask_type`, both must name the same mask. Recorded in `run_config.json`. |

The preprocessing run refuses an ambiguous mask (one that could belong to two images or two types), a mask whose header it cannot use (no Y/X of its image, non-integer pixels, axes other than Y, X and named Z/T: save each label channel as its own mask type), and two masks of one type for one image; a file in the masks folder that pairs with no image is a warning in its report. CellPatcher refuses an image the run recorded no mask of the type for (in `cell` mode, and in `random` mode when masks are read), and reads each mask's labels through the brick's `check_labels`.

**Former options (removed in v0.1.1).** `strategy`, `mask_suffixes`, `case_sensitive` and `ignore_unrequested_masks` (in `MatchConfig`, a `run_config.json`'s `matching` section, and as arguments of `validate_image_and_mask_folders`, `run_patching_pipeline`, `run_patching_condition_repeat_tree` and `run_patching_from_folders`) are refused by name with a `ValueError` saying to use `matching.mask_type`, whatever their value: an old `run_config.json` that holds them fails until they are removed from its `matching` section. `find_candidate_matches`, `filter_shared_matches`, `select_requested_masks`, `LEGACY_MATCHING_DEFAULTS` and `cellpatcher.io.image_stem` were removed with them (D86).

**Behaviour change (from CellPatcher's former `exact_or_suffix` default).** Now paired: masks of any type `<image>_<type>.tif` (former suffixes only `_mask`, `_labels`, `_label`, `_seg`, `_segmentation`), the `<type>/` folder layout, and several types in one folder with `mask_type` set. No longer paired, and refused: masks named exactly like their image, non-TIFF masks, ambiguous names as above, and a mask folder inside the image folder. In tree runs, the whole tree is paired at once; `random` mode with a mask root no longer skips images whose leaf has no mask folder.

> `config.matching` is only relevant for `mode="cell"` and `mode="random"` (the only modes that load masks from disk).

---

## `config.normalization` (`NormalizationConfig`)

Normalization with the statistics of the preprocessing brick, following the [normalization contract](normalization_contract.md) (decisions in the [plan](plan.md)). Off by default: with `enabled=False`, patch arrays and every existing `patches.csv` column are exactly as before, and the `normalization_*` columns stay empty.

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `False` | Normalize every patch. |
| `plane_histograms` | `str \| None` | `None` | Path of the preprocessing brick's `plane_histograms.csv`, unrenamed in the run's output folder beside its `run_manifest.json` (since v0.2.0 a run of preprocessing v0.4 or later, read through `RunInputs`; earlier runs are refused with a message asking to run the brick again); required for every scope except `"patch"`. Give CellPatcher the same experiment root as the preprocessing brick. |
| `manifest` | `str \| None` | `None` | The run's `run_manifest.json`, which must be the one in the table's folder (`not_the_run_output` otherwise); `None` uses it. Since v0.2.0 the run is read through the preprocessing brick's `RunInputs` (design v0.4 §6), which needs a v0.4 manifest and refuses a table, design, image or mask changed since the run (`preprocessing_run_refused`). |
| `method` | `str` | `"percentile"` | `"minmax"`, `"percentile"`, `"zscore"` or `"robust_zscore"` (MAD × 1.4826). |
| `percentile` | `(float, float)` | `(1.0, 99.8)` | Low and high percentile for `"percentile"`, `0 ≤ low < high ≤ 100`. |
| `scope` | `str` | `"image"` | Pixels the parameters come from: `"biological_repeat"` (one repeat group: the same biological repeat in every condition, technical repeats merged, as the run's published experiment description pairs them: `Fov.repeat_group` through `RunInputs`, never the derived `experiment_design.csv`; an image without one is refused, `missing_repeat_group`), `"image"` (all Z and T planes of the source image) or `"patch"` (the patch's own pixels). Per channel, except for `image_type="rgb"`. `"condition"` and `"technical_repeat"` were removed (October 2026) and are refused. |
| `per_time_point` | `bool` | `False` | Normalize each time point separately (`t` joins the group). |
| `experiment_root`, `masks_root` | `str \| None` | `None` | For `run_patching` at a CSV scope: where the run's experiment and masks folder are now, when moved since the run (owner decision of 2026-11-02); `None`: where the run recorded them. Folder and tree runs refuse them: their image folder/root and `mask_folder_path`/`mask_root_path` say it. Since v0.2.0 (CP-2b). |
| `pooling` | `str` | `"pixels"` | For `scope="biological_repeat"`: `"pixels"` (each pixel counts once) or `"conditions"` (each condition of a repeat group weighs the same). |
| `image_type` | `str` | `"fluorescence"` | `"fluorescence"` (each channel normalized separately) or `"rgb"`: colour images of three 8- or 16-bit channels, normalized with StarDist's percentile method, one offset and scale from the R, G and B pixels together (written for `C0`, `C1` and `C2`). `"rgb"` needs `method="percentile"` and `extraction.split_channels=False`. In a file run it must equal the image type the run's description declares for every image (`Fov.image_type`: `rgb`, or `channels` = `"fluorescence"`), at every scope: the declared type wins and a contradicting setting is refused, naming both (`image_type_differs`). CellPatcher supports RGB units (P24). |
| `patch_region` | `str \| None` | `None` | For `scope="patch"`: `"mask"` (the patch's own label), `"bbox"` (the label's whole bounding box, whatever the patch size) or `"patch"`; `None` means `"mask"` in `cell` mode and `"patch"` otherwise. Padding is never used. |

| `export_values` | `str` | `"normalized"` | What exported files hold: `"normalized"` (float32) or `"raw"` (the source values; `patches.csv` holds the parameters and the loaders normalize on load). A raw patch that is resized or masked is stored as float32 raw values. |
| `memory_limit` | `int` | `4 * 1024**3` | Bytes allowed for the normalized patches of eager delivery (`common.extract_arrays=True`); above it the run stops before extracting and suggests `iter_patch_arrays`. |

**Delivery** (contract §12) — all modes give the same arrays for the same configuration (raw files of resized or masked patches up to float32 rounding):

| Mode | How | Values |
|---|---|---|
| Normalized files (default) | `export.enabled=True` | float32 `.tif`/`.npy` |
| Raw files plus parameters | `export.enabled=True`, `normalization.export_values="raw"`; read back with `cellpatcher.iter_exported_patches(csv)`/`load_exported_patch` or `ExportedPatchTorchDataset`, which normalize on load with each row's `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>` columns (never recomputed; the `;`-separated `normalization_offset`/`normalization_scale` of earlier tables are read too) | Source dtype when the patch is only cropped; float32 raw values when resized or masked |
| In memory, eager | `common.extract_arrays=True` | float32 arrays in `PatchingResult.patch_arrays`, within `memory_limit` |
| In memory, lazy | Run with `extract_arrays=False` (the patches still record their normalization), then, for `run_patching`'s arrays, `cellpatcher.iter_patch_arrays(patches, images, config, masks=...)` or `patch_array(...)` (`PatchTorchDataset(images, patches, config=config)` is built on them), and for a folder or tree run `cellpatcher.iter_run_patch_arrays(patches, config)`, which reads one plane at a time through the preprocessing run | float32, one patch at a time |

```python
from cellpatcher import iter_run_patch_arrays, run_patching_condition_repeat_tree

leaves = run_patching_condition_repeat_tree(root, config=config)  # extract_arrays=False
patches = [patch for leaf in leaves for patch in leaf.result.patches]
for patch, array in iter_run_patch_arrays(patches, config):  # experiment_root=, masks_root= if moved
    ...  # one normalized patch at a time
```

**What happens** (contract §6–§9):

- The pipeline becomes generate → filter → extract → **normalize** → apply mask → resize → export. Filters see raw values; `extraction.mask_fill_value` is in normalized units (`NaN` allowed); resizing works on the float32 values. Padding added by `pad_mode="constant"` is normalized like the raw value `constant_values`.
- Each patch becomes `(pixels - offset) / scale` per channel, float32, without clipping.
- The repeat and image scopes pool the CSV rows of the patch's group (the normalization brick's `scope_groups`: the repeat group across all conditions, or the image; per channel, and per `t` with `per_time_point`), including planes that were not patched.
- Before anything is normalized, every patched plane must have its CSV row with the same labels, dtype, size, exact finite extrema and finite pixel count, and each image's size, modification time and axes must match the run's manifest. Every image without rows is listed in one message.
- Normalization needs one plane per patch: `common.t_policy` `"keep"` along a T axis longer than 1 stops the run; Z>1 is refused outright (issue #83).
- A group with no finite pixels or zero spread stops the run with the list of groups. At the patch scope, a patch whose own pixels have zero spread or no finite pixels is discarded (decision D8): it is not exported or delivered, and one `NormalizationWarning` lists every discarded patch.
- Warnings (`cellpatcher.normalization.NormalizationWarning`): approximate parameters (float images); group and image statistics include background; the `biological_repeat` scope removes differences between repeats and keeps those between conditions; a repeat group without images from some condition.

**Recorded:** written after `source_relative_path` in `patches.csv`: `normalization_method` (e.g. `percentile 1-99.8`), `normalization_scope`, `normalization_group` (the repeat group label, `<source_relative_path>`, or `<source_relative_path>::<patch_id>` for the patch scope; `::t=<t>` with `per_time_point`) and `normalization_exact` (`true`/`false`), then one `Normalization_Offset_C<i>`/`Normalization_Scale_C<i>` pair per channel, grouped by channel (full-precision numbers from the normalization brick's `Parameters.to_columns()`; a split-channel patch fills only its own channel's pair). In memory, `PatchRegion.normalization_parameters` is the brick's `Parameters` (`offset`, `scale`, `channels`); `normalization_offset`/`normalization_scale` are deprecated read-only properties giving the former `;`-separated text. `run_config.json` gains the `normalization` section, a top-level `normalization_inputs` object (`plane_histograms_sha256`, `schema_version`, `manifest_used`, `manifest_version`, `preprocessing_version`, and for `biological_repeat` `description_sha256`, the SHA-256 of the run's published description) and `normalization_settings` (the brick's `NormalizationSettings.to_dict()`); a `run_config.json` without `normalization` loads with normalization off. With normalization, the images of a run must have the same number of channels.

**In-memory runs:** `run_patching` normalizes too; for the CSV scopes each image needs `image_metadata=[{"source_relative_path": ...}]` and the right `common` labels (folder and tree runs fill both). The run is opened where it recorded the experiment and masks folder, or where `normalization.experiment_root`/`masks_root` say they are now (an experiment moved since the run; owner decision of 2026-11-02). The manifest's file checks run in folder and tree runs.

`cellpatcher.statistics` (loading `plane_histograms.csv`/`run_manifest.json`, contract §5 checks) exists since Milestone 1.

---

## Mode-to-section reference

The table below shows which config sections are written by `save_config_for_mode(mode, path)`. The common sections (`common`, `filters`, `export`, `extraction`) are always included.

| Mode | Common sections | Mode-specific sections |
|------|----------------|------------------------|
| `"cell"` | `common`, `filters`, `export`, `extraction` | `cell`, `matching` |
| `"random"` | `common`, `filters`, `export`, `extraction` | `random`, `matching` |
| `"grid"` | `common`, `filters`, `export`, `extraction` | `grid` |
| `"whole_fov"` | `common`, `filters`, `export`, `extraction` | `whole_fov` |

`save_default_config(path)` writes all sections regardless of mode.

---

## Patch identity (`patch_id`, `position_id`, `object_set`)

Owner decisions D44, D45 and D47 (2026-10-26). A patch's identity comes only from its field of view and its own coordinates, never from the place of its image in the run, so the same file gives the same `patch_id` alone, in a folder of ten files, after a file that sorts first, or in a tree run beside other wells; a rerun writes a byte-identical `patches.csv`. The code is `cellpatcher.ids`.

**`position_id`** (always filled; D45, amended by MorphoBricks D85 on 2026-10-29). The one supplied with `image_metadata=[{"position_id": ...}]` wins, kept verbatim; it must be text (a number is refused, never converted) without leading or trailing whitespace, and not a name tables read as missing (`""`, `NA`, `nan`, `none`, `null`, `n/a`, `<NA>`, any case). The same rule holds for `common.condition`/`biological_repeat`/`technical_repeat` and `image_metadata["technical_repeat"]`: the preprocessing brick's `check_identity_text`, shared by every brick (D86, owner decision D-e; before, only empty text was refused). Plane indices `t`/`z`/`c` given in `image_metadata` are Python or NumPy integers of at least 0 (`check_plane_index`; text such as `"1"`, accepted before D86, is refused). Otherwise one file is taken as one XY position, named by the **image name without its extension**, computed by `morphobricks_preprocessing.discovery.position_id` like every MorphoBricks brick (`img.ome.tif` and `img.ome.zarr` give `img`). It is unique within its technical repeat, case-insensitively, and names of one level that differ only in case (`A` and `a` as technical repeats, anywhere in the run; `Img` and `img` as positions of one technical repeat, even with different plane indices) are refused (`check_identity_case`, D-e); the FOV is the four fields `condition`, `biological_repeat`, `technical_repeat`, `position_id`. An in-memory image passed to `run_patching` with neither a `position_id` nor an `image_file` is **refused** with an error naming its place in the run: pass `image_metadata=[{"position_id": ...}, ...]`, one entry per image (an `{"image_file": ...}` entry also works: its stem then names the position). Folder and tree runs read one image per file: a file holding several images (an OME-Zarr plate or multi-position store, a TIFF with several series) is refused, whether its planes are read one at a time (a plane policy `"all"`, `"first"` or `"select"`) or the file is read whole (`"keep"`, projections), with an error naming the file and its number of images/series. Give one file per position, or pass `position_id` (with the plane indices) through `run_patching`.

**`object_set`** (cell mode; `None` in other modes). Precedence: `image_metadata[i]["object_set"]`; then `common.object_set`; then, in folder and tree runs, the mask's own name — the OME-NGFF label image name with `mask_source="ngff_labels"` (e.g. `cells`), or the mask type of the preprocessing pairing (`img_cells.tif` or `cells/img.tif` → `cells`, `img_mask.tif` → `mask`); finally `"mask"` (the mask role) when nothing names it (in-memory runs). Characters other than letters, digits, `.`, `_`, `-` in a derived name become `_`.

**`patch_id` grammar.** Levels are joined by `/`; absent levels are omitted:

```text
patch_id := condition "/" biological_repeat "/" technical_repeat "/" position_id "/"
            [object_set "/"] ["t" T "/"] ["z" Z "/"] ["c" C "/"] token
token    := "label" LABEL                                (cell)
          | "grid-y" Y0 "-x" X0 "-h" H "-w" W            (grid)
          | "random-y" Y0 "-x" X0 "-h" H "-w" W ["-n" K] (random)
          | "whole_fov"                                  (whole_fov)
```

`T` is `t` with 4 digits, `Z` is `z0` with 3 digits, `C` is `c` (no padding); each is present when the patch has that index (a plane split or selected by `t_policy`/`z_policy`, an index from `image_metadata`, or a split channel). `LABEL` is the object's own `label_id`; `Y0`/`X0`/`H`/`W` the rectangle's top-left corner and size in pixels. `K` (2, 3, ...) numbers repeated draws of the same rectangle in one random plane, in sampling order. Examples:

| Run | `patch_id` | Exported file |
|---|---|---|
| Cell mode, `ctrl/2026_01_01/B02/img.tif` with mask `img_cells.tif` | `ctrl/2026_01_01/B02/img/cells/label42` | `ctrl_2026_01_01_B02_img_cells_label42.tif` |
| Grid mode, `img.tif` in a flat folder | `single/single/single/img/grid-y0-x128-h128-w128` | `single_single_single_img_grid-y0-x128-h128-w128.tif` |
| Cell mode, time series with `t_policy="all"`, `split_channels=True`, label image `nuclei` | `ctrl/2026_01_01/B02/fov1/nuclei/t0003/c1/label7` | `ctrl_2026_01_01_B02_fov1_nuclei_t0003_c1_label7.tif` |

Exported files are named `sanitize_filename(patch_id)` plus the extension: `/` and any other character outside letters, digits, `.`, `_`, `-` become `_`. The export refuses two ids that give the same file name, and every run refuses two patches with the same `patch_id` (other than repeated random draws): for example two images given the same `position_id` without different `t`/`z`/`c` indices.

**Random mode is seeded per FOV.** Each image, or each plane with split planes, is sampled with its own seed: `numpy.random.SeedSequence([random.seed, *w])` with `w` the eight 32-bit little-endian words of the SHA-256 of the canonical identity, the JSON array `["<condition>","<biological_repeat>","<technical_repeat>","<position_id>","t=<t>","z=<z>","c=<c>"]` (compact JSON: `json.dumps(parts, ensure_ascii=False, separators=(",", ":"))`, UTF-8; the axes kept by `seed_scope`; `-` for an absent index, an index as an integer) (the whole FOV key, D85; JSON, so a label holding `|` or `,` never reads as two), from which one 64-bit state word is the generator's seed (`cellpatcher.ids.random_seed`; Python's salted `hash()` is not used). The same image therefore gets the same rectangles, and ids, alone, in a folder of ten files or after a file that sorts first; another `random.seed`, position or plane gets others. Channels split with `split_channels` share their region's rectangles. `random.seed=None` stays unseeded (different on every run).

**`image_id`** (`image_0000`, ...) is still the image's place in the run (sorted file order in a folder run, restarting in every leaf of a tree run). It joins `patches.csv` to `images.csv` and indexes a run's image list for lazy delivery; it is not an identity across runs.

**Before 2026-10-29** (D85, breaking change): a derived `position_id` was `<condition>/<biological_repeat>/<technical_repeat>/<file stem>` (`Path.stem`, so `img.ome.tif` gave `img.ome`), the four levels were not otherwise in `patch_id`, and the random seed hashed `position_id` and the plane only. Ids of derived positions keep their text; a supplied `position_id` now follows the three labels in `patch_id`, and random rectangles change.

**Before 2026-10-26** (breaking change): `patch_id` was `<image_id>_<mode>_<index>` (`image_0000_cell_00042`, `image_0000_grid_00003`, `image_0000_random_00007`, `image_0000_whole_fov_0`) plus `_t<NNN>`/`_z<NNN>` for split planes and `_c<n>` for split channels, so it changed with the run; `position_id` was only filled with `split_channels`; `binary_mask_policy` defaulted to `"connected_components"`; random mode sampled image *k* of a run with `random.seed + k` (plane *k* with split planes); unnamed in-memory images were accepted; and a multi-series TIFF was read from its first series.

## `PatchRegion` fields

Every generated patch is a `PatchRegion` dataclass. The fields below are populated for all modes; `to_dict()`/the exported `patches.csv` also include the `height`/`width` computed properties as extra convenience columns.

| Field | Type | Description |
|-------|------|-------------|
| `patch_id` | `str` | Run-independent identifier of the patch: the FOV key (`condition`, `biological_repeat`, `technical_repeat`, `position_id`), object set, plane indices and label or rectangle (see "Patch identity" above). Its sanitized form is the export filename stem. |
| `image_id` | `str \| None` | The source image's place in the run (e.g. `"image_0000"`); joins `images.csv`, not an identity across runs. |
| `mode` | `str \| None` | Patching mode that generated this patch. |
| `condition` | `str \| None` | From `config.common.condition`. See the shared CSV contract below. |
| `biological_repeat` | `str \| None` | From `config.common.biological_repeat`. See the shared CSV contract below. Formerly `repeat` (deprecated alias). |
| `technical_repeat` | `str \| None` | From `config.common.technical_repeat`, or per image from `image_metadata`. See the shared CSV contract below. |
| `position_id` | `str \| None` | Stable spatial acquisition-position identifier, shared by all Z/C/T planes from one XY position. Always filled: supplied via `image_metadata` (non-empty text, kept verbatim), else the image name without its extension (see "Patch identity" above). |
| `image_file` | `str \| None` | Source image filename. Populated automatically by `run_patching_pipeline`; `None` for direct `run_patching()` calls unless supplied via `image_metadata`. |
| `mask_file` | `str \| None` | Source mask filename (cell/random mode only, when a mask was loaded). Same population rule as `image_file`. |
| `y0`, `y1` | `int` | Vertical extent of the patch in pixels (`y0` inclusive, `y1` exclusive). |
| `x0`, `x1` | `int` | Horizontal extent of the patch in pixels (`x0` inclusive, `x1` exclusive). |
| `z0`, `z1` | `int \| None` | Selected Z-slice range (`z0` inclusive, `z1` exclusive — one slice thick). Populated when the caller passes `image_metadata=[{"z": ...}, ...]` to `run_patching`; `None` otherwise. |
| `t` | `int \| None` | Selected time index. Populated the same way via `image_metadata=[{"t": ...}, ...]`. |
| `c` | `int \| None` | Selected channel index. Populated the same way via `image_metadata=[{"c": ...}, ...]`. |
| `label_id` | `int \| None` | Cell label value from the mask (cell mode only). |
| `center_y`, `center_x` | `float \| None` | Patch centre coordinates in pixels. |
| `center_z` | `float \| None` | Reserved for future multidimensional support; not yet populated. |
| `touches_border` | `bool` | Whether the patch touches the image border. |
| `area_px` | `int \| None` | Object area in pixels (cell mode only). |
| `patch_file` | `str \| None` | Exported patch filename, stamped by `export_patches`/`export.enabled=True`. `None` when the patch was not exported to disk. |
| `output_height`, `output_width` | `int \| None` | Final exported array shape. Differs from `y1 - y0`/`x1 - x0` when `extraction.output_size` resizes the patch. `None` unless arrays were extracted (`extract_arrays=True` or `export.enabled=True`). |
| `config_file` | `str \| None` | Filename of the full run-config JSON saved alongside `patches.csv` (see below). `None` unless `export.enabled=True`. |
| `valid`, `reason` | `bool`, `str \| None` | Reserved for future rejected-patch reporting; not yet populated (`valid` is always `True`). |
| `source_relative_path` | `str \| None` | The source image's path relative to the experiment root, with `/` separators, from where it sits in the folder tree (`control/img.tif` for an image directly in `control/`, never the `single` output folders). A linked image or folder keeps the link's path under the root, not its target's, as the preprocessing brick records it (see "Trees of links" below). Links the patch to the preprocessing brick's `plane_id` ([normalization contract](normalization_contract.md) §2). Tree runs make it relative to `image_root_path`; `run_patching_pipeline` to the image folder (the file name); direct `run_patching()` calls take it from `image_metadata`, else `None`. Followed in `patches.csv` by the six `normalization_*` columns (see Normalization above). |
| `image_height`, `image_width` | `int \| None` | Y/X size of the source image (the plane, for split Z/T planes) that `y0/y1/x0/x1` refer to. Set by `run_patching` for every patch, with or without arrays. |
| `pad_top`, `pad_bottom`, `pad_left`, `pad_right` | `int \| None` | Rows/columns of `extraction.constant_values` added on each side where the rectangle extends past the image (`extraction.pad_mode="constant"`), counted before resizing; `0` when nothing was padded (`"clip"` never pads, and the built-in modes keep every rectangle inside the image). `None` unless arrays were extracted. |
| `resized` | `bool \| None` | Whether `extraction.output_size` changed the patch's Y/X shape: `False` without an output size, and for a patch that already had it (resizing then leaves the pixels unchanged). `None` unless arrays were extracted. |
| `output_axes` | `str \| None` | Axes of the delivered/exported array, in order: `common.image_axes` without the Z/T axes a plane policy resolved and, for split channels, without `C` (e.g. `"CYX"`, `"YX"`, `"ZCYX"` with `z_policy="keep"`). Exported TIFF/NPY files carry no axes of their own. `None` unless arrays were extracted. |
| `object_set` | `str \| None` | Cell mode: the object set (mask) the patch's `label_id` belongs to (see "Patch identity" above); `None` in other modes. |
| `channel_id` | `str \| None` | The canonical channel ID of the patch's `c` in its own image (CP-3): in a file run, the run's description through `RunInputs` (`Fov.channel_ids[c]`), so one channel has one ID whatever its C index in each image; in memory, `image_metadata["channel_ids"][c]` when given. `None` for a patch without `c` (every channel in one patch, or no C axis). Appended after the patch-geometry contract columns (`requested_*` ... `square_index`); `patch_id` keeps `c` and is never parsed for it. |

The geometry columns come after every earlier one, `object_set` after them, then the patch-geometry contract columns and `channel_id`, so the order of the existing columns does not change.

### Shared MorphoBricks CSV contract

`condition`, `biological_repeat`, `technical_repeat` (optional), `position_id` (when needed), `image_file`, `mask_file`, and `patch_file` are the columns of the MorphoBricks ecosystem's shared CSV contract (`AGENTS.md` §5), alongside the 0-based plane indices `z0`/`z1`, `c` and `t` (see the table above). CellPatcher populates them as follows:

- `condition`/`biological_repeat` — either set `config.common.condition`/`config.common.biological_repeat` explicitly and call `run_patching_pipeline` on a single folder, or call `run_patching_condition_repeat_tree(image_root_path, ...)` on a `raw_data/<condition>/<biological_repeat>/` root and let both be derived per leaf from the folder path (see below). A label that is not given, or set to `None`, is written as `"single"` (`SINGLE_CONDITION_LABEL`, `SINGLE_BIOLOGICAL_REPEAT_LABEL`, `SINGLE_TECHNICAL_REPEAT_LABEL`), as for a missing folder level, so the label columns are never empty.
- `technical_repeat` — the well, dish or coverslip within the biological repeat. Tree runs derive it from an optional `<condition>/<biological_repeat>/<technical_repeat>/` level and write `"single"` when a biological-repeat folder holds its images directly, so the column is never empty in a tree run. For a single folder set `config.common.technical_repeat`; when one folder holds several wells, pass `image_metadata=[{"technical_repeat": ...}, ...]` to `run_patching`.
- `position_id` — always filled (see "Patch identity" above): supplied per image via `image_metadata` (non-empty text, kept verbatim), else the image name without its extension. The canonical MorphoBricks FOV key is `condition` + `biological_repeat` + `technical_repeat` + `position_id`.
- `image_file`/`mask_file`/`patch_file` are populated automatically by `run_patching_pipeline` (folder-based runs, including each leaf of `run_patching_condition_repeat_tree`) and `export_patches` respectively.

### Per-image table (`images.csv`)

Wherever `patches.csv` is written (`export.enabled=True` with `save_metadata_csv=True`: `run_patching`, `run_patching_pipeline`, each leaf of `run_patching_condition_repeat_tree`, its combined output, and `export_patches`), `images.csv` is written beside it (`cellpatcher.export.IMAGE_TABLE_FILENAME`). It has **one row per input image the run considered**, including images that gave no patch, so a reader can tell an image without patches from one that was never patched. In memory, `PatchingResult.image_records` holds the same rows as `ImageRecord` objects (also without export), and `PatchingResult.image_table_file` names the file.

| Column | Notes |
|---|---|
| `condition`, `biological_repeat`, `technical_repeat` | The labels the image's patches get in `patches.csv` (`"single"` when not given). |
| `image_id` | `image_0000`, ... as in `patches.csv`: unique within one `patches.csv` of a folder or leaf, the only key of a direct `run_patching` call without file names. |
| `image_file`, `mask_file`, `source_relative_path` | As in `patches.csv`; empty when not known (direct `run_patching` without `image_metadata`, no mask). |
| `image_height`, `image_width` | The image's Y/X size, as in the patches' `image_height`/`image_width`. Empty for a skipped image. |
| `n_patches` | The image's number of rows in `patches.csv` (one per channel patch with `split_channels`). |
| `status` | `processed`: `n_patches` > 0. `no_patches`: the image was read and patched, but no patch was kept. `skipped`: the image had nothing to patch. |
| `reason` | Empty for `processed`. For `no_patches`, the step that left none: `the patch mode generated no region in this image` (e.g. no labeled object in its mask, every object excluded by border or area), `the filters removed all <n> region(s)`, or `normalization discarded all <n> patch(es): they cannot be normalized from their own pixels` (decision D8). For `skipped`: `the image has no plane to patch along the axes split by z_policy/t_policy` (a Z or T axis of length 0, which a plane split cannot patch). |
| `position_id` | The `position_id` every patch of the image carries in `patches.csv`, also for an image without patches: supplied via `image_metadata`, or the default of "Patch identity" above (one file is one position). Never empty since 2026-10-26. |
| `n_regions`, `n_kept`, `n_normalized` | Regions the patch mode generated, those left by the filters, and those left by normalization (equal to `n_kept` without it), counted before `split_channels`; they tell the three `no_patches` reasons apart without reading `reason`. `0` for a skipped image. |
`position_id`, `n_regions`, `n_kept`, `n_normalized` were added after the first release of the table, as its last columns, so earlier columns keep their positions. The last column of v0.1.2, `image_read_transposed`, was removed in v0.2.0 (CP-2b): since preprocessing v0.4 refuses every Y/X swap it was always `false` or empty.

There is no `failed` status: an image that cannot be read, matched to a mask or patched stops the whole run with an error before any table is written, as before, so no image is ever left out silently.

**Joining the two tables.** Every `patches.csv` row's image has exactly one `images.csv` row with the same `condition`/`biological_repeat`/`technical_repeat`/`image_id`/`image_file`, and that row has the patch's `position_id` (`image_id` alone is not unique across a combined tree CSV; `source_relative_path` is unique in folder and tree runs). An image's `n_patches` is its number of `patches.csv` rows. `export_patches` called directly writes the records it is given (`image_records=`), or else one `processed` row per image of the patches, since it cannot know about images without patches.

### Per-position table (`positions.csv`)

`images.csv` has one row per input image, which is not always one position: a position split over several time-point files (`image_metadata` with the same `position_id` and different `t`) has several. Beside `images.csv` (same places, same `save_metadata_csv` switch; `cellpatcher.export.POSITION_TABLE_FILENAME`, `export_patches(position_table_filename=)`), `positions.csv` has **exactly one row per position** of the run: the preprocessing run's FOV (`Fov.key`, as `RunInputs` gives it), grouping every source image, plane and channel of it. In memory, `cellpatcher.types.position_records(image_records)` builds the same rows as `PositionRecord` objects, and `PatchingResult.position_table_file` names the file. It imports nothing of MorphoFeature or MorphoCoverage.

The row's key is the FOV key, `condition` + `biological_repeat` + `technical_repeat` + `position_id`, with the names and values of `patches.csv` and `images.csv`, so it joins both. No label, `image_id` or file name stands in for it: equal basenames in two wells or repeats are two rows, and `image_id` (a place in the run) is not a column. A record with no `position_id` (only possible for `export_patches` called with hand-built patches that have none) is its own row with an empty `position_id`, never merged with another.

| Column | Notes |
|---|---|
| `condition`, `biological_repeat`, `technical_repeat`, `position_id` | The FOV key, as in `patches.csv`. |
| `repeat_group`, `condition_id`, `technical_repeat_id` | The canonical IDs of the preprocessing run's description (`Fov.repeat_group`, `Fov.condition_id`, `Fov.technical_repeat_id`), read from the run, never derived, parsed or guessed. Every preprocessing run has a description (a run without one is refused, `A_DESCRIPTION_REQUIRED`, Spec ADR 0002 AR10) and must give all three for every image: a missing one is refused (`described_run_missing_id`), never left empty. Empty only for in-memory runs that do not give them in `image_metadata`. The folder labels stay in the key columns; for a described run the ID can differ from the folder name (`control` / `ctrl`). Two source images of one position naming different values are refused. |
| `n_source_images` | The input images (files, planes of separate files) grouped into the row: the number of its `images.csv` rows. |
| `source_relative_paths`, `image_files`, `mask_files` | JSON arrays (in order of first appearance, without repeats) of the source images' `source_relative_path`, `image_file` and `mask_file`: provenance only, never the key. `[]` when not known (in-memory images). |
| `status` | What CellPatcher observed of the position: `analysed` (at least one patch); `analysed_no_regions` (patched; the patch mode generated no region, e.g. no labeled object); `all_regions_filtered` (regions generated, the filters removed all); `all_rejected_by_normalization` (regions kept, normalization discarded all, decision D8); `skipped` (every source image had nothing to patch); `partly_skipped` (some source images were skipped, the rest are counted). |
| `reason` | Empty for `analysed`. Otherwise the step that left no patch, worded as in `images.csv`, and, for `skipped` and `partly_skipped`, which source images were skipped and why. |
| `n_regions`, `n_kept`, `n_normalized`, `n_patches` | The sums of the position's source images' counts, with the meanings of `images.csv`; `n_patches` is the position's number of `patches.csv` rows. |
| `schema_version` | `"1"` (`POSITION_TABLE_VERSION`). |

**Versioning.** `schema_version` changes only when a column is renamed, removed or reinterpreted, or a status changes its meaning. A column added later is appended after `schema_version` and keeps the version, so readers select columns by name, as for `images.csv`. `images.csv` and `patches.csv` are unchanged (a new `ImageRecord` field is not an `images.csv` column), and `positions.csv` is a new file: a reader that ignores it loses nothing.

**What a row does not say.** CellPatcher only knows the positions it was given. A missing mask, a file that cannot be read or matched, or a failing image stops the whole run with an error before any table is written, so those are not statuses; an acquisition that failed, was never discovered or is unfinished is not a row. The master acquired-FOV ledger, with the ledger the caller supplies for those, is MorphoFeature's (MorphoFeature#194), not this table's: a position missing from `positions.csv` is *not observed*, never "analysed with zero patches".

### OME-Zarr images

Since v0.2.0 (CP-2a) folder and tree runs read OME-Zarr images and their label images through the preprocessing run like any other image (see "File runs read through the preprocessing run"); this section describes `read_array` and discovery. With the `zarr` extra (`pip install "cellpatcher[zarr]"`), folders whose names end in `.zarr` (usually `.ome.zarr`) are images for `read_array`, which reads level 0 of the store's only image, in its stored axis order, with the preprocessing brick's reader; a store with several images (positions, series or plate wells) is an error. (`read_image`, which put a store into `common.image_axes` by role, was removed in CP-2b with lazy delivery from a path.) Discovery never enters a store, and `img.ome.zarr` pairs with masks named like `img_cells.tif` or `cells/img.tif`. Without the extra, reading a store stops with the install command. OME-Zarr copies written by the preprocessing brick are not read: CellPatcher reads the source images.

**Masks from OME-NGFF labels:** a folder or tree run reads the label images the preprocessing run paired, a label image's type being its name (`matching.mask_type`); `mask_file` is `<image>.ome.zarr/labels/<name>`. The former `common.mask_source="ngff_labels"`/`mask_label_name` were removed in v0.2.0 (CP-2b).

### Condition/biological-repeat folder trees

`run_patching_condition_repeat_tree(image_root_path, *, config=None, mask_root_path=None, ...)` (in `cellpatcher.orchestration`) patches a MorphoBricks-style root once per leaf of its preprocessing run (the images' recorded condition, biological and technical repeat; `discover_condition_repeat_folders`, deprecated, is not used). Both functions keep their historical `condition_repeat` names; the level below each condition is the **biological repeat**. The MorphoBricks folder hierarchy is `experiment/condition/biological_repeat/technical_repeat/images`, with levels omitted only from the bottom up (a level's meaning comes from its position); CellPatcher reads all three levels:

- `<condition>/<biological_repeat>/<image files>` — one or more biological-repeat subfolders, each its own biological repeat.
- `<condition>/<image files>` — **no biological-repeat subfolder.** The whole condition folder is treated as a single implicit biological repeat, labeled `cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL` (`"single"`) — consistently in the discovered `biological_repeat` value, the `condition`/`biological_repeat` stamped on every patch from that folder, and (when exporting) its own `<output_dir>/<condition>/single/` output subfolder, same as an explicit biological repeat gets.
- `<image files>` directly in the root — **no condition subfolders.** The whole dataset is treated as one condition and one biological repeat, labeled `cellpatcher.SINGLE_CONDITION_LABEL`/`cellpatcher.SINGLE_BIOLOGICAL_REPEAT_LABEL` (both `"single"`), exported to `<output_dir>/single/single/`.
- `<condition>/<biological_repeat>/<technical_repeat>/<image files>` — **technical repeats.** A biological-repeat folder holding only subfolders has one subfolder per technical repeat (a well, a dish, a coverslip), each its own leaf, stamped with the subfolder name and exported to `<output_dir>/<condition>/<biological_repeat>/<technical_repeat>/`. A biological-repeat folder holding its images directly is one technical repeat, stamped `cellpatcher.SINGLE_TECHNICAL_REPEAT_LABEL` (`"single"`), and keeps the two-level output folder. The technical level needs the biological-repeat level above it: in `<condition>/<well>/<images>` the wells are read as biological repeats, so put them under a dated (or `single`) biological-repeat folder.
- The leaves, their labels and their images are found by the preprocessing brick's `discover_experiment` (see [Finding images](#finding-images)), so every brick reads the same images under the same labels. A folder mixing image files with subfolders, at any level, is refused (`[tiff_outside_leaf]`; the brick only warns), as are a fourth folder level, folder names differing only in case, a folder named `Single` (another case) or a `single` folder beside other folders, and two images of one folder with the same name apart from the extension or case. A folder with no images and no subfolders is left out with a `UserWarning` naming it. System files and folders are ignored, with the preprocessing brick's rule (`morphobricks_preprocessing.discovery.is_system_name`): names starting with `.` (`.ipynb_checkpoints`, `._img.tif`), `Thumbs.db`, `desktop.ini` and `__MACOSX`.
- **Trees of links.** Image files and folders may be symbolic links, for example a tree that gathers a subset of images stored in a hidden folder of the experiment (`.raw/`, which discovery skips) into the `<condition>/<biological_repeat>/` layout. Links are followed for reading; `image_file` is the link's name and `source_relative_path` the link's path under the root (as the preprocessing brick records it), so the patches of a linked tree are those of a copied tree. Since preprocessing v0.4.0rc4 (CellPatcher v0.2.0rc2) a link is followed only when its target stays inside the experiment (or masks) folder: a link out of it is `link_outside_folder`, one file reached by two paths (two links, or a link and its target) is `linked_twice`, and a loop is `link_loop`; the preprocessing run refuses them, and so does `RunInputs` for a link added after the run (`preprocessing_run_refused`). A link whose target does not exist raises (it is a missing image), in folder and tree runs alike. Since preprocessing v0.4.0rc5 (CellPatcher v0.2.0rc3) a mount point below the experiment or masks folder (a bind mount, or a folder or file on another device) is refused too (`mounted_folder`), by discovery and by `RunInputs`; copy the files into the experiment, or select the mounted folder itself.

When `mask_root_path` is given, it mirrors the image root in the preprocessing brick's layouts (`<mask type>/<condition>/.../<image>.tif` or `<condition>/.../<image>_<mask type>.tif`, see [`config.matching`](#configmatching-matchconfig-which-masks-to-pair)) and the whole tree is paired at once, before any image is read. A mask root is required for `mode="cell"`, optional for `mode="random"` (when given, every image of the tree needs a mask) and ignored for `mode="grid"`/`"whole_fov"`; an image without a mask aborts the whole tree run rather than being silently skipped.

Returns a list of `ConditionRepeatResult(condition, biological_repeat, image_folder, result, technical_repeat, relative_folder)`, one per leaf, sorted by condition, then biological repeat, then technical repeat.

- `combine_metadata=True` (requires `export.enabled=True` and `common.output_dir`) writes **one** `patches.csv`, **one** `images.csv` (every image of every leaf) and **one** `run_config.json` directly in `output_dir` instead of a set per leaf folder. Patch files still go to each leaf's `<output_dir>/<condition>/<biological_repeat>[/<technical_repeat>]/`, and each row's `patch_file` is the path relative to the CSV, e.g. `treated/2026_01_01/image_0000_grid_00000.tif` or `treated/2026_01_01/B02/image_0000_grid_00000.tif` — the same "relative to the CSV" meaning as in the per-leaf layout, so `ExportedPatchTorchDataset` reads the combined CSV unchanged. `patch_id`/`image_id` are only unique within a leaf; identify rows across the combined CSV by `condition`/`biological_repeat`/`technical_repeat`/`image_file` (or `patch_file`).
- `progress_callback(completed, total, leaf)` is called before each leaf and once more at the end (`leaf=None`), e.g. to drive a progress bar.

### Renamed: `repeat` → `biological_repeat`

The biological-replicate field was called `repeat` in earlier releases. It is now `biological_repeat` everywhere — the `PatchRegion` field, the `CommonConfig` field, the `patches.csv` column and the `run_config.json` key, and the `ConditionRepeatFolder`/`ConditionRepeatResult` attribute — so it cannot be confused with the MorphoBricks `technical_repeat` column (a well, dish or coverslip within a biological repeat). Nothing else changed: the values, the folder level they come from, the `"single"` label and the output layout are the same.

For a transition period the old name keeps working, and each use emits one `DeprecationWarning` pointing at the calling line:

- `CommonConfig(repeat=...)`, `PatchRegion(repeat=...)`, `ConditionRepeatFolder(repeat=...)`, `ConditionRepeatResult(repeat=...)`: the keyword is mapped onto `biological_repeat`; passing both with different values raises `TypeError`.
- `config.common.repeat`, `patch.repeat`, `leaf.repeat`: read and write `biological_repeat`.
- `apply_overrides(..., {"common.repeat": ...})` (and `overrides=` on the run functions), and `load_config()` on an older `run_config.json` whose `common` section has a `repeat` key.
- `ExportedPatchTorchDataset` reading an older `patches.csv` that has a `repeat` column and no `biological_repeat` column: the column is read as `biological_repeat` (one warning per file).

Output is identical whichever name is used; files are always written with `biological_repeat`. The constant `SINGLE_REPEAT_LABEL` stays available (not deprecated) as the same value as the new `SINGLE_BIOLOGICAL_REPEAT_LABEL`, and the function names `run_patching_condition_repeat_tree`/`discover_condition_repeat_folders` are unchanged.

### Deterministic reconstruction

Each exported `patches.csv` row, together with the `run_config.json` file it references via `config_file` (written alongside it whenever `export.enabled=True`; see `cellpatcher.RUN_CONFIG_FILENAME`), is enough to reconstruct the patch without re-deriving anything from code:

- The row's own coordinates (`y0/y1/x0/x1`, `z0/z1`, `t`, `c`) and `output_height`/`output_width` fully describe the patch geometry and final exported array shape; `image_height`/`image_width`, the `pad_*` columns, `resized` and `output_axes` say, without the source image or the config, where the rectangle lies in its image, which pixels of the array are padding, whether the array was resampled and what its axes are.
- `run_config.json` is the complete, serialized `CellPatcherConfig` for the run — strategy parameters (including `random.seed`), extraction/resize settings, the mask type, and the axes of in-memory arrays — everything needed to reproduce the same coordinates deterministically from the same inputs.
- Since v0.2.0 (CP-2b) it is **version 2**: `run_config_version: 2`, and when a preprocessing run was read (a folder or tree run, or a CSV normalization scope) a `preprocessing` section with the run (`run`), the SHA-256 of its `run_manifest.json` when the call opened it (`manifest_sha256`), which names the run's images, masks, roles and table exactly (a manifest that changes during the call is refused, and no `run_config.json` is written), and for a folder or tree run the `layout` (`image`, `mask`) of the arrays it built from the run's recorded roles (T, Z, C, then Y, X). A folder or tree run does not record `common.image_axes`, `mask_axes` or the labels (each patch's labels are in `patches.csv`). `load_config`, `read_patch_table` and the exported-patch readers (`iter_exported_patches`, `load_exported_patch`, `ExportedPatchTorchDataset`) read versions 1 (no version: written before v0.2.0, or a config file) and 2, and refuse any other naming the version.

## Typed refusals

Per-plane data refusals have their own exception types, exported from
`cellpatcher` (module `cellpatcher.errors`), so a caller can record a refused
plane without parsing messages. All derive from `CellPatcherRefusal`, itself a
`ValueError`, so `except ValueError` keeps working. They are picklable, so they
cross multiprocessing workers with their attributes.

| Exception | Raised when | Attributes |
| --- | --- | --- |
| `CellPatcherRefusal` | Base class; not raised itself. | — |
| `SamplingExhaustedError` | `random` mode with a sampling mask ran out of attempts before collecting `n_patches` draws. | `requested`, `collected`, `attempts` |
| `LabelRangeError` | A `cell`-mode mask (also in `scan_cell_objects`) holds a negative label or one above the int32 maximum. | `label` |

A caught refusal is not a measurement: no patch, fill value or fallback stands
in for the refused plane.
