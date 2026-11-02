import numpy as np


def write_tiff(path, data, axes):
    """Write *data* as an ImageJ TIFF whose header names its *axes* (``"ZYX"``, ``"CYX"``, ...).

    Images are discovered as TIFF or OME-Zarr only (issue #45); the axes in
    the header keep a stack of three planes from being read as RGB.
    """
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": axes})


def named(images, prefix="pos"):
    """One ``image_metadata`` entry with a ``position_id`` per in-memory image (``pos0``, ``pos1``, ...).

    ``run_patching`` refuses an image with neither a ``position_id`` nor an
    ``image_file`` (decision D45); tests that do not test identity name
    their images with this. None for ``images=None``.
    """
    if images is None:
        return None
    count = len(images) if isinstance(images, (list, tuple)) else 1
    return [{"position_id": f"{prefix}{index}"} for index in range(count)]


def make_synthetic_cell_image_and_labels(shape=(96, 128), seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape

    image = rng.normal(loc=0.1, scale=0.02, size=(h, w)).astype(np.float32)
    labels = np.zeros((h, w), dtype=np.int32)

    cells = {
        1: (20, 35, 20, 38),   # interior
        2: (45, 63, 55, 78),   # interior
        3: (65, 82, 90, 112),  # interior
        4: (0, 12, 8, 22),     # border-touching
        5: (88, 90, 5, 7),     # tiny (area 4)
    }

    for label_id, (y0, y1, x0, x1) in cells.items():
        labels[y0:y1, x0:x1] = label_id
        image[y0:y1, x0:x1] += 0.5 + 0.05 * label_id

    metadata = {
        "expected_valid_labels": [1, 2, 3],
        "expected_border_labels": [4],
        "expected_tiny_labels": [5],
        "tiny_label_area": 4,
    }
    return image.astype(np.float32), labels.astype(np.int32), metadata


# OME-Zarr test images, written with zarr directly (independently of the
# preprocessing brick's writer). Callers skip when zarr is not installed.
AXIS_TYPES = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}


def write_ngff(path, data, axes="zyx", *, images=1):
    """Write *data* as a one-level OME-NGFF 0.4 image; *images* > 1 repeats its multiscales entry."""
    import zarr

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if int(zarr.__version__.split(".")[0]) >= 3:
        group = zarr.open_group(str(path), mode="w", zarr_format=2)
        array = group.create_array("0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    else:
        group = zarr.open_group(str(path), mode="w")
        array = group.create_dataset("0", shape=data.shape, chunks=data.shape, dtype=data.dtype)
    array[...] = data
    multiscale = {
        "version": "0.4",
        "axes": [{"name": a, "type": AXIS_TYPES.get(a, "other")} for a in axes],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0] * data.ndim}]}],
    }
    group.attrs.update({"multiscales": [multiscale] * images})
    return path


def write_ngff_labels(path, labels, axes="zyx"):
    """Add OME-NGFF 0.4 label images to the image at *path*: *labels* maps names to arrays."""
    import zarr

    # The store is Zarr format 2 (OME-NGFF 0.4); zarr 3 must be told so.
    options = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}
    for name, data in labels.items():
        write_ngff(path / "labels" / name, data, axes)
        label_group = zarr.open_group(str(path / "labels" / name), mode="a", **options)
        label_group.attrs.update({"image-label": {"version": "0.4"}})
    zarr.open_group(str(path / "labels"), mode="a", **options).attrs.update({"labels": list(labels)})
    return path


def preprocess(root, axes="YX", *, masks=None, output=None, stored=None, **options):
    """Run the preprocessing brick on the experiment *root* and return its output folder.

    Folder and tree runs read files only through a clean preprocessing run
    (design v0.4 §6, CP-2a). *axes* is the roles of the images' axes, as
    ``morphobricks-preprocessing run --axes`` takes them (*stored*: the
    file's own axes when they differ, e.g. ``"IYX"`` for a plain stack);
    *masks* the masks folder. The output goes beside *root* unless *output* is given.
    """
    from pathlib import Path

    import morphobricks_preprocessing as mbp

    root = Path(root)
    output = Path(output) if output is not None else root.parent / f"{root.name}_preprocessing"
    result = mbp.run_experiment(
        root,
        mbp.AxisMapping(stored or axes, axes),
        output_folder=output,
        masks_folder=masks,
        **options,  # design, image_type, mask_axes, missing_as_singleton, ...
    )
    if not result.written:
        raise AssertionError("preprocessing did not run: " + "; ".join(map(str, result.issues)))
    return output


def file_run_config(config, root, axes="YX", **preprocessing):
    """Preprocess *root* (:func:`preprocess`) and set ``config.common.preprocessing_run``; return *config*."""
    config.common.preprocessing_run = str(preprocess(root, axes, **preprocessing))
    return config


def preprocessed_tree(root, *, config, mask_root_path=None, axes="YX", **kwargs):
    """Preprocess *root* (with its masks folder), then run the tree run through that run (CP-2a)."""
    from cellpatcher.orchestration import run_patching_condition_repeat_tree

    file_run_config(config, root, axes, masks=mask_root_path)
    return run_patching_condition_repeat_tree(root, config=config, mask_root_path=mask_root_path, **kwargs)


def preprocessed_folder(folder, *, config, mask_folder_path=None, axes="YX", **kwargs):
    """Preprocess *folder* (with its masks folder), then run the folder run through that run (CP-2a)."""
    from cellpatcher.orchestration import run_patching_pipeline

    file_run_config(config, folder, axes, masks=mask_folder_path)
    return run_patching_pipeline(folder, config=config, mask_folder_path=mask_folder_path, **kwargs)


def as_file_run(config, run=None):
    """A copy of *config* for a file run through *run* (default: the run of its table), axes left to it (CP-2a)."""
    import copy
    import os

    config = copy.deepcopy(config)
    config.common.image_axes = config.common.mask_axes = "YX"
    config.common.preprocessing_run = str(run or os.path.dirname(config.normalization.plane_histograms))
    return config
