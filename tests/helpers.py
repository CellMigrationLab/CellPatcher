import numpy as np

from cellpatcher.io import write_array


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


DESCRIBED_ON = "2026-11-10"
"""The confirmation date of the descriptions :func:`describe_experiment` writes."""

TEST_IDS = {"control": "ctrl"}
"""The ID a test declares for a folder label that cannot be an ID (``control``, as the Spec's ``v33``)."""


def survey_experiment(root, mapping, output, *, masks_folder=None, mask_axes=None, exclude=None, image_type=None):
    """What a description of the experiment *root* states, from preprocessing's preflight (no pixel read).

    Preprocessing v0.5.1 refuses a run without a published description (``A_DESCRIPTION_REQUIRED``),
    so a test describes its images before the run, from what pass 1 finds: the folders' labels and
    ``position_id`` of each image, each image's channel count (its header's length of the axis that
    *mapping* gives the ``C`` role; 1 without one), and the mask types found. Returns
    ``(images, mask_types, rgb)``, *images* a tuple of ``(key, relative_path, c_count)``.
    """
    from morphobricks_preprocessing.discovery import position_id
    from morphobricks_preprocessing.run.preflight import preflight

    found = preflight(root, output, exclude=exclude or (), masks_folder=masks_folder or None, mask_axes=mask_axes)
    headers = {header.relative_path: header for header in found.metadata.files}
    images = []
    for relative_path in found.discovery.images:
        header = headers[relative_path]
        own = mapping.for_axes(header.axes) if header.axes else None
        c_count = header.shape[own.roles.index("C")] if own is not None and "C" in own.roles else 1
        key = (*found.discovery.labels_for(relative_path), position_id(relative_path))
        images.append((key, relative_path, c_count))
    mask_types = tuple(found.masks.types) if found.masks is not None else ()
    return tuple(images), mask_types, image_type == "rgb"


def describe_experiment(output, survey, ids=TEST_IDS, repeats=None, representations=None):
    """Publish the experiment description of the test experiment *survey* found (:func:`survey_experiment`).

    Every run needs a published description (Spec ADR 0002 AR10), and the vendored Spec fixtures
    (``make_described_run``) write their own tiny images, so the tests that need their own images
    describe them here, before the run. **The test declares** what a user confirms in the notebook:

    - every ID is the folder label (as the Spec's ``v13_homogeneous_legacy``) or, for a label that
      cannot be an ID, the one *ids* gives (``control``: ``ctrl``, as ``v33``); any other label
      that is not a valid ID raises: name the folder differently in the test;
    - the repeats of one label are one repeat group across the conditions, unless *repeats* gives
      the repeat ID of a ``(condition, biological_repeat)`` folder (pairing different names);
    - each image's channels are ``C0``, ``C1``, ... by C index (``R``, ``G``, ``B`` for ``rgb``);
    - each mask type found is a declared mask of that ID, ``instances`` unless *representations*
      gives its representation by mask type (``{"tissue": "binary"}``): a 0/1 or 0/255 mask of
      several objects, or a sampling region, is ``binary``, never ``instances``
      (:func:`check_masks` refuses a binary mask described as ``instances``).

    Returns the bytes written to ``morphobricks.toml``.
    """
    from pathlib import Path

    from morphobricks_preprocessing.draft_ids import propose_id

    output = Path(output)
    images, mask_types, rgb = survey

    def id_of(kind, label):
        given = ids.get(label, label)
        if propose_id(kind, given) != given:
            raise AssertionError(f"the test folder {label!r} is not a valid {kind} ID: rename it or give its ID")
        return given

    units, source_repeats = {}, {}
    for key, relative_path, c_count in images:
        condition, repeat_label, technical, _ = key
        repeat = (repeats or {}).get((condition, repeat_label)) or id_of("repeats", repeat_label)
        if source_repeats.setdefault((repeat, condition), repeat_label) != repeat_label:
            raise AssertionError(f"repeat {repeat!r} names two folders of condition {condition!r}")
        channels = ("R", "G", "B") if rgb else tuple(f"C{index}" for index in range(c_count))
        unit = units.setdefault((repeat, condition, technical), {"channels": channels, "count": 0})
        if unit["channels"] != channels:
            raise AssertionError(f"{relative_path}: its unit's images have different channel counts")
        unit["count"] += 1

    def quoted(values):
        return "[" + ", ".join(f'"{value}"' for value in values) + "]"

    conditions = {label: id_of("conditions", label) for label in {key[1] for key in units}}
    technical_repeats = {label: id_of("technical_repeats", label) for label in {key[2] for key in units}}
    masks = {source_type: id_of("masks", source_type) for source_type in mask_types}
    if unknown := sorted(set(representations or {}) - set(masks)):
        raise AssertionError(f"representations names mask types the run did not record: {unknown}")
    channels = sorted({channel for unit in units.values() for channel in unit["channels"]})
    lines = ['schema = "morphobricks-experiment"', "schema_version = 1", ""]
    for label, condition in sorted(conditions.items(), key=lambda item: item[1]):
        lines += [f'[conditions."{condition}"]', f'source_folder = "{label}"', ""]
    lines += [line for channel in channels for line in (f"[channels.{channel}]", "")]
    repeat_ids = sorted({key[0] for key in units})
    for repeat in repeat_ids:
        lines += [f'[repeats."{repeat}"]', "confirmed = true", f'confirmed_on = "{DESCRIBED_ON}"', ""]
        for label in sorted({key[1] for key in units if key[0] == repeat}, key=conditions.get):
            member = {key[2]: value for key, value in units.items() if key[:2] == (repeat, label)}
            table = f'repeats."{repeat}".members."{conditions[label]}"'
            lines += [f"[{table}]", f'source_repeat = "{source_repeats[(repeat, label)]}"',
                      f"image_count = {sum(unit['count'] for unit in member.values())}", ""]
            for technical, unit in sorted(member.items(), key=lambda item: technical_repeats[item[0]]):
                lines += [f'[{table}.technical_repeats."{technical_repeats[technical]}"]',
                          f'source_folder = "{technical}"', f"channels = {quoted(unit['channels'])}"]
                lines += ['image_type = "rgb"'] if rgb else []
                lines += [""]
    for source_type, mask in sorted(masks.items(), key=lambda item: item[1]):
        representation = (representations or {}).get(source_type, "instances")
        lines += [f'[masks."{mask}"]', f'source_type = "{source_type}"', f'representation = "{representation}"', ""]
    lines += ["[used_ids]", f"conditions = {quoted(sorted(conditions.values()))}", f"channels = {quoted(channels)}",
              f"repeats = {quoted(repeat_ids)}",
              f"technical_repeats = {quoted(sorted(technical_repeats.values()))}", ""]
    data = "\n".join(lines).encode("utf-8")
    (output / "morphobricks.toml").write_bytes(data)
    return data


def check_masks(output):
    """Run preprocessing's mask step on the described run in *output* when it has masks, as a user does.

    ``RunInputs`` refuses a declared mask not yet checked (``A_MASK_UNCHECKED``). Then every mask
    the test described as ``instances`` is read: one that holds a single label value on several
    separate (4-connected) regions is a binary mask of several objects, and describing it as
    ``instances`` would state that they are one object, so the test is stopped
    (:func:`describe_experiment`'s *representations* declares it ``binary``).
    """
    from pathlib import Path

    from morphobricks_preprocessing.run.mask_step import describe_masks, prepare_mask_step
    from morphobricks_preprocessing.run_inputs import RunInputs

    if not RunInputs.open(Path(output)).mask_types():
        return
    step = describe_masks(prepare_mask_step(None, Path(output)))
    if step.issues:
        raise AssertionError("the mask step failed: " + "; ".join(map(str, step.issues)))
    _refuse_binary_masks_described_as_instances(RunInputs.open(Path(output)))


def _refuse_binary_masks_described_as_instances(inputs):
    """Raise AssertionError for an ``instances`` mask file holding one label value on several regions."""
    from morphobricks_preprocessing.run_inputs import RunInputsError
    from scipy.ndimage import label

    declared = [m.source_type for m in inputs.mask_semantics().values() if m.representation == "instances"]
    for fov in inputs.fovs():
        for mask_type in declared:
            try:
                planes = [
                    np.asarray(inputs.read_mask(fov, mask_type, z if fov.z_count > 1 else None,
                                                t if fov.t_count > 1 else None))
                    for z in range(fov.z_count) for t in range(fov.t_count)
                ]
            except RunInputsError:  # no mask of this type for this image
                continue
            values = np.unique(np.concatenate([plane[plane != 0] for plane in planes]))
            if len(values) == 1 and any(label(plane != 0)[1] > 1 for plane in planes):
                raise AssertionError(
                    f"the {mask_type!r} mask of {fov.relative_path} holds the one value {values[0]} on several "
                    "separate regions: it is a binary mask of several objects, not an instance mask. Declare it "
                    f'with representations={{"{mask_type}": "binary"}}, or write real instance labels.'
                )


def run_described(root, mapping, *, output_folder, repeats=None, representations=None, **options):
    """``morphobricks_preprocessing.run_experiment``, described: survey, :func:`describe_experiment`, run.

    The images (and their channel counts) the description is written from are what preprocessing's
    pass 1 finds (:func:`survey_experiment`, with the run's own masks folder, mask axes, excluded
    images and image type; without a masks folder, the one an earlier run in *output_folder* recorded,
    which the run takes too; *repeats*, *representations*: as :func:`describe_experiment` takes them).
    The description is published in *output_folder* before the run. Returns the run's result.
    """
    from pathlib import Path

    import morphobricks_preprocessing as mbp

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    masks_folder = options.get("masks_folder")
    manifest = Path(output_folder) / mbp.MANIFEST_FILENAME
    if masks_folder is None and manifest.is_file():  # the run takes the earlier run's masks folder
        masks_folder = (mbp.read_manifest(manifest).masks or {}).get("folder")
    survey = survey_experiment(
        root, mapping, output_folder, masks_folder=masks_folder,
        mask_axes=options.get("mask_axes"), exclude=options.get("exclude"), image_type=options.get("image_type"),
    )
    describe_experiment(output_folder, survey, repeats=repeats, representations=representations)
    result = mbp.run_experiment(root, mapping, output_folder=output_folder, **options)
    if result.written:
        check_masks(output_folder)
    return result


def make_clean_run(folder, **options):
    """Preprocessing's ``make_clean_run``: its tree (``ctrl``/``drug``, ``rep1``, images ``a`` and ``b``).

    A described run since preprocessing v0.5.1 (its ``clean_description``: channels ``ch1`` ..,
    each mask type an ``instances`` mask of that ID, the mask step run), as every run must be (AR10).
    """
    from morphobricks_preprocessing.testing import make_clean_run as make_described_clean_run

    return make_described_clean_run(folder, **options)


def preprocess(root, axes="YX", *, masks=None, output=None, stored=None, representations=None, **options):
    """Run the preprocessing brick on the experiment *root*, described, and return its output folder.

    Folder and tree runs read files only through a clean preprocessing run
    (design v0.4 §6, CP-2a) with a published description (Spec ADR 0002 AR10,
    :func:`run_described`). *axes* is the roles of the images' axes, as
    ``morphobricks-preprocessing run --axes`` takes them (*stored*: the
    file's own axes when they differ, e.g. ``"IYX"`` for a plain stack);
    *masks* the masks folder. The output goes beside *root* unless *output* is given.
    *representations* is the masks' representation by type, as :func:`describe_experiment` takes it.
    """
    from pathlib import Path

    import morphobricks_preprocessing as mbp

    root = Path(root)
    output = Path(output) if output is not None else root.parent / f"{root.name}_preprocessing"
    result = run_described(
        root,
        mbp.AxisMapping(stored or axes, axes),
        output_folder=output,
        masks_folder=masks,
        representations=representations,
        **options,  # image_type, mask_axes, missing_as_singleton, ...
    )
    if not result.written:
        raise AssertionError("preprocessing did not run: " + "; ".join(map(str, result.issues)))
    return output


def file_run_config(config, root, axes="YX", **preprocessing):
    """Preprocess *root* (:func:`preprocess`) and set ``config.common.preprocessing_run``; return *config*."""
    config.common.preprocessing_run = str(preprocess(root, axes, **preprocessing))
    return config


def preprocessed_tree(root, *, config, mask_root_path=None, axes="YX", representations=None, **kwargs):
    """Preprocess *root* (with its masks folder), then run the tree run through that run (CP-2a)."""
    from cellpatcher.orchestration import run_patching_condition_repeat_tree

    file_run_config(config, root, axes, masks=mask_root_path, representations=representations)
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

def cell_folders(tmp_path, *, empty=("b",)):
    """An image folder with images a, b, c and their masks; the masks of *empty* hold no cell."""
    image, labels, _ = make_synthetic_cell_image_and_labels()
    for name in ("a", "b", "c"):
        write_array(tmp_path / "images" / f"{name}.tif", image)
        write_array(tmp_path / "masks" / f"{name}_mask.tif", np.zeros_like(labels) if name in empty else labels)
    return tmp_path / "images", tmp_path / "masks"


def cell_tree(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels()
    empty = np.zeros_like(labels)
    layout = {
        "treated/2026_01_01/a.tif": labels,
        "treated/2026_01_01/b.tif": empty,
        "control/2026_01_01/B02/c.tif": labels,
        "control/2026_01_01/B03/c.tif": empty,
    }
    for relative, mask in layout.items():
        write_array(tmp_path / "raw" / relative, image)
        mask_path = (tmp_path / "masks" / relative).with_name(f"{relative.rsplit('/', 1)[1][:-4]}_mask.tif")
        write_array(mask_path, mask)
    return tmp_path / "raw", tmp_path / "masks"
