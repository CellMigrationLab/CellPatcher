import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellpatcher._deprecation import accept_deprecated_repeat
from cellpatcher.axes import normalize_axes

# ".zarr" is an OME-Zarr image: a folder, read with the preprocessing extra
# (docs/normalization_contract.md §2); every other extension is a file.
ZARR_SUFFIX = ".zarr"

# The images CellPatcher discovers: those of the preprocessing brick's
# discover_experiment, TIFF files and OME-Zarr folders (owner decision of
# 2026-10-30, issue #45), so every brick patches, measures and normalizes the
# same images.
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ZARR_SUFFIX}

# Image formats CellPatcher read as input before it adopted the preprocessing
# brick's discovery. Such a file where images are discovered is refused, naming
# it, never skipped: it would otherwise drop out of the dataset unnoticed.
UNSUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".npy")

# The file formats read_array and write_array handle: the input images, and
# the formats exported patches may be written in (export.patch_format).
ARRAY_EXTENSIONS = {*SUPPORTED_EXTENSIONS, *UNSUPPORTED_IMAGE_EXTENSIONS}

# Biological-repeat label stamped when a condition folder holds images
# directly, with no biological-repeat subfolder (see
# discover_condition_repeat_folders below). Treated as "this condition has
# exactly one biological repeat" rather than an error.
SINGLE_BIOLOGICAL_REPEAT_LABEL = "single"

# Name kept from before `repeat` was renamed `biological_repeat`; same value,
# still public and not deprecated.
SINGLE_REPEAT_LABEL = SINGLE_BIOLOGICAL_REPEAT_LABEL

# Condition label stamped when the root folder itself holds the images, with
# no condition subfolders: the whole dataset is one condition and one
# biological repeat.
SINGLE_CONDITION_LABEL = "single"

# Technical-repeat label stamped when a repeat folder holds its images
# directly, with no <technical_repeat> subfolders (a well, a dish): the
# repeat is one technical repeat. Never a biological repeat, which stays in
# `repeat` (see discover_condition_repeat_folders below).
SINGLE_TECHNICAL_REPEAT_LABEL = "single"


def list_supported_files(folder_path, supported_extensions=None) -> list[Path]:
    """The images directly in *folder_path* (TIFF files and OME-Zarr folders), sorted.

    Subfolders are not entered. A file in a format CellPatcher no longer
    reads as an input image (:data:`UNSUPPORTED_IMAGE_EXTENSIONS`: PNG, JPEG,
    BMP, GIF, NumPy) is refused, naming it, as is a link to a missing image.
    With *supported_extensions*, the files with those extensions are listed
    instead and nothing else is refused but missing links.
    """
    folder = Path(folder_path).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    exts = supported_extensions if supported_extensions is not None else SUPPORTED_EXTENSIONS
    exts = {e.lower() for e in exts}

    entries = _visible_entries(folder)
    if supported_extensions is None:
        _refuse_unsupported_images(folder, [entry.name for entry in entries if entry.is_file()])
    _refuse_broken_image_links(entries, exts)
    return sorted(p for p in entries if _is_image(p, exts))


def _is_system_name(name: str) -> bool:
    # The preprocessing brick's rule, so both bricks see the same images:
    # dot-files and dot-folders (.DS_Store, macOS "._img.tif" resource forks,
    # .ipynb_checkpoints), Thumbs.db, desktop.ini and __MACOSX (issue #46).
    from morphobricks_preprocessing.discovery import is_system_name

    return is_system_name(name)


def _visible_entries(folder: Path) -> list[Path]:
    """The entries of *folder*, without system files and folders (:func:`_is_system_name`)."""
    return [entry for entry in folder.iterdir() if not _is_system_name(entry.name)]


def _refuse_broken_image_links(entries, exts: set[str]) -> None:
    # Links are followed (a tree of links to images stored elsewhere is a
    # valid input), so a link whose target is gone is a missing image, not a
    # file to skip silently.
    broken = sorted(p.name for p in entries if p.suffix.lower() in exts and p.is_symlink() and not p.exists())
    if broken:
        folder = Path(entries[0]).parent
        raise ValueError(
            f"Folder '{folder}' holds {len(broken)} image link(s) whose target does not exist: "
            f"{', '.join(broken[:10])}. Fix or remove the links."
        )


def _refuse_unsupported_images(root: Path, relative_paths) -> None:
    """Refuse the files among *relative_paths* in a format CellPatcher no longer reads as an image."""
    unsupported = sorted(path for path in relative_paths if path.lower().endswith(UNSUPPORTED_IMAGE_EXTENSIONS))
    if unsupported:
        raise ValueError(
            f"'{root}' holds {len(unsupported)} image(s) in a format CellPatcher no longer reads: "
            f"{', '.join(unsupported[:10])}{', ...' if len(unsupported) > 10 else ''}. Images are found "
            "with the MorphoBricks preprocessing brick's discovery, which reads TIFF files (.tif, .tiff) "
            "and OME-Zarr folders (.zarr) only; PNG, JPEG, BMP, GIF and NumPy (.npy) images are not read "
            "(decision of 2026-10-30, issue #45). Convert them to TIFF, or move them out of the folder."
        )


def is_ome_zarr(path) -> bool:
    """True if *path* names an OME-Zarr image (a folder ending in ``.zarr``, any case)."""
    return Path(path).suffix.lower() == ZARR_SUFFIX


def image_stem(path) -> str:
    """File name without its extension, used to match images and masks.

    For an OME-Zarr image the whole ``.ome.zarr``/``.zarr`` ending is
    removed (``img.ome.zarr`` -> ``img``), so it matches ``img.tif`` or
    ``img_mask.tif``; for every other file it is ``Path.stem``.
    """
    path = Path(path)
    if not is_ome_zarr(path):
        return path.stem
    stem = path.stem
    return stem[: -len(".ome")] if stem.lower().endswith(".ome") else stem


def _is_image(path: Path, exts: set[str]) -> bool:
    # An OME-Zarr image is a folder; discovery lists it and never enters it.
    suffix = path.suffix.lower()
    if suffix not in exts:
        return False
    return path.is_dir() if suffix == ZARR_SUFFIX else path.is_file()


@accept_deprecated_repeat
@dataclass
class ConditionRepeatFolder:
    """One discovered ``<condition>/<biological_repeat>[/<technical_repeat>]`` leaf.

    ``biological_repeat`` is ``SINGLE_BIOLOGICAL_REPEAT_LABEL`` when
    *condition_dir* holds image files directly with no biological-repeat
    subfolder; in that case ``image_folder`` is the condition folder itself.
    ``technical_repeat`` is the name of a technical-repeat subfolder (a well,
    a dish) inside the biological-repeat folder, or
    ``SINGLE_TECHNICAL_REPEAT_LABEL`` when that folder holds its images
    directly. ``repeat`` is a deprecated alias of ``biological_repeat``. See
    :func:`discover_condition_repeat_folders`.
    """
    condition: str
    biological_repeat: str
    image_folder: Path
    mask_folder: Path | None = None
    technical_repeat: str = SINGLE_TECHNICAL_REPEAT_LABEL

    @property
    def relative_folder(self) -> Path:
        """The leaf's ``<condition>/<biological_repeat>[/<technical_repeat>]`` path, for
        mirrored outputs.

        The technical-repeat level appears only when the input had one, so
        a two-level tree keeps its ``<condition>/<biological_repeat>`` output layout.
        """
        path = Path(self.condition) / self.biological_repeat
        if self.technical_repeat != SINGLE_TECHNICAL_REPEAT_LABEL:
            path = path / self.technical_repeat
        return path


# Every error of discover_experiment is a refusal, "tiff_outside_leaf" (images
# in a folder that also holds subfolders) among them: an error since
# preprocessing v0.3.2 (owner decision of 2026-10-30, refused in every brick).
# Its warning "non_tiff_file" is refused for the former image formats only
# (_refuse_unsupported_images); other files were never read and stay ignored.
_EMPTY_FOLDER_CODE = "empty_folder"
_NO_IMAGES_CODE = "no_images"


def _refuse_masks_inside(image_root: Path, mask_folder) -> None:
    """Refuse a mask folder that is the image folder or inside it, as the preprocessing brick does.

    Checked before discovery: its folders would otherwise be read as image
    folders. The message is the brick's ``masks_inside_experiment``.
    """
    if mask_folder is None:
        return
    root = image_root.resolve()
    folder = Path(mask_folder).expanduser().resolve()
    if folder == root or root in folder.parents:
        raise ValueError(
            "Images and masks cannot be paired: [masks_inside_experiment] The masks folder is inside "
            "the experiment folder, where its folders would be read as conditions and repeats "
            f"({folder.relative_to(root).as_posix() or '.'}). Move the masks folder next to the image "
            "folder and select it there."
        )


def discover_images(root_path, *, mask_folder_path=None):
    """Discover the images under *root_path* with the preprocessing brick's ``discover_experiment``.

    CellPatcher does not walk folders itself (owner decision of 2026-10-30,
    issue #45): the leaves, their labels and their images are the brick's,
    so every brick reads the same images under the same labels. Images are
    TIFF files and OME-Zarr folders in folders without subfolders, at most
    three levels deep (condition, biological repeat, technical repeat).

    Refused, never skipped:

    - every error ``discover_experiment`` reports (for example
      ``duplicate_position_id``, ``case_conflict``, ``folder_too_deep``,
      ``reserved_folder_name``), except ``no_images``, which gives no leaves;
    - images in a folder that also holds subfolders (the brick's error
      ``tiff_outside_leaf``, a warning before preprocessing v0.3.2);
    - PNG, JPEG, BMP, GIF and NumPy images (:data:`UNSUPPORTED_IMAGE_EXTENSIONS`);
    - a link to a missing image;
    - a mask folder (*mask_folder_path*) that is *root_path* or inside it.

    Folders with no images and no subfolders hold nothing to patch; they are
    left out with a ``UserWarning`` naming them, as the brick warns. Other
    files (``.txt``, ``.csv``, ...) are ignored.

    Returns
    -------
    morphobricks_preprocessing.DiscoveryResult
        The brick's result; its paths are relative to *root_path*.
    """
    from morphobricks_preprocessing.discovery import discover_experiment

    root = Path(root_path).expanduser()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {root}")
    _refuse_masks_inside(root, mask_folder_path)
    result = discover_experiment(root)

    by_code = {issue.code: issue for issue in result.issues}
    if "non_tiff_file" in by_code:
        _refuse_unsupported_images(root, by_code["non_tiff_file"].paths)
    problems = [
        _issue_text(issue)
        for issue in result.issues
        if issue.is_error and issue.code != _NO_IMAGES_CODE
    ]
    if problems:
        raise ValueError(f"The image folder '{root}' cannot be used: {'; '.join(problems)}.")
    broken = [path for path in result.images if not (root / path).exists()]
    if broken:
        raise ValueError(
            f"Folder '{root}' holds {len(broken)} image link(s) whose target does not exist: "
            f"{', '.join(broken[:10])}. Fix or remove the links."
        )
    if _EMPTY_FOLDER_CODE in by_code:
        warnings.warn(_issue_text(by_code[_EMPTY_FOLDER_CODE]), UserWarning, stacklevel=2)
    return result


def discover_folder_images(image_folder_path, *, mask_folder_path=None) -> list[Path]:
    """The images of a single image folder (a folder run), found by :func:`discover_images`.

    The folder must hold its images directly. A folder holding subfolders
    of images is a tree: run it with
    :func:`~cellpatcher.orchestration.run_patching_condition_repeat_tree`.
    Returns an empty list when the folder holds no image.
    """
    folder = Path(image_folder_path).expanduser()
    result = discover_images(folder, mask_folder_path=mask_folder_path)
    nested = [leaf.relative_path for leaf in result.leaves if leaf.relative_path != "."]
    if nested:
        raise ValueError(
            f"The image folder '{folder}' holds its images in subfolders ({', '.join(nested[:10])}"
            f"{', ...' if len(nested) > 10 else ''}). A folder run reads the images directly in one "
            "folder; run a tree of condition and repeat folders with run_patching_condition_repeat_tree."
        )
    return sorted(folder / path for path in result.images)


def discover_condition_repeat_folders(
    root_path,
    *,
    mask_root_path=None,
    supported_extensions=None,
) -> list[ConditionRepeatFolder]:
    """Discover the ``<condition>/<biological_repeat>[/<technical_repeat>]`` image folders under *root_path*.

    (The function keeps its historical ``condition_repeat`` name.) The
    folders are found by the preprocessing brick's ``discover_experiment``
    (:func:`discover_images`, owner decision of 2026-10-30, issue #45), so
    the leaves and labels are the brick's:

    - ``<condition>/<biological_repeat>/<images>`` — one leaf per
      biological-repeat folder;
    - ``<condition>/<images>`` — the condition folder is one leaf whose
      biological repeat is ``SINGLE_BIOLOGICAL_REPEAT_LABEL`` (``"single"``);
    - ``<condition>/<biological_repeat>/<technical_repeat>/<images>`` — each
      technical-repeat folder (a well, a dish) is a leaf; a leaf above that
      level has the technical repeat ``SINGLE_TECHNICAL_REPEAT_LABEL``;
    - ``<images>`` directly in *root_path* — one leaf labeled
      ``SINGLE_CONDITION_LABEL`` / ``"single"`` / ``"single"``.

    A folder named ``single`` is a level with one member and must be alone
    in its place. A folder mixing images with subfolders, images in
    PNG/JPEG/BMP/GIF/NumPy format, a fourth level of folders, two images of
    one folder with the same name apart from the extension or case, and
    folder names differing only in case are refused (:func:`discover_images`
    lists every refusal). Folders without images or subfolders are left out
    with a ``UserWarning``. System files and folders are ignored: names
    starting with ``"."`` (``.ipynb_checkpoints``, ``._img.tif``),
    ``Thumbs.db``, ``desktop.ini`` and ``__MACOSX``.

    When *mask_root_path* is given, the same leaf path is mirrored under it
    for each leaf's ``mask_folder``; its existence is not checked here.
    The pairing itself reads the whole mask root (:func:`pair_masks`).

    *supported_extensions* is no longer supported: the images are the
    brick's (TIFF and OME-Zarr). Any value other than None is refused.
    """
    if supported_extensions is not None:
        raise ValueError(
            "discover_condition_repeat_folders(supported_extensions=...) is no longer supported: images are "
            "found with the preprocessing brick's discovery, which reads TIFF files and OME-Zarr folders "
            "(issue #45)."
        )
    root = Path(root_path).expanduser()
    mask_root = Path(mask_root_path).expanduser() if mask_root_path is not None else None
    if mask_root is not None and (not mask_root.exists() or not mask_root.is_dir()):
        raise ValueError(f"Folder does not exist or is not a directory: {mask_root}")

    result = discover_images(root, mask_folder_path=mask_root)
    leaves = []
    for leaf in result.leaves:
        image_folder = root if leaf.relative_path == "." else root / leaf.relative_path
        leaves.append(
            ConditionRepeatFolder(
                condition=leaf.condition,
                biological_repeat=leaf.biological_repeat,
                image_folder=image_folder,
                mask_folder=None if mask_root is None else mask_root / image_folder.relative_to(root),
                technical_repeat=leaf.technical_repeat,
            )
        )
    return leaves


def _refuse_several_series(path, series_count: int) -> None:
    """Refuse a TIFF holding several image series (positions, wells): only the first would be read."""
    if series_count > 1:
        raise ValueError(
            f"{path} holds {series_count} image series (several positions, series or wells); CellPatcher reads "
            "one image per file and would read only the first. Save each series in its own file."
        )


def read_array(file_path):
    """Read an image or mask file into a NumPy array.

    An OME-Zarr image (a ``.zarr`` folder) is read from level 0 of its only
    image, in its stored axis order; see :func:`read_ome_zarr` for its axes.
    A TIFF holding several image series is refused (with ``tifffile``), as
    a multi-image OME-Zarr store is, instead of reading its first series.
    """
    path = Path(file_path).expanduser()
    if is_ome_zarr(path):
        return read_ome_zarr(path)[0]
    if not path.exists() or not path.is_file():
        raise ValueError(f"File does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)

    if suffix in {".tif", ".tiff"}:
        # Prefer tifffile (handles multi-page/ImageJ metadata correctly);
        # fall back to imageio if it isn't installed, since either is an
        # accepted optional dependency for TIFF support.
        try:
            import tifffile
        except ImportError:
            tifffile = None
        if tifffile is not None:
            with tifffile.TiffFile(path) as tif:
                _refuse_several_series(path, len(tif.series))
                return tif.asarray()
        else:
            try:
                import imageio.v3 as iio
            except ImportError as exc:
                raise ImportError("Reading TIFF requires tifffile or imageio.") from exc
            return iio.imread(path)

    if suffix in ARRAY_EXTENSIONS:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError("Reading image files requires imageio.") from exc
        return iio.imread(path)

    raise ValueError(f"Unsupported file extension: {suffix}")


def write_array(file_path, array):
    path = Path(file_path).expanduser()
    if is_ome_zarr(path):
        raise ValueError(f"Writing OME-Zarr images is not supported: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        np.save(path, np.asarray(array))
        return

    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("Writing TIFF requires tifffile.") from exc
        tifffile.imwrite(str(path), np.asarray(array))
        return

    if suffix in ARRAY_EXTENSIONS:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError("Writing image files requires imageio.") from exc
        iio.imwrite(path, np.asarray(array))
        return

    raise ValueError(f"Unsupported file extension: {suffix}")


def channel_names(path) -> tuple[str | None, ...]:
    """The channel names an OME-TIFF or OME-Zarr image records, read from its header; () when unknown.

    Only the header is read, with the preprocessing brick's reader. Other
    formats, files whose header names no channel, and installs without the
    ``preprocessing`` extra give (); the names are for display only.
    """
    try:
        from morphobricks_preprocessing.metadata import inspect_image
    except ImportError:
        return ()
    header = inspect_image(path)
    return tuple(header.channel_names) if header.readable else ()


def read_ome_zarr(path) -> tuple[np.ndarray, str]:
    """Read level 0 of an OME-Zarr image with the preprocessing brick's reader.

    Returns
    -------
    tuple
        The pixels in their stored order, and one axis letter per dimension
        from the OME-NGFF metadata (``T``, ``C``, ``Z``, ``Y``, ``X``, or ``Q``
        for an axis without a known role).

    Raises
    ------
    cellpatcher.statistics.PreprocessingUnavailableError
        If the ``preprocessing`` extra (with ``zarr``) is not installed.
    ValueError
        If the folder is not an OME-Zarr image that can be read, or holds
        more than one image (several positions, series or wells), which is
        an error as in the preprocessing brick.
    """
    from cellpatcher.statistics import INSTALL_COMMAND, PreprocessingUnavailableError, require_preprocessing

    mbp = require_preprocessing()
    path = Path(path).expanduser()
    if not path.is_dir():
        raise ValueError(f"OME-Zarr image does not exist: {path}")
    try:
        image = mbp.open_ngff(path)
    except mbp.ZarrUnavailableError as exc:
        raise PreprocessingUnavailableError(
            f"Reading the OME-Zarr image {path.name} needs the zarr package. Install it with: {INSTALL_COMMAND}"
        ) from exc
    except mbp.NgffError as exc:
        raise ValueError(f"{path} cannot be read as an OME-Zarr image: {exc}.") from exc
    if image.image_count > 1:
        raise ValueError(
            f"{path} holds {image.image_count} images (several positions, series or wells). "
            "CellPatcher reads one image per OME-Zarr folder; save each image in its own folder."
        )
    pixels = mbp.zarr_input.read_selection(image.array, (slice(None),) * len(image.shape))
    return pixels, image.axes


def read_ngff_mask(path, label_name: str | None, axes: str) -> tuple[np.ndarray, str]:
    """Read an OME-Zarr image's label image as its mask, put into *axes* by role.

    Follows ``docs/normalization_contract.md`` §4: the label image comes from
    the image's ``labels`` group (*label_name* may be None when it has exactly
    one), level 0 is read, its values must be integers, and each of its
    dimensions must equal the image's or have length 1 (a channel axis, length
    1). A label Z or T axis with the image's length then follows the image's
    planes, and one of length 1 is shared (``common.z_policy``/``t_policy``).

    Returns
    -------
    tuple
        The mask in *axes*, and its ``mask_file`` name,
        ``<image folder name>/labels/<label name>``.

    Raises
    ------
    cellpatcher.statistics.PreprocessingUnavailableError
        If the ``preprocessing`` extra (with ``zarr``) is not installed.
    ValueError
        If the image has no such label image, the name is ambiguous, or the
        label image does not fit the image.
    """
    from cellpatcher.statistics import INSTALL_COMMAND, PreprocessingUnavailableError, require_preprocessing

    mbp = require_preprocessing()
    path = Path(path).expanduser()
    try:
        image = mbp.open_ngff(path)
        labels = mbp.open_ngff_labels(path, label_name)
    except mbp.ZarrUnavailableError as exc:
        raise PreprocessingUnavailableError(
            f"Reading the OME-Zarr image {path.name} needs the zarr package. Install it with: {INSTALL_COMMAND}"
        ) from exc
    except mbp.NgffError as exc:
        raise ValueError(
            f"The mask of {path.name} cannot be read from its OME-Zarr labels: {exc}. Set "
            "common.mask_label_name to one of its label images, or use common.mask_source='tiff_tree'."
        ) from exc
    # dataset_path is "[<image prefix>]labels/<name>/<level-0 array>".
    name = labels.dataset_path.split("/")[-2]
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"The label image {name!r} of {path.name} holds {labels.dtype} values; OME-NGFF label images "
            "hold integers."
        )
    image_lengths = dict(zip(image.axes, image.shape))
    problems = []
    for axis, length in zip(labels.axes, labels.shape):
        if axis == "C" and length != 1:
            problems.append(f"its channel axis has length {length} (must be 1)")
        elif axis in "YX" and length != image_lengths.get(axis):
            problems.append(f"its {axis} axis has length {length}, the image's {image_lengths.get(axis)}")
        elif length not in (1, image_lengths.get(axis)):
            problems.append(f"its {axis} axis has length {length}, neither 1 nor the image's {image_lengths.get(axis)}")
    if problems:
        raise ValueError(
            f"The label image {name!r} of {path.name} (axes {labels.axes!r}, shape {labels.shape}) does not fit "
            f"the image (axes {image.axes!r}, shape {image.shape}): {'; '.join(problems)}."
        )
    pixels = mbp.zarr_input.read_selection(labels.array, (slice(None),) * len(labels.shape))
    mask = conform_axes(pixels, labels.axes, axes, name=f"The label image {name!r} of {path.name}")
    return mask, f"{path.name}/labels/{name}"


def conform_axes(array, array_axes: str, target_axes: str, *, name="the image") -> np.ndarray:
    """Rearrange *array* (with axes *array_axes*) into *target_axes*, matching axes by role.

    Used for OME-Zarr images, whose axes come from their metadata. A
    dimension of length 1 that *target_axes* lacks (including an axis
    without a role, ``Q``) is dropped; a target axis the array lacks is added
    with length 1; then the dimensions are put in the target order.

    Raises
    ------
    ValueError
        If the array has a dimension longer than 1 that *target_axes* does
        not have, or the same role twice.
    """
    target = normalize_axes(target_axes)
    array = np.asarray(array)
    if len(array_axes) != array.ndim:
        raise ValueError(f"{name}: {len(array_axes)} axes ({array_axes!r}) for {array.ndim} dimensions.")
    kept, kept_axes = [], ""
    for dim, (axis, length) in enumerate(zip(array_axes, array.shape)):
        if axis in target and axis not in kept_axes:
            kept.append(dim)
            kept_axes += axis
        elif length != 1:
            role = "an axis without a known role" if axis not in target else f"a second {axis} axis"
            raise ValueError(
                f"{name} has {role} ({axis!r}, length {length}) that the axes {target!r} do not have; "
                f"its axes are {array_axes!r}. Set the axes (common.image_axes or common.mask_axes) to "
                "include every axis longer than 1."
            )
    squeezed = array.reshape([array.shape[dim] for dim in kept])
    for axis in target:
        if axis not in kept_axes:
            squeezed = squeezed[..., np.newaxis]
            kept_axes += axis
    return np.transpose(squeezed, [kept_axes.index(axis) for axis in target])


def read_image(file_path, axes: str):
    """Read an image or mask for patching with *axes* (``common.image_axes``/``mask_axes``).

    The same as :func:`read_array`, except that an OME-Zarr image is put
    into *axes* by role with :func:`conform_axes`, since its own axes come
    from its metadata rather than from the configuration.
    """
    if not is_ome_zarr(file_path):
        return read_array(file_path)
    pixels, stored_axes = read_ome_zarr(file_path)
    return conform_axes(pixels, stored_axes, axes, name=Path(file_path).name)


def read_array_shape(file_path) -> tuple[int, ...]:
    return tuple(read_array(file_path).shape)


def _warn_old_matcher(name: str) -> None:
    warnings.warn(
        f"{name} is CellPatcher's former image/mask matching, no longer used by CellPatcher and to be "
        "removed: images and masks are paired with the preprocessing brick's rule (cellpatcher.io.pair_masks, "
        "issue #45).",
        DeprecationWarning,
        stacklevel=3,
    )


def find_candidate_matches(
    image_files,
    mask_files,
    *,
    strategy="exact_or_suffix",
    mask_suffixes=("_mask", "_labels", "_label", "_seg", "_segmentation"),
    case_sensitive=False,
) -> dict[Path, list[Path]]:
    """Find candidate mask matches for each image path.

    Matching strategies:
    - ``exact``: image stem == mask stem
    - ``exact_or_suffix`` (default): exact stem or exact stem + allowed suffix
    - ``substring``: legacy permissive behavior (``image_stem in mask_stem``)

    Note:
        ``substring`` can false-match names such as ``img1`` and ``img10_mask``.
        Prefer ``exact_or_suffix`` for most datasets.

    Deprecated: CellPatcher no longer pairs with this function; it uses the
    preprocessing brick's pairing (:func:`pair_masks`, issue #45).
    """
    _warn_old_matcher("find_candidate_matches")
    if strategy not in {"exact", "exact_or_suffix", "substring"}:
        raise ValueError("strategy must be one of: 'exact', 'exact_or_suffix', 'substring'.")
    if not isinstance(mask_suffixes, (tuple, list)) or not all(isinstance(value, str) for value in mask_suffixes):
        raise ValueError("mask_suffixes must be a tuple/list of strings.")
    if not isinstance(case_sensitive, bool):
        raise ValueError("case_sensitive must be a bool.")

    # case_sensitive=False compares lower-cased stems; True compares as-is.
    normalize = (lambda value: value) if case_sensitive else (lambda value: value.lower())
    suffixes = tuple(normalize(value) for value in mask_suffixes)
    mask_names = {Path(m): normalize(image_stem(m)) for m in mask_files}
    out = {}
    # For each image, collect every mask whose stem matches under the
    # chosen strategy -- possibly more than one, which
    # validate_image_and_mask_folders then refuses.
    for image in image_files:
        image = Path(image)
        key = normalize(image_stem(image))
        matches: list[Path] = []
        for mask_path, mask_name in mask_names.items():
            if strategy == "exact":
                if mask_name == key:
                    matches.append(mask_path)
            elif strategy == "exact_or_suffix":
                if mask_name == key or any(mask_name == f"{key}{suffix}" for suffix in suffixes):
                    matches.append(mask_path)
            else:  # substring compatibility
                if key in mask_name:
                    matches.append(mask_path)
        out[image] = matches
    return out


def filter_shared_matches(candidate_matches) -> dict[Path, list[Path]]:
    """Prune shared candidates when an image has exclusive alternatives.

    Candidate generation can produce masks shared across multiple images.
    For images with multiple candidates, shared masks are removed if that image
    also has at least one exclusive (image-unique) candidate. This pruning is
    repeated until stable.

    Deprecated, kept for callers that use it directly. CellPatcher no longer
    applies it: it pairs with the preprocessing brick's rule
    (:func:`pair_masks`), which refuses every ambiguous pairing (issue #45).
    """
    _warn_old_matcher("filter_shared_matches")
    filtered = {img: list(matches) for img, matches in candidate_matches.items()}

    # Iterate to a fixed point: dropping a shared mask from one image can
    # make another mask newly exclusive for a different image, which in
    # turn may let a further shared mask be dropped, and so on.
    changed = True
    while changed:
        changed = False
        # Build the reverse mapping (mask -> images currently pointing to
        # it) fresh each pass, since the previous pass may have changed it.
        mask_to_images = {}
        for img, matches in filtered.items():
            for mask in matches:
                mask_to_images.setdefault(mask, []).append(img)

        for img, matches in filtered.items():
            if len(matches) <= 1:
                continue
            # Keep only this image's exclusive candidates, but only if that
            # doesn't throw away every candidate it has.
            exclusive = [m for m in matches if len(mask_to_images.get(m, [])) == 1]
            if exclusive and len(exclusive) != len(matches):
                filtered[img] = exclusive
                changed = True

    return filtered


def _ends_with_suffix(path, suffixes, case_sensitive: bool) -> bool:
    stem = image_stem(path) if case_sensitive else image_stem(path).lower()
    return any(stem.endswith(suffix if case_sensitive else suffix.lower()) for suffix in suffixes)


def select_requested_masks(mask_files, *, strategy, mask_suffixes, case_sensitive) -> list[Path]:
    """Keep the masks whose stem ends with one of *mask_suffixes*, before any matching.

    Used by ``ignore_unrequested_masks=True``: a mask folder may hold several
    masks of each image (``img_cells.tif``, ``img_nuclei.tif``, ...), and only
    the requested ones take part in matching, so the others are neither
    candidates nor reported as masks without an image. A mask named exactly
    like its image (no suffix) ends with no suffix and is skipped too.

    Raises
    ------
    ValueError
        If *strategy* is ``"exact"`` (it uses no suffix), *mask_suffixes* is
        empty, or no mask ends with a requested suffix.

    Deprecated: replaced by ``matching.mask_type`` with the preprocessing
    brick's pairing (:func:`pair_masks`, issue #45).
    """
    _warn_old_matcher("select_requested_masks")
    if strategy == "exact":
        raise ValueError(
            "ignore_unrequested_masks=True keeps the masks ending with one of mask_suffixes, which "
            "strategy='exact' does not use; use strategy='exact_or_suffix' (or 'substring')."
        )
    if len(mask_suffixes) == 0:
        raise ValueError("ignore_unrequested_masks=True needs the suffixes to keep, but mask_suffixes is empty.")
    kept = [Path(mask) for mask in mask_files if _ends_with_suffix(mask, mask_suffixes, case_sensitive)]
    if not kept:
        raise ValueError(
            f"No mask file ends with a requested suffix {tuple(mask_suffixes)} "
            f"({'case-sensitive' if case_sensitive else 'any case'}); all {len(mask_files)} masks end in none of "
            "them. Set mask_suffixes to the suffix of the masks to use."
        )
    return kept


# The matching options CellPatcher had before it adopted the preprocessing
# brick's pairing (issue #45), with their old defaults. They are still
# accepted at these values (an old run_config.json holds them) but have no
# effect; any other value is refused, because it asked for a pairing that no
# longer exists.
LEGACY_MATCHING_DEFAULTS = {
    "strategy": "exact_or_suffix",
    "mask_suffixes": ("_mask", "_labels", "_label", "_seg", "_segmentation"),
    "case_sensitive": False,
    "ignore_unrequested_masks": False,
}

MASK_LAYOUT_HELP = (
    "Masks are paired with the MorphoBricks preprocessing brick's rule (its find_masks): a mask is a "
    "TIFF file in the mask folder, either <mask type>/<image's folders>/<image name>.tif or "
    "<image's folders>/<image name>_<mask type>.tif (for example img_01_cells.tif), where the "
    "image's folders are its folders below the image root (none for a single image folder)."
)


def warn_legacy_matching_removed(where: str, stacklevel: int = 2) -> None:
    """Emit the ``DeprecationWarning`` for a legacy matching option given at its old default."""
    warnings.warn(
        f"{where}: the matching options strategy, mask_suffixes, case_sensitive and "
        "ignore_unrequested_masks no longer have any effect and will be removed; images and masks "
        "are paired with the preprocessing brick's rule (issue #45). Use matching.mask_type to choose "
        "the mask type.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def check_legacy_matching(where: str, **options) -> None:
    """Refuse a legacy matching option set to anything but its old default (issue #45).

    *options* maps legacy option names to the values given; None means not
    given. The old default is accepted with a ``DeprecationWarning`` (it
    selects nothing the new pairing does not do, and old ``run_config.json``
    files hold it); any other value is refused with ``ValueError``, since
    ignoring it would pair differently from what was asked.
    """
    given = {name: value for name, value in options.items() if value is not None}
    if not given:
        return
    unknown = sorted(set(given) - set(LEGACY_MATCHING_DEFAULTS))
    if unknown:
        raise TypeError(f"{where}: unknown matching option(s) {unknown}.")
    changed = {
        name: value
        for name, value in given.items()
        if (tuple(value) if isinstance(value, list) else value) != LEGACY_MATCHING_DEFAULTS[name]
    }
    if not changed:
        warn_legacy_matching_removed(where, stacklevel=3)
        return
    hint = ""
    suffixes = given.get("mask_suffixes")
    if isinstance(suffixes, (tuple, list)) and len(suffixes) == 1 and str(suffixes[0]).startswith("_"):
        hint = f" For masks named <image>{suffixes[0]}.tif, set matching.mask_type={suffixes[0][1:]!r}."
    raise ValueError(
        f"{where}: {', '.join(f'{name}={value!r}' for name, value in sorted(changed.items()))} is no longer "
        "supported. CellPatcher no longer pairs images and masks by its own rules (exact/suffix/substring "
        "matching); it uses the preprocessing brick's pairing, so every brick pairs the same files (issue "
        f"#45). {MASK_LAYOUT_HELP}{hint}"
    )


@dataclass(frozen=True)
class MaskPairing:
    """The mask of every image, from the preprocessing brick's pairing (:func:`pair_masks`).

    Attributes
    ----------
    mask_type
        The mask type used: its folder name, or the part of the mask's name
        after ``<image name>_``.
    masks
        Each image path (as given) mapped to its mask file.
    """

    mask_type: str
    masks: dict


# find_masks issue codes about a mask's header (axes, shape, pixel type, planes),
# not about which image it belongs to. They are refused for the masks of the
# type the run uses and ignored for the other types, which are never read
# (owner decision, preprocessing v0.3.2). CellPatcher still checks each mask
# against its image when it reads it.
_MASK_HEADER_CODES = frozenset(
    {"unreadable_mask", "mask_not_2d", "mask_size_mismatch", "mask_pixel_type", "mask_planes_differ"}
)
_MASK_PLANES_CODE = "mask_planes_differ"

# Reported per mask type; CellPatcher checks the type it uses itself.
_MASK_MISSING_CODE = "mask_missing"


def source_relative_path(image_path, experiment_root) -> str:
    """Path of *image_path* relative to *experiment_root*, with ``/`` separators.

    Built from where the image sits in the folder tree, so it matches the
    preprocessing brick's ``source_relative_path`` (``control/img.tif`` for an
    image directly in ``control/``). Like the brick, which walks the tree
    following links, a linked image or folder keeps its path under the root
    (the link's, not its target's). A path that is not under the root as
    written is compared by real locations.
    """
    root = Path(experiment_root).expanduser()
    image = Path(image_path).expanduser()
    try:
        # Discovery builds every image path from the root, so this holds for
        # folder and tree runs; resolving would follow links out of the tree.
        return image.absolute().relative_to(root.absolute()).as_posix()
    except ValueError:  # e.g. the same folder written through a link
        pass
    try:
        return image.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Image '{image_path}' is not inside the experiment root '{root}'.") from exc


def _discovery_of(experiment_root, relative_paths):
    """The preprocessing ``DiscoveryResult`` of images CellPatcher has already discovered.

    One leaf per image folder, holding its images' relative paths. The labels
    are the folder names, ``single`` for missing levels; ``find_masks`` reads
    only the folders and image names.
    """
    from morphobricks_preprocessing import SINGLE, DiscoveryResult, LeafFolder

    by_folder: dict[str, list[str]] = {}
    for relative in relative_paths:
        folder = relative.rsplit("/", 1)[0] if "/" in relative else "."
        by_folder.setdefault(folder, []).append(relative)
    leaves = []
    for folder, images in sorted(by_folder.items()):
        names = () if folder == "." else tuple(folder.split("/"))
        labels = (names + (SINGLE,) * 3)[:3]
        leaves.append(LeafFolder(folder, *labels, tuple(sorted(images))))
    return DiscoveryResult(root=Path(experiment_root).expanduser().resolve(), leaves=tuple(leaves))


def _issue_text(issue) -> str:
    paths = list(issue.paths)
    listed = f" ({', '.join(paths[:10])}{', ...' if len(paths) > 10 else ''})" if paths else ""
    action = f" {issue.action.rstrip('.')}" if issue.action else ""
    return f"[{issue.code}] {issue.message.rstrip('.')}{listed}.{action}"


def pair_masks(experiment_root, image_files, mask_folder_path, *, mask_type=None) -> MaskPairing:
    """Pair every image with its mask using the preprocessing brick's pairing, or raise.

    CellPatcher does not pair images and masks itself: the mask folder is
    read by ``morphobricks_preprocessing.find_masks`` (decision of
    2026-10-30, issue #45), so CellPatcher, the preprocessing brick and the
    bricks that use its discovery pair the same files. A mask is a TIFF file
    named ``<mask type>/<image's folders>/<image name>.tif`` or
    ``<image's folders>/<image name>_<mask type>.tif``, where the image's
    folders are its path below *experiment_root* and the image name is its
    ``position_id`` (D85); names are compared ignoring case.

    Refused, never resolved by a guess:

    - every problem ``find_masks`` reports about the pairing, including a
      mask that could belong to two images or two types (``ambiguous_mask``),
      a file in the mask folder that pairs with no image
      (``mask_without_image``, ``mask_outside_type_folder``: never silently
      left out) and a mask folder inside the image root;
    - several mask types when *mask_type* is None, and a *mask_type* that was
      not found;
    - an image without a mask of the type used.

    - a header problem ``find_masks`` reports about a mask of the type used
      (``_MASK_HEADER_CODES``: ``unreadable_mask``, ``mask_not_2d``,
      ``mask_size_mismatch``, ``mask_pixel_type``; ``mask_planes_differ``
      between masks of the type used). The same problems of masks of other
      types are ignored: those masks are never read.

    Each mask is also checked against its image when it is read.

    Parameters
    ----------
    experiment_root
        The folder the images' paths are relative to: the image folder of a
        folder run, the image root of a tree run.
    image_files
        The images to pair, all under *experiment_root*.
    mask_folder_path
        The mask folder, mirroring *experiment_root*'s folders; it may not be
        inside *experiment_root*.
    mask_type
        The mask type to use (``config.matching.mask_type``), or None when the
        mask folder holds exactly one type.
    """
    from morphobricks_preprocessing import find_masks
    from morphobricks_preprocessing.metadata import inspect_images

    if mask_type is not None and (not isinstance(mask_type, str) or not mask_type.strip()):
        raise ValueError(f"mask_type must be None or a non-empty name; got {mask_type!r}.")
    image_files = [Path(image) for image in image_files]
    relative = {image: source_relative_path(image, experiment_root) for image in image_files}
    discovery = _discovery_of(experiment_root, relative.values())
    # The images' headers give the Y and X sizes each mask's header is checked against.
    images = inspect_images(Path(experiment_root).expanduser(), list(relative.values())).files
    inventory = find_masks(Path(mask_folder_path).expanduser(), discovery, images, ome_zarr_labels=False)

    problems = [
        _issue_text(issue)
        for issue in inventory.issues
        if issue.code not in _MASK_HEADER_CODES and issue.code != _MASK_MISSING_CODE
    ]
    if problems:
        raise ValueError(f"Images and masks cannot be paired: {'; '.join(problems)}. {MASK_LAYOUT_HELP}")
    types = inventory.types
    if not types:
        raise ValueError(f"No mask in '{mask_folder_path}' pairs with an image. {MASK_LAYOUT_HELP}")
    if mask_type is None:
        if len(types) > 1:
            raise ValueError(
                f"The mask folder holds {len(types)} mask types ({', '.join(types)}); set "
                "matching.mask_type (or mask_type=) to the one to use. A type is never chosen for you."
            )
        mask_type = types[0]
    elif mask_type not in types:
        raise ValueError(
            f"No mask of type {mask_type!r} pairs with an image; the mask folder holds the types "
            f"{', '.join(types)} (names are compared exactly). {MASK_LAYOUT_HELP}"
        )
    _refuse_header_problems(inventory, mask_type)
    by_image = {f.image: f for f in inventory.files if f.mask_type == mask_type}
    missing = [relative[image] for image in image_files if relative[image] not in by_image]
    if missing:
        raise ValueError(
            f"Images with no {mask_type!r} mask: {', '.join(missing[:10])}"
            f"{', ...' if len(missing) > 10 else ''} ({len(missing)} of {len(image_files)}). {MASK_LAYOUT_HELP}"
        )
    folder = Path(mask_folder_path).expanduser()
    return MaskPairing(mask_type, {image: folder / by_image[relative[image]].path for image in image_files})


def _refuse_header_problems(inventory, mask_type: str) -> None:
    """Refuse the header problems ``find_masks`` reports about masks of *mask_type*, the type used.

    Its ``mask_planes_differ`` compares every mask of an image, whatever its
    type; it is recomputed between the masks of *mask_type* only, so a mask
    type the run does not use never blocks it.
    """
    used = [f for f in inventory.files if f.mask_type == mask_type]
    used_paths = {f.path for f in used}
    problems = []
    for issue in inventory.issues:
        if issue.code not in _MASK_HEADER_CODES or issue.code == _MASK_PLANES_CODE:
            continue
        paths = sorted(path for path in issue.paths if path in used_paths)
        if paths:
            problems.append(f"[{issue.code}] {issue.message.rstrip('.')} ({', '.join(paths[:10])}). {issue.action}")
    by_image: dict[str, list] = {}
    for mask in used:
        if mask.axes:
            by_image.setdefault(mask.image, []).append(mask)
    differ = sorted(
        mask.path
        for masks in by_image.values()
        if len({tuple(m.planes.items()) for m in masks}) > 1
        for mask in masks
    )
    if differ:
        planes = next(issue for issue in inventory.issues if issue.code == _MASK_PLANES_CODE)
        problems.append(f"[{_MASK_PLANES_CODE}] {planes.message.rstrip('.')} ({', '.join(differ[:10])}). {planes.action}")
    if problems:
        raise ValueError(f"The {mask_type!r} masks cannot be used: {'; '.join(problems)}")


def validate_image_and_mask_folders(
    image_folder_path,
    mask_folder_path,
    *,
    mask_type=None,
    strategy=None,
    mask_suffixes=None,
    case_sensitive=None,
    ignore_unrequested_masks=None,
) -> dict[Path, Path]:
    """Pair every image of a folder with exactly one mask, or raise listing every problem.

    The pairing is the preprocessing brick's (:func:`pair_masks`, issue #45):
    masks are TIFF files named ``<image name>_<mask type>.tif`` or
    ``<mask type>/<image name>.tif`` in *mask_folder_path*. With several mask
    types in the folder, *mask_type* says which to use.

    *strategy*, *mask_suffixes*, *case_sensitive* and
    *ignore_unrequested_masks* are the former matching options: accepted
    only at their old defaults (with a ``DeprecationWarning``), refused
    otherwise (:func:`check_legacy_matching`).

    Returns
    -------
    dict
        Each image path mapped to its mask path.
    """
    check_legacy_matching(
        "validate_image_and_mask_folders",
        strategy=strategy,
        mask_suffixes=mask_suffixes,
        case_sensitive=case_sensitive,
        ignore_unrequested_masks=ignore_unrequested_masks,
    )
    mask_folder = Path(mask_folder_path).expanduser()
    if not mask_folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {mask_folder}")
    image_files = discover_folder_images(image_folder_path, mask_folder_path=mask_folder)
    if len(image_files) == 0:
        raise ValueError("No supported image files found in image folder.")
    return pair_masks(image_folder_path, image_files, mask_folder_path, mask_type=mask_type).masks


def run_patching_from_folders(
    image_folder_path,
    *,
    config=None,
    mask_folder_path=None,
    strategy=None,
    mask_suffixes=None,
    case_sensitive=None,
):
    """Convenience alias for :func:`cellpatcher.orchestration.run_patching_pipeline`.

    See that function for full parameter documentation.
    """
    from cellpatcher.orchestration import run_patching_pipeline

    return run_patching_pipeline(
        image_folder_path,
        config=config,
        mask_folder_path=mask_folder_path,
        strategy=strategy,
        mask_suffixes=mask_suffixes,
        case_sensitive=case_sensitive,
    )
