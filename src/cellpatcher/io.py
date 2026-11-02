from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from morphobricks_normalization.records import SINGLE

from cellpatcher._deprecation import accept_deprecated_repeat
from cellpatcher.axes import normalize_axes

# ".zarr" is an OME-Zarr image: a folder, read with the preprocessing brick's
# reader and the ``zarr`` extra (docs/normalization_contract.md §2); every other
# extension is a file.
ZARR_SUFFIX = ".zarr"

# The images CellPatcher discovers: those of the preprocessing brick's
# discover_experiment, TIFF files and OME-Zarr folders (owner decision of
# 2026-10-30, issue #45), so every brick patches, measures and normalizes the
# same images.
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ZARR_SUFFIX}

# Image formats CellPatcher read as input before it adopted the preprocessing
# brick's discovery; read_array and write_array still handle them, because
# exported patches may be written in them (export.patch_format).
FORMER_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".npy")

# Image formats that are not discovered: the former ones and common microscope
# and raster formats. Such a file where images are discovered is refused,
# naming it, never skipped: it would otherwise drop out of the dataset
# unnoticed (owner decision of 2026-10-30). Other files (.txt, .csv, ...) are
# not images and stay ignored.
UNSUPPORTED_IMAGE_EXTENSIONS = (
    *FORMER_IMAGE_EXTENSIONS,
    ".npz", ".webp", ".jp2", ".czi", ".nd2", ".lif", ".lsm", ".oib", ".oif", ".oir", ".ims", ".vsi",
    ".dv", ".svs", ".ndpi",
)

# The file formats read_array and write_array handle: the input images, and
# the formats exported patches may be written in (export.patch_format).
ARRAY_EXTENSIONS = {*SUPPORTED_EXTENSIONS, *FORMER_IMAGE_EXTENSIONS}

# The label of an absent folder level is the normalization brick's SINGLE
# ("single"), shared by every MorphoBricks brick (D86, CP-1); the names below
# are public aliases of it.

# Biological-repeat label stamped when a condition folder holds images
# directly, with no biological-repeat subfolder (see
# discover_condition_repeat_folders below). Treated as "this condition has
# exactly one biological repeat" rather than an error.
SINGLE_BIOLOGICAL_REPEAT_LABEL = SINGLE

# Name kept from before `repeat` was renamed `biological_repeat`; same value,
# still public and not deprecated.
SINGLE_REPEAT_LABEL = SINGLE_BIOLOGICAL_REPEAT_LABEL

# Condition label stamped when the root folder itself holds the images, with
# no condition subfolders: the whole dataset is one condition and one
# biological repeat.
SINGLE_CONDITION_LABEL = SINGLE

# Technical-repeat label stamped when a repeat folder holds its images
# directly, with no <technical_repeat> subfolders (a well, a dish): the
# repeat is one technical repeat. Never a biological repeat, which stays in
# `repeat` (see discover_condition_repeat_folders below).
SINGLE_TECHNICAL_REPEAT_LABEL = SINGLE


def list_supported_files(folder_path, supported_extensions=None) -> list[Path]:
    """The images directly in *folder_path* (TIFF files and OME-Zarr folders), sorted.

    Subfolders are not entered. A file in a format CellPatcher no longer
    reads as an input image (:data:`UNSUPPORTED_IMAGE_EXTENSIONS`: PNG, JPEG,
    BMP, GIF, NumPy, microscope formats such as CZI) is refused, naming it, as is a link to a missing image.
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
            "and OME-Zarr folders (.zarr) only; PNG, JPEG, BMP, GIF, NumPy (.npy) and microscope formats "
            "such as .czi, .nd2 or .lif are not read (decision of 2026-10-30, issue #45). Convert them to "
            "TIFF or OME-Zarr, or move them out of the folder."
        )


def is_ome_zarr(path) -> bool:
    """True if *path* names an OME-Zarr image (a folder ending in ``.zarr``, any case)."""
    return Path(path).suffix.lower() == ZARR_SUFFIX


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
    - PNG, JPEG, BMP, GIF, NumPy and microscope-format images such as CZI, ND2 or LIF
      (:data:`UNSUPPORTED_IMAGE_EXTENSIONS`);
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
    PNG/JPEG/BMP/GIF/NumPy or microscope format, a fourth level of folders, two images of
    one folder with the same name apart from the extension or case, and
    folder names differing only in case are refused (:func:`discover_images`
    lists every refusal). Folders without images or subfolders are left out
    with a ``UserWarning``. System files and folders are ignored: names
    starting with ``"."`` (``.ipynb_checkpoints``, ``._img.tif``),
    ``Thumbs.db``, ``desktop.ini`` and ``__MACOSX``.

    When *mask_root_path* is given, the same leaf path is mirrored under it
    for each leaf's ``mask_folder``; its existence is not checked here.

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
    formats and files whose header names no channel give (); the names are
    for display only.
    """
    from morphobricks_preprocessing.metadata import inspect_image

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
        If the ``zarr`` extra is not installed.
    ValueError
        If the folder is not an OME-Zarr image that can be read, or holds
        more than one image (several positions, series or wells), which is
        an error as in the preprocessing brick.
    """
    from cellpatcher.statistics import INSTALL_COMMAND, PreprocessingUnavailableError, _preprocessing

    mbp = _preprocessing()
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


def read_array_shape(file_path) -> tuple[int, ...]:
    return tuple(read_array(file_path).shape)


# The matching options CellPatcher had before it adopted the preprocessing
# brick's pairing (issue #45). They were accepted at their old defaults with a
# DeprecationWarning until v0.1.1, which removed them (D86, owner decision of
# 2026-10-31): any use, an old run_config.json included, is refused by name.
REMOVED_MATCHING_OPTIONS = ("strategy", "mask_suffixes", "case_sensitive", "ignore_unrequested_masks")

MASK_LAYOUT_HELP = (
    "Masks are paired with the MorphoBricks preprocessing brick's rule (its find_masks): a mask is a "
    "TIFF file in the mask folder, either <mask type>/<image's folders>/<image name>.tif or "
    "<image's folders>/<image name>_<mask type>.tif (for example img_01_cells.tif), where the "
    "image's folders are its folders below the image root (none for a single image folder)."
)


def refuse_removed_matching(where: str, options) -> None:
    """Refuse the removed matching options among *options* by name, and any other unknown option.

    *options* maps the names given to their values (a function's ``**kwargs``,
    a ``matching`` section of a config file). The former options
    (:data:`REMOVED_MATCHING_OPTIONS`) raise ``ValueError`` saying to use
    ``matching.mask_type``, whatever their value; any other name raises
    ``TypeError``.

    Raises
    ------
    ValueError
        If a removed matching option is given.
    TypeError
        If another unknown option is given.
    """
    names = list(options)
    if not names:
        return
    unknown = sorted(name for name in names if name not in REMOVED_MATCHING_OPTIONS)
    if unknown:
        raise TypeError(f"{where}: unknown option(s) {unknown}.")
    hint = ""
    suffixes = options.get("mask_suffixes")
    if isinstance(suffixes, (tuple, list)) and len(suffixes) == 1 and str(suffixes[0]).startswith("_"):
        hint = f" For masks named <image>{suffixes[0]}.tif, set matching.mask_type={suffixes[0][1:]!r}."
    raise ValueError(
        f"{where}: the matching option(s) {', '.join(sorted(names))} were removed in CellPatcher v0.1.1. "
        "Images and masks are paired with the preprocessing brick's rule (issue #45); choose the mask "
        f"type with matching.mask_type (mask_type= for the functions). {MASK_LAYOUT_HELP}{hint} Remove "
        "these options, also from an old run_config.json."
    )


MASK_TRANSPOSE_HELP = (
    "To fix the masks, give each one with Y before X, as its image: transpose its last two axes, "
    "for example numpy.swapaxes(mask, -1, -2), then set common.mask_axes to those axes ('YX', "
    "or 'ZYX'/'TYX' for a stack). Nothing is ever read transposed."
)


def refuse_transposed_mask_axes(mask_axes: str) -> None:
    """Refuse ``common.mask_axes`` that put X before Y: CellPatcher cannot read a mask transposed.

    The cell and random strategies take a label plane's two dimensions as Y
    and X in their stored order, so a mask declared with X first (``'XY'``,
    ``'ZXY'``) would be taken as if Y came first: the patches would not
    match their objects (on a square image without any error).
    Refused, never ignored (owner decision of 31 October 2026). Only
    in-memory masks are declared: a folder or tree run reads each mask in the
    roles its preprocessing run recorded. :data:`MASK_TRANSPOSE_HELP` says how
    to fix the masks.
    """
    if mask_axes is None:
        return
    axes = normalize_axes(mask_axes)
    if "X" in axes and "Y" in axes and axes.index("X") < axes.index("Y"):
        raise ValueError(
            f"common.mask_axes={mask_axes!r} puts X before Y, but CellPatcher cannot read a mask "
            "transposed: it takes a label plane's dimensions as Y, X in their stored order, so the "
            f"mask would not match its image. {MASK_TRANSPOSE_HELP}"
        )


def _issue_text(issue) -> str:
    paths = list(issue.paths)
    listed = f" ({', '.join(paths[:10])}{', ...' if len(paths) > 10 else ''})" if paths else ""
    action = f" {issue.action.rstrip('.')}" if issue.action else ""
    return f"[{issue.code}] {issue.message.rstrip('.')}{listed}.{action}"


def run_patching_from_folders(
    image_folder_path,
    *,
    config=None,
    mask_folder_path=None,
    **removed_options,
):
    """Convenience alias for :func:`cellpatcher.orchestration.run_patching_pipeline`.

    See that function for full parameter documentation.
    """
    from cellpatcher.orchestration import run_patching_pipeline

    return run_patching_pipeline(
        image_folder_path,
        config=config,
        mask_folder_path=mask_folder_path,
        **removed_options,
    )
