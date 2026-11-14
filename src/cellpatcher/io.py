from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from morphobricks_normalization.records import SINGLE

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
# directly, with no biological-repeat subfolder. Treated as "this condition has
# exactly one biological repeat" rather than an error.
SINGLE_BIOLOGICAL_REPEAT_LABEL = SINGLE

# Condition label stamped when the root folder itself holds the images, with
# no condition subfolders: the whole dataset is one condition and one
# biological repeat.
SINGLE_CONDITION_LABEL = SINGLE

# Technical-repeat label stamped when a repeat folder holds its images
# directly, with no <technical_repeat> subfolders (a well, a dish): the
# repeat is one technical repeat. Never a biological repeat, which stays in
# `biological_repeat`.
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


@dataclass
class ConditionRepeatFolder:
    """One discovered ``<condition>/<biological_repeat>[/<technical_repeat>]`` leaf.

    ``biological_repeat`` is ``SINGLE_BIOLOGICAL_REPEAT_LABEL`` when
    the condition folder holds image files directly with no biological-repeat
    subfolder; in that case ``image_folder`` is the condition folder itself.
    ``technical_repeat`` is the name of a technical-repeat subfolder (a well,
    a dish) inside the biological-repeat folder, or
    ``SINGLE_TECHNICAL_REPEAT_LABEL`` when that folder holds its images
    directly. :func:`cellpatcher.run_patching_condition_repeat_tree` passes
    one to its progress callback.
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
        import tifffile

        with tifffile.TiffFile(path) as tif:
            _refuse_several_series(path, len(tif.series))
            return tif.asarray()

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
