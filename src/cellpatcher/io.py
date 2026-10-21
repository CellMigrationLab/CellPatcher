from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellpatcher._deprecation import accept_deprecated_repeat
from cellpatcher.axes import normalize_axes

# ".zarr" is an OME-Zarr image: a folder, read with the preprocessing extra
# (docs/normalization_contract.md §2); every other extension is a file.
ZARR_SUFFIX = ".zarr"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".npy", ZARR_SUFFIX}

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
    folder = Path(folder_path).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    exts = supported_extensions if supported_extensions is not None else SUPPORTED_EXTENSIONS
    exts = {e.lower() for e in exts}

    return sorted(p for p in folder.iterdir() if _is_image(p, exts))


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


def _split_folder_entries(folder: Path, exts: set[str]) -> tuple[list[Path], list[Path]]:
    # Hidden folders (e.g. Jupyter's .ipynb_checkpoints) are never
    # conditions or biological repeats, so they must not make a flat folder
    # look mixed.
    entries = list(folder.iterdir())
    loose_images = [p for p in entries if _is_image(p, exts)]
    subdirs = [p for p in entries if p.is_dir() and not p.name.startswith(".") and p not in loose_images]
    return loose_images, subdirs


def _classify_condition_subfolders(condition_dir: Path, exts: set[str]) -> list[tuple[str, Path]]:
    # A condition folder is either "flat" (images directly inside it, no
    # biological-repeat level) or "nested" (one subfolder per biological
    # repeat) -- never both, since that would leave no unambiguous way to
    # assign the loose images to a biological repeat.
    loose_images, subdirs = _split_folder_entries(condition_dir, exts)

    if loose_images and subdirs:
        raise ValueError(
            f"Condition folder '{condition_dir}' mixes image files directly inside it with "
            "subfolders. Put all images either directly in the condition folder (treated as "
            "a single implicit biological repeat) or entirely inside biological-repeat "
            "subfolders, not both."
        )
    if loose_images:
        # Flat layout: the whole condition folder counts as one biological repeat.
        return [(SINGLE_BIOLOGICAL_REPEAT_LABEL, condition_dir)]
    # Nested layout: each subfolder is its own biological repeat, sorted for a
    # deterministic, reproducible discovery order.
    return [(d.name, d) for d in sorted(subdirs, key=lambda p: p.name)]


def _classify_repeat_subfolders(repeat_dir: Path, exts: set[str]) -> list[tuple[str, Path]]:
    # A biological-repeat folder is either "flat" (its images directly inside
    # it: one technical repeat) or "nested" (one subfolder per technical
    # repeat, a well or a dish) -- never both. An empty folder stays one flat
    # leaf so the run reports it as holding no images, as before.
    loose_images, subdirs = _split_folder_entries(repeat_dir, exts)
    if loose_images and subdirs:
        raise ValueError(
            f"Biological-repeat folder '{repeat_dir}' mixes image files directly inside it "
            "with subfolders. Put all images either directly in the biological-repeat folder "
            "(one technical repeat) or entirely inside technical-repeat subfolders (one per "
            "well or dish), not both."
        )
    if subdirs:
        return [(d.name, d) for d in sorted(subdirs, key=lambda p: p.name)]
    return [(SINGLE_TECHNICAL_REPEAT_LABEL, repeat_dir)]


def discover_condition_repeat_folders(
    root_path,
    *,
    mask_root_path=None,
    supported_extensions=None,
) -> list[ConditionRepeatFolder]:
    """Discover ``<condition>/<biological_repeat>/`` image folders under *root_path*.

    (The function keeps its historical ``condition_repeat`` name; the
    folder level it reads is the biological repeat.)

    Each immediate subfolder of *root_path* is treated as a condition. Within
    a condition folder, two layouts are supported:

    - ``<condition>/<biological_repeat>/<image files>`` — one or more
      biological-repeat subfolders, each its own biological repeat.
    - ``<condition>/<image files>`` — no biological-repeat subfolder. The
      whole condition folder is treated as a single implicit biological
      repeat, labeled ``SINGLE_BIOLOGICAL_REPEAT_LABEL`` (``"single"``).
    - ``<condition>/<biological_repeat>/<technical_repeat>/<image files>`` —
      a biological-repeat folder holding only subfolders: each subfolder is a
      technical repeat (a well, a dish, a coverslip) of that biological
      repeat. A biological-repeat folder holding its images directly is one
      technical repeat, labeled ``SINGLE_TECHNICAL_REPEAT_LABEL``
      (``"single"``). The technical level needs the biological-repeat level
      above it: ``<condition>/<well>/<images>`` reads the wells as biological
      repeats.

    If *root_path* itself holds the image files and has no subfolders, the
    whole dataset is one leaf labeled ``SINGLE_CONDITION_LABEL`` /
    ``SINGLE_BIOLOGICAL_REPEAT_LABEL`` (both ``"single"``), with
    ``image_folder`` set to *root_path*.

    A root, condition or repeat folder mixing loose image files with
    subfolders is ambiguous and raises ``ValueError``. Hidden subfolders
    (names starting with ``"."``) are ignored.

    When *mask_root_path* is given, the same ``<condition>/<biological_repeat>``
    (or ``<condition>``, ``<condition>/<biological_repeat>/<technical_repeat>``,
    or root) path is mirrored under it for each discovered
    leaf's ``mask_folder`` — its existence is not checked here; callers
    (e.g. :func:`~cellpatcher.orchestration.run_patching_condition_repeat_tree`)
    decide whether a missing mask folder matters for the mode in use.
    """
    root = Path(root_path).expanduser()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {root}")

    mask_root = Path(mask_root_path).expanduser() if mask_root_path is not None else None
    if mask_root is not None and (not mask_root.exists() or not mask_root.is_dir()):
        raise ValueError(f"Folder does not exist or is not a directory: {mask_root}")

    exts = supported_extensions if supported_extensions is not None else SUPPORTED_EXTENSIONS
    exts = {e.lower() for e in exts}

    root_images, root_subdirs = _split_folder_entries(root, exts)
    if root_images and root_subdirs:
        raise ValueError(
            f"Folder '{root}' mixes image files directly inside it with subfolders. Put all "
            "images either directly in this folder (treated as a single condition and biological "
            "repeat) or entirely inside condition subfolders, not both."
        )
    if root_images:
        # Flat layout: the root folder is the only condition and biological repeat.
        return [
            ConditionRepeatFolder(
                condition=SINGLE_CONDITION_LABEL,
                biological_repeat=SINGLE_BIOLOGICAL_REPEAT_LABEL,
                image_folder=root,
                mask_folder=mask_root,
            )
        ]

    # Every immediate subfolder of `root` is a condition; sorted for a
    # deterministic discovery order across runs.
    condition_dirs = sorted(root_subdirs, key=lambda p: p.name)

    results: list[ConditionRepeatFolder] = []
    for condition_dir in condition_dirs:
        condition = condition_dir.name
        for biological_repeat, repeat_folder in _classify_condition_subfolders(condition_dir, exts):
            # A flat condition folder is its own (single) biological repeat and
            # holds images, so only an explicit biological-repeat folder can
            # nest wells.
            technical = (
                [(SINGLE_TECHNICAL_REPEAT_LABEL, repeat_folder)]
                if biological_repeat == SINGLE_BIOLOGICAL_REPEAT_LABEL
                else _classify_repeat_subfolders(repeat_folder, exts)
            )
            for technical_repeat, image_folder in technical:
                # Mirror the same input path under mask_root so callers can
                # look for a matching mask folder, without this function
                # needing to know whether the mode in use requires one.
                mask_folder = None
                if mask_root is not None:
                    mask_folder = mask_root / image_folder.relative_to(root)
                results.append(
                    ConditionRepeatFolder(
                        condition=condition,
                        biological_repeat=biological_repeat,
                        image_folder=image_folder,
                        mask_folder=mask_folder,
                        technical_repeat=technical_repeat,
                    )
                )
    return results


def read_array(file_path):
    """Read an image or mask file into a NumPy array.

    An OME-Zarr image (a ``.zarr`` folder) is read from level 0 of its only
    image, in its stored axis order; see :func:`read_ome_zarr` for its axes.
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
            return tifffile.imread(path)
        except ImportError:
            try:
                import imageio.v3 as iio
            except ImportError as exc:
                raise ImportError("Reading TIFF requires tifffile or imageio.") from exc
            return iio.imread(path)

    if suffix in SUPPORTED_EXTENSIONS:
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

    if suffix in SUPPORTED_EXTENSIONS:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError("Writing image files requires imageio.") from exc
        iio.imwrite(path, np.asarray(array))
        return

    raise ValueError(f"Unsupported file extension: {suffix}")


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
    """
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
    # chosen strategy -- possibly more than one, which is exactly what
    # filter_shared_matches/validate_image_and_mask_folders resolve next.
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

    Any remaining ambiguity is handled later by
    ``validate_image_and_mask_folders``.
    """
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


def validate_image_and_mask_folders(
    image_folder_path,
    mask_folder_path,
    *,
    strategy="exact_or_suffix",
    mask_suffixes=("_mask", "_labels", "_label", "_seg", "_segmentation"),
    case_sensitive=False,
) -> dict[Path, Path]:
    image_files = list_supported_files(image_folder_path)
    mask_files = list_supported_files(mask_folder_path)
    if len(image_files) == 0:
        raise ValueError("No supported image files found in image folder.")
    if len(mask_files) == 0:
        raise ValueError("No supported mask files found in mask folder.")

    candidate = find_candidate_matches(
        image_files,
        mask_files,
        strategy=strategy,
        mask_suffixes=mask_suffixes,
        case_sensitive=case_sensitive,
    )
    filtered = filter_shared_matches(candidate)

    # After pruning, only images with exactly one candidate are safe to
    # resolve automatically; zero or multiple candidates are collected
    # below as validation errors instead of guessing.
    no_match = [img for img, matches in filtered.items() if len(matches) == 0]
    ambiguous = [img for img, matches in filtered.items() if len(matches) > 1]

    selected = {img: matches[0] for img, matches in filtered.items() if len(matches) == 1}

    # A mask can still end up selected by more than one image (pruning only
    # removes shared candidates when an exclusive alternative exists), so
    # check for that explicitly, along with masks nothing selected at all.
    mask_to_images = {}
    for img, mask in selected.items():
        mask_to_images.setdefault(mask, []).append(img)

    shared_masks = [mask for mask, imgs in mask_to_images.items() if len(imgs) > 1]
    unmatched_masks = [mask for mask in mask_files if mask not in selected.values()]

    errors = []
    if no_match:
        errors.append("Images with no mask match: " + ", ".join(p.name for p in no_match[:10]))
    if ambiguous:
        errors.append("Images with multiple mask matches: " + ", ".join(p.name for p in ambiguous[:10]))
    if shared_masks:
        errors.append("Masks linked to multiple images: " + ", ".join(p.name for p in shared_masks[:10]))
    if unmatched_masks:
        errors.append("Masks with no matching image: " + ", ".join(p.name for p in unmatched_masks[:10]))

    if errors:
        raise ValueError("; ".join(errors))

    return selected


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
