from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".npy"}

# Repeat label stamped when a condition folder holds images directly, with
# no repeat subfolder (see discover_condition_repeat_folders below). Treated
# as "this condition has exactly one repeat" rather than an error.
SINGLE_REPEAT_LABEL = "single"


def list_supported_files(folder_path, supported_extensions=None) -> list[Path]:
    folder = Path(folder_path).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    exts = supported_extensions if supported_extensions is not None else SUPPORTED_EXTENSIONS
    exts = {e.lower() for e in exts}

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


@dataclass
class ConditionRepeatFolder:
    """One discovered ``<condition>/<repeat>`` (or single-repeat ``<condition>``) leaf.

    ``repeat`` is ``SINGLE_REPEAT_LABEL`` when *condition_dir* holds image
    files directly with no repeat subfolder; in that case ``image_folder``
    is the condition folder itself. See
    :func:`discover_condition_repeat_folders`.
    """
    condition: str
    repeat: str
    image_folder: Path
    mask_folder: Path | None = None


def _classify_condition_subfolders(condition_dir: Path, exts: set[str]) -> list[tuple[str, Path]]:
    # A condition folder is either "flat" (images directly inside it, no
    # repeat concept) or "nested" (one subfolder per repeat) -- never both,
    # since that would leave no unambiguous way to assign the loose images
    # to a repeat.
    entries = list(condition_dir.iterdir())
    loose_images = [p for p in entries if p.is_file() and p.suffix.lower() in exts]
    subdirs = [p for p in entries if p.is_dir()]

    if loose_images and subdirs:
        raise ValueError(
            f"Condition folder '{condition_dir}' mixes image files directly inside it with "
            "subfolders. Put all images either directly in the condition folder (treated as "
            "a single implicit repeat) or entirely inside repeat subfolders, not both."
        )
    if loose_images:
        # Flat layout: the whole condition folder counts as one repeat.
        return [(SINGLE_REPEAT_LABEL, condition_dir)]
    # Nested layout: each subfolder is its own repeat, sorted for a
    # deterministic, reproducible discovery order.
    return [(d.name, d) for d in sorted(subdirs, key=lambda p: p.name)]


def discover_condition_repeat_folders(
    root_path,
    *,
    mask_root_path=None,
    supported_extensions=None,
) -> list[ConditionRepeatFolder]:
    """Discover ``<condition>/<repeat>/`` image folders under *root_path*.

    Each immediate subfolder of *root_path* is treated as a condition. Within
    a condition folder, two layouts are supported:

    - ``<condition>/<repeat>/<image files>`` — one or more repeat
      subfolders, each its own repeat.
    - ``<condition>/<image files>`` — no repeat subfolder. The whole
      condition folder is treated as a single implicit repeat, labeled
      ``SINGLE_REPEAT_LABEL`` (``"single"``).

    A condition folder mixing loose image files with subfolders is
    ambiguous and raises ``ValueError``.

    When *mask_root_path* is given, the same ``<condition>/<repeat>``
    (or ``<condition>``) path is mirrored under it for each discovered
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

    # Every immediate subfolder of `root` is a condition; sorted for a
    # deterministic discovery order across runs.
    condition_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)

    results: list[ConditionRepeatFolder] = []
    for condition_dir in condition_dirs:
        condition = condition_dir.name
        for repeat, image_folder in _classify_condition_subfolders(condition_dir, exts):
            # Mirror the same condition/(repeat) path under mask_root so
            # callers can look for a matching mask folder, without this
            # function needing to know whether the mode in use requires one.
            mask_folder = None
            if mask_root is not None:
                mask_folder = mask_root / condition if repeat == SINGLE_REPEAT_LABEL else mask_root / condition / repeat
            results.append(
                ConditionRepeatFolder(
                    condition=condition, repeat=repeat, image_folder=image_folder, mask_folder=mask_folder
                )
            )
    return results


def read_array(file_path):
    path = Path(file_path).expanduser()
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
    mask_names = {Path(m): normalize(Path(m).stem) for m in mask_files}
    out = {}
    # For each image, collect every mask whose stem matches under the
    # chosen strategy -- possibly more than one, which is exactly what
    # filter_shared_matches/validate_image_and_mask_folders resolve next.
    for image in image_files:
        image = Path(image)
        key = normalize(image.stem)
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
