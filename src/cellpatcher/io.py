from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".npy"}


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


def read_array(file_path):
    path = Path(file_path).expanduser()
    if not path.exists() or not path.is_file():
        raise ValueError(f"File does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)

    if suffix in {".tif", ".tiff"}:
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


def find_candidate_matches(image_files, mask_files) -> dict[Path, list[Path]]:
    mask_names = {Path(m): Path(m).stem.lower() for m in mask_files}
    out = {}
    for image in image_files:
        image = Path(image)
        key = image.stem.lower()
        out[image] = [m for m, mname in mask_names.items() if key in mname]
    return out


def filter_shared_matches(candidate_matches) -> dict[Path, list[Path]]:
    filtered = {img: list(matches) for img, matches in candidate_matches.items()}

    changed = True
    while changed:
        changed = False
        mask_to_images = {}
        for img, matches in filtered.items():
            for mask in matches:
                mask_to_images.setdefault(mask, []).append(img)

        for img, matches in filtered.items():
            if len(matches) <= 1:
                continue
            exclusive = [m for m in matches if len(mask_to_images.get(m, [])) == 1]
            if exclusive and len(exclusive) != len(matches):
                filtered[img] = exclusive
                changed = True

    return filtered


def validate_image_and_mask_folders(image_folder_path, mask_folder_path) -> dict[Path, Path]:
    image_files = list_supported_files(image_folder_path)
    mask_files = list_supported_files(mask_folder_path)

    candidate = find_candidate_matches(image_files, mask_files)
    filtered = filter_shared_matches(candidate)

    no_match = [img for img, matches in filtered.items() if len(matches) == 0]
    ambiguous = [img for img, matches in filtered.items() if len(matches) > 1]

    selected = {img: matches[0] for img, matches in filtered.items() if len(matches) == 1}

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
