import csv
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np

from cellpatcher.types import PatchRegion


def sanitize_filename(value: str) -> str:
    text = str(value).replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    text = text.strip("_")
    return text or "patch"


def patch_metadata_to_rows(patches) -> list[dict]:
    return [patch.to_dict() for patch in patches]


def save_metadata_csv(patches, csv_path):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The CSV header follows PatchRegion's own field order, with the
    # computed height/width convenience columns inserted right after the
    # x1 coordinate they're derived from (rather than tacked on at the end).
    header = [f.name for f in fields(PatchRegion)]
    x1_index = header.index("x1")
    header = header[: x1_index + 1] + ["height", "width"] + header[x1_index + 1 :]
    rows = patch_metadata_to_rows(patches)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in header})


def save_patch_array(array, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()

    if suffix == ".npy":
        np.save(out, np.asarray(array))
        return

    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("tifffile is required for TIFF export.") from exc
        tifffile.imwrite(str(out), np.asarray(array))
        return

    raise ValueError(f"Unsupported export file extension: {out.suffix}")


def export_patches(
    patches,
    patch_arrays,
    output_dir,
    *,
    image_format="tif",
    save_metadata=True,
    metadata_filename="patches.csv",
) -> list[str]:
    if output_dir is None:
        raise ValueError("output_dir is required for export.")
    if len(patches) != len(patch_arrays):
        raise ValueError("patches and patch_arrays must have the same length.")

    fmt = str(image_format).lower()
    if fmt not in {"tif", "tiff", "npy"}:
        raise ValueError("image_format must be one of: 'tif', 'tiff', 'npy'.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanitizing patch_id for use as a filename can collapse otherwise
    # distinct ids into the same string (e.g. differing only by characters
    # that both get replaced); check for that up front so a collision fails
    # loudly instead of one patch silently overwriting another's file.
    candidate_filenames = [f"{sanitize_filename(patch.patch_id)}.{fmt}" for patch in patches]
    counts = Counter(candidate_filenames)
    duplicate_filenames = sorted(name for name, count in counts.items() if count > 1)
    if duplicate_filenames:
        preview = ", ".join(duplicate_filenames[:5])
        raise ValueError(f"Duplicate export filenames detected after sanitization: {preview}")

    saved_files: list[str] = []
    for patch, array in zip(patches, patch_arrays):
        filename = f"{sanitize_filename(patch.patch_id)}.{fmt}"
        path = out_dir / filename
        save_patch_array(array, path)
        saved_files.append(str(path))
        # Stamp patch_file before the metadata CSV is written below, so
        # that CSV records the exact filename each patch was saved as.
        patch.patch_file = filename

    if save_metadata:
        save_metadata_csv(patches, out_dir / metadata_filename)

    return saved_files
