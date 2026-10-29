import csv
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np

from cellpatcher.types import IMAGE_TABLE_COLUMNS, MASK_FRACTION_PREFIX, ImageRecord, PatchRegion

# The per-image table written beside every patches.csv (see save_image_table_csv).
IMAGE_TABLE_FILENAME = "images.csv"


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
    header = [f.name for f in fields(PatchRegion) if f.name not in ("normalization_parameters", "mask_fractions")]
    x1_index = header.index("x1")
    header = header[: x1_index + 1] + ["height", "width"] + header[x1_index + 1 :]
    exact_index = header.index("normalization_exact")
    header = header[: exact_index + 1] + normalization_columns(patches) + header[exact_index + 1 :]
    header += mask_fraction_columns(patches)
    rows = patch_metadata_to_rows(patches)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            # normalization_exact is written true/false (contract §10).
            if isinstance(row.get("normalization_exact"), bool):
                row["normalization_exact"] = "true" if row["normalization_exact"] else "false"
            writer.writerow({k: row.get(k) for k in header})


def mask_fraction_columns(patches) -> list[str]:
    """One ``mask_fraction_<name>`` column per mask any patch has a fraction for, in first-seen order."""
    names = dict.fromkeys(name for patch in patches for name in (patch.mask_fractions or {}))
    return [f"{MASK_FRACTION_PREFIX}{name}" for name in names]


def normalization_columns(patches) -> list[str]:
    """The per-channel parameter columns of *patches*, grouped by channel (normalization brick N41, N42).

    One ``Normalization_Offset_C<i>``/``Normalization_Scale_C<i>`` pair for
    every channel any patch has parameters for; a row of a split channel
    leaves the other channels' pairs blank. None without normalization.
    """
    channels = sorted({
        int(c)
        for patch in patches
        if patch.normalization_parameters is not None
        for c in patch.normalization_parameters.channels
    })
    return [name for c in channels for name in (f"Normalization_Offset_C{c}", f"Normalization_Scale_C{c}")]


def save_image_table_csv(image_records, csv_path):
    """Write the per-image table: one ``images.csv`` row per :class:`ImageRecord`.

    Lists every image the run considered, including those without patches,
    with the shared identity columns of ``patches.csv`` (AGENTS.md §5), the
    image's Y/X size, its number of rows in ``patches.csv``, a
    ``status``/``reason``, its ``position_id`` and the region counts behind
    ``n_patches`` (:data:`~cellpatcher.types.IMAGE_TABLE_COLUMNS`).
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(IMAGE_TABLE_COLUMNS))
        writer.writeheader()
        for record in image_records:
            writer.writerow(record.to_dict())


def image_records_from_patches(patches) -> list[ImageRecord]:
    """One record per source image of *patches*, for callers that give no image records.

    Only images with patches can be known from the patches themselves, so
    every record is ``processed``, and its region counts are its
    ``n_patches`` (the earlier counts are not known here). The pipeline
    passes its own records, which also list the images without patches.
    """
    records: dict[tuple, ImageRecord] = {}
    for patch in patches:
        key = (patch.condition, patch.biological_repeat, patch.technical_repeat, patch.image_id, patch.image_file)
        record = records.get(key)
        if record is None:
            record = records[key] = ImageRecord(
                image_id=patch.image_id,
                condition=patch.condition,
                biological_repeat=patch.biological_repeat,
                technical_repeat=patch.technical_repeat,
                image_file=patch.image_file,
                mask_file=patch.mask_file,
                source_relative_path=patch.source_relative_path,
                image_height=patch.image_height,
                image_width=patch.image_width,
                position_id=patch.position_id,
            )
        record.n_patches += 1
        record.n_regions = record.n_kept = record.n_normalized = record.n_patches
    return list(records.values())


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
    image_records=None,
    image_table_filename=IMAGE_TABLE_FILENAME,
) -> list[str]:
    """Save every patch array as a file named by its ``patch_id`` and, optionally, the metadata.

    With *save_metadata*, ``patches.csv`` (*metadata_filename*) and the
    per-image table ``images.csv`` (*image_table_filename*,
    :func:`save_image_table_csv`) are written in *output_dir*. The table
    lists *image_records* (the pipeline passes one per input image, with
    those that gave no patch), or, when None, the images of *patches*.
    """
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
        if image_records is None:
            image_records = image_records_from_patches(patches)
        save_image_table_csv(image_records, out_dir / image_table_filename)

    return saved_files
