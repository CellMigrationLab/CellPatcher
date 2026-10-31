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


class PatchTableError(ValueError):
    """A ``patches.csv`` cannot be read back as patches; the message lists every problem with its line."""


# Columns of tables written by earlier CellPatcher versions, read but not fields themselves:
# the ;-separated parameters of tables before 2026-10-28 (normalization brick N47).
_LEGACY_PARAMETER_COLUMNS = ("normalization_offset", "normalization_scale")
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+")
_BOOLEAN_TEXT = {"true": True, "false": False}


def _field_kinds() -> dict[str, tuple[type, bool]]:
    """PatchRegion column -> (int, float, bool or str; whether it may be None)."""
    import typing

    hints = typing.get_type_hints(PatchRegion)
    kinds = {}
    for item in fields(PatchRegion):
        if item.name in ("normalization_parameters", "mask_fractions"):
            continue
        arguments = typing.get_args(hints[item.name])
        optional = type(None) in arguments
        base = next((kind for kind in (arguments or (hints[item.name],)) if kind is not type(None)), str)
        kinds[item.name] = (base, optional)
    return kinds


def _typed(text: str, kind: type, optional: bool, column: str):
    """The value of one CSV cell, as :func:`save_metadata_csv` wrote it; ValueError otherwise."""
    if text == "":
        if optional:
            return None
        raise ValueError(f"{column} is empty")
    if kind is str:
        return text
    if kind is bool:
        value = _BOOLEAN_TEXT.get(text.lower())
        if value is None:
            raise ValueError(f"{column}={text!r} is not true or false")
        return value
    if kind is int:
        if not _INTEGER_TEXT.fullmatch(text):
            raise ValueError(f"{column}={text!r} is not an integer")
        return int(text)
    if kind is float:
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{column}={text!r} is not a number") from None
        if not np.isfinite(value):
            raise ValueError(f"{column}={text!r} is not a finite number")
        return value
    raise TypeError(f"no reader for {column} ({kind})")


def _read_run_config(csv_path: Path, names: set[str], problems: list[str]):
    """The ``run_config.json`` the rows' ``config_file`` names (relative to the CSV's folder), or None."""
    import json

    named = names - {""}
    if not named:
        return None
    if "" in names:
        problems.append("only some rows name a config_file; CellPatcher names it in every row or in none")
        return None
    if len(named) > 1:
        problems.append(f"the rows name several config files {sorted(named)}; CellPatcher writes one per table")
        return None
    path = csv_path.parent / named.pop()
    if not path.is_file():
        problems.append(f"config file {path} does not exist")
        return None
    try:
        run_config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"config file {path} cannot be read as JSON: {exc}")
        return None
    if not isinstance(run_config, dict):
        problems.append(f"config file {path} holds a {type(run_config).__name__}, not the JSON object CellPatcher writes")
        return None
    return run_config


def read_patch_table(csv_path) -> tuple[list[PatchRegion], dict | None]:
    """Read a ``patches.csv`` back as :class:`PatchRegion` objects, with its ``run_config.json``.

    The typed inverse of :func:`patch_metadata_to_rows`/:func:`save_metadata_csv`
    (D86 API C2), for the bricks that read CellPatcher's patches instead of
    parsing the table themselves. Every column is read with its
    ``PatchRegion`` field's type: integers from integer text only (``"12.0"``
    is refused), booleans from ``true``/``false`` in any case, an empty cell
    as None (refused for a field that cannot be None: ``patch_id``, the
    rectangle, ``touches_border``, ``valid``); labels and ids stay text
    (``"001"`` is not a number). ``height``/``width`` must equal the
    rectangle's; ``mask_fraction_<name>`` columns give ``mask_fractions``;
    ``Normalization_Offset_C<i>``/``Normalization_Scale_C<i>`` (or the
    ``;``-separated ``normalization_offset``/``normalization_scale`` of tables
    written before 2026-10-28) give ``normalization_parameters``, read with
    the normalization settings of the run config. A ``repeat`` column of a
    table written before the rename is read as ``biological_repeat`` with a
    ``DeprecationWarning``. A column CellPatcher does not write, and two rows
    with one ``patch_id``, are refused. A column the table lacks (a table
    written before it was added) leaves the field at its default.

    Parameters
    ----------
    csv_path
        The ``patches.csv``.

    Returns
    -------
    tuple
        The patches in row order, and the ``run_config.json`` the rows'
        ``config_file`` names, relative to the CSV's folder, as a dict (None
        when no row names one).

    Raises
    ------
    FileNotFoundError
        If the table does not exist.
    PatchTableError
        (a ``ValueError``) listing every problem: a missing column
        (``patch_id``, ``y0``, ``y1``, ``x0``, ``x1``), a cell that is not of
        its field's type, a rectangle that is not one, an unknown or
        duplicate column, a duplicate ``patch_id``, or a config file that
        is missing, not a JSON object, or not the same for every row.
    """
    from cellpatcher._deprecation import upgrade_legacy_repeat_column

    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"The patch table {path} does not exist.")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [(reader.line_num, row) for row in reader]
    lines = [line for line, _ in rows]
    rows = upgrade_legacy_repeat_column(header, [row for _, row in rows], str(path))
    header = ["biological_repeat" if name == "repeat" and "biological_repeat" not in header else name for name in header]

    kinds = _field_kinds()
    problems: list[str] = []
    duplicated = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicated:
        problems.append(f"duplicate columns {duplicated}")
    known = set(kinds) | {"height", "width", *_LEGACY_PARAMETER_COLUMNS}
    unknown = [
        name for name in header
        if name not in known
        and not name.startswith(MASK_FRACTION_PREFIX)
        and not re.fullmatch(r"Normalization_(Offset|Scale)_C[0-9]+", name)
    ]
    if unknown:
        problems.append(f"columns CellPatcher does not write: {unknown}")
    missing = [name for name in ("patch_id", "y0", "y1", "x0", "x1") if name not in header]
    if missing:
        problems.append(f"missing columns {missing}")
    if problems:
        raise PatchTableError(f"{path}: " + "; ".join(problems))

    run_config = _read_run_config(path, {row.get("config_file") or "" for row in rows}, problems)
    settings = (run_config or {}).get("normalization") or {}
    patches: list[PatchRegion] = []
    for line, row in zip(lines, rows):
        row_problems: list[str] = []
        if row.get(None):
            row_problems.append(f"{len(row[None])} more cell(s) than the header has columns")
        values = {}
        for name in header:
            if name in kinds:
                try:
                    values[name] = _typed(row[name] or "", *kinds[name], name)
                except ValueError as exc:
                    row_problems.append(str(exc))
        fractions = {}
        for name in header:
            if name.startswith(MASK_FRACTION_PREFIX) and row[name]:
                try:
                    fractions[name[len(MASK_FRACTION_PREFIX):]] = _typed(row[name], float, False, name)
                except ValueError as exc:
                    row_problems.append(str(exc))
        if fractions:
            values["mask_fractions"] = fractions
        has_parameters = any(
            row[name] for name in header
            if name in _LEGACY_PARAMETER_COLUMNS or name.startswith("Normalization_")
        )
        if has_parameters:
            if run_config is None:
                row_problems.append(
                    "normalization parameters cannot be read without the run_config.json naming the "
                    "normalization settings (no config_file)"
                )
            else:
                from cellpatcher.normalization import parameters_from_row

                try:
                    values["normalization_parameters"] = parameters_from_row(row, settings)
                except ValueError as exc:
                    row_problems.append(f"normalization parameters: {exc}")
        if not row_problems:
            try:
                patch = PatchRegion(**values)
            except ValueError as exc:
                row_problems.append(str(exc))
            else:
                for name, size in (("height", patch.height), ("width", patch.width)):
                    if name in header and row[name] != str(size):
                        row_problems.append(f"{name}={row[name]!r} is not the rectangle's {size}")
        if row_problems:
            problems.append(f"line {line} (patch_id {row.get('patch_id')!r}): " + "; ".join(row_problems))
        else:
            patches.append(patch)
    counts = Counter(patch.patch_id for patch in patches)
    repeated = sorted(patch_id for patch_id, count in counts.items() if count > 1)
    if repeated:
        problems.append(f"patch_id values on several rows: {repeated[:10]}")
    if problems:
        raise PatchTableError(f"{path}: " + "; ".join(problems))
    return patches, run_config
