"""Run-independent patch identity: ``position_id``, ``object_set`` and ``patch_id``.

A patch's ``patch_id`` is built only from the identity of its field of view
and its own coordinates, never from the position of its image in a run, so
the same image gives the same ids whether it is patched alone, in a folder of
ten images, or in a tree with other wells (owner decision D44, 2026-10-26)::

    patch_id  := position_id "/" [object_set "/"] ["t" T "/"] ["z" Z "/"] ["c" C "/"] token
    token     := "label" LABEL                                (cell)
               | "grid-y" Y0 "-x" X0 "-h" H "-w" W            (grid)
               | "random-y" Y0 "-x" X0 "-h" H "-w" W ["-n" K] (random)
               | "whole_fov"                                  (whole_fov)

- ``position_id`` is the patch's ``position_id`` (decision D45; see
  :func:`resolve_position_id`); it may itself contain ``/``.
- ``object_set`` is present in ``cell`` mode only (:func:`resolve_object_set`).
- ``T`` (4 digits), ``Z`` (3 digits) and ``C`` are the patch's ``t``, ``z0``
  and ``c`` when set: a plane split or selected by ``t_policy``/``z_policy``,
  plane indices supplied with ``image_metadata``, and the channel of a split
  channel (or a channel given in ``image_metadata``). Absent levels are
  omitted.
- ``LABEL`` is the object's own label value; ``Y0``/``X0``/``H``/``W`` the
  rectangle's top-left corner and size in pixels.
- ``K`` (from 2) numbers repeated draws of the same rectangle in one random
  plane, in sampling order; the first draw has no ``-n``.

Exported files are named ``sanitize_filename(patch_id)`` plus the extension,
which turns every ``/`` into ``_``; the export refuses ids that become the same
file name.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np

from cellpatcher.io import image_stem

# The object set of a cell-mode patch whose mask has no name of its own: the
# mask role, not a guessed object name.
DEFAULT_OBJECT_SET = "mask"
_OBJECT_SET_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
_SEPARATORS = "_-. "


def validate_object_set(value, *, name: str = "common.object_set") -> None:
    """Raise ``ValueError`` unless *value* is None or a non-empty ``[A-Za-z0-9._-]`` name.

    The name becomes one level of ``patch_id``, so it may not contain ``/``.
    """
    if value is None:
        return
    if not isinstance(value, str) or not _OBJECT_SET_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must be None or a non-empty name of letters, digits, '.', '_' or '-' "
            f"(it becomes one level of patch_id); got {value!r}."
        )


def object_set_from_mask_file(image_file, mask_file) -> str | None:
    """The object-set name a mask file's name gives, or None when it gives none.

    The part of the mask's stem after the image's stem, without leading
    separators (``img_cells.tif`` for ``img.tif`` -> ``"cells"``), with any
    other character than letters, digits, ``.``, ``_`` and ``-`` replaced by
    ``_``. None when the mask is named like its image or does not start with
    the image's stem (case-insensitively), so no name is guessed.
    """
    image, mask = image_stem(image_file), image_stem(mask_file)
    if not mask.lower().startswith(image.lower()):
        return None
    rest = re.sub(r"[^A-Za-z0-9._-]", "_", mask[len(image):].lstrip(_SEPARATORS))
    rest = rest.strip("_")
    return rest or None


def object_set_from_label_path(mask_file) -> str:
    """The label image name of an OME-NGFF mask ``<image>/labels/<name>`` (``io.read_ngff_mask``)."""
    name = str(mask_file).rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_") or DEFAULT_OBJECT_SET


def resolve_object_set(meta, common) -> str:
    """The ``object_set`` of a cell-mode image.

    Precedence: ``image_metadata["object_set"]`` (folder and tree runs put
    the mask's own name there, see ``orchestration``), then
    ``common.object_set``, then :data:`DEFAULT_OBJECT_SET` (``"mask"``).
    """
    if meta.get("object_set") is not None:
        value = str(meta["object_set"])
        validate_object_set(value, name='image_metadata["object_set"]')
        return value
    if common.object_set is not None:
        # Checked again here: an attribute set after construction is not validated.
        validate_object_set(common.object_set)
        return common.object_set
    return DEFAULT_OBJECT_SET


def array_fingerprint(image) -> str:
    """``array-<16 hex>``: a SHA-256 of an in-memory image's dtype, shape and pixels.

    The ``position_id`` fallback of an image given with neither a
    ``position_id`` nor an ``image_file``: the same pixels give the same
    value in every run, whatever their place in the run.
    """
    array = np.ascontiguousarray(np.asarray(image))
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.shape}|".encode())
    digest.update(array.tobytes())
    return f"array-{digest.hexdigest()[:16]}"


def resolve_position_id(labels, meta, image=None) -> str:
    """The ``position_id`` every patch of an image carries (decision D45).

    *labels* is the image's ``(condition, biological_repeat,
    technical_repeat)``. A ``position_id`` given in *meta*
    (``image_metadata``) wins. Otherwise one file is taken as one XY
    position: ``<condition>/<biological_repeat>/<technical_repeat>/<file
    stem>`` from ``meta["image_file"]`` (``io.image_stem``, so
    ``img.ome.zarr`` gives ``img``), or, for an in-memory image without a file
    name, the same with :func:`array_fingerprint` in place of the stem.

    Raises
    ------
    ValueError
        If neither a ``position_id``, an ``image_file`` nor the image is given.
    """
    if meta.get("position_id") is not None:
        return str(meta["position_id"])
    if meta.get("image_file"):
        source = image_stem(meta["image_file"])
    elif image is not None:
        source = array_fingerprint(image)
    else:
        raise ValueError("An image has neither a position_id, an image_file nor pixels to identify it.")
    return "/".join(str(part) for part in (*labels, source))


def patch_token(patch) -> str:
    """The last level of a patch's ``patch_id``: its label (cell) or rectangle (grid, random), or ``whole_fov``."""
    if patch.mode == "cell":
        return f"label{patch.label_id}"
    if patch.mode == "whole_fov":
        return "whole_fov"
    if patch.mode in ("grid", "random"):
        return f"{patch.mode}-y{patch.y0}-x{patch.x0}-h{patch.y1 - patch.y0}-w{patch.x1 - patch.x0}"
    raise ValueError(f"Unknown patch mode {patch.mode!r} of a patch; cannot build its patch_id.")


def build_patch_id(patch) -> str:
    """The ``patch_id`` of *patch* from its identity fields, without the random draw number."""
    if not patch.position_id:
        raise ValueError("A patch needs a position_id to build its patch_id.")
    levels = [patch.position_id]
    if patch.mode == "cell":
        levels.append(patch.object_set or DEFAULT_OBJECT_SET)
    if patch.t is not None:
        levels.append(f"t{patch.t:04d}")
    if patch.z0 is not None:
        levels.append(f"z{patch.z0:03d}")
    if patch.c is not None:
        levels.append(f"c{patch.c}")
    levels.append(patch_token(patch))
    return "/".join(levels)


def assign_patch_ids(patches) -> None:
    """Set every patch's ``patch_id`` (:func:`build_patch_id`), numbering repeated random draws.

    Repeated draws of one rectangle are numbered only within one image of
    the run (one ``image_id``), in sampling order.

    Raises
    ------
    ValueError
        If two patches get the same id otherwise: several images share a
        ``position_id`` without different plane indices.
    """
    counts: Counter = Counter()
    owners: dict[str, set] = {}
    for patch in patches:
        base = build_patch_id(patch)
        counts[base] += 1
        owners.setdefault(base, set()).add(patch.image_id)
        patch.patch_id = base if counts[base] == 1 or patch.mode != "random" else f"{base}-n{counts[base]}"
    duplicates = sorted(
        base
        for base, count in counts.items()
        if count > 1 and (len(owners[base]) > 1 or not base.rsplit("/", 1)[-1].startswith("random-"))
    )
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} patch_id(s) occur more than once, e.g. {duplicates[0]!r}: several images share a "
            "position_id without different t/z/c plane indices. Give each image its own position_id "
            '(image_metadata[i]["position_id"]) or its plane indices ("t", "z", "c").'
        )
