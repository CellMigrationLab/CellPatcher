"""Run-independent patch identity: ``position_id``, ``object_set`` and ``patch_id``.

A patch's ``patch_id`` is built only from the identity of its field of view
and its own coordinates, never from the position of its image in a run, so
the same image gives the same ids whether it is patched alone, in a folder of
ten images, or in a tree with other wells (owner decision D44, 2026-10-26)::

    patch_id  := condition "/" biological_repeat "/" technical_repeat "/" position_id "/"
                 [object_set "/"] ["t" T "/"] ["z" Z "/"] ["c" C "/"] token
    token     := "label" LABEL                                (cell)
               | "grid-y" Y0 "-x" X0 "-h" H "-w" W            (grid)
               | "random-y" Y0 "-x" X0 "-h" H "-w" W ["-n" K] (random)
               | "whole_fov"                                  (whole_fov)
               | "whole_fov/sq" N                             (whole_fov_square)

- ``condition``, ``biological_repeat``, ``technical_repeat`` and
  ``position_id`` are the patch's FOV key (MorphoBricks decision D85,
  2026-10-29): ``position_id`` is the image name without its extension
  (:func:`resolve_position_id`), unique within its technical repeat, so the
  four levels make ``patch_id`` unique across a whole run or tree. Ids are
  never parsed: every level is also a field of the patch.
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
- ``N`` (from 0) is a whole-FOV square's place along the long side
  (``whole_fov.square_cover``); its ``patch_id`` is its parent's plus
  ``/sq<N>``, and ``parent_id`` is the parent's ``patch_id``, built from the
  parent's fields (:func:`parent_patch_id`), not cut out of the square's id.

Exported files are named ``sanitize_filename(patch_id)`` plus the extension,
which turns every ``/`` into ``_``; the export refuses ids that become the same
file name.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

import numpy as np

from cellpatcher.io import image_stem

# The object set of a cell-mode patch whose mask has no name of its own: the
# mask role, not a guessed object name.
DEFAULT_OBJECT_SET = "mask"
_OBJECT_SET_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
_SEPARATORS = "_-. "


def identity_text(value, name: str) -> str:
    """Return *value* unchanged if it is non-empty text, else raise ``ValueError``.

    FOV labels, ``position_id`` and ``object_set`` are identity: they are never converted (a
    number ``7`` is not the label ``"7"``, D85), so a value of another type is refused.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text (identity is never converted, D85); got {value!r}.")
    return value


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


def object_set_from_mask_type(mask_type) -> str | None:
    """The object-set name of a mask type from the preprocessing pairing, or None for an empty one.

    The type is the mask's folder or the part of its name after
    ``<image name>_`` (``img_cells.tif`` -> ``"cells"``), with any other
    character than letters, digits, ``.``, ``_`` and ``-`` replaced by ``_``,
    exactly as :func:`object_set_from_mask_file` treats the end of a name.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(mask_type)).strip("_") or None


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
        value = meta["object_set"]
        validate_object_set(value, name='image_metadata["object_set"]')  # text only, unchanged
        return value
    if common.object_set is not None:
        # Checked again here: an attribute set after construction is not validated.
        validate_object_set(common.object_set)
        return common.object_set
    return DEFAULT_OBJECT_SET


def resolve_position_id(labels, meta, *, image_index=None) -> str:
    """The ``position_id`` every patch of an image carries (MorphoBricks D85, amending D45).

    A ``position_id`` given in *meta* (``image_metadata``) is kept verbatim.
    Otherwise one file is taken as one XY position, named by the image name
    without its extension, as **every** brick names it: computed by
    ``morphobricks_preprocessing.discovery.position_id`` (``.ome.tiff``,
    ``.ome.tif``, ``.ome.zarr``, ``.tiff``, ``.tif`` and ``.zarr``, any
    case: ``ctrl/rep1/img_01.ome.tif`` gives ``img_01``). The image's
    *labels* (``condition``, ``biological_repeat``, ``technical_repeat``)
    are not packed into it: they are the other three fields of the FOV key,
    and lead the ``patch_id`` (:func:`build_patch_id`).

    Raises
    ------
    ValueError
        If a given ``position_id`` is not non-empty text (it is never coerced), or
        if neither a ``position_id`` nor an ``image_file`` is given: an
        in-memory image has no identity of its own, and its place in the run
        (*image_index*, named in the message) is not one.
    """
    if meta.get("position_id") is not None:
        value = meta["position_id"]
        if not isinstance(value, str) or not value.strip():
            which = "an image" if image_index is None else f"image {image_index} of the run"
            raise ValueError(
                f"The position_id given for {which} must be non-empty text (kept verbatim, D85); "
                f"got {value!r}."
            )
        return value
    if meta.get("image_file"):
        from morphobricks_preprocessing.discovery import position_id

        return position_id(str(meta["image_file"]))
    which = "An image" if image_index is None else f"Image {image_index} of the run"
    raise ValueError(
        f"{which} has neither a position_id nor an image_file, so its patches have no identity (its place "
        'in the run is not one). Pass image_metadata=[{"position_id": ...}, ...] to run_patching, one entry '
        'per image (or {"image_file": ...}: its file stem then names the position).'
    )


def plane_index(meta, axis: str) -> int | None:
    """The plane index *meta* gives along *axis* (``"t"``, ``"z"`` or ``"c"``) as patches carry it.

    ``None`` when absent, else the integer: the one representation stamped on patches, hashed
    into random seeds and compared for uniqueness, so ``1`` and ``"1"`` are one plane. Only a
    non-negative integer (not a bool) or its canonical decimal text is accepted; ``True``,
    ``1.5``, ``1.0``, ``"1.0"``, ``"01"`` and negatives raise ``ValueError`` (identity is never
    coerced, D85).
    """
    value = meta.get(axis)
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        return int(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0:
        return int(value)
    raise ValueError(
        f"image_metadata gives {axis}={value!r}: a plane index is a non-negative integer (or its "
        "decimal text, e.g. 1 or \"1\"); it is never rounded or converted from another value."
    )


def check_unique_position(seen: dict, labels, meta, index) -> None:
    """Refuse a source image whose FOV key another image of the run already has (D85).

    ``position_id`` is unique within its technical repeat whatever its case, so the key is
    ``(condition, biological_repeat, technical_repeat, position_id.casefold())`` plus the
    plane indices (``t``, ``z``, ``c``) *meta* gives, if any: images of one position told
    apart by their planes are legal. Called once per source image, before any plane split,
    geometry or filter, so an image that yields no patch is checked too. *seen* maps the
    keys met so far to their image index and is updated.
    """
    key = (*map(str, labels), str(meta["position_id"]).casefold(), *(plane_index(meta, axis) for axis in "tzc"))
    if key in seen:
        raise ValueError(
            f"Images {seen[key]} and {index} of the run have the same position_id within technical repeat "
            f"{key[:3]!r} ({meta['position_id']!r}; position_id is unique within its technical repeat "
            "whatever its case, D85) and no different t/z/c plane indices. Give each image its own "
            'position_id (image_metadata[i]["position_id"]) or its plane indices ("t", "z", "c").'
        )
    seen[key] = index


# The plane indices each random seed scope includes (config.RandomConfig.seed_scope).
SEED_SCOPE_AXES = {"plane_channel": ("t", "z", "c"), "plane": ("t", "z"), "position": ()}


def random_seed(seed, meta, scope: str = "plane_channel", labels=None) -> int | None:
    """The seed random mode samples one image (plane) with, from the run seed and the image's identity.

    Derived from ``seed`` and the canonical identity, the compact JSON array
    ``["condition","biological_repeat","technical_repeat","position_id","t=<t>","z=<z>","c=<c>"]``
    (*labels* and *meta*, the image's metadata; ``-`` for an absent index; JSON, so a
    label holding any character, ``|`` or ``,`` included, never reads as two;
    the whole FOV key, so two repeats' images of one name differ, D85) through SHA-256 and
    ``numpy.random.SeedSequence``, so the same image (plane) gets the same
    rectangles in every run, whatever its place in the run, and another
    ``seed`` gives others. None when ``seed`` is None (unseeded sampling).
    *scope* (``RandomConfig.seed_scope``) keeps only some indices:
    ``"plane_channel"`` all three (the default, unchanged), ``"plane"`` t and
    z, ``"position"`` none (the FOV key alone).
    """
    if seed is None:
        return None
    if scope not in SEED_SCOPE_AXES:  # e.g. set after the config was validated
        raise ValueError(f"random.seed_scope must be one of {list(SEED_SCOPE_AXES)}; got {scope!r}.")
    if labels is None:
        raise ValueError("random_seed needs the image's (condition, biological_repeat, technical_repeat).")
    parts = [*(str(label) for label in labels), str(meta["position_id"])]
    indices = {axis: plane_index(meta, axis) for axis in SEED_SCOPE_AXES[scope]}
    parts += [f"{axis}={'-' if value is None else value}" for axis, value in indices.items()]
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, len(digest), 4)]
    return int(np.random.SeedSequence([int(seed), *words]).generate_state(1, dtype=np.uint64)[0])


def patch_token(patch) -> str:
    """The last level of a patch's ``patch_id``: its label (cell) or rectangle (grid, random), or ``whole_fov``."""
    if patch.mode == "cell":
        return f"label{patch.label_id}"
    if patch.mode == "whole_fov":
        return "whole_fov"
    if patch.mode == "whole_fov_square":
        if patch.square_index is None:
            raise ValueError("A whole_fov_square patch needs its square_index to build its patch_id.")
        return f"whole_fov/sq{int(patch.square_index)}"
    if patch.mode in ("grid", "random"):
        return f"{patch.mode}-y{patch.y0}-x{patch.x0}-h{patch.y1 - patch.y0}-w{patch.x1 - patch.x0}"
    raise ValueError(f"Unknown patch mode {patch.mode!r} of a patch; cannot build its patch_id.")


def _fov_key(patch) -> list[str]:
    """The patch's FOV key: condition, biological and technical repeat, position_id (D85)."""
    key = [patch.condition, patch.biological_repeat, patch.technical_repeat, patch.position_id]
    if not all(isinstance(part, str) and part for part in key):
        raise ValueError(
            "A patch needs its condition, biological_repeat, technical_repeat and position_id "
            f"as text to build its patch_id (never converted, D85); got {key}."
        )
    return key


def build_patch_id(patch) -> str:
    """The ``patch_id`` of *patch* from its identity fields, without the random draw number."""
    levels = _fov_key(patch)
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


def parent_patch_id(patch) -> str:
    """The ``patch_id`` of a whole-FOV square's parent, from the square's own fields."""
    levels = _fov_key(patch)
    levels += [f"t{patch.t:04d}"] if patch.t is not None else []
    levels += [f"z{patch.z0:03d}"] if patch.z0 is not None else []
    levels += [f"c{patch.c}"] if patch.c is not None else []
    return "/".join([*levels, "whole_fov"])


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
    random_bases: set[str] = set()
    for patch in patches:
        base = build_patch_id(patch)
        counts[base] += 1
        # Exact duplicates, repeated random draws excepted; positions differing only in
        # case were refused per source image before any geometry (check_unique_position, D85).
        owners.setdefault(base, set()).add(patch.image_id)
        if patch.mode == "random":
            random_bases.add(base)
        patch.patch_id = base if counts[base] == 1 or patch.mode != "random" else f"{base}-n{counts[base]}"
        if patch.mode == "whole_fov_square":
            patch.parent_id = parent_patch_id(patch)
    duplicates = sorted(
        base
        for base, count in counts.items()
        if count > 1 and (len(owners[base]) > 1 or base not in random_bases)
    )
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} patch_id(s) occur more than once, e.g. {duplicates[0]!r}: several images share a "
            "position_id without different t/z/c plane indices. Give each image its own position_id "
            '(image_metadata[i]["position_id"]) or its plane indices ("t", "z", "c").'
        )
