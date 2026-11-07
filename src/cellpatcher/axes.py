import numpy as np

from cellpatcher.errors import ZStackNotSupportedError

_ALLOWED_AXES = {"T", "C", "Z", "Y", "X"}


def normalize_axes(axes: str, ndim: int | None = None) -> str:
    if not isinstance(axes, str):
        raise ValueError("axes must be a string.")

    # Canonicalize case/whitespace so "yx", "Y X", and "YX" all compare equal.
    normalized = axes.replace(" ", "").upper()
    if any(axis not in _ALLOWED_AXES for axis in normalized):
        raise ValueError("axes may only contain T, C, Z, Y, X.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("axes must not contain duplicate entries.")
    # When the caller knows the array's ndim, cross-check the axes string
    # actually has one letter per dimension.
    if ndim is not None and len(normalized) != ndim:
        raise ValueError(f"axes length ({len(normalized)}) must match ndim ({ndim}).")
    return normalized


def axis_index(axes: str, axis: str) -> int | None:
    normalized = normalize_axes(axes)
    if not isinstance(axis, str) or len(axis) != 1:
        raise ValueError("axis must be a single character string.")
    axis = axis.upper()
    if axis not in _ALLOWED_AXES:
        raise ValueError("axis must be one of T, C, Z, Y, X.")
    # None (not -1) signals "axis absent" so callers can distinguish that
    # from a real position without an extra membership check.
    return normalized.find(axis) if axis in normalized else None


def axes_dict(axes: str) -> dict[str, int | None]:
    normalized = normalize_axes(axes)
    return {axis: (normalized.find(axis) if axis in normalized else None) for axis in ["T", "C", "Z", "Y", "X"]}


def require_spatial_axes(axes: str) -> tuple[int, int]:
    normalized = normalize_axes(axes)
    y_idx = axis_index(normalized, "Y")
    x_idx = axis_index(normalized, "X")
    if y_idx is None or x_idx is None:
        raise ValueError("axes must include both Y and X spatial axes.")
    return y_idx, x_idx


def move_axes(array, source_axes: str, target_axes: str):
    src = normalize_axes(source_axes, ndim=array.ndim)
    tgt = normalize_axes(target_axes, ndim=array.ndim)

    if set(src) != set(tgt):
        raise ValueError("source_axes and target_axes must contain the same axes.")

    # For each target position, find where that axis currently lives in
    # `src`; np.transpose then permutes dimensions into that order.
    order = [src.index(axis) for axis in tgt]
    return np.transpose(array, axes=order)



def mapping_to_axes(mapping, ndim: int | None = None) -> str:
    """Convert notebook/UI axis mapping dicts into compact axis strings.

    This is a convenience helper (currently not used by the core pipeline) for
    turning mappings like ``{0: "C", 1: "Y", 2: "X"}`` into ``"CYX"``.
    Values ``"Other"`` and ``None`` are ignored, so
    ``{0: "Other", 1: "Y", 2: "X"}`` resolves to ``"YX"``.
    """
    if not hasattr(mapping, "items"):
        raise ValueError("mapping must be a dict-like object of dimension index to axis label.")

    # First pass: coerce keys to ints and values to upper-case strings (or
    # None), while tracking the highest index seen so ndim can be inferred
    # if the caller didn't pass one.
    normalized_mapping: dict[int, str | None] = {}
    max_idx = -1
    for key, value in mapping.items():
        try:
            idx = int(key)
        except Exception as exc:
            raise ValueError(f"Invalid mapping index: {key}") from exc
        if idx < 0:
            raise ValueError(f"Invalid mapping index: {idx}")

        max_idx = max(max_idx, idx)
        if value is None:
            normalized_mapping[idx] = None
        else:
            normalized_mapping[idx] = str(value).strip().upper()

    if ndim is None:
        ndim = max_idx + 1

    for idx in normalized_mapping:
        if idx >= ndim:
            raise ValueError(f"Mapping index {idx} is out of range for ndim={ndim}.")

    # Second pass: walk dimensions in order and keep only the ones with a
    # real axis label, dropping "Other"/unset entries entirely rather than
    # keeping a placeholder for them.
    labels = []
    for idx in range(ndim):
        label = normalized_mapping.get(idx)
        if label is None or label in {"OTHER", ""}:
            continue
        if label not in _ALLOWED_AXES:
            raise ValueError(f"Unsupported axis label in mapping: {label}")
        labels.append(label)

    if not labels:
        raise ValueError("Mapping resolved to empty axes.")

    axes = "".join(labels)
    return normalize_axes(axes)


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


IMAGE_TRANSPOSE_HELP = (
    "To fix the images, give each one with Y before X: swap its two spatial axes, for example "
    "numpy.swapaxes(image, -1, -2) when they are the last two, then set common.image_axes to the "
    "new order ('YX', 'CYX', ...). Nothing is ever read transposed."
)


def refuse_transposed_image_axes(image_axes: str) -> None:
    """Refuse ``common.image_axes`` that put X before Y for in-memory images.

    In-memory arrays are taken only in the canonical layout, Y before X (design v0.4 §6: a
    canonical-only layout argument, no swaps), as in-memory masks already are
    (:func:`refuse_transposed_mask_axes`). With X first (``'XY'``, ``'CXY'``) every patch's
    ``y0``/``y1`` would index the array's second spatial dimension and ``x0``/``x1`` its
    first, so ``patches.csv`` would give the coordinates of the transposed image, not of the
    array given. Folder and tree runs declare no axes: they read each file in the roles its
    preprocessing run recorded.

    Parameters
    ----------
    image_axes
        The declared axes of the in-memory images.

    Raises
    ------
    ValueError
        When X comes before Y, with :data:`IMAGE_TRANSPOSE_HELP`.
    """
    axes = normalize_axes(image_axes)
    if "X" in axes and "Y" in axes and axes.index("X") < axes.index("Y"):
        raise ValueError(
            f"common.image_axes={image_axes!r} puts X before Y, but CellPatcher takes in-memory images only "
            "with Y before X (design v0.4 §6): the patch coordinates y0/y1 and x0/x1 would refer to the "
            f"array's swapped dimensions. {IMAGE_TRANSPOSE_HELP}"
        )


def refuse_transposed_inputs(image_axes: str, mask_axes: str | None, images: object, masks: object) -> None:
    """Refuse the declared axes of in-memory images and masks that put X before Y.

    Parameters
    ----------
    image_axes, mask_axes
        ``common.image_axes`` and ``common.mask_axes``.
    images, masks
        The in-memory images and masks; the axes of each are checked when it is given.

    Raises
    ------
    ValueError
        From :func:`refuse_transposed_image_axes` or :func:`refuse_transposed_mask_axes`.
    """
    if images is not None:
        refuse_transposed_image_axes(image_axes)
    if masks is not None:
        refuse_transposed_mask_axes(mask_axes)


def refuse_z_stack(shape: tuple[int, ...], axes: str, what: str = "The image") -> None:
    """Refuse an in-memory array of *shape* with more than one Z plane, whatever ``z_policy`` says (issue #83).

    The guard of the in-memory route, the counterpart of the file route's check in
    ``cellpatcher.run_reading.fov_layout``: MorphoBricks v1 analyses 2-D planes, so a depth stack is
    neither patched plane by plane, kept whole nor reduced here. An array with no Z axis, or one Z
    plane, passes.

    Raises
    ------
    ZStackNotSupportedError
        If the Z axis of *axes* has more than one plane in *shape*.
    """
    position = axis_index(normalize_axes(axes, len(shape)), "Z")
    if position is not None and int(shape[position]) > 1:
        length = int(shape[position])
        raise ZStackNotSupportedError(
            f"{what} has {length} Z planes, but CellPatcher works on 2-D planes only, whatever z_policy says. "
            "A Z projection must be done as a recorded projection step in morphobricks-preprocessing "
            "before patching.",
            axis="Z",
            length=length,
        )


def refuse_unsupported_inputs(image_axes: str, mask_axes: str | None, images: object, masks: object) -> None:
    """Refuse in-memory images and masks with X before Y, or with more than one Z plane (issue #83).

    Parameters
    ----------
    image_axes, mask_axes
        ``common.image_axes`` and ``common.mask_axes``.
    images, masks
        The in-memory images and masks (a single array or a list/tuple); each one given is checked.

    Raises
    ------
    ValueError
        From :func:`refuse_transposed_inputs`.
    ZStackNotSupportedError
        If an image or mask has more than one Z plane, whatever ``z_policy`` says.
    """
    refuse_transposed_inputs(image_axes, mask_axes, images, masks)
    for array, array_axes, what in ((images, image_axes, "The image"), (masks, mask_axes, "The mask")):
        if array is not None:
            for one in array if isinstance(array, (list, tuple)) else [array]:
                if one is not None:  # a missing mask is refused where it is needed
                    refuse_z_stack(one.shape, array_axes, what)
