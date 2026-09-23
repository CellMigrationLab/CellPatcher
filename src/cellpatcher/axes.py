import numpy as np

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
