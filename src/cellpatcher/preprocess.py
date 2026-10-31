from dataclasses import dataclass

import numpy as np

from cellpatcher.axes import axis_index, normalize_axes, require_spatial_axes


@dataclass(frozen=True)
class PreparedPlane:
    array: np.ndarray
    axes: str
    metadata: dict
    plane_id: str


@dataclass(frozen=True)
class PreparedPlanePair:
    image: np.ndarray
    image_axes: str
    labels: np.ndarray
    label_axes: str
    metadata: dict
    plane_id: str


def select_axis_index(array, axes, axis, index):
    arr = np.asarray(array)
    normalized_axes = normalize_axes(axes, arr.ndim)
    axis = str(axis).upper()
    idx = axis_index(normalized_axes, axis)
    if idx is None:
        raise ValueError(f"Axis '{axis}' is not present in axes '{normalized_axes}'.")

    size = arr.shape[idx]
    sel = int(index)
    if sel < 0:
        sel += size
    if sel < 0 or sel >= size:
        raise ValueError(f"Index {index} is out of range for axis '{axis}' of size {size}.")

    selected = np.take(arr, sel, axis=idx)
    selected_axes = normalized_axes[:idx] + normalized_axes[idx + 1 :]
    return selected, selected_axes


def project_axis(array, axes, axis, method):
    arr = np.asarray(array)
    normalized_axes = normalize_axes(axes, arr.ndim)
    axis = str(axis).upper()
    idx = axis_index(normalized_axes, axis)
    if idx is None:
        raise ValueError(f"Axis '{axis}' is not present in axes '{normalized_axes}'.")

    method = str(method).lower()
    if method == "first":
        out = np.take(arr, 0, axis=idx)
    elif method == "max":
        out = np.max(arr, axis=idx)
    elif method == "mean":
        out = np.mean(arr, axis=idx)
    elif method == "sum":
        out = np.sum(arr, axis=idx)
    else:
        raise ValueError("method must be one of: 'first', 'max', 'mean', 'sum'.")

    out_axes = normalized_axes[:idx] + normalized_axes[idx + 1 :]
    return out, out_axes


def repeat_to_rgb(image, axes):
    arr = np.asarray(image)
    normalized_axes = normalize_axes(axes, arr.ndim)
    require_spatial_axes(normalized_axes)

    c_idx = axis_index(normalized_axes, "C")
    if c_idx is None:
        # No channel axis at all: treat as single-channel grayscale and add
        # a new trailing C axis of size 3.
        if normalized_axes != "YX":
            raise ValueError("repeat_to_rgb without a C axis requires axes='YX'.")
        return np.repeat(arr[..., None], 3, axis=2), "YXC"

    c_size = arr.shape[c_idx]
    if c_size == 3:
        # Already RGB-shaped: nothing to do.
        return arr, normalized_axes
    if c_size == 1:
        # Single-channel with an explicit C axis: repeat it 3x in place.
        return np.repeat(arr, 3, axis=c_idx), normalized_axes
    raise ValueError(
        "repeat_to_rgb only supports grayscale/single-channel or already-3-channel data; "
        f"got C size {c_size}."
    )


def _apply_axis_policy(
    arr,
    axes,
    *,
    axis,
    policy,
    index=None,
    allowed_project_methods=None,
    label_context=False,
    param_name=None,
    index_param_name=None,
):
    # If the array doesn't have this axis at all, there's nothing to
    # collapse -- return it unchanged rather than erroring.
    idx = axis_index(axes, axis)
    if idx is None:
        return arr, axes

    # Error messages should name the actual caller-facing parameter (e.g.
    # "channel_policy" instead of the generic axis name) when the caller
    # supplied one.
    _policy_param = param_name if param_name is not None else f"{axis.lower()}_policy"
    _index_param = index_param_name if index_param_name is not None else f"{axis.lower()}_index"

    size = arr.shape[idx]
    if policy == "error":
        # 'error' is a safety net: a size-1 axis is unambiguous to collapse,
        # but anything larger means the caller must decide explicitly how
        # to reduce it (select/project), rather than this silently picking
        # a plane for them.
        if size > 1:
            raise ValueError(
                f"{axis} axis has size {size}; explicit selection/projection is required "
                f"(got {_policy_param}='error')."
            )
        return select_axis_index(arr, axes, axis, 0)

    if policy == "first":
        return select_axis_index(arr, axes, axis, 0)

    if policy == "select":
        if index is None:
            raise ValueError(f"{_policy_param}='select' requires {_index_param}.")
        return select_axis_index(arr, axes, axis, index)

    # Projection policies (max/mean/sum/...) collapse the whole axis via a
    # reduction instead of picking one index; which methods are valid
    # depends on the axis (e.g. labels only ever allow 'max').
    if allowed_project_methods and policy in allowed_project_methods:
        return project_axis(arr, axes, axis, policy)

    if label_context:
        raise ValueError(f"Unsupported {_policy_param} for labels: {policy}")
    raise ValueError(f"Unsupported {_policy_param}: {policy}")


def prepare_image_for_patching(
    image,
    axes,
    *,
    z_policy="error",
    z_index=None,
    t_policy="error",
    t_index=None,
    channel_policy="keep",
    channel_index=None,
):
    """Resolve the T, Z and C axes of an image to one 2D (optionally multichannel) plane.

    The former per-image normalization (``normalize``/``percentile``) was
    removed in v0.1.1: normalize patches with ``config.normalization``.
    """
    return _prepare_image(
        image, axes, z_policy=z_policy, z_index=z_index, t_policy=t_policy, t_index=t_index,
        channel_policy=channel_policy, channel_index=channel_index,
    )


def _prepare_image(
    image,
    axes,
    *,
    z_policy,
    z_index,
    t_policy,
    t_index,
    channel_policy,
    channel_index,
):
    arr = np.asarray(image)
    ax = normalize_axes(axes, arr.ndim)

    if t_policy == "all" or z_policy == "all":
        raise ValueError("'all' policies are not supported in this function; use iter_image_planes_for_patching for plane expansion.")

    # Collapse T then Z down to single planes (each call is a no-op if the
    # array has no such axis); patching itself only ever operates on 2D
    # Y/X (+ optional channel).
    arr, ax = _apply_axis_policy(arr, ax, axis="T", policy=t_policy, index=t_index)
    arr, ax = _apply_axis_policy(
        arr, ax, axis="Z", policy=z_policy, index=z_index, allowed_project_methods={"max", "mean", "sum"}
    )

    # Channel handling is its own policy set (not a generic axis policy)
    # because 'repeat_to_rgb' has behavior _apply_axis_policy doesn't cover.
    if channel_policy == "keep":
        pass
    elif channel_policy == "repeat_to_rgb":
        arr, ax = repeat_to_rgb(arr, ax)
    elif channel_policy in {"select", "first"}:
        idx = 0 if channel_policy == "first" else channel_index
        arr, ax = _apply_axis_policy(
            arr, ax, axis="C", policy="select", index=idx,
            param_name="channel_policy", index_param_name="channel_index",
        )
    elif channel_policy in {"max", "mean"}:
        arr, ax = _apply_axis_policy(
            arr, ax, axis="C", policy=channel_policy,
            allowed_project_methods={"max", "mean"},
            param_name="channel_policy",
        )
    else:
        raise ValueError(f"Unsupported channel_policy: {channel_policy}")

    # After T/Z/C handling, exactly one Y and one X axis must remain.
    require_spatial_axes(ax)
    return arr, ax


def prepare_labels_for_cell_patching(
    labels,
    axes,
    *,
    z_policy="error",
    z_index=None,
    t_policy="error",
    t_index=None,
    channel_policy="error",
    channel_index=None,
):
    """Prepare labels for cell patching to a single YX plane.

    Notes:
        z_policy='max' may be useful for non-overlapping labels across Z, but is
        not always biologically safe and should be used explicitly.
    """
    arr = np.asarray(labels)
    ax = normalize_axes(axes, arr.ndim)

    if t_policy == "all" or z_policy == "all":
        raise ValueError("'all' policies are not supported in this function; use iter_cell_planes_for_patching for plane expansion.")

    # Same T/Z collapsing as prepare_image_for_patching, but Z only allows
    # 'max' projection for labels (mean/sum would produce meaningless,
    # non-integer "label" values).
    arr, ax = _apply_axis_policy(arr, ax, axis="T", policy=t_policy, index=t_index, label_context=True)
    arr, ax = _apply_axis_policy(
        arr, ax, axis="Z", policy=z_policy, index=z_index, allowed_project_methods={"max"}, label_context=True
    )

    if channel_policy == "error":
        arr, ax = _apply_axis_policy(
            arr, ax, axis="C", policy="error", label_context=True,
            param_name="channel_policy",
        )
    elif channel_policy == "first":
        arr, ax = _apply_axis_policy(
            arr, ax, axis="C", policy="first", label_context=True,
            param_name="channel_policy",
        )
    elif channel_policy == "select":
        arr, ax = _apply_axis_policy(
            arr, ax, axis="C", policy="select", index=channel_index, label_context=True,
            param_name="channel_policy", index_param_name="channel_index",
        )
    else:
        raise ValueError(f"Unsupported channel_policy for labels: {channel_policy}")

    # Unlike prepare_image_for_patching, labels must end up as *exactly*
    # 'YX' (no leftover channel axis), since downstream cell-mode code
    # expects a plain 2D label map.
    if ax != "YX":
        raise ValueError(f"Prepared labels must resolve to 'YX', got '{ax}'.")

    unique_values = np.unique(arr)
    if np.any(unique_values < 0):
        raise ValueError("labels must be non-negative.")
    if np.issubdtype(arr.dtype, np.floating):
        if np.any((unique_values != 0.0) & (unique_values != 1.0)):
            raise ValueError("Non-binary float labels are ambiguous. Use integer labels or binary float masks.")

    return arr, "YX"


def _axis_size(array, axes, axis):
    idx = axis_index(axes, axis)
    return None if idx is None else int(np.asarray(array).shape[idx])


def _plane_id_for_metadata(metadata: dict) -> str:
    parts = []
    if "t" in metadata:
        parts.append(f"t{int(metadata['t']):03d}")
    if "z" in metadata:
        parts.append(f"z{int(metadata['z']):03d}")
    return "_".join(parts) if parts else "plane000"


def iter_image_planes_for_patching(
    image,
    axes,
    *,
    z_policy="error",
    z_index=None,
    t_policy="error",
    t_index=None,
    channel_policy="keep",
    channel_index=None,
):
    arr = np.asarray(image)
    ax = normalize_axes(axes, arr.ndim)

    valid_z = {"error", "first", "select", "max", "mean", "sum", "all"}
    valid_t = {"error", "first", "select", "all"}
    if z_policy not in valid_z:
        raise ValueError(f"Unsupported z_policy: {z_policy}")
    if t_policy not in valid_t:
        raise ValueError(f"Unsupported t_policy: {t_policy}")

    z_size = _axis_size(arr, ax, "Z")
    t_size = _axis_size(arr, ax, "T")

    # 'all' expands into one iteration per index along that axis; any other
    # policy is handled once per (T, Z) combination, i.e. a single pass with
    # a sentinel None value.
    t_values = range(t_size) if t_policy == "all" and t_size is not None else [None]
    z_values = range(z_size) if z_policy == "all" and z_size is not None else [None]

    for t_val in t_values:
        for z_val in z_values:
            metadata = {}
            # When a policy is 'all', re-express this iteration's index as
            # an explicit 'select' for prepare_image_for_patching (which
            # doesn't understand 'all' itself), and record it in metadata so
            # the caller can trace each yielded plane back to its T/Z index.
            eff_t_policy = t_policy
            eff_t_index = t_index
            if t_policy == "all":
                eff_t_policy = "select"
                eff_t_index = t_val
                metadata["t"] = int(t_val)

            eff_z_policy = z_policy
            eff_z_index = z_index
            if z_policy == "all":
                eff_z_policy = "select"
                eff_z_index = z_val
                metadata["z"] = int(z_val)

            out_arr, out_axes = _prepare_image(
                arr,
                ax,
                z_policy=eff_z_policy,
                z_index=eff_z_index,
                t_policy=eff_t_policy,
                t_index=eff_t_index,
                channel_policy=channel_policy,
                channel_index=channel_index,
            )
            yield PreparedPlane(array=out_arr, axes=out_axes, metadata=metadata, plane_id=_plane_id_for_metadata(metadata))


def iter_cell_planes_for_patching(
    image,
    image_axes,
    labels,
    label_axes,
    *,
    z_policy="error",
    z_index=None,
    t_policy="error",
    t_index=None,
    image_channel_policy="keep",
    image_channel_index=None,
    label_channel_policy="error",
    label_channel_index=None,
):
    img = np.asarray(image)
    lab = np.asarray(labels)
    i_ax = normalize_axes(image_axes, img.ndim)
    l_ax = normalize_axes(label_axes, lab.ndim)

    valid_z = {"error", "first", "select", "max", "all"}
    valid_t = {"error", "first", "select", "all"}
    if z_policy not in valid_z:
        raise ValueError(f"Unsupported z_policy for cell iterator: {z_policy}")
    if t_policy not in valid_t:
        raise ValueError(f"Unsupported t_policy for cell iterator: {t_policy}")

    img_z, lab_z = _axis_size(img, i_ax, "Z"), _axis_size(lab, l_ax, "Z")
    img_t, lab_t = _axis_size(img, i_ax, "T"), _axis_size(lab, l_ax, "T")

    # 'all' requires the image and its labels to walk the same T/Z indices
    # in lockstep, so both must actually carry that axis, at matching sizes.
    if z_policy == "all":
        if img_z is None or lab_z is None:
            raise ValueError("z_policy='all' requires both image and labels to contain a Z axis.")
        if img_z != lab_z:
            raise ValueError(f"Image/label Z sizes must match for z_policy='all' (got {img_z} vs {lab_z}).")
    if t_policy == "all":
        if img_t is None or lab_t is None:
            raise ValueError("t_policy='all' requires both image and labels to contain a T axis.")
        if img_t != lab_t:
            raise ValueError(f"Image/label T sizes must match for t_policy='all' (got {img_t} vs {lab_t}).")

    t_values = range(img_t) if t_policy == "all" else [None]
    z_values = range(img_z) if z_policy == "all" else [None]

    for t_val in t_values:
        for z_val in z_values:
            # Same 'all' -> explicit 'select' re-expression as
            # iter_image_planes_for_patching, applied to both the image and
            # its labels so they select the same T/Z plane.
            metadata = {}
            eff_t_policy = t_policy
            eff_t_index = t_index
            if t_policy == "all":
                eff_t_policy = "select"
                eff_t_index = t_val
                metadata["t"] = int(t_val)

            eff_z_policy_img = z_policy
            eff_z_policy_lab = z_policy
            eff_z_index = z_index
            if z_policy == "all":
                eff_z_policy_img = "select"
                eff_z_policy_lab = "select"
                eff_z_index = z_val
                metadata["z"] = int(z_val)

            prepared_image, prepared_image_axes = _prepare_image(
                img,
                i_ax,
                z_policy=eff_z_policy_img,
                z_index=eff_z_index,
                t_policy=eff_t_policy,
                t_index=eff_t_index,
                channel_policy=image_channel_policy,
                channel_index=image_channel_index,
            )
            prepared_labels, prepared_label_axes = prepare_labels_for_cell_patching(
                lab,
                l_ax,
                z_policy=eff_z_policy_lab,
                z_index=eff_z_index,
                t_policy=eff_t_policy,
                t_index=eff_t_index,
                channel_policy=label_channel_policy,
                channel_index=label_channel_index,
            )

            # Image and label preparation ran independently above; verify
            # they actually agree on the resulting spatial shape before
            # yielding the pair as a matched plane.
            p_i_ax = normalize_axes(prepared_image_axes, np.asarray(prepared_image).ndim)
            y_idx, x_idx = require_spatial_axes(p_i_ax)
            if np.asarray(prepared_image).shape[y_idx] != prepared_labels.shape[0] or np.asarray(prepared_image).shape[x_idx] != prepared_labels.shape[1]:
                raise ValueError(
                    "Spatial shape mismatch after plane preparation: "
                    f"image YX=({np.asarray(prepared_image).shape[y_idx]}, {np.asarray(prepared_image).shape[x_idx]}) "
                    f"vs labels YX={prepared_labels.shape}."
                )

            yield PreparedPlanePair(
                image=prepared_image,
                image_axes=prepared_image_axes,
                labels=prepared_labels,
                label_axes=prepared_label_axes,
                metadata=metadata,
                plane_id=_plane_id_for_metadata(metadata),
            )
