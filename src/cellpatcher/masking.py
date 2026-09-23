import numpy as np

from cellpatcher.axes import normalize_axes, require_spatial_axes


def _label_to_2d(label_patch, label_axes: str) -> np.ndarray:
    label_arr = np.asarray(label_patch)
    normalized = normalize_axes(label_axes, label_arr.ndim)
    y_idx, x_idx = require_spatial_axes(normalized)

    if label_arr.shape[y_idx] <= 0 or label_arr.shape[x_idx] <= 0:
        raise ValueError("label_patch Y/X dimensions must be positive.")

    # Bring Y/X to the front regardless of their original axis order, so
    # the rest of this function can assume a simple (Y, X, ...) layout.
    moved = np.moveaxis(label_arr, (y_idx, x_idx), (0, 1))
    if moved.ndim == 2:
        return moved

    # Any remaining (non-spatial) dimensions must be singleton -- e.g. a
    # label array with a size-1 channel axis -- so they can be dropped
    # without losing information.
    extra_shape = moved.shape[2:]
    if any(dim != 1 for dim in extra_shape):
        raise ValueError(
            "label_patch must be 2D after extraction or have only singleton non-spatial dimensions. "
            f"Got extra shape {extra_shape}."
        )
    return moved.reshape(moved.shape[0], moved.shape[1])


def apply_label_mask_to_patch(
    image_patch,
    label_patch,
    *,
    label_id,
    image_axes="YX",
    label_axes="YX",
    fill_value=0,
):
    if label_id is None:
        raise ValueError("label_id is required for mask application and must not be None.")

    img = np.asarray(image_patch)
    if img.ndim < 2:
        raise ValueError(f"image_patch must be at least 2D, got ndim={img.ndim}.")

    normalized_image_axes = normalize_axes(image_axes, img.ndim)
    img_y_idx, img_x_idx = require_spatial_axes(normalized_image_axes)
    label_2d = _label_to_2d(label_patch, label_axes)

    if img.shape[img_y_idx] != label_2d.shape[0] or img.shape[img_x_idx] != label_2d.shape[1]:
        raise ValueError(
            "Spatial shape mismatch between image_patch and label_patch: "
            f"image YX=({img.shape[img_y_idx]}, {img.shape[img_x_idx]}) vs label YX={label_2d.shape}."
        )

    # Build a boolean mask of "this pixel belongs to label_id", then reshape
    # it to a shape with 1s everywhere except the image's own Y/X axes so it
    # broadcasts correctly against images with extra axes (e.g. channels).
    mask_2d = label_2d == label_id
    mask_shape = [1] * img.ndim
    mask_shape[img_y_idx] = mask_2d.shape[0]
    mask_shape[img_x_idx] = mask_2d.shape[1]
    broadcast_mask = mask_2d.reshape(mask_shape)

    return np.where(broadcast_mask, img, fill_value).astype(img.dtype, copy=False)


def apply_label_masks_to_patches(
    image_patches,
    label_patches,
    regions,
    *,
    image_axes="YX",
    label_axes="YX",
    fill_value=0,
):
    if not (len(image_patches) == len(label_patches) == len(regions)):
        raise ValueError("image_patches, label_patches, and regions must have the same length.")

    masked = []
    for image_patch, label_patch, region in zip(image_patches, label_patches, regions):
        if region.label_id is None:
            raise ValueError("Mask application requires regions with label_id; found region with label_id=None.")
        masked.append(
            apply_label_mask_to_patch(
                image_patch,
                label_patch,
                label_id=region.label_id,
                image_axes=image_axes,
                label_axes=label_axes,
                fill_value=fill_value,
            )
        )
    return masked
