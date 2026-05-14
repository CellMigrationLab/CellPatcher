import numpy as np

from cellpatcher.axes import normalize_axes, require_spatial_axes
from cellpatcher.types import PatchRegion


def extract_patch(image, region: PatchRegion, *, axes: str = "YX", pad_mode: str = "clip", constant_values=0):
    array = np.asarray(image)
    normalized_axes = normalize_axes(axes, array.ndim)
    y_idx, x_idx = require_spatial_axes(normalized_axes)

    y0, y1 = int(region.y0), int(region.y1)
    x0, x1 = int(region.x0), int(region.x1)

    if pad_mode not in {"clip", "constant"}:
        raise ValueError("pad_mode must be either 'clip' or 'constant'.")

    h = array.shape[y_idx]
    w = array.shape[x_idx]

    src_y0 = max(0, y0)
    src_y1 = min(h, y1)
    src_x0 = max(0, x0)
    src_x1 = min(w, x1)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        if pad_mode == "clip":
            raise ValueError("Requested patch is completely out of image bounds.")
        # constant mode: return a fully-padded array of the requested shape
        out_shape = list(array.shape)
        out_shape[y_idx] = y1 - y0
        out_shape[x_idx] = x1 - x0
        return np.full(out_shape, constant_values, dtype=array.dtype)

    slices = [slice(None)] * array.ndim
    slices[y_idx] = slice(src_y0, src_y1)
    slices[x_idx] = slice(src_x0, src_x1)
    cropped = array[tuple(slices)]

    if pad_mode == "clip":
        return cropped

    out_h = y1 - y0
    out_w = x1 - x0
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Requested region must have positive height and width.")

    pad_top = max(0, src_y0 - y0)
    pad_bottom = max(0, y1 - src_y1)
    pad_left = max(0, src_x0 - x0)
    pad_right = max(0, x1 - src_x1)

    pad_width = [(0, 0)] * array.ndim
    pad_width[y_idx] = (pad_top, pad_bottom)
    pad_width[x_idx] = (pad_left, pad_right)

    return np.pad(cropped, pad_width=pad_width, mode="constant", constant_values=constant_values)


def extract_patches(image, regions, *, axes: str = "YX", pad_mode: str = "clip", constant_values=0) -> list[np.ndarray]:
    return [
        extract_patch(image, region, axes=axes, pad_mode=pad_mode, constant_values=constant_values)
        for region in regions
    ]
