import numpy as np

from cellpatcher.axes import normalize_axes, require_spatial_axes


def normalize_output_size(output_size) -> tuple[int, int] | None:
    if output_size is None:
        return None
    if isinstance(output_size, int):
        if output_size <= 0:
            raise ValueError("output_size int must be > 0.")
        return (output_size, output_size)
    if isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        h, w = int(output_size[0]), int(output_size[1])
        if h <= 0 or w <= 0:
            raise ValueError("output_size tuple/list values must be > 0.")
        return (h, w)
    raise ValueError("output_size must be None, a positive int, or a tuple/list of two positive ints.")


def resize_patch(patch, *, axes: str = "YX", output_size=None, order: int = 1):
    array = np.asarray(patch)
    normalized_axes = normalize_axes(axes, array.ndim)
    y_idx, x_idx = require_spatial_axes(normalized_axes)

    out_size = normalize_output_size(output_size)
    if out_size is None:
        return array

    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be an int >= 0.")

    out_h, out_w = out_size
    in_h = array.shape[y_idx]
    in_w = array.shape[x_idx]

    # Only the spatial axes get a non-1.0 zoom factor; every other axis
    # (channel, Z, T, ...) is left untouched.
    zoom_factors = [1.0] * array.ndim
    zoom_factors[y_idx] = out_h / in_h
    zoom_factors[x_idx] = out_w / in_w

    from scipy.ndimage import zoom as scipy_zoom

    return scipy_zoom(array, zoom=zoom_factors, order=order)


def resize_patches(patches, *, axes: str = "YX", output_size=None, order: int = 1) -> list[np.ndarray]:
    return [
        resize_patch(p, axes=axes, output_size=output_size, order=order)
        for p in patches
    ]
