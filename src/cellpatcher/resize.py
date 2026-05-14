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


def _resize_nearest_fallback(array: np.ndarray, y_idx: int, x_idx: int, out_h: int, out_w: int):
    in_h = array.shape[y_idx]
    in_w = array.shape[x_idx]
    y_coords = np.clip(np.round(np.linspace(0, in_h - 1, out_h)).astype(int), 0, in_h - 1)
    x_coords = np.clip(np.round(np.linspace(0, in_w - 1, out_w)).astype(int), 0, in_w - 1)
    tmp = np.take(array, y_coords, axis=y_idx)
    out = np.take(tmp, x_coords, axis=x_idx)
    return out


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

    zoom_factors = [1.0] * array.ndim
    zoom_factors[y_idx] = out_h / in_h
    zoom_factors[x_idx] = out_w / in_w

    try:
        from scipy.ndimage import zoom as scipy_zoom

        return scipy_zoom(array, zoom=zoom_factors, order=order)
    except ImportError:
        if order != 0:
            raise ImportError("scipy is required for interpolation orders > 0 during resize.")
        return _resize_nearest_fallback(array, y_idx, x_idx, out_h, out_w)


def resize_patches(patches, *, axes: str = "YX", output_size=None, order: int = 1) -> list[np.ndarray]:
    return [
        resize_patch(p, axes=axes, output_size=output_size, order=order)
        for p in patches
    ]
