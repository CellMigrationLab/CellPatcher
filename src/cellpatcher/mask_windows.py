"""Patch mask fractions counted from a mask read by windows, never whole.

``run_patching(..., fraction_masks=[{name: mask}])`` takes a mask plane as an array, or as a
:class:`WindowedMask`: a **mask window reader**, a callable ``read(bounds) -> ndarray`` with
``bounds = (y0, y1, x0, x1)`` half-open ``pixel_edge`` bounds in the plane frame (Y before X), the
plane's ``shape``, the mask's pixel ``dtype`` and a ``memory_limit`` in bytes. CellPatcher reads no
file: the caller's reader does (for a preprocessing run, ``RunInputs.read_mask_window``).

:func:`windowed_foreground_counts` gives each patch rectangle's number of foreground (non-zero)
pixels, the integer the whole-array path takes from its summed-area table, so every
``mask_fractions`` value (``count / area``) is bitwise the same. The rectangles' bounding box is cut
into disjoint blocks (full-width row bands, or pieces of one row when a row is over the limit),
each holding at most ``memory_limit // (itemsize + 1)`` pixels: the window read plus its boolean
foreground. A block no rectangle meets is not read; of the others only the bounding box of the
rectangles' parts inside it is read. So every pixel is read at most once per mask, and only pixels
in or between patches. Each rectangle's count is the sum of its parts' counts in the blocks.

Pure geometry and counting: no file is opened here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_FOREGROUND_BYTES = 1
"""Bytes per pixel of the boolean foreground made from each window (``window != 0``)."""


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class WindowedMask:
    """A 2-D mask plane given by a window reader instead of an array.

    Parameters
    ----------
    read
        ``read((y0, y1, x0, x1)) -> ndarray``: the mask's pixels in that half-open ``pixel_edge``
        window of the plane, as a ``(y1 - y0, x1 - x0)`` array of ``dtype``; bounds are plain ints.
    shape
        ``(height, width)`` of the plane: the image's Y/X shape.
    dtype
        The pixel type every window has (for a preprocessing run, ``mask_window_plan``'s ``dtype``):
        boolean or integer, as every region (Spec ADR 0005 BM7). A window of another type is
        refused, never cast.
    memory_limit
        Most bytes one read may hold with its boolean foreground: ``h * w * (itemsize + 1)``.

    Raises
    ------
    ValueError
        If ``read`` is not callable, ``shape`` is not two positive integers, or ``memory_limit``
        is not an integer that holds one pixel.
    morphobricks_normalization.regions.RegionMaskError
        ``MASK_REGION_INVALID`` (reason ``dtype``, a ``ValueError``) when ``dtype`` is not boolean
        or integer (a float probability map is thresholded by the user first), before any read.
    """

    read: Callable[[tuple[int, int, int, int]], Any]
    shape: tuple[int, int]
    dtype: Any
    memory_limit: int

    def __post_init__(self) -> None:
        if not callable(self.read):
            raise ValueError(f"WindowedMask.read must be callable (bounds -> array); got {self.read!r}.")
        shape = self.shape
        if not isinstance(shape, (tuple, list)) or len(shape) != 2 or not all(
            _is_integer(n) and n > 0 for n in shape
        ):
            raise ValueError(f"WindowedMask.shape must be two positive integers (height, width); got {shape!r}.")
        object.__setattr__(self, "shape", (int(shape[0]), int(shape[1])))
        dtype = np.dtype(self.dtype)
        _check_window(np.zeros((1, 1), dtype=dtype), (1, 1))  # the region rule on the declared type, no read
        object.__setattr__(self, "dtype", dtype)
        if not _is_integer(self.memory_limit) or self.memory_limit < self.pixel_bytes:
            raise ValueError(
                f"WindowedMask.memory_limit must be an integer of at least {self.pixel_bytes} bytes (one "
                f"{dtype} pixel and its foreground); got {self.memory_limit!r}."
            )
        object.__setattr__(self, "memory_limit", int(self.memory_limit))

    @property
    def pixel_bytes(self) -> int:
        """Bytes one pixel of a read holds: the window's itemsize and its boolean foreground."""
        return np.dtype(self.dtype).itemsize + _FOREGROUND_BYTES


def mask_blocks(
    box: tuple[int, int, int, int], max_pixels: int
) -> list[tuple[int, int, int, int]]:
    """Return disjoint blocks ``(y0, y1, x0, x1)`` covering *box*, each at most *max_pixels* pixels.

    Full-width row bands when a row of *box* fits, else pieces of single rows; in row-major order.
    """
    y0, y1, x0, x1 = box
    width = x1 - x0
    if width <= max_pixels:
        band, step = max_pixels // width, width
    else:
        band, step = 1, max_pixels
    return [
        (top, min(top + band, y1), left, min(left + step, x1))
        for top in range(y0, y1, band)
        for left in range(x0, x1, step)
    ]


def _check_window(window: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return a window's foreground ``window != 0`` by normalization's region checker (``MASK_REGION_INVALID``)."""
    from morphobricks_normalization.regions import check_region

    return check_region(window, shape, name="the windowed mask")


def _read_window(mask: WindowedMask, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Read one window, refuse one that is not the declared shape and type, and return its foreground."""
    window = mask.read(bounds)
    expected = (bounds[1] - bounds[0], bounds[3] - bounds[2])
    if not isinstance(window, np.ndarray) or window.shape != expected or window.dtype != mask.dtype:
        found = (type(window).__name__, getattr(window, "shape", None), getattr(window, "dtype", None))
        raise ValueError(
            f"the mask window reader returned {found} for bounds {bounds}; expected an ndarray of "
            f"shape {expected} and dtype {mask.dtype}."
        )
    return _check_window(window, expected)


def windowed_foreground_counts(mask: WindowedMask, rectangles: Sequence[Sequence[int]]) -> list[int]:
    """Return each rectangle's number of non-zero mask pixels, reading the mask by windows.

    Parameters
    ----------
    mask
        The mask plane's window reader (:class:`WindowedMask`).
    rectangles
        ``(y0, y1, x0, x1)`` per patch: half-open, non-empty and inside ``mask.shape``.

    Returns
    -------
    list of int
        One count per rectangle, in their order: exactly the counts of the whole mask's slices.

    Raises
    ------
    ValueError
        If a rectangle is not inside the plane, or a window is not the declared shape and dtype
        (only the windows read are checked).
    """
    if not isinstance(mask, WindowedMask):
        raise ValueError(f"mask must be a WindowedMask; got {type(mask).__name__}.")
    boxes = np.array([[int(v) for v in rect] for rect in rectangles], dtype=np.int64).reshape(-1, 4)
    height, width = mask.shape
    bad = (boxes[:, 0] < 0) | (boxes[:, 1] > height) | (boxes[:, 2] < 0) | (boxes[:, 3] > width)
    bad |= (boxes[:, 1] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 2])
    if bad.any():
        raise ValueError(f"rectangle {tuple(boxes[np.argmax(bad)])} is not a non-empty part of the plane {mask.shape}.")
    counts = [0] * len(boxes)
    if not len(boxes):
        return counts
    box = (int(boxes[:, 0].min()), int(boxes[:, 1].max()), int(boxes[:, 2].min()), int(boxes[:, 3].max()))
    for block in mask_blocks(box, mask.memory_limit // mask.pixel_bytes):
        parts = np.stack(
            [
                np.maximum(boxes[:, 0], block[0]), np.minimum(boxes[:, 1], block[1]),
                np.maximum(boxes[:, 2], block[2]), np.minimum(boxes[:, 3], block[3]),
            ],
            axis=1,
        )
        meets = np.flatnonzero((parts[:, 1] > parts[:, 0]) & (parts[:, 3] > parts[:, 2]))
        if not len(meets):
            continue
        inside = parts[meets]
        window = (int(inside[:, 0].min()), int(inside[:, 1].max()), int(inside[:, 2].min()), int(inside[:, 3].max()))
        foreground = _read_window(mask, window)
        for index, (y0, y1, x0, x1) in zip(meets.tolist(), inside.tolist(), strict=True):
            counts[index] += int(
                np.count_nonzero(foreground[y0 - window[0]:y1 - window[0], x0 - window[2]:x1 - window[2]])
            )
    return counts
