"""Share of each patch inside named masks (``run_patching(..., fraction_masks=...)``).

For every patch the caller lists masks for, ``PatchRegion.mask_fractions[name]``
is the number of the mask's foreground (non-zero) pixels inside the patch's
``y0/y1/x0/x1`` divided by the patch's area, computed exactly in integers from
one summed-area table per mask and image. Cell patches also carry
``"own_object"`` (set by the cell strategy), so mask names may not be that.

A mask may instead be a :class:`~cellpatcher.mask_windows.WindowedMask` (a window
reader, the plane's shape and dtype, and a memory limit): its counts are taken from
windows of the patches only (:func:`~cellpatcher.mask_windows.windowed_foreground_counts`),
never from the whole plane, and the fractions are bitwise those of the array.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np

from cellpatcher.mask_windows import WindowedMask, windowed_foreground_counts

OWN_OBJECT = "own_object"
_NAME = re.compile(r"[A-Za-z0-9._-]+")


def check_unit_fraction(value: object, name: str) -> None:
    """Refuse a setting that is not a real number in [0, 1]; ``None`` means unset.

    ``bool`` and ``np.bool_`` are refused (``True`` is not 1.0 and ``False`` is not an explicit
    0), as are strings, NaN and infinities. Ints and floats (and NumPy numbers) pass.

    Raises
    ------
    ValueError
        With ``name`` when ``value`` is not ``None`` or a finite real number in [0, 1].
    """
    if value is None:
        return
    is_number = isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
    if not is_number or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be None or a number with 0 <= {name} <= 1; got {value!r}.")


def check_real_number(value: object, name: str, *, finite: bool = True) -> None:
    """Refuse a setting that is not a real number (a finite one unless ``finite=False``).

    ``bool`` and ``np.bool_`` (``True`` is not 1.0), strings, NaN and infinities are refused; ints
    and floats (and NumPy numbers) pass; NaN and infinities pass only with ``finite=False`` (a fill
    value). Nothing is converted with ``float()``.

    Raises
    ------
    ValueError
        With ``name`` when ``value`` is not a (finite) real number.
    """
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or (finite and not math.isfinite(value)):
        raise ValueError(f"{name} must be a {'finite ' if finite else ''}number; got {value!r}.")


def check_integer(value: object, name: str, *, minimum: int | None = None) -> None:
    """Refuse a setting that is not an integer >= *minimum* (no minimum when it is None).

    ``bool``, floats such as 16.0 and text are refused.

    Raises
    ------
    ValueError
        With ``name`` when ``value`` is not an ``int`` (or NumPy integer) >= *minimum*.
    """
    is_integer = isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
    if not is_integer or (minimum is not None and value < minimum):
        floor = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{floor}; got {value!r}.")


def check_choice(value: object, choices: Sequence[str], name: str) -> None:
    """Refuse a setting that is not text equal to one of *choices* (a number, list or None is refused, not matched).

    Raises
    ------
    ValueError
        With ``name`` and the choices when ``value`` is not a ``str`` among them.
    """
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}; got {value!r}.")


def check_strict_bool(value: object, name: str) -> None:
    """Refuse a setting that is not literally ``True`` or ``False`` (``np.bool_``, 0, 1 and text are refused).

    Raises
    ------
    ValueError
        With ``name`` when ``value`` is not a ``bool``.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False; got {value!r}.")


def check_fraction_masks(fraction_masks, image_list, image_axes: str) -> list[dict]:
    """The fraction masks as one ``{name: 2-D bool array or WindowedMask}`` per image, after checking them.

    Raises
    ------
    ValueError
        If there is not one mapping per image, a name is not a plain name or
        is ``"own_object"``, or a mask (or a ``WindowedMask``'s ``shape``) is
        not 2-D with the image's Y/X shape.
    """
    from cellpatcher.strategies.base import _spatial_shape_from_axes

    if len(fraction_masks) != len(image_list):
        raise ValueError("fraction_masks must have exactly one mapping per image (or be None).")
    checked = []
    for index, (masks, image) in enumerate(zip(fraction_masks, image_list)):
        shape = _spatial_shape_from_axes(image.shape, axes=image_axes)
        entry = {}
        for name, mask in dict(masks or {}).items():
            if not isinstance(name, str) or not _NAME.fullmatch(name) or name == OWN_OBJECT:
                raise ValueError(
                    f"fraction mask name {name!r} of image {index} must be letters, digits, '.', '_' or '-', "
                    f"and not {OWN_OBJECT!r}."
                )
            entry[name] = _checked_mask(mask, f"fraction mask {name!r} of image {index}", tuple(shape))
        checked.append(entry)
    return checked


def _checked_mask(mask: object, what: str, shape: tuple) -> np.ndarray | WindowedMask:
    """Return *mask* as its boolean foreground, or the ``WindowedMask`` itself, once its shape is the image's."""
    if isinstance(mask, WindowedMask):
        if mask.shape != shape:
            raise ValueError(f"{what} is a WindowedMask of shape {mask.shape}; the image's Y/X shape is {shape}.")
        return mask
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"{what} has shape {array.shape}; the image's Y/X shape is {shape}.")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{what} holds NaN or infinity.")
    return array != 0


def _summed_area(mask: np.ndarray) -> np.ndarray:
    """The summed-area table of *mask*, with a leading row and column of zeros."""
    table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(mask, axis=0, dtype=np.int64), axis=1, out=table[1:, 1:])
    return table


def _foreground_counts(mask: np.ndarray | WindowedMask, rectangles: list[tuple[int, int, int, int]]) -> list[int]:
    """Return each rectangle's number of foreground pixels: by windows, or from the summed-area table."""
    if isinstance(mask, WindowedMask):
        return windowed_foreground_counts(mask, rectangles)
    table = _summed_area(mask)
    return [int(table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]) for y0, y1, x0, x1 in rectangles]


def stamp_mask_fractions(patches, fraction_masks: list[dict]) -> None:
    """Add each patch's share inside its image's fraction masks to ``mask_fractions``."""
    by_image: dict[int, list] = {}
    for patch in patches:
        by_image.setdefault(int(patch.image_id.rsplit("_", 1)[1]), []).append(patch)
    for index, image_patches in by_image.items():
        masks = fraction_masks[index]
        if not masks:
            continue
        rectangles = [(patch.y0, patch.y1, patch.x0, patch.x1) for patch in image_patches]
        counts = {name: _foreground_counts(mask, rectangles) for name, mask in masks.items()}
        for position, patch in enumerate(image_patches):
            area = (patch.y1 - patch.y0) * (patch.x1 - patch.x0)
            fractions = dict(patch.mask_fractions or {})
            for name in masks:
                fractions[name] = counts[name][position] / area
            patch.mask_fractions = fractions
