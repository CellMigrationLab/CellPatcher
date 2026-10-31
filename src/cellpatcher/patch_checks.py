"""Checks of a patch rectangle and a region mask handed to a measuring brick (D86 API C3).

MorphoFeature passes CellPatcher's patches to the feature bricks as
``bounds = (y0, y1, x0, x1)`` with the 2D plane they lie in, and region masks
of that plane. MorphoObjects, MorphoRelate, fiberfeatures and MorphoEmbed each
kept a copy of these checks, and MorphoEmbed's took float bounds; the rule now
lives here, strict for every brick.
"""

from __future__ import annotations

import numpy as np


class PatchBoundsError(ValueError):
    """A patch rectangle or region mask a brick cannot measure on."""


def _plane_shape(shape) -> tuple[int, int]:
    values = tuple(shape) if isinstance(shape, (tuple, list)) else None
    if values is None or len(values) != 2 or not all(_is_integer(v) and v > 0 for v in values):
        raise PatchBoundsError(f"the plane shape {shape!r} is not two positive integers (Y, X)")
    return int(values[0]), int(values[1])


def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def check_patch_bounds(bounds, shape, *, name: str = "the patch") -> tuple[int, int, int, int]:
    """Return ``(y0, y1, x0, x1)`` as plain ints when they are a non-empty rectangle inside *shape*.

    Parameters
    ----------
    bounds
        ``(y0, y1, x0, x1)``, half-open like ``PatchRegion``'s rectangle: a
        tuple or list of four Python or NumPy integers. Floats (even whole
        ones such as ``4.0``), booleans and text are refused, never converted.
    shape
        The ``(height, width)`` of the 2D plane the rectangle lies in.
    name
        How the message names the patch, e.g. ``"patch 'p1'"``.

    Raises
    ------
    PatchBoundsError
        (a ``ValueError``) if *bounds* are not four integers, the rectangle
        is empty (``y1 <= y0`` or ``x1 <= x0``), or it leaves the plane.
    """
    height, width = _plane_shape(shape)
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 4 or not all(_is_integer(v) for v in bounds):
        raise PatchBoundsError(f"{name}: bounds {bounds!r} are not 4 integers (y0, y1, x0, x1)")
    y0, y1, x0, x1 = (int(v) for v in bounds)
    if y1 <= y0 or x1 <= x0:
        raise PatchBoundsError(f"{name}: bounds {[y0, y1, x0, x1]} are empty (y1 must exceed y0, x1 x0)")
    if not (0 <= y0 and y1 <= height and 0 <= x0 and x1 <= width):
        raise PatchBoundsError(f"{name}: bounds {[y0, y1, x0, x1]} leave the {(height, width)} plane")
    return y0, y1, x0, x1


def check_region_mask(mask, shape, *, name: str = "the region mask") -> np.ndarray:
    """Return *mask* when it is a 2D integer or boolean array of the plane's *shape*.

    A region mask selects the pixels to measure (non-zero inside). A missing
    mask, one of another shape, and one holding floats are refused, never
    converted or resized.

    Raises
    ------
    PatchBoundsError
        (a ``ValueError``) naming the mask and the problem.
    """
    plane = _plane_shape(shape)
    if not isinstance(mask, np.ndarray):
        raise PatchBoundsError(f"{name} is missing or not an array (got {type(mask).__name__})")
    if mask.shape != plane:
        raise PatchBoundsError(f"{name} has shape {mask.shape}, not the plane's {plane}")
    if not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == np.bool_):
        raise PatchBoundsError(f"{name} must hold integers or booleans, not {mask.dtype}")
    return mask
