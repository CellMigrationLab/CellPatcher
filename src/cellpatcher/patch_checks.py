"""Checks of a patch rectangle and a region mask handed to a measuring brick (D86 API C3).

MorphoFeature passes CellPatcher's patches to the feature bricks as
``bounds = (y0, y1, x0, x1)`` with the 2D plane they lie in, and region masks
of that plane. MorphoObjects, MorphoRelate, fiberfeatures and MorphoEmbed each
kept a copy of these checks, and MorphoEmbed's took float bounds; the rule now
lives here, strict for every brick.

:func:`context_bounds` gives the execution window of a patch (MorphoBricks-Spec
ADR 0004, decisions EW4 and EW5): the patch grown by a context margin, clipped
at the plane edge, with the margin actually kept per side. Every rectangle is
``pixel_edge`` and half-open in the plane frame (EW1); only ``local_interior``
is relative to the window's ``origin``. Pure geometry: no pixel is read.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class WindowRecord:
    """The window record of one patch (Spec ADR 0004, EW4/EW5); every value a plain int.

    Attributes
    ----------
    context_bounds
        ``(y0, y1, x0, x1)`` of the window: the patch grown by the margin and
        clipped at the plane edge (half-open, plane frame).
    origin
        ``(y, x)`` of the window's first pixel in the plane, i.e.
        ``(context_bounds[0], context_bounds[2])``.
    margin_requested
        ``(top, bottom, left, right)``: the margin asked for, ``(my, my, mx, mx)``.
    margin_clipped
        ``(top, bottom, left, right)``: the margin kept; smaller than the
        requested one only where the plane edge cuts it.
    local_interior
        ``(y0, y1, x0, x1)`` of the patch in window-local coordinates (the
        patch bounds minus ``origin``).
    plane_shape
        ``(height, width)`` of the plane the window lies in (ADR 0004 EW4).
    """

    context_bounds: tuple[int, int, int, int]
    origin: tuple[int, int]
    margin_requested: tuple[int, int, int, int]
    margin_clipped: tuple[int, int, int, int]
    local_interior: tuple[int, int, int, int]
    plane_shape: tuple[int, int]

    def __post_init__(self) -> None:
        """Refuse a record that :func:`context_bounds` would not return for its own patch.

        Every field is a tuple of plain ``int`` (not a list, a NumPy scalar or a boolean), the
        requested margin is ``(my, my, mx, mx)``, and every field equals what
        :func:`context_bounds` gives for the patch ``origin + local_interior``, that margin and
        ``plane_shape``. So a record built directly is immutable and consistent.
        """
        for name, count in _RECORD_FIELDS.items():
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != count or any(type(v) is not int for v in values):
                raise TypeError(f"WindowRecord.{name} must be a tuple of {count} plain ints, got {values!r}")
        top, bottom, left, right = self.margin_requested
        if top != bottom or left != right:
            raise PatchBoundsError(f"margin_requested {self.margin_requested} is not (my, my, mx, mx)")
        (oy, ox), (ly0, ly1, lx0, lx1) = self.origin, self.local_interior
        expected = _window_fields((oy + ly0, oy + ly1, ox + lx0, ox + lx1), top, left, self.plane_shape)
        wrong = [name for name, value in expected.items() if getattr(self, name) != value]
        if wrong:
            raise PatchBoundsError(f"WindowRecord fields {wrong} disagree with its patch, margin and plane")


_RECORD_FIELDS = {
    "context_bounds": 4,
    "origin": 2,
    "margin_requested": 4,
    "margin_clipped": 4,
    "local_interior": 4,
    "plane_shape": 2,
}


def _window_fields(
    bounds: tuple[int, ...] | list[int], my: int, mx: int, shape: tuple[int, int]
) -> dict[str, tuple[int, ...]]:
    """Return the fields of the window record of a checked patch (ADR 0004 formulas)."""
    height, width = shape
    y0, y1, x0, x1 = check_patch_bounds(bounds, shape)
    window = (max(0, y0 - my), min(height, y1 + my), max(0, x0 - mx), min(width, x1 + mx))
    origin_y, origin_x = window[0], window[2]
    return {
        "context_bounds": window,
        "origin": (origin_y, origin_x),
        "margin_requested": (my, my, mx, mx),
        "margin_clipped": (y0 - window[0], window[1] - y1, x0 - window[2], window[3] - x1),
        "local_interior": (y0 - origin_y, y1 - origin_y, x0 - origin_x, x1 - origin_x),
        "plane_shape": (height, width),
    }


def _integer_tuple(name: str, values: object, count: int) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise TypeError(f"{name} must be a tuple of {count} integers, got {values!r}")
    if not all(_is_integer(v) for v in values):
        raise TypeError(f"{name} must hold integers only (not floats or booleans), got {values!r}")
    return tuple(int(v) for v in values)


def context_bounds(
    bounds: tuple[int, int, int, int] | list[int],
    margin: tuple[int, int] | list[int],
    shape: tuple[int, int] | list[int],
) -> WindowRecord:
    """Return the window record of a patch grown by *margin* and clipped at the plane edge.

    Parameters
    ----------
    bounds
        The patch ``(y0, y1, x0, x1)``, half-open ``pixel_edge`` in the plane
        frame: a non-empty rectangle inside the plane.
    margin
        ``(my, mx)``: the context margin in pixels per axis, non-negative.
    shape
        ``(H, W)``: the plane's Y/X shape.

    Every value is a Python or NumPy integer; floats (even whole ones) and
    booleans are refused, never converted.

    Raises
    ------
    TypeError
        If an argument is not a tuple or list of the right length, or holds a
        value that is not an integer.
    PatchBoundsError
        (a ``ValueError``) if the margin is negative, the shape is not
        positive, or *bounds* are not a non-empty rectangle inside the plane.
    """
    _integer_tuple("bounds", bounds, 4)
    my, mx = _integer_tuple("margin", margin, 2)
    height, width = _plane_shape(_integer_tuple("shape", shape, 2))
    if my < 0 or mx < 0:
        raise PatchBoundsError(f"a context margin is never negative, got {tuple(margin)!r}")
    return WindowRecord(**_window_fields(bounds, my, mx, (height, width)))
