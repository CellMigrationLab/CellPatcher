"""Random mode: fixed-size squares at uniformly drawn positions of one seeded stream.

The draw sequence (since v0.2.0, matrix finding F8):

1. The generator is ``numpy.random.default_rng(seed)``; the pipeline derives each image's
   (plane's) ``seed`` from the run seed and its identity (:func:`cellpatcher.ids.random_seed`).
2. Draw *k* takes ``y0 = rng.integers(0, H - P + 1)`` and then
   ``x0 = rng.integers(0, W - P + 1)`` (``H``, ``W`` the plane's Y/X size, ``P`` the
   patch size). The stream depends only on the seed, the plane's shape and the patch size,
   never on a mask.
3. ``min_mask_fraction`` explicitly ``0``: the first ``n_patches`` draws are the patches. A sampling
   mask is checked (2-D, the plane's shape) but excludes nothing, so the rectangles are
   the same with and without one.
4. ``min_mask_fraction`` ``None`` (the default) with a mask: a draw is kept when it overlaps the
   mask by at least one pixel (the rule before the option existed, as a fraction of
   ``1 / patch_size**2`` would give). Without a mask every draw is kept.
5. ``min_mask_fraction > 0`` (needs a mask): the draws are taken in stream order and a
   draw is kept when its share of mask pixels is at least the fraction; the patches are
   the first ``n_patches`` kept draws. A rejected draw is still consumed, so the patches
   are the draws of the mask-free stream that pass the filter, in order. After
   ``max(1000, 50 * n_patches)`` draws the plane is refused
   (:class:`cellpatcher.SamplingExhaustedError`).

Sampling is with replacement: a rectangle can be drawn twice (``cellpatcher.ids`` numbers
the repeats). ``seed=None`` uses NumPy's nondeterministic initialization.
"""

from __future__ import annotations

import warnings

import numpy as np

from cellpatcher.config import RandomConfig
from cellpatcher.errors import SamplingExhaustedError
from cellpatcher.setting_checks import check_random_settings
from cellpatcher.strategies.base import BasePatchStrategy, _spatial_shape_from_axes
from cellpatcher.types import PatchRegion

ZERO_FRACTION_WARNING = (
    "random mode: min_mask_fraction is explicitly 0, so the sampling mask excludes no rectangle and the "
    "rectangles are those drawn without a mask (since CellPatcher 0.2.0). Leave min_mask_fraction "
    "unset to keep the draws that overlap the mask, or set it > 0 to sample inside the mask."
)
"""The warning of a sampling mask given with ``min_mask_fraction == 0``: it filters nothing."""


def _sampling_mask(labels: object, fraction: float | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Return the boolean sampling mask, or None when it can exclude no draw.

    Parameters
    ----------
    labels
        The sampling mask (any array whose nonzero pixels are foreground), or None.
    fraction
        ``RandomConfig.min_mask_fraction`` (None: any overlap).
    shape
        The plane's ``(height, width)``.

    Returns
    -------
    numpy.ndarray or None
        The mask as booleans unless *fraction* is 0, else None (the mask is checked only, and
        a ``UserWarning`` says that it excludes nothing).

    Raises
    ------
    ValueError
        If *fraction* is > 0 without a mask, or the mask is not 2-D of the plane's shape.
    """
    if labels is None:
        if fraction:
            raise ValueError("min_mask_fraction needs a sampling mask (masks=...) in random mode.")
        return None
    mask = np.asarray(labels).astype(bool)
    if mask.ndim != 2:
        raise ValueError("labels sampling mask must be 2D for random mode.")
    if mask.shape != shape:
        raise ValueError(f"labels shape {mask.shape} must match image YX shape {shape}.")
    if fraction != 0:
        return mask
    warnings.warn(ZERO_FRACTION_WARNING, UserWarning, stacklevel=3)
    return None


def _region(y0: int, x0: int, size: int, shape: tuple[int, int], image_id: object, mode: str) -> PatchRegion:
    """Return the region of the square of side *size* at ``(y0, x0)`` in a plane of *shape*."""
    y1, x1 = y0 + size, x0 + size
    touches_border = y0 == 0 or x0 == 0 or y1 == shape[0] or x1 == shape[1]
    return PatchRegion(
        patch_id=f"random-y{y0}-x{x0}-h{size}-w{size}",
        image_id=image_id,
        mode=mode,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        center_y=(y0 + y1) / 2,
        center_x=(x0 + x1) / 2,
        area_px=size * size,
        touches_border=touches_border,
        requested_y0=y0,
        requested_y1=y1,
        requested_x0=x0,
        requested_x1=x1,
        patch_touches_image_border=touches_border,
    )


class RandomPatchStrategy(BasePatchStrategy):
    """Random mode's strategy: ``n_patches`` squares of one seeded draw sequence (module docstring)."""

    mode = "random"

    def __init__(self, config: RandomConfig | None = None) -> None:
        self.config = config if config is not None else RandomConfig()

    def generate_regions(
        self, *, image_shape: tuple[int, ...], labels: object = None, image_id: object = None, axes: str = "YX"
    ) -> list[PatchRegion]:
        """Return the plane's random squares, in draw order.

        Parameters
        ----------
        image_shape
            The image's shape; its Y/X size is read with *axes*.
        labels
            Optional sampling mask of the plane's Y/X shape. It excludes draws only when
            ``min_mask_fraction`` > 0.
        image_id
            Stamped on every region.
        axes
            The axes of *image_shape*.

        Returns
        -------
        list of PatchRegion
            ``n_patches`` regions, the kept draws in stream order.

        Raises
        ------
        ValueError
            If ``min_mask_fraction`` is not ``None`` or a number in [0, 1] (a bool, text, NaN), the
            patch size or count is not positive, the patch is larger than the plane,
            or the mask is missing (fraction > 0), not 2-D or of another shape.
        SamplingExhaustedError
            If fewer than ``n_patches`` draws reach ``min_mask_fraction`` within the attempt cap.
        """
        check_random_settings(self.config)  # the config is mutable: re-checked where it is used
        shape = _spatial_shape_from_axes(image_shape, axes=axes)
        size, n_patches = int(self.config.patch_size), int(self.config.n_patches)
        if size <= 0:
            raise ValueError("patch_size must be > 0.")
        if n_patches <= 0:
            raise ValueError("n_patches must be > 0.")
        if size > shape[0] or size > shape[1]:
            raise ValueError(
                f"patch_size ({size}) must be <= image spatial dimensions Y={shape[0]}, X={shape[1]}."
            )
        fraction = self.config.min_mask_fraction
        mask = _sampling_mask(labels, fraction, shape)
        corners = self._kept_corners(np.random.default_rng(self.config.seed), shape, mask, fraction)
        return [_region(y0, x0, size, shape, image_id, self.mode) for y0, x0 in corners]

    def _kept_corners(
        self, rng: np.random.Generator, shape: tuple[int, int], mask: np.ndarray | None, fraction: float | None
    ) -> list[tuple[int, int]]:
        """Return the top-left corners of the first ``n_patches`` kept draws of *rng*'s stream.

        Every draw takes ``y0`` and then ``x0`` from *rng*, whether it is kept or not. Without
        *mask* every draw is kept; with it, a draw whose share of mask pixels is below
        *fraction* (any overlap when None) is dropped, and after ``max(1000, 50 * n_patches)``
        draws the plane is refused.
        """
        size, n_patches = int(self.config.patch_size), int(self.config.n_patches)
        y_max, x_max = shape[0] - size, shape[1] - size
        max_attempts = max(1000, n_patches * 50)
        corners: list[tuple[int, int]] = []
        attempts = 0
        while len(corners) < n_patches:
            if mask is not None and attempts >= max_attempts:
                raise SamplingExhaustedError(
                    "Could not sample enough valid random patches from the provided mask. "
                    f"Requested {n_patches}, collected {len(corners)} after {attempts} attempts.",
                    requested=n_patches,
                    collected=len(corners),
                    attempts=attempts,
                )
            y0 = int(rng.integers(0, y_max + 1))
            x0 = int(rng.integers(0, x_max + 1))
            attempts += 1
            if mask is None or self._reaches(mask[y0:y0 + size, x0:x0 + size], fraction):
                corners.append((y0, x0))
        return corners

    @staticmethod
    def _reaches(window: np.ndarray, fraction: float | None) -> bool:
        """Return whether a draw's mask *window* overlaps (None) or reaches *fraction* of the mask."""
        return bool(window.any()) if fraction is None else bool(window.mean() >= fraction)
