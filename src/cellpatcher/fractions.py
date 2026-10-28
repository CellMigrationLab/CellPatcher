"""Share of each patch inside named masks (``run_patching(..., fraction_masks=...)``).

For every patch the caller lists masks for, ``PatchRegion.mask_fractions[name]``
is the number of the mask's foreground (non-zero) pixels inside the patch's
``y0/y1/x0/x1`` divided by the patch's area, computed exactly in integers from
one summed-area table per mask and image. Cell patches also carry
``"own_object"`` (set by the cell strategy), so mask names may not be that.
"""

from __future__ import annotations

import re

import numpy as np

OWN_OBJECT = "own_object"
_NAME = re.compile(r"[A-Za-z0-9._-]+")


def check_fraction_masks(fraction_masks, image_list, image_axes: str) -> list[dict]:
    """The fraction masks as one ``{name: 2-D bool array}`` per image, after checking them.

    Raises
    ------
    ValueError
        If there is not one mapping per image, a name is not a plain name or
        is ``"own_object"``, or a mask is not 2-D with the image's Y/X shape.
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
            array = np.asarray(mask)
            if array.shape != tuple(shape):
                raise ValueError(
                    f"fraction mask {name!r} of image {index} has shape {array.shape}; the image's Y/X shape "
                    f"is {tuple(shape)}."
                )
            entry[name] = array != 0
        checked.append(entry)
    return checked


def _summed_area(mask: np.ndarray) -> np.ndarray:
    """The summed-area table of *mask*, with a leading row and column of zeros."""
    table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(mask, axis=0, dtype=np.int64), axis=1, out=table[1:, 1:])
    return table


def stamp_mask_fractions(patches, fraction_masks: list[dict]) -> None:
    """Add each patch's share inside its image's fraction masks to ``mask_fractions``."""
    tables = [{name: _summed_area(mask) for name, mask in masks.items()} for masks in fraction_masks]
    for patch in patches:
        masks = tables[int(patch.image_id.rsplit("_", 1)[1])]
        if not masks:
            continue
        area = (patch.y1 - patch.y0) * (patch.x1 - patch.x0)
        fractions = dict(patch.mask_fractions or {})
        for name, table in masks.items():
            inside = (
                table[patch.y1, patch.x1] - table[patch.y0, patch.x1] - table[patch.y1, patch.x0]
                + table[patch.y0, patch.x0]
            )
            fractions[name] = int(inside) / area
        patch.mask_fractions = fractions
