"""Patches before and after normalization, for the notebook's preview (layer: pure).

The notebook shows a few patches of the previewed image twice: as the run gives them without
normalization and as it gives them with it, each with its intensity histogram. Nothing here
normalizes or bins anything itself: the two sets of arrays are the patching runs' own (the
preview patches the image with normalization off and on, :mod:`cellpatcher.notebook_session`),
and the histograms are the normalization brick's ``compute_plane_histogram``, the one that
binned ``plane_histograms.csv`` (design v0.4 §9: the maths stays with its owner). Patches are
paired by ``patch_id``, never by their order, so a patch the normalization discarded (its own
pixels have no spread, decision D8) is simply not shown.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from morphobricks_normalization import compute_plane_histogram

if TYPE_CHECKING:
    from morphobricks_normalization import PlaneHistogram

    from cellpatcher.types import PatchRegion

DISPLAY_BINS = 64
"""Bins of the histograms shown (for display only; the statistics are never computed from them)."""
PAIRS_SHOWN = 4
"""Patches shown before and after normalization."""


@dataclass(frozen=True)
class BeforeAfter:
    """One patch without and with normalization, as the notebook draws it.

    Attributes
    ----------
    patch_id
        The patch's identifier (the same in both runs).
    label_id
        The object's label in cell mode, else None.
    channel
        The image channel shown: the chosen index of the patch's C axis, or the channel a
        split-channel patch holds (``patch.c``); None for a patch without channels.
    raw, normalized
        The Y/X plane shown, from the patch without and with normalization.
    raw_histogram, normalized_histogram
        The brick's histogram of each plane (:data:`DISPLAY_BINS` bins; NaN pixels, such as
        a NaN mask fill, are left out and counted by the brick).
    """

    patch_id: str
    label_id: int | None
    channel: int | None
    raw: np.ndarray
    normalized: np.ndarray
    raw_histogram: PlaneHistogram
    normalized_histogram: PlaneHistogram


def display_plane(array: np.ndarray, axes: str, channel: int = 0) -> np.ndarray:
    """Return the Y/X plane of a patch array: *channel* of its C axis, the first Z and T plane.

    Parameters
    ----------
    array
        A patch array, as delivered.
    axes
        Its axes (``PatchRegion.output_axes``), one letter per dimension, Y before X.
    channel
        The channel shown when the array has a C axis.

    Returns
    -------
    numpy.ndarray
        A 2-D view of *array*, never transposed.

    Raises
    ------
    ValueError
        When *axes* does not describe *array*, puts X before Y, or *channel* is not one of its
        channels.
    """
    array = np.asarray(array)
    if len(axes) != array.ndim or "Y" not in axes or "X" not in axes:
        raise ValueError(f"The axes {axes!r} do not describe a patch array of shape {array.shape}.")
    if axes.index("X") < axes.index("Y"):
        raise ValueError(f"The axes {axes!r} put X before Y; a patch array is never shown transposed.")
    if "C" in axes and not 0 <= channel < array.shape[axes.index("C")]:
        raise ValueError(f"Channel {channel} is not a channel of the patch (axes {axes!r}, shape {array.shape}).")
    index = tuple(slice(None) if axis in "YX" else channel if axis == "C" else 0 for axis in axes)
    return array[index]


def before_after(
    raw_patches: Sequence[PatchRegion],
    raw_arrays: Sequence[np.ndarray],
    normalized_patches: Sequence[PatchRegion],
    normalized_arrays: Sequence[np.ndarray],
    *,
    channel: int = 0,
    limit: int = PAIRS_SHOWN,
) -> tuple[BeforeAfter, ...]:
    """Pair the first normalized patches with the same patches without normalization.

    Parameters
    ----------
    raw_patches, raw_arrays
        The patches and arrays of the run without normalization.
    normalized_patches, normalized_arrays
        The same image's patches and arrays with normalization, in their order.
    channel
        The channel shown of patches with a C axis.
    limit
        The most pairs returned.

    Returns
    -------
    tuple of BeforeAfter
        In the order of *normalized_patches*; a patch missing from the raw run is left out.

    Raises
    ------
    ValueError
        When a run's patches and arrays differ in number, or a pair has different axes.
    """
    for patches, arrays in ((raw_patches, raw_arrays), (normalized_patches, normalized_arrays)):
        if len(patches) != len(arrays):
            raise ValueError(f"{len(patches)} patches but {len(arrays)} arrays: the arrays were not kept.")
    raw = {patch.patch_id: (patch, array) for patch, array in zip(raw_patches, raw_arrays, strict=True)}
    pairs = []
    for patch, array in zip(normalized_patches, normalized_arrays, strict=True):
        if len(pairs) >= limit:
            break
        if patch.patch_id not in raw:
            continue
        raw_patch, raw_array = raw[patch.patch_id]
        if raw_patch.output_axes != patch.output_axes:
            raise ValueError(
                f"The patch {patch.patch_id} has the axes {raw_patch.output_axes!r} without normalization and "
                f"{patch.output_axes!r} with it."
            )
        before = display_plane(raw_array, raw_patch.output_axes, channel)
        after = display_plane(array, patch.output_axes, channel)
        shown = channel if "C" in patch.output_axes else patch.c
        pairs.append(BeforeAfter(
            patch_id=patch.patch_id,
            label_id=patch.label_id,
            channel=shown,
            raw=before,
            normalized=after,
            raw_histogram=compute_plane_histogram(before, DISPLAY_BINS),
            normalized_histogram=compute_plane_histogram(after, DISPLAY_BINS),
        ))
    return tuple(pairs)
