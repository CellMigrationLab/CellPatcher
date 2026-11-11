"""Bounding boxes of the objects of a 2D instance mask, computed in one pass.

The patch geometry other MorphoBricks bricks need from a mask (MorphoBricks
decision D86, API C1): each object's bounding-box side, and the objects that
touch the image border. MorphoFeature (``object_extent.object_sides``,
``assembly.border_labels``) and MorphoEmbed (``provider_crop_scale``) kept
byte-for-byte copies of this; :func:`cellpatcher.scan_cell_objects` uses the
same boxes, so a patch size fitted downstream and CellPatcher's own scan agree.

Labels are checked with the preprocessing brick's ``check_labels`` (owner
decision D-b): whole-number labels from 0 (background) to the int32 maximum;
a float mask is refused even when its values are whole numbers.

It also holds **the one shared labeller** (Spec ADR 0002 AR5, experiment-v1 §4.5c):
:func:`label_objects` turns a 2D ``binary`` mask into object labels for the explicit,
per-request ``binary_mask_policy`` the resolver returns (``MaskSelection.conversion``). Every
route that measures a ``binary`` mask as objects calls it (CellPatcher's cell mode,
MorphoFeature, the direct runners), so the routes find the same objects; no route labels a
mask with its own code, and nothing labels a mask without the policy. An ``instances`` mask
is never passed to it: it is used as stored (decision D47: an ``instances`` mask holding 0
and 1 is one object).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONNECTED_COMPONENTS = "connected_components"
BINARY_MASK_POLICIES = (CONNECTED_COMPONENTS,)
"""The ``binary_mask_policy`` values (P27; the resolver contract's ``BINARY_MASK_POLICIES``).
There is no default: a caller without an explicit per-request policy does not call the labeller."""

CONNECTIVITIES = {1: "4-connected", 2: "8-connected"}
"""The ``connectivity`` of :func:`label_objects`, as ``scipy.ndimage.generate_binary_structure(2, c)``:
1 joins a pixel to its 4 edge neighbours (up, down, left, right); 2 also to its 4 corner
neighbours, so two blobs that touch only at a corner are one object at 2 and two at 1."""

BINARY_MASK_CONNECTIVITY = 1
"""The connectivity of ``connected_components`` on the shared 2-D routes (4-connected, the Stage 2
plan's rule and CellPatcher's ``cell.binary_mask_connectivity`` default). Recorded with the policy."""


@dataclass(frozen=True)
class ObjectBoxes:
    """The bounding box of every object of a mask, one entry per label in increasing order.

    Attributes
    ----------
    labels
        The non-zero labels, increasing.
    area
        Pixels per object.
    y_min, y_max, x_min, x_max
        The first and last row and column of each object (inclusive).
    shape
        The mask's ``(height, width)``.
    """

    labels: np.ndarray
    area: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    shape: tuple[int, int]

    @property
    def height(self) -> np.ndarray:
        """Bounding-box height of each object, in pixels."""
        return self.y_max - self.y_min + 1

    @property
    def width(self) -> np.ndarray:
        """Bounding-box width of each object, in pixels."""
        return self.x_max - self.x_min + 1

    @property
    def side(self) -> np.ndarray:
        """Bounding-box side (the larger of height and width) of each object, in pixels."""
        return np.maximum(self.height, self.width)

    @property
    def touches_border(self) -> np.ndarray:
        """Whether each object has a pixel on the first or last row or column."""
        height, width = self.shape
        return (self.y_min == 0) | (self.x_min == 0) | (self.y_max == height - 1) | (self.x_max == width - 1)


def _checked_mask(labels, name: str) -> np.ndarray:
    from morphobricks_preprocessing.masks import check_labels

    mask = np.asarray(labels)
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D instance mask (Y, X); got {mask.ndim} dimensions, shape {mask.shape}.")
    if mask.size == 0:
        raise ValueError(f"{name} is empty (shape {mask.shape}).")
    check_labels(mask, name)
    return mask


def object_boxes(labels, name: str = "the mask") -> ObjectBoxes:
    """Bounding boxes and areas of every object of a 2D instance mask.

    Parameters
    ----------
    labels
        2D integer (or boolean) instance mask; 0 is background.
    name
        How messages name the mask.

    Raises
    ------
    ValueError
        If the mask is not 2D, is empty, or its labels are not whole numbers
        from 0 to the int32 maximum (preprocessing ``check_labels``).
    """
    mask = _checked_mask(labels, name)
    rows, cols = np.nonzero(mask)
    values = mask[rows, cols].astype(np.int64)
    order = np.argsort(values, kind="stable")
    values, rows, cols = values[order], rows[order], cols[order]
    unique, starts = np.unique(values, return_index=True)
    if unique.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return ObjectBoxes(empty, empty, empty, empty, empty, empty, mask.shape)
    area = np.diff(np.append(starts, values.size))
    return ObjectBoxes(
        labels=unique,
        area=area.astype(np.int64),
        y_min=np.minimum.reduceat(rows, starts).astype(np.int64),
        y_max=np.maximum.reduceat(rows, starts).astype(np.int64),
        x_min=np.minimum.reduceat(cols, starts).astype(np.int64),
        x_max=np.maximum.reduceat(cols, starts).astype(np.int64),
        shape=(int(mask.shape[0]), int(mask.shape[1])),
    )


def border_labels(labels, name: str = "the mask") -> frozenset[int]:
    """The non-zero labels with at least one pixel on the image border.

    Raises
    ------
    ValueError
        As :func:`object_boxes`.
    """
    mask = _checked_mask(labels, name)
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return frozenset(int(label) for label in np.unique(edges) if label != 0)


def object_sides(labels, name: str = "the mask") -> dict[int, int]:
    """Label -> bounding-box side (the larger of height and width) of every interior object.

    Objects with a pixel on the image border are left out (their box is cut
    by the image edge), as MorphoFeature's crop-extent fit and MorphoEmbed's
    crops need.

    Raises
    ------
    ValueError
        As :func:`object_boxes`.
    """
    boxes = object_boxes(labels, name)
    return {
        int(label): int(side)
        for label, side, border in zip(boxes.labels, boxes.side, boxes.touches_border)
        if not border
    }


def label_objects(mask: np.ndarray, policy: str, connectivity: int) -> np.ndarray:
    """Return the object labels of a 2D ``binary`` mask under an explicit ``binary_mask_policy``.

    The one shared labeller of MorphoBricks (Spec ADR 0002 AR5): every route that uses a
    ``binary`` mask as an object set calls this function with the policy the request recorded
    (``MaskSelection.conversion``) and records the policy and *connectivity* beside the mask ID.
    It is pure: it reads no file and keeps no state.

    Parameters
    ----------
    mask
        A 2D ``binary`` mask: boolean, or integers holding 0 (background) and at most one
        positive foreground value (0/1, 0/255). The foreground is ``mask != 0``. Never an
        ``instances`` mask: those are used as stored (D47), not relabelled.
    policy
        ``"connected_components"`` (:data:`BINARY_MASK_POLICIES`). Required: there is no default,
        and None is refused.
    connectivity
        1 (4-connected) or 2 (8-connected), :data:`CONNECTIVITIES`. Required.
        :data:`BINARY_MASK_CONNECTIVITY` is the one the shared 2-D routes use.

    Returns
    -------
    numpy.ndarray
        ``int32`` labels of the mask's shape: 0 for background, then 1, 2, ... for each
        connected component of the foreground, numbered in row-major (reading) order of each
        component's first pixel (``scipy.ndimage.label``). Deterministic for a given input.

    Raises
    ------
    ValueError
        For a policy outside :data:`BINARY_MASK_POLICIES` (None included), a connectivity other
        than 1 or 2, a mask that is not 2D or is empty, a float or other non-integer, non-boolean
        mask, or an integer mask with a negative value or more than one foreground value (it is
        not ``binary``: declare it ``instances``, or fix it).
    """
    from scipy.ndimage import generate_binary_structure
    from scipy.ndimage import label as scipy_label

    if not isinstance(policy, str) or policy not in BINARY_MASK_POLICIES:
        raise ValueError(
            f"binary_mask_policy must be one of {BINARY_MASK_POLICIES}, not {policy!r}: a binary mask is labelled "
            "only by an explicit per-request policy (A_MASK_NOT_INSTANCES without one)."
        )
    if isinstance(connectivity, bool) or not isinstance(connectivity, (int, np.integer)) or connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 (4-connected) or 2 (8-connected), not {connectivity!r}.")
    foreground = _binary_foreground(mask)
    labels, _ = scipy_label(foreground, structure=generate_binary_structure(2, int(connectivity)))
    return labels.astype(np.int32, copy=False)


def _binary_foreground(mask: np.ndarray) -> np.ndarray:
    """Return ``mask != 0`` of a 2D boolean or 0/one-value integer mask, refusing anything else."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"The binary mask must be 2D (Y, X); got {array.ndim} dimensions, shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"The binary mask is empty (shape {array.shape}).")
    if array.dtype == bool:
        return array
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"The binary mask has pixel type {array.dtype}; save it as integers (0 and 1) or booleans.")
    values = np.unique(array[array != 0])
    if values.size > 1 or (values.size == 1 and values[0] < 0):
        shown = ", ".join(str(int(value)) for value in values[:5])
        raise ValueError(
            f"The binary mask holds the foreground values {shown}{', ...' if values.size > 5 else ''}: a binary mask "
            "holds 0 and one positive value. A mask whose values name objects is an instances mask (used as stored)."
        )
    return array != 0
