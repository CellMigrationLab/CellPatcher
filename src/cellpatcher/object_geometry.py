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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
