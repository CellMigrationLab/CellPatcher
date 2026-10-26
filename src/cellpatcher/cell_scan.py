from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cellpatcher.axes import normalize_axes, require_spatial_axes
from cellpatcher.config import CellConfig
from cellpatcher.strategies.cell import CellSegmentationStrategy


@dataclass
class CellObjectRecord:
    image_id: str
    label_id: int
    area_px: int
    bbox_height: int
    bbox_width: int
    square_size: int
    touches_border: bool
    included: bool
    reason: str | None = None


@dataclass
class CellPatchSizeSummary:
    records: list[CellObjectRecord]
    n_images: int
    n_objects_total: int
    n_included: int
    n_skipped_border: int
    n_skipped_area: int
    max_bbox_height: int
    max_bbox_width: int
    min_square_size: int
    max_square_size: int
    mean_square_size: float
    median_square_size: float
    percentile_90_square_size: float
    percentile_95_square_size: float

    def recommended_size(self, method: str = "max") -> int:
        if self.n_included == 0:
            return 0

        method_map = {
            "max": self.max_square_size,
            "median": self.median_square_size,
            "mean": self.mean_square_size,
            "p90": self.percentile_90_square_size,
            "p95": self.percentile_95_square_size,
        }
        if method not in method_map:
            raise ValueError("method must be one of: 'max', 'median', 'mean', 'p90', 'p95'.")
        return int(math.ceil(float(method_map[method])))


def _normalize_shape_inputs(image_shapes, labels_list):
    if labels_list is None:
        raise ValueError("labels_list is required.")

    # Accept either a single label array or a list of them.
    if isinstance(labels_list, (list, tuple)):
        label_items = list(labels_list)
    else:
        label_items = [labels_list]

    # image_shapes is ambiguous on its own: a single shape tuple like (H, W)
    # looks the same as "a list of shapes" once wrapped, so detect the
    # single-shape case explicitly (a tuple of plain ints) before falling
    # back to treating it as a list of shape tuples.
    if isinstance(image_shapes, tuple) and all(isinstance(v, int) for v in image_shapes):
        shape_items = [tuple(image_shapes)]
    elif isinstance(image_shapes, (list, tuple)):
        shape_items = [tuple(shape) for shape in image_shapes]
    else:
        raise ValueError("image_shapes must be a shape tuple or a list/tuple of shape tuples.")

    if len(shape_items) != len(label_items):
        raise ValueError("Number of image_shapes and labels_list entries must match.")

    return shape_items, label_items


def _prepare_labeled_plane(labels, config: CellConfig, mask_axes: str) -> np.ndarray:
    # Reuse the cell strategy's own channel-selection logic so scanning sees
    # exactly the same 2D label plane a real patching run would use.
    strategy = CellSegmentationStrategy(config=config, mask_axes=mask_axes)
    selected = strategy._select_label_plane(labels)

    if selected.ndim != 2:
        raise ValueError("Selected labels must be 2D after channel selection.")
    if selected.size == 0:
        raise ValueError("labels must not be empty.")

    # The same rules as a patching run (decision D47): integer masks are
    # instance labels as given; binary masks need the connected-component
    # opt-in; labels above the int32 maximum are refused.
    labeled, _ = strategy.label_objects(selected)
    return labeled


def scan_cell_objects(*, image_shape, labels, image_id=None, config=None, image_axes="YX", mask_axes="YX") -> list[CellObjectRecord]:
    """Scan a single image/label pair and return per-object size records.

    For each non-zero label in *labels*, one :class:`CellObjectRecord` is
    returned containing the object's bounding-box dimensions, area, border
    status, and whether it passes the ``config`` inclusion criteria
    (``min_area``, ``max_area``, ``exclude_border``).

    Parameters
    ----------
    image_shape:
        Shape tuple of the image (used to determine spatial dimensions).
    labels:
        Integer label array (2D ``YX``, or multidimensional when
        ``mask_axes`` and ``config.mask_channel`` are set).
    image_id:
        Optional identifier string embedded in each returned record.
        Defaults to ``"image_0000"``.
    config:
        :class:`~cellpatcher.CellConfig` instance.  Defaults to
        ``CellConfig()`` (all defaults).
    image_axes:
        Axes string for *image_shape* (e.g. ``"YX"``, ``"CYX"``).
    mask_axes:
        Axes string for *labels*.

    Returns
    -------
    list[CellObjectRecord]
        One record per non-zero label, sorted by label id.
    """
    local_config = config if config is not None else CellConfig()

    # Resolve the image's Y/X spatial extent so per-object border checks
    # below know where the image edges actually are.
    normalized_image_axes = normalize_axes(image_axes, len(image_shape))
    y_idx, x_idx = require_spatial_axes(normalized_image_axes)
    image_h = int(image_shape[y_idx])
    image_w = int(image_shape[x_idx])
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got Y={image_h}, X={image_w}.")

    labeled = _prepare_labeled_plane(labels, local_config, mask_axes)

    if labeled.shape != (image_h, image_w):
        raise ValueError(f"labels shape {labeled.shape} must match image YX shape {(image_h, image_w)}.")

    object_ids = [int(v) for v in np.unique(labeled) if int(v) != 0]
    records: list[CellObjectRecord] = []
    image_id_value = image_id if image_id is not None else "image_0000"

    # One record per labeled object, regardless of whether it would pass
    # the size/border filters -- scan_cell_patch_size needs the excluded
    # objects too, to report why they were skipped.
    for label_id in sorted(object_ids):
        ys, xs = np.where(labeled == label_id)
        if ys.size == 0:
            continue

        area_px = int(ys.size)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        bbox_height = y_max - y_min + 1
        bbox_width = x_max - x_min + 1
        square_size = int(max(bbox_height, bbox_width))
        touches_border = y_min == 0 or x_min == 0 or y_max == image_h - 1 or x_max == image_w - 1

        # Border exclusion takes priority over the area check when deciding
        # why an object is excluded, matching CellSegmentationStrategy's
        # own precedence.
        included = True
        reason = None
        if local_config.exclude_border and touches_border:
            included = False
            reason = "border"
        elif area_px < local_config.min_area or (
            local_config.max_area is not None and area_px > local_config.max_area
        ):
            included = False
            reason = "area"

        records.append(
            CellObjectRecord(
                image_id=image_id_value,
                label_id=label_id,
                area_px=area_px,
                bbox_height=bbox_height,
                bbox_width=bbox_width,
                square_size=square_size,
                touches_border=touches_border,
                included=included,
                reason=reason,
            )
        )

    return records


def scan_cell_patch_size(image_shapes, labels_list, config=None, image_axes="YX", mask_axes="YX") -> CellPatchSizeSummary:
    local_config = config if config is not None else CellConfig()
    shape_items, label_items = _normalize_shape_inputs(image_shapes, labels_list)

    # Scan every image/label pair independently, then pool all per-object
    # records together for the dataset-level statistics below.
    all_records: list[CellObjectRecord] = []
    for idx, (shape, labels) in enumerate(zip(shape_items, label_items)):
        image_id = f"image_{idx:04d}"
        all_records.extend(
            scan_cell_objects(
                image_shape=shape,
                labels=labels,
                image_id=image_id,
                config=local_config,
                image_axes=image_axes,
                mask_axes=mask_axes,
            )
        )

    included = [r for r in all_records if r.included]
    n_skipped_border = sum(1 for r in all_records if r.reason == "border")
    n_skipped_area = sum(1 for r in all_records if r.reason == "area")

    # Size statistics are only meaningful over objects that would actually
    # be patched; skipped objects (border/area) are excluded here even
    # though they're still present in `records` for inspection.
    if included:
        bbox_heights = np.asarray([r.bbox_height for r in included], dtype=np.int32)
        bbox_widths = np.asarray([r.bbox_width for r in included], dtype=np.int32)
        square_sizes = np.asarray([r.square_size for r in included], dtype=np.float64)

        max_bbox_height = int(np.max(bbox_heights))
        max_bbox_width = int(np.max(bbox_widths))
        min_square_size = int(np.min(square_sizes))
        max_square_size = int(np.max(square_sizes))
        mean_square_size = float(np.mean(square_sizes))
        median_square_size = float(np.median(square_sizes))
        percentile_90_square_size = float(np.percentile(square_sizes, 90))
        percentile_95_square_size = float(np.percentile(square_sizes, 95))
    else:
        max_bbox_height = 0
        max_bbox_width = 0
        min_square_size = 0
        max_square_size = 0
        mean_square_size = 0.0
        median_square_size = 0.0
        percentile_90_square_size = 0.0
        percentile_95_square_size = 0.0

    return CellPatchSizeSummary(
        records=all_records,
        n_images=len(shape_items),
        n_objects_total=len(all_records),
        n_included=len(included),
        n_skipped_border=n_skipped_border,
        n_skipped_area=n_skipped_area,
        max_bbox_height=max_bbox_height,
        max_bbox_width=max_bbox_width,
        min_square_size=min_square_size,
        max_square_size=max_square_size,
        mean_square_size=mean_square_size,
        median_square_size=median_square_size,
        percentile_90_square_size=percentile_90_square_size,
        percentile_95_square_size=percentile_95_square_size,
    )
