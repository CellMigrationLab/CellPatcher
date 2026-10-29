import warnings
from collections import deque

import numpy as np

from cellpatcher.axes import axis_index, normalize_axes
from cellpatcher.config import CellConfig
from cellpatcher.errors import LabelRangeError
from cellpatcher.strategies.base import BasePatchStrategy, _spatial_shape_from_axes
from cellpatcher.types import PatchRegion

_INT32_MAX = int(np.iinfo(np.int32).max)


class CellSegmentationStrategy(BasePatchStrategy):
    mode = "cell"
    # Coordinates are generated in Y/X patch space and clipped to image bounds.
    # Therefore, fixed patch_size near borders can yield smaller PatchRegion extents.
    # If a fixed downstream array size is needed, use extraction padding/resizing.

    def __init__(self, config: CellConfig | None = None, mask_axes: str = "YX") -> None:
        self.config = config if config is not None else CellConfig()
        self.mask_axes = normalize_axes(mask_axes)

    @staticmethod
    def _label_connected_components(binary: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import label as scipy_label

            labeled, _ = scipy_label(binary)
            return labeled.astype(np.int32)
        except ImportError:
            # scipy is not installed; fall back to the built-in BFS implementation below
            pass

        # Manual 4-connected flood fill: scan every pixel in reading order,
        # and whenever an unlabeled foreground pixel is found, BFS outward
        # from it assigning the same new label to its whole connected blob.
        h, w = binary.shape
        labeled = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(h):
            for x in range(w):
                if not binary[y, x] or labeled[y, x] != 0:
                    continue
                current_label += 1
                queue = deque([(y, x)])
                labeled[y, x] = current_label
                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labeled[ny, nx] == 0:
                            labeled[ny, nx] = current_label
                            queue.append((ny, nx))
        return labeled

    def label_objects(self, selected: np.ndarray, *, image_id=None) -> tuple[np.ndarray, bool]:
        """The object labels of a 2D mask plane as int32, and whether they were computed.

        Decision D47: an integer mask is instance labels, used as given (a
        0/1 or 0/255 mask is one object). A boolean mask, or a float mask
        of 0.0/1.0, has no object ids: it is refused unless
        ``binary_mask_policy="connected_components"``, which labels its
        connected components with a warning (the second value is then
        True: the labels are not the mask's values).

        Raises
        ------
        LabelRangeError
            (a ``ValueError``) for negative labels or labels above the int32
            maximum.
        ValueError
            For a float
            mask with other values than 0.0/1.0, a binary mask without the
            opt-in, or another dtype.
        """
        where = f" (image {image_id})" if image_id is not None else ""
        if selected.dtype == bool or np.issubdtype(selected.dtype, np.floating):
            if selected.dtype != bool and not np.all(np.isin(np.unique(selected), [0.0, 1.0])):
                raise ValueError(
                    f"Non-binary float label arrays are ambiguous{where}. Float masks must contain only 0.0 "
                    "and 1.0. For labeled masks, use an integer dtype."
                )
            if self.config.binary_mask_policy != "connected_components":
                raise ValueError(
                    f"The mask{where} is binary ({selected.dtype}), so it has no object ids. Give integer "
                    "instance labels (each object its own value; mask.astype(np.uint8) makes the whole "
                    "foreground one object, label 1), or opt in to connected-component labelling with "
                    "cell.binary_mask_policy='connected_components'."
                )
            warnings.warn(
                f"The binary mask{where} is labelled by 4-connected components "
                "(cell.binary_mask_policy='connected_components'): its label_id values are computed, "
                "not taken from the mask.",
                UserWarning,
                stacklevel=3,
            )
            return self._label_connected_components(selected.astype(bool)), True
        if not np.issubdtype(selected.dtype, np.integer):
            raise ValueError(f"labels must hold integers (instance labels){where}; got dtype {selected.dtype}.")
        if selected.min() < 0:
            raise LabelRangeError(
                "labels must be non-negative. Found negative values in the label array.",
                label=int(selected.min()),
            )
        if int(selected.max()) > _INT32_MAX:
            raise LabelRangeError(
                f"The mask{where} holds label {int(selected.max())}, above the int32 maximum ({_INT32_MAX}) "
                "CellPatcher stores labels in; relabel the mask with smaller values.",
                label=int(selected.max()),
            )
        return selected.astype(np.int32, copy=False), False

    def _select_label_plane(self, labels):
        # A plain 2D label array needs no channel handling at all.
        arr = np.asarray(labels)
        if arr.ndim == 2:
            return arr
        if arr.ndim < 2:
            raise ValueError(f"labels must be at least 2D, got ndim={arr.ndim}.")
        # arr.ndim > 2: a channel (or other extra) axis must be resolved
        # down to 2D explicitly -- there's no safe default to guess here.
        if self.config.mask_channel is None:
            raise ValueError(
                "mask_channel is required for multidimensional labels. "
                "Set config.mask_channel and config.common.mask_axes (e.g. mask_axes='CYX')."
            )
        if len(self.mask_axes) != arr.ndim:
            raise ValueError(
                f"mask_axes length ({len(self.mask_axes)}) must match labels.ndim ({arr.ndim})."
            )
        c_axis = axis_index(self.mask_axes, "C")
        if c_axis is None:
            raise ValueError(
                f"mask_axes '{self.mask_axes}' does not contain a 'C' axis, "
                "which is required for multidimensional label channel selection."
            )
        c_size = arr.shape[c_axis]
        ch = int(self.config.mask_channel)
        if ch < 0 or ch >= c_size:
            raise ValueError(
                f"mask_channel {ch} is out of bounds for C axis of size {c_size} "
                f"(mask_axes='{self.mask_axes}')."
            )
        selected = np.take(arr, ch, axis=c_axis)
        if selected.ndim != 2:
            raise ValueError(
                f"After selecting channel {ch} from axis {c_axis}, "
                f"the result is {selected.ndim}D; expected 2D."
            )
        return selected

    def generate_regions(self, *, image_shape, labels=None, image_id=None, axes: str = "YX") -> list[PatchRegion]:
        image_h, image_w = _spatial_shape_from_axes(image_shape, axes=axes)

        if labels is None:
            raise ValueError("labels is required for cell mode.")

        selected = self._select_label_plane(labels)
        if selected.ndim != 2:
            raise ValueError("Selected labels must be 2D after channel selection.")
        if selected.shape != (image_h, image_w):
            raise ValueError(f"labels shape {selected.shape} must match image YX shape {(image_h, image_w)}.")
        if selected.size == 0:
            raise ValueError("labels must not be empty.")

        labeled, _ = self.label_objects(selected, image_id=image_id)

        object_ids = [int(v) for v in np.unique(labeled) if int(v) != 0]
        patches: list[PatchRegion] = []

        # One patch per labeled object; objects are skipped (not raised on)
        # when they fail the size/border criteria below, since a dataset
        # naturally mixes objects that should and shouldn't be patched.
        for label_id in sorted(object_ids):
            ys, xs = np.where(labeled == label_id)
            if ys.size == 0:
                continue

            area_px = int(ys.size)
            if area_px < self.config.min_area:
                continue
            if self.config.max_area is not None and area_px > self.config.max_area:
                continue

            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            obj_h = y_max - y_min + 1
            obj_w = x_max - x_min + 1

            touches_border = y_min == 0 or x_min == 0 or y_max == image_h - 1 or x_max == image_w - 1
            if self.config.exclude_border and touches_border:
                continue

            # 'auto' sizes the patch to just cover the object's own bounding
            # box (a square large enough for its longer side); otherwise
            # every object gets the same fixed patch size.
            if self.config.patch_size == "auto":
                patch_size = int(max(obj_h, obj_w))
            elif isinstance(self.config.patch_size, int):
                patch_size = int(self.config.patch_size)
            else:
                raise ValueError("patch_size must be 'auto' or a positive integer.")
            if patch_size <= 0:
                raise ValueError("patch_size must be > 0.")

            if self.config.center_on == "centroid":
                center_y = float(np.mean(ys))
                center_x = float(np.mean(xs))
            elif self.config.center_on == "bbox_center":
                center_y = (y_min + y_max + 1) / 2.0
                center_x = (x_min + x_max + 1) / 2.0
            else:
                raise ValueError(f"Unsupported center_on: {self.config.center_on}")

            # Build a patch_size x patch_size square centered on the object,
            # then clip it to the image bounds -- this is why a fixed
            # patch_size can still yield a smaller region near an edge (see
            # the class-level note above).
            y0 = int(round(center_y - patch_size / 2.0))
            x0 = int(round(center_x - patch_size / 2.0))
            y1 = y0 + patch_size
            x1 = x0 + patch_size

            requested = {"requested_y0": y0, "requested_y1": y1, "requested_x0": x0, "requested_x1": x1}
            y0 = max(0, y0)
            x0 = max(0, x0)
            y1 = min(image_h, y1)
            x1 = min(image_w, x1)

            if y1 <= y0 or x1 <= x0:
                continue
            own = float(np.count_nonzero(labeled[y0:y1, x0:x1] == label_id)) / ((y1 - y0) * (x1 - x0))

            patch_id = f"label{label_id}"
            patches.append(
                PatchRegion(
                    patch_id=patch_id,
                    image_id=image_id,
                    mode=self.mode,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    label_id=label_id,
                    center_y=center_y,
                    center_x=center_x,
                    touches_border=touches_border,
                    area_px=area_px,
                    **requested,
                    patch_touches_image_border=y0 == 0 or x0 == 0 or y1 == image_h or x1 == image_w,
                    # touches_border above is this one: the object's own pixels.
                    object_touches_image_border=touches_border,
                    mask_fractions={"own_object": own},
                )
            )

        return sorted(patches, key=lambda p: int(p.label_id or 0))
