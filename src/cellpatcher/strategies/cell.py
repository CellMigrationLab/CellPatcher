from collections import deque

import numpy as np

from cellpatcher.axes import axis_index, normalize_axes
from cellpatcher.config import CellConfig
from cellpatcher.strategies.base import BasePatchStrategy
from cellpatcher.types import PatchRegion


class CellSegmentationStrategy(BasePatchStrategy):
    mode = "cell"

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

    def _select_label_plane(self, labels):
        arr = np.asarray(labels)
        if arr.ndim == 2:
            return arr
        if arr.ndim < 2:
            raise ValueError(f"labels must be at least 2D, got ndim={arr.ndim}.")
        # arr.ndim > 2
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

    def generate_regions(self, *, image_shape, labels=None, image_id=None) -> list[PatchRegion]:
        if len(image_shape) < 2:
            raise ValueError("image_shape must have at least 2 dimensions (Y, X).")

        image_h = int(image_shape[-2])
        image_w = int(image_shape[-1])
        if image_h <= 0 or image_w <= 0:
            raise ValueError(f"Spatial dimensions must be positive, got Y={image_h}, X={image_w}.")

        if labels is None:
            raise ValueError("labels is required for cell mode.")

        selected = self._select_label_plane(labels)
        if selected.ndim != 2:
            raise ValueError("Selected labels must be 2D after channel selection.")
        if selected.shape != (image_h, image_w):
            raise ValueError(f"labels shape {selected.shape} must match image YX shape {(image_h, image_w)}.")
        if selected.size == 0:
            raise ValueError("labels must not be empty.")

        unique_values = np.unique(selected)
        foreground_mask = selected.astype(bool)

        if np.any(unique_values < 0):
            raise ValueError(
                "labels must be non-negative. Found negative values in the label array."
            )

        if np.issubdtype(selected.dtype, np.floating):
            if np.any((unique_values != 0.0) & (unique_values != 1.0)):
                raise ValueError(
                    "Non-binary float label arrays are ambiguous. "
                    "Float masks must contain only 0.0 and 1.0. "
                    "For labeled masks, use an integer dtype."
                )

        if np.all(np.isin(unique_values, [0, 1, False, True])):
            if self.config.binary_mask_policy == "connected_components":
                labeled = self._label_connected_components(foreground_mask)
            elif self.config.binary_mask_policy == "single_object":
                labeled = foreground_mask.astype(np.int32)
            else:
                raise ValueError(f"Unsupported binary_mask_policy: {self.config.binary_mask_policy}")
        else:
            labeled = selected.astype(np.int32)

        object_ids = [int(v) for v in np.unique(labeled) if int(v) != 0]
        patches: list[PatchRegion] = []

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

            y0 = int(round(center_y - patch_size / 2.0))
            x0 = int(round(center_x - patch_size / 2.0))
            y1 = y0 + patch_size
            x1 = x0 + patch_size

            y0 = max(0, y0)
            x0 = max(0, x0)
            y1 = min(image_h, y1)
            x1 = min(image_w, x1)

            if y1 <= y0 or x1 <= x0:
                continue

            patch_id = f"{image_id}_cell_{label_id:05d}" if image_id is not None else f"cell_{label_id:05d}"
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
                )
            )

        return sorted(patches, key=lambda p: int(p.label_id or 0))
