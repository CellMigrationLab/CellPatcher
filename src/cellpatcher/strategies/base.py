from abc import ABC, abstractmethod

from cellpatcher.axes import normalize_axes, require_spatial_axes
from cellpatcher.types import PatchRegion


def _spatial_shape_from_axes(image_shape, axes: str = "YX") -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must have at least 2 dimensions.")
    normalized_axes = normalize_axes(axes, ndim=len(image_shape))
    y_idx, x_idx = require_spatial_axes(normalized_axes)
    height = int(image_shape[y_idx])
    width = int(image_shape[x_idx])
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got Y={height}, X={width}.")
    return height, width


class BasePatchStrategy(ABC):
    mode: str

    @abstractmethod
    def generate_regions(self, *, image_shape, labels=None, image_id=None, axes: str = "YX") -> list[PatchRegion]:
        ...
