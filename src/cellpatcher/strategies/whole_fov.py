from cellpatcher.strategies.base import BasePatchStrategy, _spatial_shape_from_axes
from cellpatcher.types import PatchRegion


class WholeFOVStrategy(BasePatchStrategy):
    mode = "whole_fov"

    def generate_regions(self, *, image_shape, labels=None, image_id=None, axes: str = "YX") -> list[PatchRegion]:
        y_size, x_size = _spatial_shape_from_axes(image_shape, axes=axes)

        patch_id = f"{image_id}_whole_fov_0" if image_id is not None else "whole_fov_0"
        return [
            PatchRegion(
                patch_id=patch_id,
                image_id=image_id,
                mode=self.mode,
                y0=0,
                y1=y_size,
                x0=0,
                x1=x_size,
                center_y=y_size / 2,
                center_x=x_size / 2,
                area_px=y_size * x_size,
                touches_border=True,
            )
        ]
