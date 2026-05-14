from cellpatcher.strategies.base import BasePatchStrategy
from cellpatcher.types import PatchRegion


class WholeFOVStrategy(BasePatchStrategy):
    mode = "whole_fov"

    def __init__(self, config=None) -> None:
        self.config = config

    def generate_regions(self, *, image_shape, labels=None, image_id=None) -> list[PatchRegion]:
        if len(image_shape) < 2:
            raise ValueError("image_shape must have at least 2 dimensions (Y, X).")

        y_size = int(image_shape[-2])
        x_size = int(image_shape[-1])
        if y_size <= 0 or x_size <= 0:
            raise ValueError(f"Spatial dimensions must be positive, got Y={y_size}, X={x_size}.")

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
