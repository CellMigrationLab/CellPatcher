from cellpatcher.config import WholeFOVConfig
from cellpatcher.strategies.base import BasePatchStrategy, _spatial_shape_from_axes
from cellpatcher.types import PatchRegion


def square_cover_starts(long_side: int, short_side: int) -> list[int]:
    """Start positions along the long side of the squares (of the short side) covering an image.

    ⌈long ÷ short⌉ squares, evenly spaced: the first at 0, the last flush
    with the far edge, the others at the rounded even steps between.
    """
    count = -(-long_side // short_side)
    if count == 1:
        return [0]
    span = long_side - short_side
    return [round(index * span / (count - 1)) for index in range(count)]


class WholeFOVStrategy(BasePatchStrategy):
    mode = "whole_fov"

    def __init__(self, config: WholeFOVConfig | None = None) -> None:
        self.config = config if config is not None else WholeFOVConfig()

    def generate_regions(self, *, image_shape, labels=None, image_id=None, axes: str = "YX") -> list[PatchRegion]:
        y_size, x_size = _spatial_shape_from_axes(image_shape, axes=axes)

        patch_id = "whole_fov"
        # A single region spanning the entire image; touches_border=True
        # unconditionally since the region's edges are the image's edges by
        # definition.
        regions = [
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
                requested_y0=0,
                requested_y1=y_size,
                requested_x0=0,
                requested_x1=x_size,
                patch_touches_image_border=True,
            )
        ]
        if self.config.square_cover:
            regions += self._squares(y_size, x_size, image_id)
        return regions

    @staticmethod
    def _squares(y_size: int, x_size: int, image_id) -> list[PatchRegion]:
        """The square cover of a ``y_size`` x ``x_size`` image (:class:`WholeFOVConfig`)."""
        side = min(y_size, x_size)
        squares = []
        for index, start in enumerate(square_cover_starts(max(y_size, x_size), side)):
            y0, x0 = (start, 0) if y_size > x_size else (0, start)
            squares.append(
                PatchRegion(
                    # parent_id and the final patch_id are set with the ids
                    # (cellpatcher.ids.assign_patch_ids).
                    patch_id=f"whole_fov/sq{index}",
                    image_id=image_id,
                    mode="whole_fov_square",
                    y0=y0,
                    y1=y0 + side,
                    x0=x0,
                    x1=x0 + side,
                    center_y=y0 + side / 2,
                    center_x=x0 + side / 2,
                    area_px=side * side,
                    touches_border=True,
                    requested_y0=y0,
                    requested_y1=y0 + side,
                    requested_x0=x0,
                    requested_x1=x0 + side,
                    patch_touches_image_border=True,
                    square_index=index,
                )
            )
        return squares
