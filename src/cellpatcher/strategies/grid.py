from cellpatcher.config import GridConfig
from cellpatcher.strategies.base import BasePatchStrategy, _spatial_shape_from_axes
from cellpatcher.types import PatchRegion


class GridStrategy(BasePatchStrategy):
    mode = "grid"

    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config if config is not None else GridConfig()

    @staticmethod
    def _starts(
        size: int, tile_size: int, stride: int, include_partial: bool, partial_mode: str = "flush"
    ) -> list[int]:
        # A single tile already covers (or exceeds) the whole dimension.
        if include_partial and tile_size >= size:
            return [0]

        if not include_partial:
            # Only start positions whose full tile fits inside the image;
            # anything past `size - tile_size` would run off the edge.
            return list(range(0, size - tile_size + 1, stride))

        # include_partial=True: step across the whole dimension. The tiles
        # past `size - tile_size` are clipped by the far edge.
        starts = list(range(0, size, stride))
        if partial_mode == "clipped":
            # Clipped edge tiles only: no extra full tile flush with the far
            # edge, so with overlap=0 the tiles partition the dimension.
            return starts

        # partial_mode="flush": also make sure the full tile flush with the
        # far edge is always included even if the regular stride would skip
        # over it (it overlaps its neighbours).
        final_start = max(0, size - tile_size)
        if final_start not in starts:
            starts.append(final_start)
        return sorted(set(starts))

    def generate_regions(self, *, image_shape, labels=None, image_id=None, axes: str = "YX") -> list[PatchRegion]:
        height, width = _spatial_shape_from_axes(image_shape, axes=axes)

        tile_size = int(self.config.tile_size)
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0.")

        # overlap=0 gives stride == tile_size (tiles touch); overlap closer
        # to 1 shrinks the stride so consecutive tiles increasingly overlap.
        stride = max(1, round(tile_size * (1 - self.config.overlap)))

        if not self.config.include_partial and (tile_size > height or tile_size > width):
            return []

        self.config.check_partial_mode()
        include_partial = self.config.include_partial
        partial_mode = self.config.partial_mode
        y_starts = self._starts(height, tile_size, stride, include_partial, partial_mode)
        x_starts = self._starts(width, tile_size, stride, include_partial, partial_mode)

        patches: list[PatchRegion] = []
        for y0 in y_starts:
            y1 = min(y0 + tile_size, height)
            # _starts can still produce a start whose tile would be clipped
            # short by the image edge; skip those explicitly unless partial
            # tiles were requested.
            if not self.config.include_partial and (y1 - y0) != tile_size:
                continue

            for x0 in x_starts:
                x1 = min(x0 + tile_size, width)
                if not self.config.include_partial and (x1 - x0) != tile_size:
                    continue

                tile_h = y1 - y0
                tile_w = x1 - x0
                patch_id = f"grid-y{y0}-x{x0}-h{tile_h}-w{tile_w}"
                touches_border = y0 == 0 or x0 == 0 or y1 == height or x1 == width

                patches.append(
                    PatchRegion(
                        patch_id=patch_id,
                        image_id=image_id,
                        mode=self.mode,
                        y0=y0,
                        y1=y1,
                        x0=x0,
                        x1=x1,
                        center_y=(y0 + y1) / 2,
                        center_x=(x0 + x1) / 2,
                        area_px=tile_h * tile_w,
                        touches_border=touches_border,
                        # A partial tile is the full tile clipped to the image.
                        requested_y0=y0,
                        requested_y1=y0 + tile_size,
                        requested_x0=x0,
                        requested_x1=x0 + tile_size,
                        patch_touches_image_border=touches_border,
                    )
                )

        return patches
