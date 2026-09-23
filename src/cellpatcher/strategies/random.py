import numpy as np

from cellpatcher.config import RandomConfig
from cellpatcher.strategies.base import BasePatchStrategy, _spatial_shape_from_axes
from cellpatcher.types import PatchRegion


class RandomPatchStrategy(BasePatchStrategy):
    mode = "random"
    # Sampling is with replacement: duplicate coordinates are allowed.
    # seed=None uses NumPy's nondeterministic RNG initialization.

    def __init__(self, config: RandomConfig | None = None) -> None:
        self.config = config if config is not None else RandomConfig()

    def generate_regions(self, *, image_shape, labels=None, image_id=None, axes: str = "YX") -> list[PatchRegion]:
        height, width = _spatial_shape_from_axes(image_shape, axes=axes)
        patch_size = int(self.config.patch_size)
        n_patches = int(self.config.n_patches)
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if n_patches <= 0:
            raise ValueError("n_patches must be > 0.")
        if patch_size > height or patch_size > width:
            raise ValueError(
                f"patch_size ({patch_size}) must be <= image spatial dimensions Y={height}, X={width}."
            )

        rng = np.random.default_rng(self.config.seed)
        # The valid range for a top-left corner so the full patch stays
        # inside the image.
        y_max = height - patch_size
        x_max = width - patch_size

        # An optional mask restricts sampling to only the region(s) it
        # covers; without one, any position in [0, y_max] x [0, x_max] is
        # accepted immediately (see the "if mask is not None" check below).
        mask = None
        if labels is not None:
            mask = np.asarray(labels).astype(bool)
            if mask.ndim != 2:
                raise ValueError("labels sampling mask must be 2D for random mode.")
            if mask.shape != (height, width):
                raise ValueError(f"labels shape {mask.shape} must match image YX shape {(height, width)}.")

        patches: list[PatchRegion] = []
        attempts = 0
        # A sparse mask can make random sampling take many tries per hit;
        # cap the attempts so a bad mask fails fast with a clear error
        # instead of looping indefinitely.
        max_attempts = max(1000, n_patches * 50)

        while len(patches) < n_patches:
            if mask is not None and attempts >= max_attempts:
                raise ValueError(
                    "Could not sample enough valid random patches from the provided mask. "
                    f"Requested {n_patches}, collected {len(patches)} after {attempts} attempts."
                )

            y0 = int(rng.integers(0, y_max + 1))
            x0 = int(rng.integers(0, x_max + 1))
            y1 = y0 + patch_size
            x1 = x0 + patch_size

            attempts += 1
            # Reject (without counting toward n_patches) any draw that
            # doesn't overlap the sampling mask at all.
            if mask is not None and not np.any(mask[y0:y1, x0:x1]):
                continue

            idx = len(patches)
            patch_id = f"{image_id}_random_{idx:05d}" if image_id is not None else f"random_{idx:05d}"
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
                    area_px=patch_size * patch_size,
                    touches_border=touches_border,
                )
            )

        return patches
