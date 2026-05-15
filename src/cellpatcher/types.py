from dataclasses import asdict, dataclass


@dataclass
class PatchRegion:
    """Coordinate-first patch descriptor used across the backend.

    Active spatial coordinates in the current backend are 2D ``y0/y1/x0/x1``.

    Reserved metadata fields for future multidimensional workflows:
    - ``z0/z1``, ``t``, ``c``, ``center_z``

    Reserved metadata fields for future rejected-region diagnostics:
    - ``valid``, ``reason``

    Current pipeline behavior:
    - only accepted patches are returned;
    - rejected-patch diagnostics are not populated yet.
    """
    patch_id: str
    image_id: str | None = None
    mode: str | None = None
    y0: int = 0
    y1: int = 0
    x0: int = 0
    x1: int = 0
    # Reserved for future multidimensional workflows (Z-stack, time series, channel selection).
    z0: int | None = None
    z1: int | None = None
    t: int | None = None
    c: int | None = None
    label_id: int | None = None
    center_y: float | None = None
    center_x: float | None = None
    center_z: float | None = None
    touches_border: bool = False
    area_px: int | None = None
    # Reserved for future rejected-patch reporting (e.g. background suppression reason).
    valid: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.y1 <= self.y0:
            raise ValueError(f"Invalid y coordinates: y1 ({self.y1}) must be greater than y0 ({self.y0}).")
        if self.x1 <= self.x0:
            raise ValueError(f"Invalid x coordinates: x1 ({self.x1}) must be greater than x0 ({self.x0}).")

        if (self.z0 is None) != (self.z1 is None):
            raise ValueError("z0 and z1 must both be provided together for 3D regions.")
        if self.z0 is not None and self.z1 is not None and self.z1 <= self.z0:
            raise ValueError(f"Invalid z coordinates: z1 ({self.z1}) must be greater than z0 ({self.z0}).")

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def has_z(self) -> bool:
        return self.z0 is not None and self.z1 is not None

    def to_dict(self) -> dict:
        return asdict(self)
