from dataclasses import asdict, dataclass

from cellpatcher._deprecation import accept_deprecated_repeat


@accept_deprecated_repeat
@dataclass
class PatchRegion:
    """Coordinate-first patch descriptor used across the backend.

    Active spatial coordinates in the current backend are 2D ``y0/y1/x0/x1``.

    Multidimensional-workflow metadata (populated when a plane was selected
    via ``run_patching(..., image_metadata=...)``; ``None`` otherwise):
    - ``z0/z1``, ``t``, ``c``

    Reserved metadata fields for future multidimensional workflows:
    - ``center_z``

    Reserved metadata fields for future rejected-region diagnostics:
    - ``valid``, ``reason``

    Current pipeline behavior:
    - only accepted patches are returned;
    - ``valid`` is always ``True`` and ``reason`` is always ``None`` for every
      patch currently returned by the pipeline — these fields are reserved for
      future rejected-patch reporting and are not yet populated.

    Ecosystem/provenance fields (populated by ``run_patching_pipeline``/
    ``export_patches`` when the corresponding information is available;
    ``None`` when a caller uses the low-level ``run_patching`` directly
    without supplying it). ``condition``/``biological_repeat``/``technical_repeat``/
    ``image_file``/``mask_file``/``patch_file`` are the shared MorphoBricks CSV columns
    (see AGENTS.md §5); ``output_height``/``output_width``/``config_file``
    exist so a single CSV row is enough to deterministically reconstruct the
    exported patch array without re-deriving it from the run's config file.
    - ``condition``, ``biological_repeat``: experimental group / biological
      replicate label, from ``CellPatcherConfig.common``. ``repeat`` is a
      deprecated alias of ``biological_repeat`` (constructor keyword and
      attribute; each use emits a ``DeprecationWarning``).
    - ``technical_repeat``: technical replicate within the biological repeat
      (a well, a dish, a coverslip), from ``CellPatcherConfig.common`` or
      per image from ``image_metadata``; never a biological repeat. Tree runs
      stamp ``SINGLE_TECHNICAL_REPEAT_LABEL`` (``"single"``) when a biological
      repeat folder holds its images directly.
    - ``position_id``: optional stable spatial acquisition-position identifier.
      Use the same value for all Z/C/T planes from one XY position.
    - ``image_file``, ``mask_file``: source filenames, from folder-based runs.
    - ``patch_file``: exported patch filename, from ``export_patches``.
    - ``output_height``, ``output_width``: final exported array shape, which
      can differ from ``height``/``width`` (the raw region size) when
      ``ExtractionConfig.output_size`` resizes the patch.
    - ``config_file``: path to the full run config JSON saved alongside the
      metadata CSV, capturing every parameter (strategy, seed, extraction,
      matching) needed to reproduce this run's coordinates deterministically.
    - ``source_relative_path``: the source image's path relative to the
      experiment root, with ``/`` separators, from its real location
      (``control/img.tif`` for an image directly in ``control/``; never the
      ``single`` output folders). Links the patch to the preprocessing
      brick's ``plane_id``. Filled by folder and tree runs, or per image from
      ``image_metadata``; ``None`` otherwise.
    - ``normalization_method``, ``normalization_scope``, ``normalization_group``,
      ``normalization_offset``, ``normalization_scale``, ``normalization_exact``:
      the normalization applied to the patch (``config.normalization``):
      ``(pixels - offset) / scale`` per channel; offsets and scales are
      ``;``-separated per channel. ``None`` when normalization is off.
    """
    patch_id: str
    image_id: str | None = None
    mode: str | None = None
    condition: str | None = None
    biological_repeat: str | None = None
    technical_repeat: str | None = None
    position_id: str | None = None
    image_file: str | None = None
    mask_file: str | None = None
    y0: int = 0
    y1: int = 0
    x0: int = 0
    x1: int = 0
    # Multidimensional workflows (Z-stack, time series, channel selection).
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
    patch_file: str | None = None
    output_height: int | None = None
    output_width: int | None = None
    config_file: str | None = None
    # Reserved for future rejected-patch reporting (e.g. background suppression reason).
    valid: bool = True
    reason: str | None = None
    # Appended after every earlier field so existing CSV columns keep their
    # order (docs/normalization_contract.md §10).
    source_relative_path: str | None = None
    # How the patch was normalized (docs/normalization_contract.md §10); all
    # None when normalization is off. Offsets and scales are one value per
    # channel of the patch, in channel order, joined by ";" (full precision).
    normalization_method: str | None = None
    normalization_scope: str | None = None
    normalization_group: str | None = None
    normalization_offset: str | None = None
    normalization_scale: str | None = None
    normalization_exact: bool | None = None

    def __post_init__(self) -> None:
        # y0/x0 are always required (defaulted to 0), so their ranges are
        # validated unconditionally.
        if self.y1 <= self.y0:
            raise ValueError(f"Invalid y coordinates: y1 ({self.y1}) must be greater than y0 ({self.y0}).")
        if self.x1 <= self.x0:
            raise ValueError(f"Invalid x coordinates: x1 ({self.x1}) must be greater than x0 ({self.x0}).")

        # z0/z1 are optional (a patch may not carry Z information at all),
        # but if either is set both must be, and as a half-open range like
        # y0/y1 and x0/x1.
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
        data = asdict(self)
        data["height"] = self.height
        data["width"] = self.width
        return data
