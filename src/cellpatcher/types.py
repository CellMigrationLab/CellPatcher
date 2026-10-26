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
      experiment root, with ``/`` separators, from where it sits in the
      folder tree (``control/img.tif`` for an image directly in ``control/``;
      never the ``single`` output folders; a linked image or folder keeps the
      link's path, as in the preprocessing brick). Links the patch to the preprocessing
      brick's ``plane_id``. Filled by folder and tree runs, or per image from
      ``image_metadata``; ``None`` otherwise.
    - ``normalization_method``, ``normalization_scope``, ``normalization_group``,
      ``normalization_offset``, ``normalization_scale``, ``normalization_exact``:
      the normalization applied to the patch (``config.normalization``):
      ``(pixels - offset) / scale`` per channel; offsets and scales are
      ``;``-separated per channel. ``None`` when normalization is off.
    - ``image_height``, ``image_width``: the Y/X size of the source image
      (plane) the rectangle ``y0/y1/x0/x1`` lies in; set by ``run_patching``
      for every patch, whether or not arrays are extracted.
    - ``pad_top``, ``pad_bottom``, ``pad_left``, ``pad_right``: rows/columns
      of ``extraction.constant_values`` added on each side of the patch where
      its rectangle extends past the image (``extraction.pad_mode="constant"``),
      before any resizing; 0 when nothing was padded (``"clip"`` never pads).
    - ``resized``: whether ``extraction.output_size`` changed the patch's Y/X
      shape (``False`` when no output size is set, or when the patch already
      had that size and resizing left it unchanged).
    - ``output_axes``: the axes of the delivered/exported array, in order
      (e.g. ``"CYX"``; ``"YX"`` for a split channel; a kept Z or T axis stays).
      Exported TIFF/NPY files carry no axes of their own.
    The padding, ``resized`` and ``output_axes`` fields, like
    ``output_height``/``output_width``, are ``None`` unless arrays were
    extracted (``common.extract_arrays=True`` or ``export.enabled=True``).
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
    # Geometry of the source image and of the delivered/exported array, so a
    # reader of the exported files needs neither the source image nor
    # CellPatcher's rules to know what each array holds. Appended after every
    # earlier field so existing CSV columns keep their order.
    image_height: int | None = None
    image_width: int | None = None
    pad_top: int | None = None
    pad_bottom: int | None = None
    pad_left: int | None = None
    pad_right: int | None = None
    resized: bool | None = None
    output_axes: str | None = None

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


# Columns of the per-image table (``images.csv``), written beside every
# ``patches.csv``. The identity columns use exactly the names ``patches.csv``
# uses (AGENTS.md §5), so the two tables join on them.
IMAGE_TABLE_COLUMNS = (
    "condition",
    "biological_repeat",
    "technical_repeat",
    "image_id",
    "image_file",
    "mask_file",
    "source_relative_path",
    "image_height",
    "image_width",
    "n_patches",
    "status",
    "reason",
    # Appended after the first release of the table, so earlier columns keep
    # their positions: the image's position (AGENTS.md §5, empty when neither
    # supplied nor defaulted) and its counts before n_patches, which tell a
    # mode that generated no region from filters or normalization removing all.
    "position_id",
    "n_regions",
    "n_kept",
    "n_normalized",
)


@dataclass
class ImageRecord:
    """One input image of a run, and what became of it: a row of ``images.csv``.

    Every image the run considered gets one record, including images that
    gave no patch, so a reader of ``patches.csv`` can tell an image without
    patches from one that was never patched. The counts follow the image's
    regions through the pipeline: ``n_regions`` generated by the patch mode,
    ``n_kept`` left by the filters, ``n_normalized`` left by normalization
    (decision D8 discards the patches that cannot be normalized; equal to
    ``n_kept`` without normalization) and ``n_patches`` rows written to
    ``patches.csv`` (one per channel with ``split_channels``).
    ``position_id`` is the one every patch of the image carries: supplied
    via ``image_metadata``, the ``split_channels`` default, or None.

    ``status`` (see :attr:`status`) is ``"processed"`` (at least one patch),
    ``"no_patches"`` (patched, none kept; :attr:`reason` says at which step)
    or ``"skipped"`` (``skip_reason`` set: the image had nothing to patch).
    """

    image_id: str
    condition: str | None = None
    biological_repeat: str | None = None
    technical_repeat: str | None = None
    image_file: str | None = None
    mask_file: str | None = None
    source_relative_path: str | None = None
    image_height: int | None = None
    image_width: int | None = None
    n_regions: int = 0
    n_kept: int = 0
    n_normalized: int = 0
    n_patches: int = 0
    skip_reason: str | None = None
    position_id: str | None = None

    @property
    def status(self) -> str:
        if self.skip_reason is not None:
            return "skipped"
        return "processed" if self.n_patches > 0 else "no_patches"

    @property
    def reason(self) -> str | None:
        """Why the image has no patch; ``None`` when it has patches."""
        if self.skip_reason is not None:
            return self.skip_reason
        if self.n_patches > 0:
            return None
        if self.n_regions == 0:
            return "the patch mode generated no region in this image"
        if self.n_kept == 0:
            return f"the filters removed all {self.n_regions} region(s)"
        if self.n_normalized == 0:
            return (
                f"normalization discarded all {self.n_kept} patch(es): "
                "they cannot be normalized from their own pixels"
            )
        return "no patch was kept"

    def to_dict(self) -> dict:
        """The ``images.csv`` row: :data:`IMAGE_TABLE_COLUMNS` only."""
        return {column: getattr(self, column) for column in IMAGE_TABLE_COLUMNS}
