from dataclasses import dataclass, field


@dataclass
class CommonConfig:
    image_axes: str = "YX"
    mask_axes: str = "YX"
    output_dir: str | None = None
    extract_arrays: bool = False
    # Stamped onto every PatchRegion/metadata-CSV row for this run when set.
    # These are the MorphoBricks shared-CSV-contract `condition`/`repeat`/
    # `technical_repeat` columns (see AGENTS.md §5). run_patching()/
    # run_patching_pipeline() never infer them; set them explicitly, or use
    # orchestration.run_patching_condition_repeat_tree to derive them per
    # leaf from a <condition>/<repeat>[/<technical_repeat>]/ folder tree.
    # technical_repeat is the well/dish/coverslip within the biological
    # repeat; per-image image_metadata["technical_repeat"] overrides it.
    condition: str | None = None
    repeat: str | None = None
    technical_repeat: str | None = None


@dataclass
class CellConfig:
    patch_size: int | str = "auto"
    mask_channel: int | None = None
    center_on: str = "bbox_center"
    exclude_border: bool = True
    min_area: int = 1
    max_area: int | None = None
    binary_mask_policy: str = "connected_components"

    def __post_init__(self) -> None:
        if self.center_on not in {"centroid", "bbox_center"}:
            raise ValueError("center_on must be either 'centroid' or 'bbox_center'.")
        if self.binary_mask_policy not in {"connected_components", "single_object"}:
            raise ValueError("binary_mask_policy must be either 'connected_components' or 'single_object'.")
        if self.min_area < 1:
            raise ValueError("min_area must be >= 1.")
        if self.max_area is not None and self.max_area < self.min_area:
            raise ValueError("max_area must be >= min_area when provided.")


@dataclass
class RandomConfig:
    patch_size: int = 128
    n_patches: int = 100
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if self.n_patches <= 0:
            raise ValueError("n_patches must be > 0.")


@dataclass
class GridConfig:
    tile_size: int = 128
    overlap: float = 0.0
    include_partial: bool = False

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be > 0.")
        if not (0 <= self.overlap < 1):
            raise ValueError("overlap must satisfy 0 <= overlap < 1.")


@dataclass
class FilterConfig:
    min_foreground_fraction: float | None = None
    foreground_threshold: float | str | None = None
    min_mean_intensity: float | None = None

    def __post_init__(self) -> None:
        if self.min_foreground_fraction is not None:
            value = float(self.min_foreground_fraction)
            if value < 0 or value > 1:
                raise ValueError("min_foreground_fraction must be in [0, 1].")
        if self.min_mean_intensity is not None and not isinstance(self.min_mean_intensity, (int, float)):
            raise ValueError("min_mean_intensity must be a scalar number or None.")
        if self.foreground_threshold is not None and not isinstance(self.foreground_threshold, (int, float, str)):
            raise ValueError("foreground_threshold must be None, a scalar number, or a string.")


@dataclass
class ExportConfig:
    enabled: bool = False
    save_metadata_csv: bool = True
    image_format: str = "tif"
    # Write run_config.json next to the exported patches. Disabled per folder
    # by run_patching_condition_repeat_tree(combine_metadata=True), which
    # writes one run_config.json for the whole tree instead.
    save_run_config: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool.")
        if not isinstance(self.save_metadata_csv, bool):
            raise ValueError("save_metadata_csv must be a bool.")
        if not isinstance(self.save_run_config, bool):
            raise ValueError("save_run_config must be a bool.")
        if str(self.image_format).lower() not in {"tif", "tiff", "npy"}:
            raise ValueError("image_format must be one of: 'tif', 'tiff', 'npy'.")


@dataclass
class ExtractionConfig:
    pad_mode: str = "clip"
    constant_values: float = 0.0
    output_size: int | tuple[int, int] | None = None
    resize_order: int = 1
    apply_mask: bool = False
    mask_fill_value: float = 0.0

    def __post_init__(self) -> None:
        if self.pad_mode not in {"clip", "constant"}:
            raise ValueError("pad_mode must be either 'clip' or 'constant'.")

        # output_size accepts three shapes: None (no resize), a single int
        # (square output), or an explicit (height, width) pair.
        if self.output_size is not None:
            if isinstance(self.output_size, int):
                if self.output_size <= 0:
                    raise ValueError("output_size int must be > 0.")
            elif isinstance(self.output_size, (tuple, list)) and len(self.output_size) == 2:
                h, w = int(self.output_size[0]), int(self.output_size[1])
                if h <= 0 or w <= 0:
                    raise ValueError("output_size tuple/list values must be > 0.")
            else:
                raise ValueError("output_size must be None, a positive int, or tuple/list of two positive ints.")

        if not isinstance(self.resize_order, int) or self.resize_order < 0:
            raise ValueError("resize_order must be an int >= 0.")
        if not isinstance(self.apply_mask, bool):
            raise ValueError("apply_mask must be a bool.")
        if not isinstance(self.mask_fill_value, (int, float)):
            raise ValueError("mask_fill_value must be a scalar number.")


@dataclass
class MatchConfig:
    """Preferred image↔mask filename matching options.

    ``validate_image_and_mask_folders`` is a standalone helper that accepts
    matching options explicitly; it does not automatically consume
    ``CellPatcherConfig.matching``.

    The package-backed notebook/workflows should pass these fields through when
    calling folder-validation helpers.
    """
    strategy: str = "exact_or_suffix"
    mask_suffixes: tuple[str, ...] = ("_mask", "_labels", "_label", "_seg", "_segmentation")
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in {"exact", "exact_or_suffix", "substring"}:
            raise ValueError("strategy must be one of: 'exact', 'exact_or_suffix', 'substring'.")
        if not isinstance(self.mask_suffixes, (tuple, list)) or not all(
            isinstance(value, str) for value in self.mask_suffixes
        ):
            raise ValueError("mask_suffixes must be a tuple/list of strings.")
        if not isinstance(self.case_sensitive, bool):
            raise ValueError("case_sensitive must be a bool.")


@dataclass
class CellPatcherConfig:
    """Top-level configuration for a CellPatcher run.

    All settings are plain dataclass fields, so the simplest way to customise a
    run is to build a config and set attributes directly — you get IDE
    autocomplete and immediate validation::

        from cellpatcher import CellPatcherConfig

        cfg = CellPatcherConfig(mode="grid")
        cfg.grid.tile_size = 64
        cfg.grid.overlap = 0.1
        cfg.common.output_dir = "./patches"
        cfg.export.enabled = True

    For batch runs where you want to keep a shared base config immutable, use
    :func:`~cellpatcher.apply_overrides` to produce independent copies::

        from cellpatcher import CellPatcherConfig, apply_overrides

        base = CellPatcherConfig(mode="grid")
        for tile_size in [64, 128, 256]:
            cfg = apply_overrides(base, {"grid.tile_size": tile_size})
            run_patching_pipeline("./images", config=cfg)

    Parameters
    ----------
    mode:
        Patch strategy to use.  One of ``'cell'``, ``'random'``, ``'grid'``,
        ``'whole_fov'``.
    common:
        Shared settings (axes, output directory, array extraction).
    cell:
        Parameters for ``mode='cell'`` (cell-centred patching).
    random:
        Parameters for ``mode='random'`` (randomly sampled patches).
    grid:
        Parameters for ``mode='grid'`` (regular tiled grid).
    filters:
        Foreground-intensity filters applied after region generation.
    export:
        File export settings (format, metadata CSV).
    extraction:
        Array extraction settings (padding, resizing, mask application).
    matching:
        Image↔mask filename matching strategy, used by
        :func:`~cellpatcher.run_patching_pipeline` and
        :func:`~cellpatcher.validate_image_and_mask_folders`.
    """

    mode: str = "cell"
    common: CommonConfig = field(default_factory=CommonConfig)
    cell: CellConfig = field(default_factory=CellConfig)
    random: RandomConfig = field(default_factory=RandomConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)

    def __post_init__(self) -> None:
        valid_modes = {"cell", "random", "whole_fov", "grid"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}.")
