from dataclasses import dataclass, field


@dataclass
class CommonConfig:
    image_axes: str = "YX"
    mask_axes: str = "YX"
    preserve_non_spatial_axes: bool = True
    output_dir: str | None = None
    extract_arrays: bool = False
    n_workers: int = 1

    def __post_init__(self) -> None:
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1.")


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

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool.")
        if not isinstance(self.save_metadata_csv, bool):
            raise ValueError("save_metadata_csv must be a bool.")
        if str(self.image_format).lower() not in {"tif", "tiff", "npy"}:
            raise ValueError("image_format must be one of: 'tif', 'tiff', 'npy'.")


@dataclass
class ExtractionConfig:
    pad_mode: str = "clip"
    constant_values: float = 0.0
    preserve_non_spatial_axes: bool = True
    output_size: int | tuple[int, int] | None = None
    resize_order: int = 1
    keep_range: bool = True

    def __post_init__(self) -> None:
        if self.pad_mode not in {"clip", "constant"}:
            raise ValueError("pad_mode must be either 'clip' or 'constant'.")
        if not isinstance(self.preserve_non_spatial_axes, bool):
            raise ValueError("preserve_non_spatial_axes must be a bool.")

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
        if not isinstance(self.keep_range, bool):
            raise ValueError("keep_range must be a bool.")


@dataclass
class CellPatcherConfig:
    mode: str = "cell"
    common: CommonConfig = field(default_factory=CommonConfig)
    cell: CellConfig = field(default_factory=CellConfig)
    random: RandomConfig = field(default_factory=RandomConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    def __post_init__(self) -> None:
        valid_modes = {"cell", "random", "whole_fov", "grid"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}.")
