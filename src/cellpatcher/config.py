import math
from dataclasses import dataclass, field
from numbers import Integral

from cellpatcher._deprecation import accept_deprecated_repeat
from cellpatcher.io import (
    SINGLE_BIOLOGICAL_REPEAT_LABEL,
    SINGLE_CONDITION_LABEL,
    SINGLE_TECHNICAL_REPEAT_LABEL,
)
from cellpatcher.ids import validate_object_set
from cellpatcher.planes import validate_plane_policies

MASK_SOURCES = ("tiff_tree", "ngff_labels")


@accept_deprecated_repeat
@dataclass
class CommonConfig:
    image_axes: str = "YX"
    mask_axes: str = "YX"
    output_dir: str | None = None
    extract_arrays: bool = False
    # Stamped onto every PatchRegion/metadata-CSV row for this run. These are
    # the MorphoBricks shared-CSV-contract `condition`/`biological_repeat`/
    # `technical_repeat` columns (see AGENTS.md §5). A label that is not
    # given is "single", as for a missing folder level; None is stamped as
    # "single" too. run_patching()/run_patching_pipeline() never infer them
    # from paths; set them explicitly, or use
    # orchestration.run_patching_condition_repeat_tree to derive them per
    # leaf from a <condition>/<biological_repeat>[/<technical_repeat>]/
    # folder tree. technical_repeat is the well/dish/coverslip within the
    # biological repeat; per-image image_metadata["technical_repeat"]
    # overrides it. `repeat` stays accepted as a deprecated alias of
    # biological_repeat (DeprecationWarning).
    condition: str | None = SINGLE_CONDITION_LABEL
    biological_repeat: str | None = SINGLE_BIOLOGICAL_REPEAT_LABEL
    technical_repeat: str | None = SINGLE_TECHNICAL_REPEAT_LABEL
    # How a Z or T axis of the images becomes 2D planes before patching
    # (cellpatcher.planes). "keep": every patch keeps all planes along the
    # axis (the behaviour before these options existed). "all": one patch
    # per plane; "first"/"select" (with z_index/t_index): one plane; Z only:
    # "max"/"mean"/"sum" project the stack. Masks follow the image's planes
    # when they have the same Z/T length and are shared otherwise.
    z_policy: str = "keep"
    z_index: int | None = None
    t_policy: str = "keep"
    t_index: int | None = None
    # Where folder and tree runs read masks (docs/normalization_contract.md
    # §4): "tiff_tree", a mask folder (tree) mirroring the images and matched
    # by file name with `matching`; or "ngff_labels", the label image named
    # mask_label_name in each OME-Zarr image's own labels group (the name may
    # be left None when every image has exactly one).
    mask_source: str = "tiff_tree"
    mask_label_name: str | None = None
    # The object set of cell-mode patches (the `object_set` column and a
    # level of patch_id, cellpatcher.ids). None: folder and tree runs name it
    # after the mask (its OME-NGFF label name, or its file name's suffix after
    # the image's stem), and "mask" when the mask gives no name (in-memory
    # run_patching, or a mask named like its image). Set it to name it
    # explicitly; image_metadata["object_set"] overrides it per image.
    object_set: str | None = None

    def __post_init__(self) -> None:
        validate_plane_policies(self)
        if self.mask_source not in MASK_SOURCES:
            raise ValueError(f"mask_source must be one of {list(MASK_SOURCES)}; got {self.mask_source!r}.")
        if self.mask_label_name is not None and not isinstance(self.mask_label_name, str):
            raise ValueError("mask_label_name must be a string or None.")
        validate_object_set(self.object_set)


BINARY_MASK_POLICIES = ("single_object", "connected_components")


@dataclass
class CellConfig:
    """Parameters of ``mode='cell'``.

    Masks are instance labels (decision D47, 2026-10-26): an integer mask is
    used as given, each non-zero value one object (a 0/1 integer mask is one
    object, label 1). ``binary_mask_policy`` only matters for a boolean mask
    (or a float mask of 0.0/1.0), which has no object ids: ``"single_object"``
    (default) refuses it; ``"connected_components"`` is the explicit opt-in
    that labels its connected components (4-connected), with a warning, and
    is recorded in ``run_config.json``. Labels above the int32 maximum are
    refused.
    """

    patch_size: int | str = "auto"
    mask_channel: int | None = None
    center_on: str = "bbox_center"
    exclude_border: bool = True
    min_area: int = 1
    max_area: int | None = None
    binary_mask_policy: str = "single_object"

    def __post_init__(self) -> None:
        if self.center_on not in {"centroid", "bbox_center"}:
            raise ValueError("center_on must be either 'centroid' or 'bbox_center'.")
        if self.binary_mask_policy not in BINARY_MASK_POLICIES:
            raise ValueError("binary_mask_policy must be either 'single_object' or 'connected_components'.")
        if self.min_area < 1:
            raise ValueError("min_area must be >= 1.")
        if self.max_area is not None and self.max_area < self.min_area:
            raise ValueError("max_area must be >= min_area when provided.")


SEED_SCOPES = ("plane_channel", "plane", "position")


@dataclass
class RandomConfig:
    """Parameters of ``mode='random'``.

    ``seed_scope`` chooses the identity each image's seed is derived from
    (:func:`cellpatcher.ids.random_seed`): ``"plane_channel"`` (the default,
    the behaviour before the option existed) ``position_id``, t, z and c, so
    every plane and channel gets other rectangles; ``"plane"`` ``position_id``,
    t and z, so every channel of a plane gets the same rectangles;
    ``"position"`` ``position_id`` only, so every plane of a position gets the
    same rectangles (a position followed over time).

    ``min_mask_fraction`` (with a sampling mask) is the least share of a
    draw's pixels that must lie in the mask's foreground. 0 (the default)
    keeps the behaviour before the option existed: any overlap is accepted.
    """

    patch_size: int = 128
    n_patches: int = 100
    seed: int | None = 0
    seed_scope: str = "plane_channel"
    min_mask_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if self.n_patches <= 0:
            raise ValueError("n_patches must be > 0.")
        if self.seed_scope not in SEED_SCOPES:
            raise ValueError(f"seed_scope must be one of {list(SEED_SCOPES)}; got {self.seed_scope!r}.")
        if not 0.0 <= self.min_mask_fraction <= 1.0:
            raise ValueError("min_mask_fraction must satisfy 0 <= min_mask_fraction <= 1.")


@dataclass
class WholeFOVConfig:
    """Parameters of ``mode='whole_fov'``.

    ``square_cover`` adds, after each whole-FOV region, the squares that cover
    it for square-input consumers: ⌈long side ÷ short side⌉ squares of the
    short side, evenly spaced along the long side (the first at 0, the last
    flush with the far edge), mode ``"whole_fov_square"``, each recording its
    whole-FOV region in ``parent_id``. A square image gives one square equal
    to the region.
    """

    square_cover: bool = False


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
    # Export one single-channel patch per channel of each region instead of
    # one multichannel patch (requires a "C" axis in common.image_axes):
    # True for every channel, or a list of channel indices for only those
    # (CellPatcher follow-up C5).
    split_channels: bool | list[int] = False

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
        channel_selection(self.split_channels)

    def split_channel_indices(self, channel_count: int) -> tuple[int, ...]:
        """The channels that get a single-channel patch of their own, for an image with *channel_count* channels.

        Every channel for ``split_channels=True``, the selected ones (in
        increasing order) for a list, and none for False.

        Raises
        ------
        ValueError
            If a selected index is not a channel of the image.
        """
        selection = channel_selection(self.split_channels)
        if selection is None:
            return tuple(range(channel_count)) if self.split_channels else ()
        outside = [channel for channel in selection if channel >= channel_count]
        if outside:
            raise ValueError(
                f"extraction.split_channels selects channel {', '.join(map(str, outside))}, but the image has "
                f"{channel_count} channel{'' if channel_count == 1 else 's'} (indices 0 to {channel_count - 1}). "
                "Select channels the images have, or True for every channel."
            )
        return selection


def channel_selection(split_channels) -> tuple[int, ...] | None:
    """The channel indices a list ``split_channels`` selects, sorted; None for True or False.

    Raises
    ------
    ValueError
        If *split_channels* is neither a bool nor a non-empty list of distinct
        non-negative channel indices.
    """
    if isinstance(split_channels, bool):
        return None
    if (
        not isinstance(split_channels, (list, tuple))
        or not split_channels
        or any(isinstance(channel, bool) or not isinstance(channel, Integral) or channel < 0
               for channel in split_channels)
    ):
        raise ValueError(
            "split_channels must be a bool or a non-empty list of channel indices (0, 1, ...): True gives one "
            "patch per channel, a list one patch per selected channel, False one multichannel patch; got "
            f"{split_channels!r}."
        )
    if len(set(split_channels)) != len(split_channels):
        raise ValueError(f"split_channels lists a channel more than once: {list(split_channels)!r}.")
    return tuple(sorted(int(channel) for channel in split_channels))


NORMALIZATION_METHODS = ("minmax", "percentile", "zscore", "robust_zscore")
NORMALIZATION_SCOPES = ("biological_repeat", "image", "patch")
# Removed without a deprecation period (normalization brick decision N29).
REMOVED_NORMALIZATION_SCOPES = ("condition", "technical_repeat")
POOLING_WEIGHTS = ("pixels", "conditions")
IMAGE_TYPES = ("fluorescence", "rgb")
# StarDist's percentile normalization is the one RGB method (normalization brick decision N33).
RGB_METHODS = ("percentile",)
PATCH_REGIONS = ("mask", "bbox", "patch")
EXPORT_VALUES = ("normalized", "raw")


def unknown_scope_message(scope) -> str:
    """The message for a ``normalization.scope`` that does not exist, as the normalization brick words it."""
    message = (
        f"Scope '{scope}' does not exist. Choose 'biological_repeat' (one biological repeat across all "
        "conditions), 'image' or 'patch'."
    )
    if scope in REMOVED_NORMALIZATION_SCOPES:
        message += (
            f" The '{scope}' scope was removed (October 2026): normalizing per {scope.replace('_', ' ')} "
            "removed real differences between them. Raw patches exported with it still normalize on load."
        )
    return message


@dataclass
class NormalizationConfig:
    """Normalization with the statistics of the preprocessing brick (``docs/normalization_contract.md``).

    Off by default, which keeps patching exactly as before. When enabled,
    each patch is normalized after extraction, before masking and resizing,
    as ``(pixels - offset) / scale`` per channel, in float32 and without
    clipping; the parameters are written to ``patches.csv``.

    Attributes
    ----------
    enabled
        Normalize the patches.
    plane_histograms
        Path of the preprocessing brick's ``plane_histograms.csv``; needed for
        every scope except ``"patch"``.
    manifest
        Path of its ``run_manifest.json``, for extra checks; None uses the one
        next to the CSV when there is one.
    method
        ``"minmax"``, ``"percentile"`` (default), ``"zscore"`` or
        ``"robust_zscore"`` (MAD x 1.4826); only ``"percentile"`` for
        ``image_type="rgb"``.
    percentile
        Low and high percentile of ``"percentile"``, ``0 <= low < high <= 100``.
    clip
        ``(low, high)`` limits applied right after normalizing, before masking
        and resizing (so resizing interpolates clipped values), or None (the
        default: no clip). For example ``(0, 1)`` with ``"percentile"`` or
        ``"minmax"``.
    scope
        Pixels the parameters come from: ``"biological_repeat"`` (one repeat
        group: the same biological repeat in every condition, technical
        repeats merged, as paired in the preprocessing brick's
        ``experiment_design.csv``), ``"image"`` (default: all Z and T planes
        of the source image) or ``"patch"`` (the patch's own pixels). Per
        channel, except for ``image_type="rgb"``. The former ``"condition"``
        and ``"technical_repeat"`` scopes are refused.
    per_time_point
        Normalize each time point separately (adds ``t`` to the group).
    patch_region
        For ``scope="patch"``: ``"mask"`` (the patch's own label), ``"bbox"``
        (the label's bounding box) or ``"patch"`` (all patch pixels); None
        means ``"mask"`` in ``cell`` mode and ``"patch"`` otherwise.
    export_values
        What exported files hold: ``"normalized"`` (default, float32) or
        ``"raw"``: the source values, with the parameters in ``patches.csv``
        for normalizing on load (``cellpatcher.iter_exported_patches``). A
        raw patch that is resized or masked is stored as float32 raw values.
    memory_limit
        Bytes allowed for the normalized patches of eager delivery
        (``common.extract_arrays=True``); above it the run stops and suggests
        ``cellpatcher.iter_patch_arrays``, which gives one patch at a time.
    experiment_design
        For ``scope="biological_repeat"``: path of the preprocessing brick's
        ``experiment_design.csv`` (the repeat groups and their confirmation);
        None uses the one next to ``plane_histograms``.
    pooling
        For ``scope="biological_repeat"``: how much each condition weighs in
        its repeat group, ``"pixels"`` (default: each pixel once) or
        ``"conditions"`` (each condition equally).
    image_type
        ``"fluorescence"`` (default: each channel normalized separately) or
        ``"rgb"``: colour images of three 8- or 16-bit channels, normalized
        with StarDist's percentile method, one offset and scale from the R, G
        and B pixels pooled together (normalization brick decisions N33, N34).
        With a preprocessing manifest that records the image type, this must
        be the same type. ``"rgb"`` keeps the three channels in one patch, so
        ``extraction.split_channels`` must be False.
    """

    enabled: bool = False
    plane_histograms: str | None = None
    manifest: str | None = None
    method: str = "percentile"
    percentile: tuple[float, float] = (1.0, 99.8)
    clip: tuple[float, float] | None = None
    scope: str = "image"
    per_time_point: bool = False
    patch_region: str | None = None
    export_values: str = "normalized"
    memory_limit: int = 4 * 1024**3
    # Appended after the earlier fields so run_config.json keeps their order.
    experiment_design: str | None = None
    pooling: str = "pixels"
    image_type: str = "fluorescence"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("normalization.enabled must be a bool.")
        if self.method not in NORMALIZATION_METHODS:
            raise ValueError(f"normalization.method must be one of {list(NORMALIZATION_METHODS)}; got {self.method!r}.")
        if self.scope not in NORMALIZATION_SCOPES:
            raise ValueError(f"normalization.scope: {unknown_scope_message(self.scope)}")
        if self.pooling not in POOLING_WEIGHTS:
            raise ValueError(
                f"normalization.pooling must be 'pixels' (each pixel counts once) or 'conditions' (each "
                f"condition weighs the same); got {self.pooling!r}."
            )
        if self.image_type not in IMAGE_TYPES:
            raise ValueError(
                f"normalization.image_type must be 'fluorescence' (each channel normalized separately) or 'rgb' "
                f"(colour images); got {self.image_type!r}."
            )
        if self.image_type == "rgb" and self.method not in RGB_METHODS:
            raise ValueError(
                "normalization.image_type='rgb' is normalized with StarDist's percentile method only, which scales "
                f"the three colour channels together; got method={self.method!r}. Use method='percentile'."
            )
        if self.patch_region is not None and self.patch_region not in PATCH_REGIONS:
            raise ValueError(
                f"normalization.patch_region must be one of {list(PATCH_REGIONS)} or None; got {self.patch_region!r}."
            )
        if not isinstance(self.per_time_point, bool):
            raise ValueError("normalization.per_time_point must be a bool.")
        # A JSON round trip turns the tuple into a list.
        try:
            low, high = (float(value) for value in self.percentile)
        except (TypeError, ValueError):
            raise ValueError("normalization.percentile must be two numbers (low, high).") from None
        if not 0 <= low < high <= 100:
            raise ValueError(f"normalization.percentile must satisfy 0 <= low < high <= 100; got ({low}, {high}).")
        self.percentile = (low, high)
        if self.clip is not None:
            try:
                low, high = (float(value) for value in self.clip)
            except (TypeError, ValueError):
                raise ValueError("normalization.clip must be None or two numbers (low, high).") from None
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"normalization.clip must be finite with low < high; got ({low}, {high}).")
            self.clip = (low, high)
        if self.export_values not in EXPORT_VALUES:
            raise ValueError(
                f"normalization.export_values must be one of {list(EXPORT_VALUES)}; got {self.export_values!r}."
            )
        if isinstance(self.memory_limit, bool) or not isinstance(self.memory_limit, int) or self.memory_limit <= 0:
            raise ValueError(f"normalization.memory_limit must be a positive int (bytes); got {self.memory_limit!r}.")


@dataclass
class MatchConfig:
    """Preferred image↔mask filename matching options.

    ``validate_image_and_mask_folders`` is a standalone helper that accepts
    matching options explicitly; it does not automatically consume
    ``CellPatcherConfig.matching``.

    The package-backed notebook/workflows should pass these fields through when
    calling folder-validation helpers.

    ``ignore_unrequested_masks`` leaves out, before matching, every mask whose
    stem ends in none of ``mask_suffixes``: for a mask folder holding several
    masks per image (``_cells``, ``_nuclei``, ...), set
    ``mask_suffixes=("_cells",)`` and ``ignore_unrequested_masks=True``. Off
    by default, so every mask in the folder must match an image.
    """
    strategy: str = "exact_or_suffix"
    mask_suffixes: tuple[str, ...] = ("_mask", "_labels", "_label", "_seg", "_segmentation")
    case_sensitive: bool = False
    ignore_unrequested_masks: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ignore_unrequested_masks, bool):
            raise ValueError("ignore_unrequested_masks must be a bool.")
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
    whole_fov:
        Parameters for ``mode='whole_fov'`` (the square cover).
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
    normalization:
        Normalization with the preprocessing brick's statistics; off by
        default (:class:`NormalizationConfig`).
    """

    mode: str = "cell"
    common: CommonConfig = field(default_factory=CommonConfig)
    cell: CellConfig = field(default_factory=CellConfig)
    random: RandomConfig = field(default_factory=RandomConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    whole_fov: WholeFOVConfig = field(default_factory=WholeFOVConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    def __post_init__(self) -> None:
        valid_modes = {"cell", "random", "whole_fov", "grid"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}.")
