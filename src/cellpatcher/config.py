from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from cellpatcher._deprecation import accept_deprecated_repeat
from cellpatcher.fractions import check_choice
from cellpatcher.io import (
    REMOVED_MATCHING_OPTIONS,
    SINGLE_BIOLOGICAL_REPEAT_LABEL,
    SINGLE_CONDITION_LABEL,
    SINGLE_TECHNICAL_REPEAT_LABEL,
    refuse_removed_matching,
)
from cellpatcher.setting_checks import (
    MODES,
    channel_selection,
    check_cell_settings,
    check_common_settings,
    check_export_settings,
    check_extraction_settings,
    check_filter_settings,
    check_grid_settings,
    check_match_settings,
    check_normalization_settings,
    check_partial_mode,
    check_random_settings,
    check_whole_fov_settings,
)


@accept_deprecated_repeat
@dataclass
class CommonConfig:
    # The axes of in-memory arrays given to run_patching (canonical, Y before X). Folder and
    # tree runs read each image and mask in the roles the preprocessing run recorded, refuse
    # other values than "YX" and do not record them in run_config.json (design v0.4 §6).
    image_axes: str = "YX"
    mask_axes: str = "YX"
    output_dir: str | None = None
    extract_arrays: bool = False
    # The output folder of the clean preprocessing run (v0.4) a folder or tree run reads its
    # images, masks and labels through (RunInputs, design v0.4 §6); required for file runs.
    preprocessing_run: str | None = None
    # The shared-CSV-contract labels (AGENTS.md §5) of in-memory run_patching; None is "single",
    # as for a missing folder level. Folder and tree runs take each image's from the run and
    # refuse other values. technical_repeat (well/dish/coverslip) is overridden per image by
    # image_metadata["technical_repeat"]; `repeat` is a deprecated alias of biological_repeat.
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
    # The object set of cell-mode patches (the `object_set` column and a level of patch_id,
    # cellpatcher.ids). None: folder and tree runs name it after the mask type the run recorded
    # (a folder name, or an OME-NGFF label name), in-memory run_patching "mask". Set it to name
    # it explicitly; image_metadata["object_set"] overrides it per image.
    object_set: str | None = None

    def __post_init__(self) -> None:
        check_common_settings(self)


@dataclass
class CellConfig:
    """Parameters of ``mode='cell'``.

    Masks are instance labels (decision D47, 2026-10-26): an integer mask is
    used as given, each non-zero value one object (a 0/1 integer mask is one
    object, label 1). ``binary_mask_policy`` only matters for a boolean mask,
    which has no object ids: ``"single_object"`` (default) refuses it;
    ``"connected_components"`` is the explicit opt-in that labels its
    connected components (4-connected), with a warning, and is recorded in
    ``run_config.json``. Labels are checked with the preprocessing brick's
    ``check_labels`` (decision D-b): negative labels and labels above the
    int32 maximum are refused, and so is a float mask, even of whole numbers
    or 0.0/1.0.
    """

    patch_size: int | str = "auto"
    mask_channel: int | None = None
    center_on: str = "bbox_center"
    exclude_border: bool = True
    min_area: int = 1
    max_area: int | None = None
    binary_mask_policy: str = "single_object"

    def __post_init__(self) -> None:
        check_cell_settings(self)


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
    draw's pixels in the mask's foreground. ``None`` (the default, F8): any
    overlap; an explicit ``0``: the mask excludes no draw. Draw sequence:
    :mod:`cellpatcher.strategies.random`.
    """

    patch_size: int = 128
    n_patches: int = 100
    seed: int | None = 0
    seed_scope: str = "plane_channel"
    min_mask_fraction: float | None = None

    def __post_init__(self) -> None:
        check_random_settings(self)


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

    def __post_init__(self) -> None:
        check_whole_fov_settings(self)


@dataclass
class GridConfig:
    tile_size: int = 128
    overlap: float = 0.0
    include_partial: bool = False
    # How include_partial=True covers the far edge. "flush" (the historical
    # behaviour) keeps the clipped edge tiles and adds a full tile flush with
    # the far edge, which overlaps its neighbours. "clipped" keeps only the
    # clipped edge tiles: with overlap=0 every pixel is in exactly one tile.
    partial_mode: str = "flush"

    def __post_init__(self) -> None:
        check_grid_settings(self)

    def check_partial_mode(self) -> None:
        """Refuse an unknown ``partial_mode``, or ``"clipped"`` without ``include_partial``.

        Also called by the grid strategy, since fields edited after
        construction (``apply_overrides``, attribute assignment) skip
        ``__post_init__``.
        """
        check_partial_mode(self)


@dataclass
class FilterConfig:
    min_foreground_fraction: float | None = None
    foreground_threshold: float | str | None = None
    min_mean_intensity: float | None = None

    def __post_init__(self) -> None:
        check_filter_settings(self)


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
        check_export_settings(self)


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
        # output_size: None (no resize), an int (square) or a (height, width) pair of ints.
        check_extraction_settings(self)

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
        The preprocessing brick's ``plane_histograms.csv``, in its output
        folder, read through its ``RunInputs``; needed except for ``"patch"``.
    manifest
        The run's ``run_manifest.json``: None, or the one in the table's
        folder (another file is refused).
    experiment_root, masks_root
        For ``run_patching`` at a CSV scope: where the run's experiment and masks
        folder are now, if moved since the run (None: where it recorded them).
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
        For ``scope="biological_repeat"``: the run's ``experiment_design.csv``
        (the repeat groups and their confirmation): None, or the one next to
        ``plane_histograms`` (another file is refused).
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
    experiment_root: str | None = None
    masks_root: str | None = None
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
        check_normalization_settings(self)
        # A JSON round trip turns the tuple into a list. The numbers were checked above (a bool, text
        # or NaN is refused), so float() only stores a valid number as a Python float: run_config.json
        # then records 1.0, not 1 or the text "1.0" of a NumPy float32 (json.dump with default=str).
        self.percentile = tuple(float(value) for value in self.percentile)
        if self.clip is not None:
            self.clip = tuple(float(value) for value in self.clip)


# The default of the removed options below: "not given".
_NOT_GIVEN = object()


@dataclass
class MatchConfig:
    """Which masks to pair with the images.

    Images and masks are paired by the preprocessing run (its ``find_masks``,
    design v0.4 §6), so every brick reads the same pairs: a mask is a TIFF file
    ``<mask type>/<image's folders>/<image name>.tif`` or
    ``<image's folders>/<image name>_<mask type>.tif`` in the mask folder, or
    an OME-Zarr image's label image (its type is its name).

    ``mask_type`` names the mask type to use (``"cells"`` for
    ``img_cells.tif`` or ``cells/img.tif``). None (default) uses the run's only
    type, and refuses a run holding several.

    The former options ``strategy``, ``mask_suffixes``, ``case_sensitive``
    and ``ignore_unrequested_masks`` were removed in v0.1.1: given here, set
    later or found in an old ``run_config.json``, they are refused by name
    with a ``ValueError`` saying to use ``mask_type``.
    """

    mask_type: str | None = None
    strategy: InitVar[object] = _NOT_GIVEN
    mask_suffixes: InitVar[object] = _NOT_GIVEN
    case_sensitive: InitVar[object] = _NOT_GIVEN
    ignore_unrequested_masks: InitVar[object] = _NOT_GIVEN

    def __post_init__(self, strategy, mask_suffixes, case_sensitive, ignore_unrequested_masks) -> None:
        given = dict(zip(REMOVED_MATCHING_OPTIONS, (strategy, mask_suffixes, case_sensitive, ignore_unrequested_masks)))
        refuse_removed_matching(
            "MatchConfig", {name: value for name, value in given.items() if value is not _NOT_GIVEN}
        )
        check_match_settings(self)

    def __setattr__(self, name, value) -> None:
        # The former options are not fields: setting one after construction
        # (cfg.matching.strategy = "substring") must not be silently ignored.
        if name in REMOVED_MATCHING_OPTIONS:
            refuse_removed_matching(f"matching.{name}", {name: value})
        super().__setattr__(name, value)


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
        Which mask type to pair with the images (:class:`MatchConfig`);
        the pairing itself is the preprocessing brick's.
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
        check_choice(self.mode, MODES, "mode")
