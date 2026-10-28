from __future__ import annotations

import copy
import functools
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
from typing import Any

from cellpatcher._deprecation import accept_deprecated_repeat
from cellpatcher.config import (
    CellConfig,
    CellPatcherConfig,
    CommonConfig,
    ExportConfig,
    ExtractionConfig,
    FilterConfig,
    GridConfig,
    MatchConfig,
    NormalizationConfig,
    RandomConfig,
    WholeFOVConfig,
)
from cellpatcher.io import (
    SINGLE_TECHNICAL_REPEAT_LABEL,
    ConditionRepeatFolder,
    discover_condition_repeat_folders,
    list_supported_files,
    is_ome_zarr,
    read_image,
    read_ngff_mask,
    validate_image_and_mask_folders,
)
from cellpatcher.export import IMAGE_TABLE_FILENAME, save_image_table_csv, save_metadata_csv
from cellpatcher.ids import object_set_from_label_path, object_set_from_mask_file
from cellpatcher.normalization import CSV_SCOPES, cached_statistics, load_experiment_design, validate_normalization
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, PatchingResult, _patch_planes, run_config_data, run_patching
from cellpatcher.statistics import check_images
from cellpatcher.planes import iter_file_planes, splits_planes

# Sections always written regardless of mode
_COMMON_SECTIONS = ("common", "filters", "export", "extraction", "normalization")

# Mode-specific sections (beyond the common ones above)
_MODE_SECTIONS: dict[str, tuple[str, ...]] = {
    "cell": ("cell", "matching"),
    "random": ("random", "matching"),
    "grid": ("grid",),
    "whole_fov": ("whole_fov",),
}


def _config_to_mode_dict(config: CellPatcherConfig) -> dict:
    """Serialize only the sections relevant to *config.mode*."""
    full = asdict(config)
    result: dict = {"mode": full["mode"]}
    for section in _COMMON_SECTIONS:
        result[section] = full[section]
    for section in _MODE_SECTIONS.get(config.mode, ()):
        result[section] = full[section]
    return result


def apply_overrides(config: CellPatcherConfig, overrides: dict[str, Any]) -> CellPatcherConfig:
    """Return a copy of *config* with dotted-path *overrides* applied.

    Each key in *overrides* must be either a top-level ``CellPatcherConfig``
    field (e.g. ``"mode"``) or a two-part dotted path ``"<section>.<field>"``
    (e.g. ``"grid.tile_size"``).

    **When to use this vs. direct attribute editing**

    For a one-off run, setting attributes directly on a config object is
    simpler and gives you IDE autocomplete::

        cfg = CellPatcherConfig(mode="grid")
        cfg.grid.tile_size = 64               # clear, IDE-autocompleted
        cfg.common.output_dir = "./patches"
        run_patching_pipeline("./images", config=cfg)

    Use ``apply_overrides`` when you want to keep a *shared base config
    immutable* and derive multiple independent copies — for example during a
    parameter sweep::

        base = CellPatcherConfig(mode="grid")
        for tile_size in [64, 128, 256]:
            cfg = apply_overrides(base, {"grid.tile_size": tile_size})
            run_patching_pipeline("./images", config=cfg)

    Parameters
    ----------
    config:
        Base config to derive from (not mutated).
    overrides:
        Mapping of dotted-path keys to new values.  Example::

            overrides={"mode": "grid", "grid.tile_size": 64, "common.output_dir": "./out"}

    Returns
    -------
    CellPatcherConfig
        A new config instance with all overrides applied.

    Raises
    ------
    ValueError
        If a key refers to an unknown section or field.
    """
    # Deep-copy so the caller's original config is never mutated, even
    # though sub-configs are ordinary mutable dataclasses.
    cfg = copy.deepcopy(config)
    for key, value in overrides.items():
        # A bare key ("mode") sets a top-level field directly; a dotted key
        # ("grid.tile_size") reaches into the named sub-config section.
        parts = key.split(".", 1)
        if len(parts) == 1:
            if not hasattr(cfg, key):
                raise ValueError(
                    f"Unknown top-level config field: '{key}'. "
                    f"Valid fields: {[f.name for f in cfg.__dataclass_fields__.values()]}."  # type: ignore[attr-defined]
                )
            setattr(cfg, key, value)
        else:
            section, field = parts
            if not hasattr(cfg, section):
                raise ValueError(
                    f"Unknown config section: '{section}'. "
                    f"Valid sections: {[f.name for f in cfg.__dataclass_fields__.values()]}."  # type: ignore[attr-defined]
                )
            sub = getattr(cfg, section)
            # Checked on the class so a deprecated alias property (e.g.
            # "common.repeat") is recognised without its getter warning; the
            # setattr below then warns exactly once.
            if field not in getattr(sub, "__dataclass_fields__", {}) and not hasattr(type(sub), field):
                raise ValueError(
                    f"Unknown field '{field}' in config section '{section}'."
                )
            setattr(sub, field, value)
    return cfg


def default_config() -> CellPatcherConfig:
    """Return a fresh default :class:`~cellpatcher.CellPatcherConfig` instance.

    This is the recommended starting point for building a config in code.
    Attributes can be edited directly on the returned object — you get IDE
    autocomplete and validation fires immediately::

        from cellpatcher import default_config, run_patching_pipeline

        cfg = default_config()          # mode='cell', all defaults
        cfg.mode = "grid"
        cfg.grid.tile_size = 64
        cfg.grid.overlap = 0.1
        cfg.common.output_dir = "./patches"
        cfg.export.enabled = True

        result = run_patching_pipeline("./images", config=cfg)

    Returns
    -------
    CellPatcherConfig
        A new config with all fields set to their defaults.
    """
    return CellPatcherConfig()


def save_default_config(path) -> Path:
    """Serialize a default ``CellPatcherConfig`` to a JSON file.

    The resulting file can be opened in a text editor, tweaked, and then
    loaded back with :func:`load_config`.

    Parameters
    ----------
    path:
        Destination file path (e.g. ``"./cellpatcher_config.json"``).

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    config = default_config()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(config), indent=2))
    return out


def save_config_for_mode(mode: str, path) -> Path:
    """Serialize a mode-specific default config to a JSON file.

    Unlike :func:`save_default_config`, only the sections relevant to *mode*
    are written, producing a clean template that users can open, edit, and
    load back with :func:`load_config`.

    The common sections (``common``, ``filters``, ``export``, ``extraction``)
    are always included.  Mode-specific sections:

    * ``cell``   → ``cell``, ``matching``
    * ``random`` → ``random``, ``matching``
    * ``grid``   → ``grid``
    * ``whole_fov`` → ``whole_fov``

    Parameters
    ----------
    mode:
        One of ``'cell'``, ``'random'``, ``'grid'``, ``'whole_fov'``.
    path:
        Destination file path (e.g. ``"./grid_config.json"``).

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    valid_modes = {"cell", "random", "whole_fov", "grid"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}.")
    config = CellPatcherConfig(mode=mode)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_config_to_mode_dict(config), indent=2))
    return out


def load_config(path) -> CellPatcherConfig:
    """Load a :class:`~cellpatcher.CellPatcherConfig` from a JSON file.

    The file is typically created by :func:`save_default_config` or
    :func:`save_config_for_mode`.  All nested sub-configs are reconstructed as
    proper dataclass instances, so the returned object is fully usable with
    :func:`run_patching` and :func:`run_patching_pipeline`.

    After loading you can edit any attribute before passing the config to a
    run function — there is no need to re-serialise::

        from cellpatcher import load_config, run_patching_pipeline

        cfg = load_config("./grid_config.json")
        cfg.grid.tile_size = 64          # tweak in-place, IDE autocomplete works
        cfg.common.output_dir = "./out"

        result = run_patching_pipeline("./images", config=cfg)

    For immutable base configs that are reused across multiple runs, use
    :func:`apply_overrides` to produce independent copies::

        base = load_config("./grid_config.json")
        for tile_size in [64, 128, 256]:
            cfg = apply_overrides(base, {"grid.tile_size": tile_size})
            run_patching_pipeline("./images", config=cfg)

    Parameters
    ----------
    path:
        Path to the JSON config file.

    Returns
    -------
    CellPatcherConfig
        A fully reconstructed config instance.
    """
    return _config_from_dict(json.loads(Path(path).read_text()))


def _config_from_dict(data: dict) -> CellPatcherConfig:
    """Build a config from the data of a config or ``run_config.json`` file (extra keys are ignored)."""
    # Each section is reconstructed only if present in the file (a
    # mode-specific save omits irrelevant sections), otherwise it falls
    # back to that section's own defaults.
    return CellPatcherConfig(
        mode=data.get("mode", "cell"),
        common=CommonConfig(**data["common"]) if "common" in data else CommonConfig(),
        cell=CellConfig(**data["cell"]) if "cell" in data else CellConfig(),
        random=RandomConfig(**data["random"]) if "random" in data else RandomConfig(),
        grid=GridConfig(**data["grid"]) if "grid" in data else GridConfig(),
        whole_fov=WholeFOVConfig(**data["whole_fov"]) if "whole_fov" in data else WholeFOVConfig(),
        filters=FilterConfig(**data["filters"]) if "filters" in data else FilterConfig(),
        export=ExportConfig(**data["export"]) if "export" in data else ExportConfig(),
        extraction=ExtractionConfig(**data["extraction"]) if "extraction" in data else ExtractionConfig(),
        matching=MatchConfig(**data["matching"]) if "matching" in data else MatchConfig(),
        # A run_config.json written before normalization existed means it was off.
        normalization=(
            NormalizationConfig(**data["normalization"]) if "normalization" in data else NormalizationConfig()
        ),
    )


def _mask_loaders(image_folder_path, mask_folder_path, image_files, config, strategy, mask_suffixes, case_sensitive):
    """Resolve the mask of every image, without reading any: one loader per image, or None.

    Each loader returns ``(mask, mask_file name)``. Masks come from a mask
    folder matched by file name (``common.mask_source='tiff_tree'``, with
    ``config.matching`` unless *strategy*/*mask_suffixes*/*case_sensitive*
    override it), or from each OME-Zarr image's own label image
    (``'ngff_labels'``). Returns None when the run uses no masks: modes
    ``grid``/``whole_fov``, or ``random`` without a mask folder.

    Raises
    ------
    ValueError
        If the mode and the mask inputs do not fit, or images and masks do
        not match one to one.
    """
    if config.mode in {"grid", "whole_fov"}:
        if mask_folder_path is not None:
            raise ValueError(
                f"mask_folder_path is not supported for mode='{config.mode}'. "
                "Remove it, or switch to 'cell' or 'random' mode."
            )
        return None
    if config.common.mask_source == "ngff_labels":
        if mask_folder_path is not None:
            raise ValueError(
                "common.mask_source='ngff_labels' reads the masks from the OME-Zarr images' own labels; "
                "remove mask_folder_path (or mask_root_path), or use common.mask_source='tiff_tree'."
            )
        not_zarr = [image.name for image in image_files if not is_ome_zarr(image)]
        if not_zarr:
            raise ValueError(
                "common.mask_source='ngff_labels' needs OME-Zarr images, which hold their labels, but "
                f"{len(not_zarr)} image(s) are not OME-Zarr: {', '.join(not_zarr[:10])}. Use "
                "common.mask_source='tiff_tree' with a mask folder for them."
            )
        return [
            functools.partial(read_ngff_mask, image, config.common.mask_label_name, config.common.mask_axes)
            for image in image_files
        ]
    if mask_folder_path is None:
        if config.mode == "cell":
            raise ValueError("mask_folder_path is required when config.mode='cell'.")
        return None  # random mode: masks are optional
    matching = config.matching
    selected = validate_image_and_mask_folders(
        image_folder_path,
        mask_folder_path,
        strategy=matching.strategy if strategy is None else strategy,
        mask_suffixes=matching.mask_suffixes if mask_suffixes is None else mask_suffixes,
        case_sensitive=matching.case_sensitive if case_sensitive is None else case_sensitive,
        ignore_unrequested_masks=matching.ignore_unrequested_masks,
    )
    return [functools.partial(_read_mask_file, selected[image], config.common.mask_axes) for image in image_files]


def _read_mask_file(path: Path, axes: str):
    return read_image(path, axes), path.name


def _check_preprocessing_inputs(config: CellPatcherConfig, experiment_root, image_files) -> None:
    """Check the images against the preprocessing outputs before any patching (contract §5).

    Only with normalization at a CSV scope: every image must have rows in
    ``plane_histograms.csv`` (all images without rows in one message) and,
    with a manifest, the size, modification time and axes recorded for it.
    The planes themselves are checked as they are patched.
    """
    settings = config.normalization
    if not settings.enabled or settings.scope not in CSV_SCOPES:
        return
    validate_normalization(config)
    statistics = cached_statistics(settings.plane_histograms, settings.manifest)
    if settings.scope == "biological_repeat":
        load_experiment_design(settings, statistics)  # a design problem stops the run before any image is read
    image_axes = {_source_relative_path(path, experiment_root): config.common.image_axes for path in image_files}
    check_images(statistics, image_axes, experiment_root=Path(experiment_root).expanduser())


def _source_relative_path(image_path: Path, experiment_root) -> str:
    """Path of *image_path* relative to *experiment_root*, with ``/`` separators.

    Built from where the image sits in the folder tree, so it matches the
    preprocessing brick's ``source_relative_path`` (``control/img.tif`` for an
    image directly in ``control/``), unlike
    ``ConditionRepeatFolder.relative_folder``, which inserts ``single`` for
    missing levels. Like the brick, which walks the tree following links,
    a linked image or folder keeps its path under the root (the link's, not
    its target's), so a tree of links to images stored elsewhere can be
    patched. A path that is not under the root as written is compared by real
    locations, as before.
    """
    root = Path(experiment_root).expanduser()
    image = Path(image_path).expanduser()
    try:
        # Discovery builds every image path from the root, so this holds for
        # folder and tree runs; resolving would follow links out of the tree.
        return image.absolute().relative_to(root.absolute()).as_posix()
    except ValueError:  # e.g. the same folder written through a link
        pass
    try:
        return image.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Image '{image_path}' is not inside the experiment root '{root}'.") from exc


def _mask_object_set(config: CellPatcherConfig, image_path: Path, mask_file: str | None) -> str | None:
    """The object set a cell-mode mask's own name gives, or None (``common.object_set`` or ``"mask"`` then).

    Only when ``common.object_set`` is not set: an OME-NGFF mask gives its
    label image name; a mask file the part of its name after the image's
    stem (``img_cells.tif`` -> ``"cells"``), and nothing when named like its
    image.
    """
    if config.mode != "cell" or mask_file is None or config.common.object_set is not None:
        return None
    if config.common.mask_source == "ngff_labels":
        return object_set_from_label_path(mask_file)
    return object_set_from_mask_file(image_path.name, mask_file)


def _image_metadata(image_path: Path, mask_file: str | None, experiment_root, config: CellPatcherConfig) -> dict:
    """Build one image's ``run_patching(image_metadata=...)`` entry.

    Carries the source file names (and, with *experiment_root*, the image's
    ``source_relative_path``) through so they land on every ``PatchRegion``
    (and the exported metadata CSV) *before* ``export_patches`` writes it —
    passing this into ``run_patching`` rather than stamping the result
    afterwards is what makes that ordering guarantee hold. The image file
    name also gives the image's default ``position_id``, and the mask's name
    its ``object_set`` (:func:`_mask_object_set`).
    """
    meta = {"image_file": image_path.name}
    if mask_file is not None:
        meta["mask_file"] = mask_file
    if experiment_root is not None:
        meta["source_relative_path"] = _source_relative_path(image_path, experiment_root)
    object_set = _mask_object_set(config, image_path, mask_file)
    if object_set is not None:
        meta["object_set"] = object_set
    return meta


def run_patching_pipeline(
    image_folder_path,
    *,
    config: CellPatcherConfig | None = None,
    mask_folder_path=None,
    strategy=None,
    mask_suffixes=None,
    case_sensitive=None,
    overrides: dict[str, Any] | None = None,
):
    """Orchestrate a folder-based patching run end-to-end.

    Reads all supported image files from *image_folder_path*, optionally
    matches and loads mask files, then calls :func:`run_patching` with the
    supplied config.

    **Typical usage — direct attribute editing (recommended for interactive use)**

    Build a config, set attributes, and pass it in.  This gives you IDE
    autocomplete and immediate validation::

        from cellpatcher import CellPatcherConfig, run_patching_pipeline

        cfg = CellPatcherConfig(mode="grid")
        cfg.grid.tile_size = 64
        cfg.grid.overlap = 0.1
        cfg.common.output_dir = "./patches"
        cfg.export.enabled = True

        result = run_patching_pipeline("./images", config=cfg)

    For cell mode you also need to supply a mask folder::

        cfg = CellPatcherConfig(mode="cell")
        cfg.cell.patch_size = 128
        cfg.common.output_dir = "./cell_patches"
        cfg.export.enabled = True

        result = run_patching_pipeline(
            "./images",
            mask_folder_path="./masks",
            config=cfg,
        )

    **Batch / sweep usage — apply_overrides**

    When you want to keep a shared base config immutable and run multiple
    variants, use :func:`apply_overrides` to derive independent copies before
    calling this function::

        from cellpatcher import CellPatcherConfig, apply_overrides, run_patching_pipeline

        base = CellPatcherConfig(mode="grid")
        for tile_size in [64, 128, 256]:
            cfg = apply_overrides(base, {"grid.tile_size": tile_size,
                                         "common.output_dir": f"./patches_{tile_size}"})
            run_patching_pipeline("./images", config=cfg)

    Alternatively the *overrides* keyword provides an inline shorthand that
    behaves identically::

        run_patching_pipeline(
            "./images",
            config=base,
            overrides={"grid.tile_size": 64, "common.output_dir": "./patches_64"},
        )

    Parameters
    ----------
    image_folder_path:
        Path to the folder containing image files.
    config:
        A :class:`~cellpatcher.CellPatcherConfig` instance.  Defaults to
        ``CellPatcherConfig()`` (cell mode with all defaults).
    mask_folder_path:
        Path to the folder containing mask files.

        * **Required** when ``config.mode='cell'``, unless
          ``config.common.mask_source='ngff_labels'``, which reads each
          OME-Zarr image's own label image instead (then it must not be
          given).
        * **Optional** when ``config.mode='random'`` — masks restrict the
          spatial region from which patches are sampled.
        * **Must not be provided** for ``'grid'`` or ``'whole_fov'`` modes
          (those modes do not consume masks; a ``ValueError`` is raised if
          this argument is set for those modes).
    strategy, mask_suffixes, case_sensitive:
        Override the corresponding ``config.matching`` fields when resolving
        image↔mask filename pairs. Unused for modes that do not load masks.
    overrides:
        Optional dotted-path overrides applied on top of *config* before
        running.  Keys follow the same ``"<section>.<field>"`` convention as
        :func:`apply_overrides`.  The original *config* object is not
        mutated.  Example::

            overrides={"grid.tile_size": 64, "common.output_dir": "./out"}

    Returns
    -------
    PatchingResult
        Regions, optional patch arrays, and exported file paths. Each
        patch's ``source_relative_path`` is its image's file name: the image
        folder is the experiment root of this run.
    """
    if config is None:
        config = CellPatcherConfig()
    if not isinstance(config, CellPatcherConfig):
        raise TypeError("config must be a CellPatcherConfig instance.")
    if overrides:
        config = apply_overrides(config, overrides)
    _check_preprocessing_inputs(config, image_folder_path, list_supported_files(image_folder_path))
    return _run_folder_patching(
        image_folder_path,
        config=config,
        mask_folder_path=mask_folder_path,
        strategy=strategy,
        mask_suffixes=mask_suffixes,
        case_sensitive=case_sensitive,
        experiment_root=image_folder_path,
    )


def _run_folder_patching(
    image_folder_path,
    *,
    config: CellPatcherConfig,
    mask_folder_path,
    strategy,
    mask_suffixes,
    case_sensitive,
    experiment_root,
) -> PatchingResult:
    """Body of :func:`run_patching_pipeline` for one image folder.

    *experiment_root* is the folder each image's ``source_relative_path`` is
    relative to: the image folder itself for a flat run, the tree root for
    :func:`run_patching_condition_repeat_tree`.
    """
    image_files = list_supported_files(image_folder_path)
    if len(image_files) == 0:
        raise ValueError("No supported image files found in image folder.")
    # image_files' sorted order fixes the image_0000, image_0001, ...
    # image_id values (a run-local index); patch_id does not use them.
    loaders = _mask_loaders(
        image_folder_path, mask_folder_path, image_files, config, strategy, mask_suffixes, case_sensitive
    )
    common = config.common

    if splits_planes(common):
        # One image at a time: its mask, then only its chosen planes
        # (read plane by plane with the preprocessing extra).
        def sources():
            for index, image_path in enumerate(image_files):
                mask, mask_file = loaders[index]() if loaders is not None else (None, None)
                yield (
                    _image_metadata(image_path, mask_file, experiment_root, config),
                    iter_file_planes(
                        image_path, common.image_axes, mask, common.mask_axes, common,
                        single_planes=config.normalization.enabled,
                    ),
                )

        return _patch_planes(config, sources())

    masks, mask_files = None, [None] * len(image_files)
    if loaders is not None:
        masks, mask_files = (list(values) for values in zip(*(load() for load in loaders)))
    # OME-Zarr images are put into common.image_axes by role (read_image).
    images = [read_image(path, common.image_axes) for path in image_files]
    image_metadata = [
        _image_metadata(path, mask_file, experiment_root, config) for path, mask_file in zip(image_files, mask_files)
    ]
    return run_patching(images=images, masks=masks, config=config, image_metadata=image_metadata)


@accept_deprecated_repeat
@dataclass
class ConditionRepeatResult:
    """One leaf's outcome from :func:`run_patching_condition_repeat_tree`.

    ``technical_repeat`` is the leaf's technical-repeat folder (a well, a
    dish), or ``SINGLE_TECHNICAL_REPEAT_LABEL`` when the biological-repeat
    folder held its images directly; ``relative_folder`` is where its outputs
    go under ``output_dir``. ``repeat`` is a deprecated alias of
    ``biological_repeat``.
    """
    condition: str
    biological_repeat: str
    image_folder: str
    result: PatchingResult
    technical_repeat: str = SINGLE_TECHNICAL_REPEAT_LABEL
    relative_folder: str = ""


def run_patching_condition_repeat_tree(
    image_root_path,
    *,
    config: CellPatcherConfig | None = None,
    mask_root_path=None,
    strategy=None,
    mask_suffixes=None,
    case_sensitive=None,
    overrides: dict[str, Any] | None = None,
    combine_metadata: bool = False,
    progress_callback=None,
) -> list[ConditionRepeatResult]:
    """Run :func:`run_patching_pipeline` once per
    ``<condition>/<biological_repeat>[/<technical_repeat>]`` leaf.

    (The function keeps its historical ``condition_repeat`` name; the
    folder level it reads below each condition is the biological repeat.)

    Walks *image_root_path* with
    :func:`~cellpatcher.io.discover_condition_repeat_folders`, which accepts
    both the full MorphoBricks layout
    (``<condition>/<biological_repeat_YYYY_MM_DD>/``) and a condition folder
    holding images directly with no biological-repeat subfolder — the latter
    is treated as a single implicit biological repeat, labeled
    ``cellpatcher.io.SINGLE_BIOLOGICAL_REPEAT_LABEL`` (``"single"``),
    consistently across input discovery, processing (the
    ``config.common.biological_repeat`` value stamped on every patch from
    that condition), and output (its own output subfolder, same as an
    explicit biological repeat would get). A biological-repeat folder may in
    turn hold one subfolder per technical repeat (a well, a dish):
    ``<condition>/<biological_repeat>/<technical_repeat>/``; a
    biological-repeat folder holding its images directly is one technical
    repeat, labeled ``cellpatcher.io.SINGLE_TECHNICAL_REPEAT_LABEL``
    (``"single"``).

    For each discovered leaf, a copy of *config* is made with
    ``common.condition``/``common.biological_repeat``/``common.technical_repeat``
    set to the values discovered from the folder path. When
    ``config.export.enabled`` and ``config.common.output_dir`` are set, each
    leaf's ``output_dir`` becomes ``<output_dir>/<condition>/<biological_repeat>``,
    or ``<output_dir>/<condition>/<biological_repeat>/<technical_repeat>`` for a
    technical-repeat subfolder, mirroring the input layout. ``image_id``
    restarts at ``image_0000`` in every leaf; ``patch_id`` starts with the
    image's ``position_id`` (``<condition>/<biological_repeat>/<technical_repeat>/<stem>``),
    so it is unique across the whole tree.

    Parameters
    ----------
    image_root_path:
        Root folder containing one subfolder per condition.
    config:
        Base :class:`~cellpatcher.CellPatcherConfig`, applied to every leaf
        (with ``common.condition``/``.biological_repeat``/``.output_dir`` overridden
        per leaf as described above). Defaults to ``CellPatcherConfig()``.
    mask_root_path:
        Root folder mirroring *image_root_path*'s ``<condition>/<biological_repeat>``
        (or ``<condition>``, or ``<condition>/<biological_repeat>/<technical_repeat>``)
        structure for masks. Required for
        ``mode='cell'`` (each leaf's mask subfolder must exist); optional
        for ``mode='random'``; must not be provided for ``'grid'``/
        ``'whole_fov'`` (same rules as :func:`run_patching_pipeline`, applied
        per leaf).
    strategy, mask_suffixes, case_sensitive, overrides:
        Same as :func:`run_patching_pipeline`, applied to every leaf.
    combine_metadata:
        When ``True`` (requires ``config.export.enabled`` and
        ``config.common.output_dir``), a single ``patches.csv``, a single
        ``images.csv`` (every image of every leaf) and a single
        ``run_config.json`` are written directly in ``output_dir`` instead
        of one set per leaf folder. The patch files themselves still go to
        each leaf's ``<output_dir>/<condition>/<biological_repeat>[/<technical_repeat>]/``,
        and each row's ``patch_file`` is that file's path relative to
        ``output_dir`` (e.g.
        ``"treated/2026_01_01/treated_2026_01_01_single_img_grid-y0-x0-h128-w128.tif"``,
        or ``"treated/2026_01_01/B02/treated_2026_01_01_B02_img_grid-y0-x0-h128-w128.tif"``), so it
        stays relative to the CSV exactly as in the per-leaf layout. Every
        leaf's ``result.metadata_file`` (and ``result.image_table_file``)
        then points to the combined CSV.
    progress_callback:
        Optional ``callable(completed, total, leaf)``, called before each
        leaf is processed (``leaf`` is the
        :class:`~cellpatcher.io.ConditionRepeatFolder` about to run) and
        once more at the end with ``completed == total`` and ``leaf=None``.

    Returns
    -------
    list[ConditionRepeatResult]
        One entry per discovered leaf, in the order
        :func:`~cellpatcher.io.discover_condition_repeat_folders` returns
        them (conditions, then biological repeats, then technical repeats,
        each sorted by name).

    Raises
    ------
    ValueError
        If no condition subfolders with images are found, or (propagated
        from :func:`run_patching_pipeline`) if a leaf fails validation, e.g.
        a missing required mask folder. The first failing leaf aborts the
        whole run rather than silently skipping it.
    """
    if config is None:
        config = CellPatcherConfig()
    if not isinstance(config, CellPatcherConfig):
        raise TypeError("config must be a CellPatcherConfig instance.")
    if overrides:
        config = apply_overrides(config, overrides)
    if combine_metadata and (not config.export.enabled or config.common.output_dir is None):
        raise ValueError(
            "combine_metadata=True requires config.export.enabled=True and config.common.output_dir to be set."
        )

    if config.common.mask_source == "ngff_labels" and mask_root_path is not None:
        raise ValueError(
            "common.mask_source='ngff_labels' reads the masks from the OME-Zarr images' own labels; "
            "remove mask_root_path, or use common.mask_source='tiff_tree'."
        )

    leaves = discover_condition_repeat_folders(image_root_path, mask_root_path=mask_root_path)
    if not leaves:
        raise ValueError(f"No condition subfolders with images found under: {image_root_path}")
    # Checked once for the whole tree, so every image without rows is listed together.
    _check_preprocessing_inputs(
        config, image_root_path, [path for leaf in leaves for path in list_supported_files(leaf.image_folder)]
    )

    base_output_dir = Path(config.common.output_dir) if config.common.output_dir is not None else None
    # grid/whole_fov never accept a mask_folder_path (run_patching_pipeline
    # raises if given one), so a mask folder existing on disk for those
    # modes must be ignored rather than passed through.
    masks_allowed = config.mode not in {"grid", "whole_fov"}

    # With combined metadata, the per-leaf CSV and run config are replaced by
    # one pair written in output_dir after all leaves have run.
    leaf_export = (
        replace(config.export, save_metadata_csv=False, save_run_config=False) if combine_metadata else config.export
    )

    results: list[ConditionRepeatResult] = []
    for index, leaf in enumerate(leaves):
        if progress_callback is not None:
            progress_callback(index, len(leaves), leaf)
        # Derive this leaf's config from the shared base: same settings
        # everywhere, but condition/biological_repeat/technical_repeat/
        # output_dir specific to this leaf.
        leaf_common = replace(
            config.common,
            condition=leaf.condition,
            biological_repeat=leaf.biological_repeat,
            technical_repeat=leaf.technical_repeat,
        )
        if base_output_dir is not None:
            leaf_common.output_dir = str(base_output_dir / leaf.relative_folder)
        leaf_config = replace(config, common=leaf_common, export=leaf_export)

        # Only pass a mask folder through if the mode can use one AND it
        # actually exists -- e.g. random mode treats a missing mask folder
        # as "no mask" rather than an error, same as calling
        # run_patching_pipeline directly without mask_folder_path.
        leaf_mask_folder = None
        if masks_allowed and leaf.mask_folder is not None and leaf.mask_folder.exists():
            leaf_mask_folder = str(leaf.mask_folder)

        # source_relative_path is relative to the tree root, as in the
        # preprocessing brick, not to the leaf folder.
        result = _run_folder_patching(
            str(leaf.image_folder),
            config=leaf_config,
            mask_folder_path=leaf_mask_folder,
            strategy=strategy,
            mask_suffixes=mask_suffixes,
            case_sensitive=case_sensitive,
            experiment_root=image_root_path,
        )
        results.append(
            ConditionRepeatResult(
                condition=leaf.condition,
                biological_repeat=leaf.biological_repeat,
                image_folder=str(leaf.image_folder),
                result=result,
                technical_repeat=leaf.technical_repeat,
                relative_folder=leaf.relative_folder.as_posix(),
            )
        )

    if combine_metadata:
        _write_combined_metadata(config, base_output_dir, results)
    if progress_callback is not None:
        progress_callback(len(leaves), len(leaves), None)
    return results


def _write_combined_metadata(config: CellPatcherConfig, output_dir: Path, results: list[ConditionRepeatResult]) -> None:
    """Write one run_config.json, one patches.csv and one images.csv covering every leaf of a tree run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / RUN_CONFIG_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(run_config_data(config), f, indent=2, default=str)

    all_patches = []
    for leaf_result in results:
        for patch in leaf_result.result.patches:
            # Leaf patch files live in
            # <output_dir>/<condition>/<biological_repeat>[/<technical_repeat>]/,
            # so prefix that folder to keep patch_file relative to the CSV.
            if patch.patch_file:
                patch.patch_file = (Path(leaf_result.relative_folder) / patch.patch_file).as_posix()
            patch.config_file = RUN_CONFIG_FILENAME
        all_patches.extend(leaf_result.result.patches)

    if config.export.save_metadata_csv:
        csv_path = output_dir / "patches.csv"
        save_metadata_csv(all_patches, csv_path)
        # Every image of every leaf, with those that gave no patch.
        image_table_path = output_dir / IMAGE_TABLE_FILENAME
        save_image_table_csv(
            [record for leaf_result in results for record in leaf_result.result.image_records], image_table_path
        )
        for leaf_result in results:
            leaf_result.result.metadata_file = str(csv_path)
            leaf_result.result.image_table_file = str(image_table_path)
