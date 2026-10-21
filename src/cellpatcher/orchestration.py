from __future__ import annotations

import copy
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
    RandomConfig,
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
from cellpatcher.export import save_metadata_csv
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, PatchingResult, run_patching

# Sections always written regardless of mode
_COMMON_SECTIONS = ("common", "filters", "export", "extraction")

# Mode-specific sections (beyond the common ones above)
_MODE_SECTIONS: dict[str, tuple[str, ...]] = {
    "cell": ("cell", "matching"),
    "random": ("random", "matching"),
    "grid": ("grid",),
    "whole_fov": (),
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
    * ``whole_fov`` → *(none beyond common)*

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
    data = json.loads(Path(path).read_text())
    # Each section is reconstructed only if present in the file (a
    # mode-specific save omits irrelevant sections), otherwise it falls
    # back to that section's own defaults.
    return CellPatcherConfig(
        mode=data.get("mode", "cell"),
        common=CommonConfig(**data["common"]) if "common" in data else CommonConfig(),
        cell=CellConfig(**data["cell"]) if "cell" in data else CellConfig(),
        random=RandomConfig(**data["random"]) if "random" in data else RandomConfig(),
        grid=GridConfig(**data["grid"]) if "grid" in data else GridConfig(),
        filters=FilterConfig(**data["filters"]) if "filters" in data else FilterConfig(),
        export=ExportConfig(**data["export"]) if "export" in data else ExportConfig(),
        extraction=ExtractionConfig(**data["extraction"]) if "extraction" in data else ExtractionConfig(),
        matching=MatchConfig(**data["matching"]) if "matching" in data else MatchConfig(),
    )


def _load_matched_masks(image_folder_path, mask_folder_path, image_files, config, strategy, mask_suffixes, case_sensitive):
    """Validate image↔mask folder pairing and return masks ordered like *image_files*.

    Parameters
    ----------
    image_folder_path:
        Path to the image folder (passed through to
        :func:`validate_image_and_mask_folders`).
    mask_folder_path:
        Path to the mask folder.
    image_files:
        Pre-sorted list of image :class:`~pathlib.Path` objects whose order
        determines the order of the returned mask arrays.
    config:
        Active ``CellPatcherConfig``; its ``matching`` sub-config provides
        default strategy, suffixes, and case-sensitivity.
    strategy, mask_suffixes, case_sensitive:
        When not ``None``, these override the corresponding ``config.matching``
        fields.

    Returns
    -------
    tuple
        ``(masks, mask_paths)`` — mask arrays and their source
        :class:`~pathlib.Path` objects, both in the same order as
        *image_files*.
    """
    matching = config.matching
    selected = validate_image_and_mask_folders(
        image_folder_path,
        mask_folder_path,
        strategy=matching.strategy if strategy is None else strategy,
        mask_suffixes=matching.mask_suffixes if mask_suffixes is None else mask_suffixes,
        case_sensitive=matching.case_sensitive if case_sensitive is None else case_sensitive,
    )
    mask_paths = [selected[path] for path in image_files]
    return [read_image(path, config.common.mask_axes) for path in mask_paths], mask_paths


def _source_relative_path(image_path: Path, experiment_root) -> str:
    """Path of *image_path* relative to *experiment_root*, with ``/`` separators.

    Built from the image's real location, so it matches the preprocessing
    brick's ``source_relative_path`` (``control/img.tif`` for an image
    directly in ``control/``), unlike ``ConditionRepeatFolder.relative_folder``,
    which inserts ``single`` for missing levels.
    """
    root = Path(experiment_root).expanduser().resolve()
    try:
        return Path(image_path).expanduser().resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Image '{image_path}' is not inside the experiment root '{root}'.") from exc


def _load_ngff_masks(image_files, config, mask_folder_path):
    """Read each OME-Zarr image's label image as its mask (``common.mask_source='ngff_labels'``).

    Returns ``(masks, mask names)`` in the order of *image_files*.
    """
    if mask_folder_path is not None:
        raise ValueError(
            "common.mask_source='ngff_labels' reads the masks from the OME-Zarr images' own labels; "
            "remove mask_folder_path (or mask_root_path), or use common.mask_source='tiff_tree'."
        )
    not_zarr = [path.name for path in image_files if not is_ome_zarr(path)]
    if not_zarr:
        raise ValueError(
            "common.mask_source='ngff_labels' needs OME-Zarr images, which hold their labels, but "
            f"{len(not_zarr)} image(s) are not OME-Zarr: {', '.join(not_zarr[:10])}. Use "
            "common.mask_source='tiff_tree' with a mask folder for them."
        )
    masks, names = [], []
    for path in image_files:
        mask, name = read_ngff_mask(path, config.common.mask_label_name, config.common.mask_axes)
        masks.append(mask)
        names.append(name)
    return masks, names


def _image_metadata_for_filenames(image_files, mask_paths=None, experiment_root=None) -> list[dict]:
    """Build the per-image ``run_patching(image_metadata=...)`` payload.

    Carries source filenames (and, with *experiment_root*, each image's
    ``source_relative_path``) through so they land on every ``PatchRegion``
    (and the exported metadata CSV) *before* ``export_patches`` writes it —
    passing this into ``run_patching`` rather than stamping the result
    afterwards is what makes that ordering guarantee hold.
    """
    if mask_paths is None:
        mask_paths = [None] * len(image_files)
    metadata = []
    for image_path, mask_path in zip(image_files, mask_paths):
        meta = {"image_file": image_path.name}
        if mask_path is not None:
            # A label image of an OME-Zarr image is named by a string
            # (<image>/labels/<name>), a mask file by its path.
            meta["mask_file"] = mask_path if isinstance(mask_path, str) else mask_path.name
        if experiment_root is not None:
            meta["source_relative_path"] = _source_relative_path(image_path, experiment_root)
        metadata.append(meta)
    return metadata


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
    # ids used everywhere downstream (run_patching, filename stamping).
    # OME-Zarr images are put into common.image_axes by role (read_image).
    images = [read_image(path, config.common.image_axes) for path in image_files]

    # Each mode has different mask requirements, so dispatch on mode before
    # deciding whether/how to load masks at all.
    if config.mode in {"grid", "whole_fov"}:
        if mask_folder_path is not None:
            raise ValueError(
                f"mask_folder_path is not supported for mode='{config.mode}'. "
                "Remove it, or switch to 'cell' or 'random' mode."
            )
        image_metadata = _image_metadata_for_filenames(image_files, experiment_root=experiment_root)
        return run_patching(images=images, masks=None, config=config, image_metadata=image_metadata)

    if config.mode == "cell":
        if config.common.mask_source == "ngff_labels":
            masks, mask_paths = _load_ngff_masks(image_files, config, mask_folder_path)
        elif mask_folder_path is None:
            raise ValueError("mask_folder_path is required when config.mode='cell'.")
        else:
            masks, mask_paths = _load_matched_masks(
                image_folder_path, mask_folder_path, image_files, config,
                strategy, mask_suffixes, case_sensitive,
            )
        image_metadata = _image_metadata_for_filenames(image_files, mask_paths, experiment_root)
        return run_patching(images=images, masks=masks, config=config, image_metadata=image_metadata)

    # random mode: masks are optional (restrict patch placement region)
    masks = None
    mask_paths = None
    if config.common.mask_source == "ngff_labels":
        masks, mask_paths = _load_ngff_masks(image_files, config, mask_folder_path)
    elif mask_folder_path is not None:
        masks, mask_paths = _load_matched_masks(
            image_folder_path, mask_folder_path, image_files, config,
            strategy, mask_suffixes, case_sensitive,
        )
    image_metadata = _image_metadata_for_filenames(image_files, mask_paths, experiment_root)
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
    technical-repeat subfolder, mirroring the input layout (patch ids
    restart at ``image_0000`` in every leaf, so leaves never share a folder).

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
        ``config.common.output_dir``), a single ``patches.csv`` and a single
        ``run_config.json`` are written directly in ``output_dir`` instead
        of one pair per leaf folder. The patch files themselves still go to
        each leaf's ``<output_dir>/<condition>/<biological_repeat>[/<technical_repeat>]/``,
        and each row's ``patch_file`` is that file's path relative to
        ``output_dir`` (e.g. ``"treated/2026_01_01/image_0000_grid_00000.tif"``,
        or ``"treated/2026_01_01/B02/image_0000_grid_00000.tif"``), so it
        stays relative to the CSV exactly as in the per-leaf layout. Every
        leaf's ``result.metadata_file`` then points to the combined CSV.
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
    """Write one run_config.json and one patches.csv covering every leaf of a tree run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / RUN_CONFIG_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, default=str)

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
        for leaf_result in results:
            leaf_result.result.metadata_file = str(csv_path)
