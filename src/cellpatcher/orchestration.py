from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
from typing import Any

from cellpatcher._deprecation import accept_deprecated_repeat
from cellpatcher.config import CellPatcherConfig
from cellpatcher.file_runs import FileRun, leaves, masks_used, open_file_run, patch_fovs, run_config_for
from cellpatcher.io import SINGLE_TECHNICAL_REPEAT_LABEL, ConditionRepeatFolder, refuse_removed_matching
from cellpatcher.export import IMAGE_TABLE_FILENAME, save_image_table_csv, save_metadata_csv
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, PatchingResult
from cellpatcher.run_config_file import config_from_dict, run_config_data
from cellpatcher.run_session import statistics_session

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
    return config_from_dict(json.loads(Path(path).read_text()), str(path))


def run_patching_pipeline(
    image_folder_path: str | os.PathLike,
    *,
    config: CellPatcherConfig | None = None,
    mask_folder_path: str | os.PathLike | None = None,
    overrides: dict[str, Any] | None = None,
    **removed_options: object,
) -> PatchingResult:
    """Patch the images of a clean preprocessing run of one condition and biological repeat.

    The run is ``config.common.preprocessing_run`` (its output folder), opened
    with ``RunInputs.open`` with *image_folder_path* as the experiment root
    (:mod:`cellpatcher.file_runs`, design v0.4 §6): the images patched are the
    run's, read with the roles it recorded, with their labels and
    ``position_id`` as it recorded them, and their masks are those it paired.
    Without a run, a ``PreprocessingCheckError`` (``no_preprocessing_run``)
    gives the preprocessing command. A run with several conditions or
    biological repeats is refused: use :func:`run_patching_condition_repeat_tree`.

    **Typical usage — direct attribute editing (recommended for interactive use)**

    Build a config, set attributes, and pass it in.  This gives you IDE
    autocomplete and immediate validation::

        from cellpatcher import CellPatcherConfig, run_patching_pipeline

        cfg = CellPatcherConfig(mode="grid")
        cfg.common.preprocessing_run = "./images_MorphoBricks_Preprocessing"
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
        The experiment root: where the preprocessed experiment is now (it
        may have been moved or copied since the run; its images must be
        unchanged).
    config:
        A :class:`~cellpatcher.CellPatcherConfig` instance.  Defaults to
        ``CellPatcherConfig()`` (cell mode with all defaults).
    mask_folder_path:
        Where the run's masks folder is now; None: where the run recorded
        it. The masks are those the run paired (masks-folder TIFFs and
        OME-Zarr label images), of the type ``config.matching.mask_type``
        (None: the run's only type).

        * ``'cell'`` mode always reads masks.
        * ``'random'`` mode reads them when this or ``matching.mask_type``
          is given — masks restrict where patches are sampled.
        * ``'grid'`` and ``'whole_fov'`` read none; given, it only says
          where the run's masks folder is now (its masks are checked there,
          as for the tree run).
    strategy, mask_suffixes, case_sensitive:
        Former matching options, removed in v0.1.1 and refused by name:
        the masks are those the preprocessing run paired; choose the mask
        type with ``config.matching.mask_type``.
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
        patch's ``source_relative_path`` is its image's path relative to the
        experiment root, as the run recorded it.

    Raises
    ------
    PreprocessingCheckError
        Without ``common.preprocessing_run``, or when ``RunInputs`` refuses
        the run (an image or mask changed or missing, a run before v0.4).
    ValueError
        For ``common.image_axes``/``mask_axes`` other than ``"YX"``, labels
        other than ``"single"``, or ``normalization.experiment_root``/``masks_root``:
        file runs read these from the run and its arguments (the message says so).
    """
    config = _checked_config(config, overrides)
    refuse_removed_matching("run_patching_pipeline", removed_options)
    with statistics_session():
        # grid/whole_fov read no masks: a masks folder given for them only says where the run's is now.
        use_masks = masks_used(config, mask_folder_path)
        run = open_file_run(
            config, image_folder_path, mask_folder_path, use_masks=use_masks, where="run_patching_pipeline"
        )
        groups = leaves(run)
        labels = {(leaf.condition, leaf.biological_repeat) for leaf, _ in groups}
        if len(labels) > 1:
            raise ValueError(
                f"The preprocessing run has {len(labels)} conditions or biological repeats; "
                "run_patching_pipeline patches one. Use run_patching_condition_repeat_tree for the whole "
                "experiment."
            )
        ((condition, biological_repeat),) = labels
        leaf_config = run_config_for(config, run, (condition, biological_repeat, SINGLE_TECHNICAL_REPEAT_LABEL))
        return patch_fovs(run, run.fovs, leaf_config)


def _checked_config(config: CellPatcherConfig | None, overrides: dict[str, Any] | None) -> CellPatcherConfig:
    """Return *config* (default: ``CellPatcherConfig()``) with *overrides* applied."""
    if config is None:
        config = CellPatcherConfig()
    if not isinstance(config, CellPatcherConfig):
        raise TypeError("config must be a CellPatcherConfig instance.")
    return apply_overrides(config, overrides) if overrides else config


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
    image_root_path: str | os.PathLike,
    *,
    config: CellPatcherConfig | None = None,
    mask_root_path: str | os.PathLike | None = None,
    overrides: dict[str, Any] | None = None,
    combine_metadata: bool = False,
    progress_callback: Callable[[int, int, ConditionRepeatFolder | None], object] | None = None,
    **removed_options: object,
) -> list[ConditionRepeatResult]:
    """Patch every image of a clean preprocessing run, one leaf per
    ``<condition>/<biological_repeat>[/<technical_repeat>]``.

    (The function keeps its historical ``condition_repeat`` name; the
    folder level it reads below each condition is the biological repeat.)

    The run is ``config.common.preprocessing_run``, opened with
    ``RunInputs.open`` with *image_root_path* as the experiment root and
    *mask_root_path* as its masks folder (:mod:`cellpatcher.file_runs`,
    design v0.4 §6). Its images are grouped by the labels the run recorded
    for them (``"single"`` for a level an experiment does not have, as the
    preprocessing brick names it), read with the roles it recorded, and
    their masks are those it paired; no name is parsed and no axis is
    given. Without a run, a ``PreprocessingCheckError``
    (``no_preprocessing_run``) gives the preprocessing command.

    Each leaf gets a copy of *config* with its labels. When
    ``config.export.enabled`` and ``config.common.output_dir`` are set, each
    leaf's ``output_dir`` becomes ``<output_dir>/<condition>/<biological_repeat>``,
    or ``<output_dir>/<condition>/<biological_repeat>/<technical_repeat>`` for a
    technical repeat other than ``"single"``, mirroring the input layout. ``image_id``
    restarts at ``image_0000`` in every leaf; ``patch_id`` starts with the
    image's FOV key (``<condition>/<biological_repeat>/<technical_repeat>/<position_id>``,
    D85), so it is unique across the whole tree.

    Parameters
    ----------
    image_root_path:
        The experiment root: where the preprocessed experiment is now (it
        may have been moved or copied since the run; its images must be
        unchanged).
    config:
        Base :class:`~cellpatcher.CellPatcherConfig`, applied to every leaf
        (with the labels and ``.output_dir`` set per leaf as described
        above). Defaults to ``CellPatcherConfig()``.
    mask_root_path:
        Where the run's masks folder is now; None: where the run recorded
        it. ``mode='cell'`` reads the masks of type
        ``config.matching.mask_type`` (None: the run's only type);
        ``mode='random'`` reads them when this or ``matching.mask_type`` is
        given, and every image must then have one; ``'grid'``/``'whole_fov'``
        read none.
    overrides:
        Same as :func:`run_patching_pipeline`, applied to every leaf. The
        removed matching options are refused by name, as there.
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
        One entry per leaf: conditions, then biological repeats, then
        technical repeats, each sorted by name.

    Raises
    ------
    PreprocessingCheckError, ValueError
        As :func:`run_patching_pipeline`, before any image is read. A
        failing leaf aborts the whole run rather than being skipped.
    """
    config = _checked_config(config, overrides)
    if combine_metadata and (not config.export.enabled or config.common.output_dir is None):
        raise ValueError(
            "combine_metadata=True requires config.export.enabled=True and config.common.output_dir to be set."
        )
    refuse_removed_matching("run_patching_condition_repeat_tree", removed_options)
    use_masks = masks_used(config, mask_root_path)
    with statistics_session():
        run = open_file_run(
            config, image_root_path, mask_root_path, use_masks=use_masks, where="run_patching_condition_repeat_tree"
        )
        groups = leaves(run)
        results = _patch_leaves(config, run, groups, combine_metadata, progress_callback)
        if combine_metadata:  # in the session: its normalization_inputs come from the run opened above
            tree_config = run_config_for(config, run, (SINGLE_TECHNICAL_REPEAT_LABEL,) * 3)
            _write_combined_metadata(tree_config, Path(config.common.output_dir), results)
    if progress_callback is not None:
        progress_callback(len(groups), len(groups), None)
    return results


def _patch_leaves(
    config: CellPatcherConfig,
    run: FileRun,
    groups: list[tuple[ConditionRepeatFolder, list]],
    combine_metadata: bool,
    progress_callback: Callable | None,
) -> list[ConditionRepeatResult]:
    """Patch each leaf of a tree run with its own labels and output folder.

    Every leaf gets the base settings, its labels from the run, and, with
    ``common.output_dir``, ``<output_dir>/<condition>/<biological_repeat>[/<technical_repeat>]``.
    With combined metadata the per-leaf CSV and run config are replaced by one
    pair written in ``output_dir`` after all leaves have run.
    """
    base_output_dir = Path(config.common.output_dir) if config.common.output_dir is not None else None
    leaf_export = (
        replace(config.export, save_metadata_csv=False, save_run_config=False) if combine_metadata else config.export
    )
    results: list[ConditionRepeatResult] = []
    for index, (leaf, fovs) in enumerate(groups):
        if progress_callback is not None:
            progress_callback(index, len(groups), leaf)
        leaf_config = run_config_for(config, run, (leaf.condition, leaf.biological_repeat, leaf.technical_repeat))
        if base_output_dir is not None:
            leaf_config.common.output_dir = str(base_output_dir / leaf.relative_folder)
        result = patch_fovs(run, fovs, replace(leaf_config, export=leaf_export))
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
