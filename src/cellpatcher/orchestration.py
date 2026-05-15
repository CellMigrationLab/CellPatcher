from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

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
from cellpatcher.io import list_supported_files, read_array, validate_image_and_mask_folders
from cellpatcher.pipeline import run_patching


def default_config() -> CellPatcherConfig:
    """Return a fresh default config instance for easy editing."""
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


def load_config(path) -> CellPatcherConfig:
    """Load a ``CellPatcherConfig`` from a JSON file written by :func:`save_default_config`.

    All nested sub-configs (``CommonConfig``, ``GridConfig``, etc.) are
    reconstructed as proper dataclass instances so the returned object is
    fully usable with :func:`run_patching` and :func:`run_patching_pipeline`.

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
    list
        Mask arrays in the same order as *image_files*.
    """
    matching = config.matching
    selected = validate_image_and_mask_folders(
        image_folder_path,
        mask_folder_path,
        strategy=matching.strategy if strategy is None else strategy,
        mask_suffixes=matching.mask_suffixes if mask_suffixes is None else mask_suffixes,
        case_sensitive=matching.case_sensitive if case_sensitive is None else case_sensitive,
    )
    return [read_array(selected[path]) for path in image_files]


def run_patching_pipeline(
    image_folder_path,
    *,
    config: CellPatcherConfig | None = None,
    mask_folder_path=None,
    strategy=None,
    mask_suffixes=None,
    case_sensitive=None,
):
    """Orchestrate a folder-based patching run end-to-end.

    Reads all supported image files from *image_folder_path*, optionally
    matches and loads mask files, then calls :func:`run_patching` with the
    supplied config.

    Parameters
    ----------
    image_folder_path:
        Path to the folder containing image files.
    config:
        A ``CellPatcherConfig`` instance. Defaults to ``CellPatcherConfig()``.
    mask_folder_path:
        Path to the folder containing mask files.

        * **Required** when ``config.mode='cell'``.
        * **Optional** when ``config.mode='random'`` — masks restrict the
          spatial region from which patches are sampled.
        * **Must not be provided** for ``'grid'`` or ``'whole_fov'`` modes
          (those modes do not consume masks; a ``ValueError`` is raised if
          this argument is set for those modes).
    strategy, mask_suffixes, case_sensitive:
        Override the corresponding ``config.matching`` fields when resolving
        image↔mask filename pairs. Unused for modes that do not load masks.

    Returns
    -------
    PatchingResult
        Regions, optional patch arrays, and exported file paths.
    """
    if config is None:
        config = CellPatcherConfig()
    if not isinstance(config, CellPatcherConfig):
        raise TypeError("config must be a CellPatcherConfig instance.")

    image_files = list_supported_files(image_folder_path)
    if len(image_files) == 0:
        raise ValueError("No supported image files found in image folder.")
    images = [read_array(path) for path in image_files]

    if config.mode in {"grid", "whole_fov"}:
        if mask_folder_path is not None:
            raise ValueError(
                f"mask_folder_path is not supported for mode='{config.mode}'. "
                "Remove it, or switch to 'cell' or 'random' mode."
            )
        return run_patching(images=images, masks=None, config=config)

    if config.mode == "cell":
        if mask_folder_path is None:
            raise ValueError("mask_folder_path is required when config.mode='cell'.")
        masks = _load_matched_masks(
            image_folder_path, mask_folder_path, image_files, config,
            strategy, mask_suffixes, case_sensitive,
        )
        return run_patching(images=images, masks=masks, config=config)

    # random mode: masks are optional (restrict patch placement region)
    masks = None
    if mask_folder_path is not None:
        masks = _load_matched_masks(
            image_folder_path, mask_folder_path, image_files, config,
            strategy, mask_suffixes, case_sensitive,
        )
    return run_patching(images=images, masks=masks, config=config)
