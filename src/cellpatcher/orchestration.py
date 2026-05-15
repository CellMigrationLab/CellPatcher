from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from cellpatcher.config import CellPatcherConfig
from cellpatcher.io import list_supported_files, read_array, validate_image_and_mask_folders
from cellpatcher.pipeline import run_patching


def default_config() -> CellPatcherConfig:
    """Return a fresh default config instance for easy editing."""
    return CellPatcherConfig()


def save_default_config(path) -> Path:
    """Write default config as JSON for users to edit and reload."""
    config = default_config()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(config), indent=2))
    return out


def load_config(path) -> CellPatcherConfig:
    """Load config JSON created from save_default_config()."""
    data = json.loads(Path(path).read_text())
    return CellPatcherConfig(**data)


def run_patching_pipeline(
    image_folder_path,
    *,
    config: CellPatcherConfig | None = None,
    mask_folder_path=None,
    strategy=None,
    mask_suffixes=None,
    case_sensitive=None,
):
    """Full folder-based orchestration for the package pipeline."""
    if config is None:
        config = CellPatcherConfig()
    if not isinstance(config, CellPatcherConfig):
        raise TypeError("config must be a CellPatcherConfig instance.")

    image_files = list_supported_files(image_folder_path)
    if len(image_files) == 0:
        raise ValueError("No supported image files found in image folder.")
    images = [read_array(path) for path in image_files]

    if config.mode != "cell":
        return run_patching(images=images, masks=None, config=config)

    if mask_folder_path is None:
        raise ValueError("mask_folder_path is required when config.mode='cell'.")

    matching = config.matching
    selected = validate_image_and_mask_folders(
        image_folder_path,
        mask_folder_path,
        strategy=matching.strategy if strategy is None else strategy,
        mask_suffixes=matching.mask_suffixes if mask_suffixes is None else mask_suffixes,
        case_sensitive=matching.case_sensitive if case_sensitive is None else case_sensitive,
    )
    masks = [read_array(selected[path]) for path in image_files]
    return run_patching(images=images, masks=masks, config=config)
