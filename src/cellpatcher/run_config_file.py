"""The ``run_config.json`` written beside ``patches.csv``, and reading it back (layer: orchestration).

Version 2 (CellPatcher v0.2.0, CP-2b; design v0.4 §6, §8) adds ``run_config_version`` and,
when a preprocessing run was read, a ``preprocessing`` section:

- ``run``: the run's output folder, as given (``common.preprocessing_run``, or the folder of
  ``normalization.plane_histograms`` for a CSV scope of ``run_patching``);
- ``manifest_sha256``: the SHA-256 of its ``run_manifest.json`` when this call opened it, which
  names the run's images, masks, roles and table exactly (a manifest changed since is refused
  when the file is written: :func:`cellpatcher.run_open.refuse_changed_manifest`);
- for a folder or tree run, ``layout``: the order of the arrays CellPatcher built from the roles
  the run recorded (T, Z, C, then Y, X; ``image`` and ``mask``), which delivers exported patches.

A folder or tree run does not record ``common.image_axes``, ``mask_axes`` or the labels: they
are the run's, not settings (each patch's labels are in ``patches.csv``), so the file loads back
as a config a file run accepts. Version 1 is a ``run_config.json`` written before v0.2.0, or a
config file without a version (:func:`cellpatcher.save_default_config`). Any other version is
refused by every reader.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cellpatcher.config import (
    REMOVED_NORMALIZATION_SCOPES,
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
from cellpatcher.normalization import CSV_SCOPES, normalization_inputs, run_settings
from cellpatcher.run_open import refuse_changed_manifest
from cellpatcher.run_session import recalled

RUN_CONFIG_VERSION = 2
READ_VERSIONS = (1, RUN_CONFIG_VERSION)

# Settings removed in v0.2.0 (CP-2b), with the value that had no effect.
_REMOVED_COMMON = {"mask_source": "tiff_tree", "mask_label_name": None}

# The common settings a folder or tree run takes from the preprocessing run, not recorded.
_RUN_SETTINGS = ("image_axes", "mask_axes", "condition", "biological_repeat", "technical_repeat")


def run_config_data(config: CellPatcherConfig) -> dict:
    """Return the ``run_config.json`` content of a run with *config* (version 2).

    The config, ``run_config_version``, the ``preprocessing`` section when a run was read
    (see the module), and the normalization brick's records: ``normalization_inputs`` when a
    CSV is used, ``normalization_settings`` when normalization is on.

    Raises
    ------
    RuntimeError
        If the preprocessing run read was not opened in this call (an internal error: its
        manifest's SHA-256 is recorded when it is opened).
    PreprocessingCheckError
        If the run's ``run_manifest.json`` changed since this call opened it.
    """
    data: dict[str, Any] = {"run_config_version": RUN_CONFIG_VERSION, **asdict(config)}
    preprocessing = _preprocessing_section(config)
    if preprocessing is not None:
        data["preprocessing"] = preprocessing
    if preprocessing is not None and "layout" in preprocessing:
        for name in _RUN_SETTINGS:
            data["common"].pop(name)
    inputs = normalization_inputs(config)
    if inputs is not None:
        data["normalization_inputs"] = inputs
    settings = run_settings(config)
    if settings is not None:
        data["normalization_settings"] = settings
    return data


def _preprocessing_section(config: CellPatcherConfig) -> dict | None:
    """Return the ``preprocessing`` section: the run read, its manifest's SHA-256 and a file run's layout."""
    common, settings = config.common, config.normalization
    file_run = common.preprocessing_run is not None and recalled("file_run", common.preprocessing_run) is not None
    if file_run:
        run = common.preprocessing_run
    elif settings.enabled and settings.scope in CSV_SCOPES and settings.plane_histograms is not None:
        run = str(Path(settings.plane_histograms).expanduser().parent)
    else:
        return None
    sha256 = recalled("manifest_sha256", run)
    if sha256 is None:
        raise RuntimeError(f"The preprocessing run {run} was not opened in this call, so it cannot be recorded.")
    refuse_changed_manifest(run, sha256)
    section: dict[str, Any] = {"run": str(run), "manifest_sha256": sha256}
    if file_run:
        section["layout"] = {"image": common.image_axes, "mask": common.mask_axes}
    return section


def run_config_version(data: Mapping, where: str) -> int:
    """Return the version of a config or ``run_config.json`` content, refusing one this CellPatcher cannot read.

    Raises
    ------
    ValueError
        For a ``run_config_version`` other than 1 or 2.
    """
    version = data.get("run_config_version", 1)
    if type(version) is not int or version not in READ_VERSIONS:
        raise ValueError(
            f"{where} has run_config_version {version!r}; this CellPatcher reads versions "
            f"{', '.join(map(str, READ_VERSIONS))}. It was written by another CellPatcher version: read it with "
            "that version, or run the patching again."
        )
    return version


def config_from_dict(data: Mapping, where: str = "The config") -> CellPatcherConfig:
    """Build a config from the content of a config or ``run_config.json`` file (other keys are ignored).

    Each section is built only when present (a mode-specific save leaves others out), otherwise
    from its defaults; a file written before normalization existed means it was off.

    Raises
    ------
    ValueError
        For a version this CellPatcher cannot read (:func:`run_config_version`).
    """
    run_config_version(data, where)
    sections = {
        "common": CommonConfig,
        "cell": CellConfig,
        "random": RandomConfig,
        "grid": GridConfig,
        "whole_fov": WholeFOVConfig,
        "filters": FilterConfig,
        "export": ExportConfig,
        "extraction": ExtractionConfig,
        "matching": MatchConfig,
        "normalization": NormalizationConfig,
    }
    data = {**data, "common": _without_removed_settings(data.get("common", {}), where)}
    built = {name: kind(**data[name]) if name in data else kind() for name, kind in sections.items()}
    return CellPatcherConfig(mode=data.get("mode", "cell"), **built)


def _without_removed_settings(common: Mapping, where: str) -> dict:
    """Return the ``common`` section without the settings removed in v0.2.0, refusing one that was used.

    ``mask_source`` and ``mask_label_name`` chose where folder and tree runs read masks; a file
    written before v0.2.0 holds their defaults, which had no effect and are left out.
    """
    common = dict(common)
    for name, default in _REMOVED_COMMON.items():
        value = common.pop(name, default)
        if value != default:
            raise ValueError(
                f"{where}: common.{name}={value!r} was removed in CellPatcher v0.2.0: a folder or tree run reads "
                "the masks its preprocessing run paired, masks-folder TIFFs and OME-Zarr label images alike. "
                "Remove it and choose the mask type with matching.mask_type (a label image's type is its name)."
            )
    return common


def delivery_config(data: Mapping, where: str) -> CellPatcherConfig:
    """Return the config of a ``run_config.json`` for delivering its exported patches.

    A folder or tree run's arrays are in its recorded ``preprocessing.layout``, put back into
    ``common.image_axes``/``mask_axes`` (the axes of its patches). A run normalized at a scope
    removed since (``condition``, ``technical_repeat``; normalization brick N29) still loads: its
    parameters are in ``patches.csv``, and only the patch axes are needed.
    """
    normalization = data.get("normalization")
    if isinstance(normalization, Mapping) and normalization.get("scope") in REMOVED_NORMALIZATION_SCOPES:
        data = {**data, "normalization": {**normalization, "scope": "image"}}
    config = config_from_dict(data, where)
    layout = (data.get("preprocessing") or {}).get("layout")
    if layout is not None:
        config.common.image_axes, config.common.mask_axes = layout["image"], layout["mask"]
    return config
