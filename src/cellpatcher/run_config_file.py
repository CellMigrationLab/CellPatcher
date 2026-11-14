"""The ``run_config.json`` written beside ``patches.csv``, and reading it back (layer: orchestration).

Version 2 (CellPatcher v0.2.0, CP-2b; design v0.4 §6, §8) adds ``run_config_version`` and,
when a preprocessing run was read, a ``preprocessing`` section:

- ``run``: the run's output folder, as given (``common.preprocessing_run``, or the folder of
  ``normalization.plane_histograms`` for a CSV scope of ``run_patching``);
- ``manifest_sha256``: the SHA-256 of its ``run_manifest.json`` when this call opened it, which
  names the run's images, masks, roles and table exactly (a manifest changed since is refused
  when the file is written: :func:`cellpatcher.run_open.refuse_changed_manifest`);
- ``mask_conversions`` when random mode read a ``parent_labels`` mask as an area (W3,
  experiment-v1 §4.5b): ``[{"mask_id": ..., "role": "region", "conversion": "parent_labels_as_area"}]``;
  absent otherwise;
- for a folder or tree run, ``layout``: the order of the arrays CellPatcher built from the roles
  the run recorded (T, Z, C, then Y, X; ``image`` and ``mask``), which delivers exported patches.
  Each image is read in its own roles (audit A-M1); when the images' layouts differ, ``image``
  and ``mask`` are the axes any of them has and ``images`` maps each image's path to its own
  ``image`` and ``mask`` layout (:func:`delivery_config` reads a patch's from there).

A folder or tree run does not record ``common.image_axes``, ``mask_axes`` or the labels: they
are the run's, not settings (each patch's labels are in ``patches.csv``), so the file loads back
as a config a file run accepts. Config files (:func:`cellpatcher.save_default_config`) carry the
same version. Every reader refuses any other version, and a file without one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from cellpatcher.normalization import CSV_SCOPES, normalization_inputs, run_settings
from cellpatcher.run_open import refuse_changed_manifest
from cellpatcher.run_request import PARENT_LABELS_AS_AREA
from cellpatcher.run_session import recalled, recalled_layouts

RUN_CONFIG_VERSION = 2

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
    area = recalled(PARENT_LABELS_AS_AREA, run) if file_run else None
    if area is not None:
        # experiment-v1 §4.5b: a parent_labels mask read as an area is recorded with its mask ID.
        section["mask_conversions"] = [{"mask_id": area, "role": "region", "conversion": PARENT_LABELS_AS_AREA}]
    if file_run:
        section["layout"] = {"image": common.image_axes, "mask": common.mask_axes}
        images = recalled_layouts(run) or {}
        if any(layout != section["layout"] for layout in images.values()):
            section["layout"]["images"] = images
    return section


def run_config_version(data: Mapping, where: str) -> int:
    """Return the version of a config or ``run_config.json`` content, refusing one this CellPatcher cannot read.

    Raises
    ------
    ValueError
        For a ``run_config_version`` other than 2, or none.
    """
    if "run_config_version" not in data:
        raise ValueError(
            f"{where} has no run_config_version; this CellPatcher reads version {RUN_CONFIG_VERSION} only. "
            f"Add \"run_config_version\": {RUN_CONFIG_VERSION} to a config file you wrote, or save it again "
            "with save_default_config/save_config_for_mode; run the patching again for an older run."
        )
    version = data["run_config_version"]
    if type(version) is not int or version != RUN_CONFIG_VERSION:
        raise ValueError(
            f"{where} has run_config_version {version!r}; this CellPatcher reads version {RUN_CONFIG_VERSION} "
            "only. It was written by another CellPatcher version: read it with that version, or run the "
            "patching again."
        )
    return version


def config_from_dict(data: Mapping, where: str = "The config") -> CellPatcherConfig:
    """Build a config from the content of a config or ``run_config.json`` file (other keys are ignored).

    Each section is built only when present (a mode-specific save leaves others out), otherwise
    from its defaults. A key a section does not have is refused (``TypeError``).

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
    built = {name: kind(**data[name]) if name in data else kind() for name, kind in sections.items()}
    return CellPatcherConfig(mode=data.get("mode", "cell"), **built)


def delivery_config(data: Mapping, where: str, source: str | None = None) -> CellPatcherConfig:
    """Return the config of a ``run_config.json`` for delivering its exported patches.

    A folder or tree run's arrays are in its recorded ``preprocessing.layout``, put back into
    ``common.image_axes``/``mask_axes`` (the axes of its patches): the layout of the image at
    *source* (a patch's ``source_relative_path``) when the run recorded one per image.

    Raises
    ------
    ValueError
        If the run recorded a layout per image and *source* is not one of its images: the run's
        union of axes is not that patch's layout, so none is guessed.
    """
    config = config_from_dict(data, where)
    layout = (data.get("preprocessing") or {}).get("layout")
    if layout is not None:
        images = layout.get("images")
        if images is not None:
            if source not in images:
                raise ValueError(
                    f"{where} records each image's layout, and {source!r} is not one of its images: "
                    "the patch's axes are unknown (is its source_relative_path missing or edited?)."
                )
            layout = images[source]
        config.common.image_axes, config.common.mask_axes = layout["image"], layout["mask"]
    return config
