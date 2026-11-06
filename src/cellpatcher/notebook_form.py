"""The settings form of the CellPatcher notebook: widgets in, a ``CellPatcherConfig`` out (layer: ui).

Every control maps to one setting of :class:`cellpatcher.CellPatcherConfig`; nothing is
computed here. The form offers only the settings of a file run since v0.2.0: no axes, labels
or mask source, which are the preprocessing run's (design v0.4 §6). Needs the ``notebook``
extra (``ipywidgets``) and the preprocessing brick's ``notebook_style``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import ipywidgets as widgets
from morphobricks_preprocessing import notebook_style as style

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
)
from cellpatcher.notebook_session import RunSummary

MODES = (
    ("Cell: one patch per segmented object", "cell"),
    ("Grid: regular tiles", "grid"),
    ("Random: patches at random positions", "random"),
    ("Whole FOV: one patch per image", "whole_fov"),
)
CELL_SIZES = (
    ("Per object (fits each object)", "auto"),
    ("Fixed size", "fixed"),
    ("Largest object of the dataset", "max"),
    ("95th percentile of the dataset", "p95"),
    ("90th percentile of the dataset", "p90"),
    ("Median object of the dataset", "median"),
)
PLANE_POLICIES = (
    ("Keep all planes in each patch", "keep"),
    ("One patch per plane", "all"),
    ("First plane", "first"),
    ("One plane (index below)", "select"),
)
SCOPES = (
    ("Image: all planes of the patch's image (recommended)", "image"),
    ("Biological repeat: the repeat group across conditions", "biological_repeat"),
    ("Patch: each patch's own pixels", "patch"),
)
METHODS = (
    ("Percentile range", "percentile"),
    ("Min-max", "minmax"),
    ("Z-score", "zscore"),
    ("Robust z-score", "robust_zscore"),
)
WIDE = {"description_width": "200px"}


def _int(description: str, value: int, minimum: int = 0) -> widgets.BoundedIntText:
    """Return a whole-number box with a wide label."""
    return widgets.BoundedIntText(value=value, min=minimum, max=10**9, description=description, style=WIDE)


def _float(description: str, value: float, maximum: float = 10**12) -> widgets.BoundedFloatText:
    """Return a number box with a wide label."""
    return widgets.BoundedFloatText(value=value, min=0, max=maximum, description=description, style=WIDE)


def _choice(description: str, options: tuple, value: str) -> widgets.Dropdown:
    """Return a dropdown of ``(label, value)`` options with a wide label."""
    return widgets.Dropdown(options=options, value=value, description=description, style=WIDE)


def _check(description: str, value: bool = False) -> widgets.Checkbox:
    """Return a checkbox that keeps its whole label."""
    return widgets.Checkbox(value=value, description=description, indent=False)


@dataclass
class CellControls:
    """The ``cell`` mode settings."""

    size: widgets.Dropdown
    fixed: widgets.BoundedIntText
    exclude_border: widgets.Checkbox
    min_area: widgets.BoundedIntText
    max_area: widgets.BoundedIntText
    apply_mask: widgets.Checkbox
    components: widgets.Checkbox

    @classmethod
    def build(cls) -> CellControls:
        """Return the controls at the defaults of :class:`cellpatcher.CellConfig`."""
        return cls(
            size=_choice("Patch size", CELL_SIZES, "auto"),
            fixed=_int("Fixed size (pixels)", 64, 1),
            exclude_border=_check("Skip objects touching the image border", True),
            min_area=_int("Minimum object area (px)", 1, 1),
            max_area=_int("Maximum object area (px, 0: none)", 0),
            apply_mask=_check("Blank the pixels outside the object (apply mask)"),
            components=_check("Masks are boolean: label their connected components"),
        )

    def controls(self) -> list[widgets.Widget]:
        """Return the controls in display order."""
        return [self.size, self.fixed, self.exclude_border, self.min_area, self.max_area, self.apply_mask,
                self.components]

    def config(self, dataset_size: Callable[[str], int | str]) -> CellConfig:
        """Return the cell settings; *dataset_size* gives a dataset statistic's size (``"auto"``: none yet)."""
        choice = self.size.value
        size: int | str = choice if choice == "auto" else self.fixed.value
        if choice not in {"auto", "fixed"}:
            size = dataset_size(choice)
        return CellConfig(
            patch_size=size,
            exclude_border=self.exclude_border.value,
            min_area=self.min_area.value,
            max_area=self.max_area.value or None,
            binary_mask_policy="connected_components" if self.components.value else "single_object",
        )


@dataclass
class PlaneControls:
    """The T plane policy (shown when the run's images have time points; Z stacks are refused, issue #83)."""

    t_policy: widgets.Dropdown
    t_index: widgets.BoundedIntText

    @classmethod
    def build(cls) -> PlaneControls:
        """Return the controls at the defaults of :class:`cellpatcher.CommonConfig`."""
        return cls(
            t_policy=_choice("T (time) points", PLANE_POLICIES, "keep"),
            t_index=_int("Time point index", 0),
        )

    def show_for(self, layout: str) -> None:
        """Show the controls when the axes *layout* has time points."""
        for control in (self.t_policy, self.t_index):
            style.set_visible(control, "T" in layout)

    def common(self, summary: RunSummary) -> CommonConfig:
        """Return the common settings: the run, the time policy and nothing about files."""
        return CommonConfig(
            preprocessing_run=str(summary.run),
            t_policy=self.t_policy.value if "T" in summary.layout else "keep",
            t_index=self.t_index.value if self.t_policy.value == "select" else None,
        )


@dataclass
class NormalizationControls:
    """The normalization settings (off by default)."""

    enabled: widgets.Checkbox
    scope: widgets.Dropdown
    method: widgets.Dropdown
    low: widgets.BoundedFloatText
    high: widgets.BoundedFloatText
    per_time_point: widgets.Checkbox
    raw: widgets.Checkbox

    @classmethod
    def build(cls) -> NormalizationControls:
        """Return the controls at the defaults of :class:`cellpatcher.NormalizationConfig`."""
        return cls(
            enabled=_check("Normalize the patches"),
            scope=_choice("Statistics from", SCOPES, "image"),
            method=_choice("Method", METHODS, "percentile"),
            low=_float("Low percentile", 1.0, 100),
            high=_float("High percentile", 99.8, 100),
            per_time_point=_check("Normalize each time point separately"),
            raw=_check("Export the raw values (parameters in patches.csv)"),
        )

    def controls(self) -> list[widgets.Widget]:
        """Return the controls in display order."""
        return [self.enabled, self.scope, self.method, self.low, self.high, self.per_time_point, self.raw]

    def config(self, summary: RunSummary) -> NormalizationConfig:
        """Return the normalization settings; the CSV scopes read the run's own table."""
        if not self.enabled.value:
            return NormalizationConfig()
        return NormalizationConfig(
            enabled=True,
            scope=self.scope.value,
            method=self.method.value,
            percentile=(self.low.value, self.high.value),
            per_time_point=self.per_time_point.value,
            export_values="raw" if self.raw.value else "normalized",
            image_type=summary.image_type or "fluorescence",
        )


@dataclass
class SettingsForm:
    """Every setting of a run, as widgets, and the ``CellPatcherConfig`` they give."""

    mode: widgets.Dropdown
    mask_type: widgets.Dropdown
    cell: CellControls
    grid_size: widgets.BoundedIntText
    grid_overlap: widgets.BoundedFloatText
    grid_partial: widgets.Checkbox
    random_size: widgets.BoundedIntText
    random_count: widgets.BoundedIntText
    random_seed: widgets.BoundedIntText
    random_masks: widgets.Checkbox
    planes: PlaneControls
    min_mean: widgets.BoundedFloatText
    output_size: widgets.BoundedIntText
    pad: widgets.Checkbox
    channels: widgets.SelectMultiple
    normalization: NormalizationControls
    file_format: widgets.Dropdown

    @classmethod
    def build(cls) -> SettingsForm:
        """Return the form at CellPatcher's defaults."""
        return cls(
            mode=_choice("Patching strategy", MODES, "cell"),
            mask_type=widgets.Dropdown(options=[], description="Mask type", style=WIDE),
            cell=CellControls.build(),
            grid_size=_int("Tile size (pixels)", 128, 1),
            grid_overlap=_float("Overlap (fraction, 0 to <1)", 0.0, 0.99),
            grid_partial=_check("Keep partial tiles at the edges"),
            random_size=_int("Patch size (pixels)", 128, 1),
            random_count=_int("Patches per image", 100, 1),
            random_seed=_int("Seed (reproducible sampling)", 0),
            random_masks=_check("Sample only inside the masks"),
            planes=PlaneControls.build(),
            min_mean=_float("Minimum mean intensity (0: off)", 0.0),
            output_size=_int("Resize patches to (pixels, 0: no)", 0),
            pad=_check("Pad patches crossing the image edge to full size"),
            channels=widgets.SelectMultiple(options=[], description="Split channels", style=WIDE),
            normalization=NormalizationControls.build(),
            file_format=_choice("Patch files", (("TIFF (.tif)", "tif"), ("NumPy (.npy)", "npy")), "tif"),
        )

    def set_run(self, summary: RunSummary) -> None:
        """Offer the run's mask types, channels and plane axes."""
        self.mask_type.options = list(summary.mask_types)
        self.mask_type.value = summary.mask_types[0] if summary.mask_types else None
        self.channels.options = [(name, index) for index, name in enumerate(summary.channels)]
        self.channels.value = ()
        self.planes.show_for(summary.layout)

    def mode_widgets(self) -> dict[str, list[widgets.Widget]]:
        """Return the controls shown only in each mode."""
        return {
            "cell": self.cell.controls(),
            "grid": [self.grid_size, self.grid_overlap, self.grid_partial],
            "random": [self.random_size, self.random_count, self.random_seed, self.random_masks],
            "whole_fov": [],
        }

    def uses_masks(self) -> bool:
        """Return whether the chosen settings read masks."""
        return self.mode.value == "cell" or (self.mode.value == "random" and self.random_masks.value)

    def config(self, summary: RunSummary, dataset_size: Callable[[str], int | str]) -> CellPatcherConfig:
        """Return the settings as a file-run ``CellPatcherConfig`` for *summary*'s run.

        Raises
        ------
        ValueError
            From the config's own checks (for example a minimum area above the maximum).
        """
        mask_type = self.mask_type.value if self.uses_masks() else None
        return CellPatcherConfig(
            mode=self.mode.value,
            common=self.planes.common(summary),
            cell=self.cell.config(dataset_size),
            grid=GridConfig(tile_size=self.grid_size.value, overlap=self.grid_overlap.value,
                            include_partial=self.grid_partial.value),
            random=RandomConfig(patch_size=self.random_size.value, n_patches=self.random_count.value,
                                seed=self.random_seed.value),
            filters=FilterConfig(min_mean_intensity=self.min_mean.value or None),
            extraction=ExtractionConfig(
                output_size=self.output_size.value or None,
                pad_mode="constant" if self.pad.value else "clip",
                apply_mask=self.mode.value == "cell" and self.cell.apply_mask.value,
                split_channels=sorted(self.channels.value) or False,
            ),
            export=ExportConfig(enabled=True, image_format=self.file_format.value),
            matching=MatchConfig(mask_type=mask_type),
            normalization=self.normalization.config(summary),
        )
