"""The CellPatcher notebook: one page per step, built from the package (layer: ui).

``notebooks/CellPatcher/CellPatcher.ipynb`` only creates :class:`CellPatcherNotebook` and calls
one step per cell; every action is a call to :mod:`cellpatcher.notebook_session`, and this
module only shows its results. Since v0.2.0 the notebook reads every file through the
preprocessing run (design v0.4 §6): it asks for the run's output folder, and for where the
experiment and masks folder are now only when they were moved since the run. The look is the
preprocessing brick's ``notebook_style`` (its declared embedding API), so the bricks' notebooks
look alike. Needs the ``notebook`` extra (``ipywidgets``, ``matplotlib``).
"""

from __future__ import annotations

import html
import traceback
from collections.abc import Callable
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Rectangle
from morphobricks_preprocessing import notebook_style as style
from morphobricks_preprocessing.run_inputs import RunInputsError

from cellpatcher import __version__
from cellpatcher.cell_scan import CellPatchSizeSummary
from cellpatcher.config import CellPatcherConfig
from cellpatcher.normalization_review import BeforeAfter
from cellpatcher.notebook_form import SettingsForm
from cellpatcher.notebook_session import (
    GroupParameters,
    Preview,
    RunSummary,
    export_patches_of_run,
    group_parameters_of_run,
    preview_image,
    scan_object_sizes,
    summarize_run,
)
from cellpatcher.statistics import PreprocessingCheckError

GALLERY_SIZE = 12
"""Patches shown in the preview gallery."""


def issue_card(error: Exception) -> str:
    """Return the HTML status card of a refusal: each issue's message, files and action.

    A :class:`~cellpatcher.statistics.PreprocessingCheckError` lists its issues; the brick's
    ``RunInputsError`` (a file refused while it is read) its message as a refused run; a
    ``ValueError`` or ``TypeError`` (a setting refused) or ``FileNotFoundError`` (a file a
    table names is missing) its message; anything else, which
    is a bug, its traceback.
    """
    if isinstance(error, PreprocessingCheckError):
        lines = []
        for issue in error.issues:
            files = f" ({', '.join(issue.paths[:5])}{', ...' if len(issue.paths) > 5 else ''})" if issue.paths else ""
            action = f" <i>What to do:</i> {html.escape(issue.action)}" if issue.action else ""
            lines.append(f"<b>[{html.escape(issue.code)}]</b> {html.escape(issue.message)}{html.escape(files)}{action}")
        return style.status("error", "The preprocessing run cannot be used as it is", lines, raw=True)
    if isinstance(error, RunInputsError):  # refused while reading, e.g. a file changed since step 1
        return style.status("error", "The preprocessing run cannot be used as it is", [str(error)])
    if isinstance(error, (ValueError, TypeError, FileNotFoundError)):
        return style.status("error", "Please change the settings", [str(error)])
    trace = html.escape("".join(traceback.format_exception(error)))
    return style.status("error", f"Unexpected error: {type(error).__name__}", [f"<pre>{trace}</pre>"], raw=True)


def guarded(target: widgets.HTML, action: Callable[[], object]) -> object | None:
    """Run *action*; on an error show :func:`issue_card` in *target* and return None.

    Nothing is continued after an error: the step stops there and says what to do.
    """
    try:
        return action()
    except Exception as error:  # every error is shown in the page, with its action or traceback
        target.value = issue_card(error)
        return None


def path_row(text: widgets.Text, message: widgets.HTML) -> widgets.HBox:
    """Return a path box with a Browse… button that opens the system's folder dialog."""
    browse = style.button("Browse…", icon="folder-open")

    def choose(_: object) -> None:
        chosen = style.ask_directory(style.clean_path(text.value))
        if chosen is None:
            message.value = style.no_dialog_status("folder", "the folder")
        elif chosen:
            text.value = chosen

    browse.on_click(choose)
    return widgets.HBox([text, browse])


def _folder_box(description: str, placeholder: str) -> widgets.Text:
    """Return a wide text box for a folder path."""
    return widgets.Text(description=description, placeholder=placeholder, style={"description_width": "200px"},
                        layout=widgets.Layout(width="80%"))


def summary_html(summary: RunSummary) -> str:
    """Return the tiles and table that describe an opened run."""
    conditions = {group[0] for group, _ in summary.groups}
    numbers = style.tiles([
        ("🖼️", f"{len(summary.images)}", "images"),
        ("🧪", f"{len(conditions)}", "conditions"),
        ("📁", f"{len(summary.groups)}", "output folders"),
        ("🎨", f"{len(summary.channels)}", "channels"),
    ])
    lines = [
        f"Experiment read from: {summary.experiment_root}" + (" (given)" if summary.moved_experiment else ""),
        f"Masks folder: {summary.masks_root or 'none'}" + (" (given)" if summary.moved_masks else ""),
        f"Channels: {', '.join(summary.channels)}; array layout {summary.layout}",
        f"Mask types: {', '.join(summary.mask_types) or 'none (grid, random and whole FOV only)'}",
        f"Image type: {summary.image_type or 'not recorded (fluorescence)'}",
    ]
    rows = [["Condition", "Biological repeat", "Technical repeat", "Images"]]
    rows += [[*group, str(count)] for group, count in summary.groups]
    return (style.status("ok", "The preprocessing run is ready", lines) + numbers + style.table(rows, numeric_from=3))


def scan_html(scan: CellPatchSizeSummary) -> str:
    """Return the object counts and sizes of a dataset scan."""
    rows = [
        ["Statistic", "Pixels"],
        ["Smallest", str(scan.min_square_size)],
        ["Median", f"{scan.median_square_size:.0f}"],
        ["90th percentile", f"{scan.percentile_90_square_size:.0f}"],
        ["95th percentile", f"{scan.percentile_95_square_size:.0f}"],
        ["Largest", str(scan.max_square_size)],
    ]
    title = f"{scan.n_included} of {scan.n_objects_total} objects kept in {scan.n_images} images"
    lines = [f"{scan.n_skipped_border} touch the border, {scan.n_skipped_area} are outside the area limits."]
    return style.status("ok", title, lines) + style.table(rows, numeric_from=1)


def parameters_html(table: GroupParameters) -> str:
    """Return the normalization parameters per group and channel, and the groups that stop a run."""
    has_t = any(row["t"] is not None for row in table.rows)
    rows = [["Group", "Channel", *(["Time point"] if has_t else []), "Planes", "Offset", "Scale", "Exact"]]
    problems = []
    for row in table.rows:
        values = [row["group"], f"C{row['channel']}", *([str(row["t"])] if has_t else []), str(row["planes"])]
        if "problem" in row:
            problems.append(f"{row['group']} (C{row['channel']}): {row['problem']}")
            values += ["-", "-", row["problem"]]
        else:
            values += [f"{row['offset']:.6g}", f"{row['scale']:.6g}", "yes" if row["exact"] else "approximate"]
        rows.append(values)
    groups = len({(row["group"], row["t"]) for row in table.rows})
    lines = [f"Method: {table.method}; scope: {table.scope}. Each patch becomes (pixels - offset) / scale, "
             "per channel, with its group's offset and scale.", *table.notes]
    text = style.status("ok", f"Parameters per group ({groups} groups)", lines) + style.table(rows, numeric_from=2)
    if problems:
        text += style.status("error", "These groups cannot be normalized; a run stops on them",
                             [*problems, "Choose a larger scope or another method."])
    return text


class CellPatcherNotebook:
    """The notebook's pages: the run, the settings, a preview and the export."""

    def __init__(self) -> None:
        self.summary: RunSummary | None = None
        self.scan: CellPatchSizeSummary | None = None
        self.form = SettingsForm.build()
        self.run_folder = _folder_box("Preprocessing output folder", "…/my_experiment_MorphoBricks_Preprocessing")
        self.moved = widgets.Checkbox(description="The experiment was moved or copied since preprocessing",
                                      indent=False)
        self.experiment_folder = _folder_box("Experiment folder now", "only if moved")
        self.masks_folder = _folder_box("Masks folder now", "only if moved")
        self.output_folder = _folder_box("Output folder", "…/my_experiment_CellPatcher")
        self.run_message = widgets.HTML()
        self.settings_message = widgets.HTML()
        self.preview_message = widgets.HTML()
        self.export_message = widgets.HTML()
        self.check_button = style.button("Check the preprocessing run", style="primary", main=True)
        self.scan_button = style.button("Scan the object sizes of the dataset", icon="search")
        self.parameters_button = style.button("Show parameters per group", icon="table")
        self.parameters_message = widgets.HTML()
        self.preview_button = style.button("Show the preview", style="primary", main=True)
        self.export_button = style.button("Patch and export every image", style="success", main=True)
        self.image_choice = widgets.Dropdown(options=[], description="Image", style={"description_width": "200px"})
        self.preview_output = widgets.Output()
        self.progress = widgets.IntProgress(value=0, min=0, max=1, layout=widgets.Layout(width="99%"))
        self._connect()

    def _connect(self) -> None:
        """Attach the buttons and the controls that change what is shown."""
        self.check_button.on_click(lambda _: self.check_run())
        self.scan_button.on_click(lambda _: self.scan_objects())
        self.parameters_button.on_click(lambda _: self.show_group_parameters())
        self.preview_button.on_click(lambda _: self.show_preview())
        self.export_button.on_click(lambda _: self.export())
        self.moved.observe(lambda _: self._show_moved(), names="value")
        self.form.mode.observe(lambda _: self._show_mode(), names="value")
        self.form.cell.size.observe(lambda _: self._show_mode(), names="value")
        for control in (self.form.cell.exclude_border, self.form.cell.min_area, self.form.cell.max_area,
                        self.form.mask_type):
            control.observe(lambda _: self._forget_scan(), names="value")
        self._show_moved()
        self._show_mode()

    def _show_moved(self) -> None:
        """Show the experiment and masks folders only for a moved experiment."""
        for box in (self.experiment_folder, self.masks_folder):
            style.set_visible(box, self.moved.value)

    def _show_mode(self) -> None:
        """Show the settings of the chosen mode."""
        for mode, controls in self.form.mode_widgets().items():
            for control in controls:
                style.set_visible(control, mode == self.form.mode.value)
        style.set_visible(self.form.mask_type, self.form.mode.value in {"cell", "random"})
        dataset = self.form.mode.value == "cell" and self.form.cell.size.value not in {"auto", "fixed"}
        style.set_visible(self.form.cell.fixed, self.form.mode.value == "cell" and self.form.cell.size.value == "fixed")
        style.set_visible(self.scan_button, dataset)

    def _forget_scan(self) -> None:
        """Drop a dataset scan made with other object settings."""
        self.scan = None

    def welcome(self) -> None:
        """Show that the notebook is ready, with the package's version."""
        display(style.page("CellPatcher is ready", f"CellPatcher {html.escape(__version__)}", [
            widgets.HTML(style.status("ok", "Run the next cells from top to bottom.", [
                "Step 1: the preprocessing run. Step 2: the settings. Step 3: a preview. Step 4: the export.",
            ])),
        ]))

    def run_step(self) -> None:
        """Show step 1: the preprocessing run and, if it was moved, where the experiment is now."""
        display(style.page("Step 1: the preprocessing run", "CellPatcher reads every image and mask through it.", [
            style.card("Preprocessing run", [
                path_row(self.run_folder, self.run_message),
                style.help_text("The output folder the preprocessing notebook wrote (it holds "
                                "<b>run_manifest.json</b> and <b>plane_histograms.csv</b>)."),
                self.moved,
                path_row(self.experiment_folder, self.run_message),
                path_row(self.masks_folder, self.run_message),
                style.help_text("Only if the experiment (or its masks folder) is no longer where it was "
                                "preprocessed. Leave the masks folder empty if only the experiment moved."),
                self.check_button,
                self.run_message,
            ]),
        ]))

    def check_run(self) -> RunSummary | None:
        """Open the run (``RunInputs.open``) and show what it recorded, or why it cannot be used."""
        self.run_message.value = style.working("Opening the preprocessing run and checking every file…")
        moved = self.moved.value
        summary = guarded(self.run_message, lambda: summarize_run(
            style.clean_path(self.run_folder.value),
            style.clean_path(self.experiment_folder.value) if moved else None,
            style.clean_path(self.masks_folder.value) if moved else None,
        ))
        self.summary, self.scan = summary, None
        if summary is not None:
            self.form.set_run(summary)
            self.image_choice.options = list(summary.images)
            self.image_choice.value = summary.images[0]
            self.run_message.value = summary_html(summary)
            if not self.output_folder.value:
                root = summary.experiment_root
                self.output_folder.value = str(root.parent / f"{root.name}_CellPatcher")
        return summary

    def settings_step(self) -> None:
        """Show step 2: the patching strategy and processing settings."""
        form = self.form
        display(style.page("Step 2: the settings", "How the images are cut into patches, and what is done to them.", [
            style.card("Patching strategy", [form.mode, form.mask_type, *form.cell.controls(), self.scan_button,
                                             form.grid_size, form.grid_overlap, form.grid_partial, form.random_size,
                                             form.random_count, form.random_seed, form.random_masks]),
            style.card("Planes", [form.planes.z_policy, form.planes.z_index, form.planes.t_policy,
                                  form.planes.t_index,
                                  style.help_text("Shown for images with Z or T planes. Normalization needs one "
                                                  "plane per patch.")]),
            style.card("Processing", [form.min_mean, form.output_size, form.pad, form.channels,
                                      style.help_text("Split channels: one patch per selected channel; none "
                                                      "selected keeps all channels in each patch.")]),
            style.card("Normalization", [*form.normalization.controls(), style.help_text(
                "The image and biological-repeat scopes use the preprocessing run's plane_histograms.csv and "
                "experiment_design.csv; the image type is the one confirmed in preprocessing. "
                "<b>Show parameters per group</b> lists the offset and scale every group gets."),
                self.parameters_button, self.parameters_message]),
            style.card("Export", [form.file_format]),
            self.settings_message,
        ]))

    def _dataset_size(self, statistic: str) -> int:
        """Return the patch size of a dataset statistic, from the object scan."""
        if self.scan is None:
            raise ValueError("The patch size comes from the dataset's objects: press 'Scan the object sizes of the "
                             "dataset' in step 2 first, or choose another patch size.")
        return self.scan.recommended_size(method=statistic)

    def config(self) -> CellPatcherConfig:
        """Return the settings as a ``CellPatcherConfig`` (step 1 must have opened a run)."""
        if self.summary is None:
            raise ValueError("Check the preprocessing run in step 1 first.")
        return self.form.config(self.summary, self._dataset_size)

    def scan_objects(self) -> CellPatchSizeSummary | None:
        """Measure the objects of every mask of the run, for a dataset patch size."""
        self.settings_message.value = style.working("Reading every mask through the preprocessing run…")
        scan = guarded(self.settings_message, lambda: scan_object_sizes(self.config_for_scan(), self._summary()))
        self.scan = scan
        if scan is not None:
            self.settings_message.value = scan_html(scan)
        return scan

    def show_group_parameters(self) -> GroupParameters | None:
        """Show the normalization parameters of every group of the run's table, before any patch."""
        self.parameters_message.value = style.working("Reading the preprocessing run's plane_histograms.csv…")
        table = guarded(self.parameters_message, lambda: group_parameters_of_run(self.config_for_scan(),
                                                                               self._summary()))
        if table is not None:
            self.parameters_message.value = parameters_html(table)
        return table

    def config_for_scan(self) -> CellPatcherConfig:
        """Return the settings with a per-object size, to scan objects before a dataset size exists."""
        return self.form.config(self._summary(), lambda _: 0)

    def _summary(self) -> RunSummary:
        """Return the opened run, or refuse before step 1."""
        if self.summary is None:
            raise ValueError("Check the preprocessing run in step 1 first.")
        return self.summary

    def preview_step(self) -> None:
        """Show step 3: one image patched with the settings, nothing written."""
        display(style.page("Step 3: preview", "One image, patched with the settings of step 2. Nothing is written.", [
            style.card("Preview", [self.image_choice, self.preview_button, self.preview_message, self.preview_output]),
        ]))

    def show_preview(self) -> Preview | None:
        """Patch the chosen image in memory and draw its patches."""
        self.preview_message.value = style.working("Patching the image…")
        preview = guarded(self.preview_message, lambda: preview_image(self.config(), self._summary(),
                                                                      self.image_choice.value))
        self.preview_output.clear_output()
        if preview is not None:
            patches = preview.result.patches
            lines = ["Rectangles: where the patches are, on the first plane and channel. Below: the first patches."]
            if preview.before_after:
                lines.append("Then the first patches without and with normalization, with their intensity "
                             "histograms.")
            self.preview_message.value = style.status("ok", f"{len(patches)} patches in {preview.image}", lines)
            with self.preview_output:
                draw_preview(preview)
                draw_before_after(preview.before_after)
        return preview

    def export_step(self) -> None:
        """Show step 4: patch every image of the run and export the patches."""
        style.set_visible(self.progress, False)
        display(style.page("Step 4: patch and export", "Every image of the preprocessing run, with the settings.", [
            style.card("Output", [
                path_row(self.output_folder, self.export_message),
                style.help_text("One patches.csv, images.csv and run_config.json for the whole experiment, and "
                                "the patch files in one subfolder per condition and biological repeat. Keep it "
                                "outside the experiment folder (on Google Drive on Colab)."),
                self.export_button, self.progress, self.export_message,
            ]),
        ]))

    def export(self) -> list | None:
        """Patch and export every image of the run, showing progress per output folder."""
        self.export_message.value = style.working("Patching…")
        style.set_visible(self.progress, True)

        def progress(done: int, total: int, leaf: object) -> None:
            self.progress.max, self.progress.value = max(total, 1), done
            if leaf is not None:
                self.export_message.value = style.working(f"Folder {done + 1} of {total}: {leaf.relative_folder}")

        results = guarded(self.export_message, lambda: export_patches_of_run(
            self.config(), self._summary(), style.clean_path(self.output_folder.value), progress))
        if results is not None:
            rows = [["Condition", "Biological repeat", "Technical repeat", "Patches"]]
            rows += [[r.condition, r.biological_repeat, r.technical_repeat, str(len(r.result.patches))]
                     for r in results]
            total = sum(len(r.result.patches) for r in results)
            output = Path(style.clean_path(self.output_folder.value))
            self.export_message.value = style.status("ok", f"Exported {total} patches", [
                f"Table of every patch: {output / 'patches.csv'}", f"Settings used: {output / 'run_config.json'}",
            ]) + style.table(rows, numeric_from=3)
        return results


def _display_range(plane: np.ndarray) -> tuple[float, float]:
    """Return the 1st and 99.8th percentiles of a plane, for display only."""
    low, high = np.percentile(plane, (1.0, 99.8))
    return float(low), float(max(high, low + 1e-9))


def draw_preview(preview: Preview) -> None:
    """Draw the image's first plane with the patch rectangles, then a gallery of patches."""
    plane = preview.plane[0]
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(plane, cmap="gray", vmin=_display_range(plane)[0], vmax=_display_range(plane)[1])
    for patch in preview.result.patches:
        axis.add_patch(Rectangle((patch.x0 - 0.5, patch.y0 - 0.5), patch.x1 - patch.x0, patch.y1 - patch.y0,
                                 fill=False, edgecolor="#F59E0B", linewidth=1))
    axis.set_title(preview.image)
    axis.axis("off")
    plt.show()
    arrays = (preview.result.patch_arrays or [])[:GALLERY_SIZE]
    if arrays:
        figure, axes = plt.subplots(1, len(arrays), figsize=(2 * len(arrays), 2.2), squeeze=False)
        for axis, array, patch in zip(axes[0], arrays, preview.result.patches, strict=False):
            shown = np.asarray(array, dtype=float).reshape(-1, *np.shape(array)[-2:])[0]
            axis.imshow(shown, cmap="gray")
            axis.set_title(patch.label_id if patch.label_id is not None else "", fontsize=8)
            axis.axis("off")
        plt.show()
    plt.close("all")


def draw_before_after(pairs: tuple[BeforeAfter, ...]) -> None:
    """Draw each patch without and with normalization, each beside its intensity histogram."""
    if not pairs:
        return
    figure, axes = plt.subplots(len(pairs), 4, figsize=(10, 2.3 * len(pairs)), squeeze=False)
    for number, (row, pair) in enumerate(zip(axes, pairs, strict=True), start=1):
        # The patch_id is shown whole or not at all: an identity is never parsed (here, cut at a "/").
        name = f"label {pair.label_id}" if pair.label_id is not None else f"patch {number}"
        channel = "" if pair.channel is None else f", C{pair.channel}"
        for column, (plane, histogram, state) in enumerate((
            (pair.raw, pair.raw_histogram, "without normalization"),
            (pair.normalized, pair.normalized_histogram, "normalized"),
        )):
            image_axis, histogram_axis = row[2 * column], row[2 * column + 1]
            image_axis.imshow(plane, cmap="gray")
            image_axis.set_title(f"{name}{channel}: {state}", fontsize=8)
            image_axis.axis("off")
            if histogram.is_empty:
                histogram_axis.set_title("no finite pixels", fontsize=8)
            else:
                histogram_axis.fill_between(histogram.representatives, histogram.counts, step="mid", color="#2563EB")
                histogram_axis.set_title(f"intensities, {state}", fontsize=8)
            histogram_axis.tick_params(labelsize=7)
    figure.suptitle("Before and after normalization (first plane shown)", fontsize=10)
    plt.tight_layout()
    plt.show()
    plt.close(figure)
