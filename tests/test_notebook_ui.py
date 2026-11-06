"""The notebook's pages (cellpatcher.notebook_ui, CP-3): its buttons call the session and show the result.

Skipped without the notebook extra (ipywidgets, matplotlib).
"""

import json

import numpy as np
import pytest
import tifffile

pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")

import matplotlib

matplotlib.use("Agg")

from cellpatcher.notebook_ui import CellPatcherNotebook, issue_card
from cellpatcher.statistics import PreprocessingCheckError


@pytest.fixture
def clean(tmp_path):
    from morphobricks_preprocessing.testing import make_clean_run

    return make_clean_run(tmp_path / "clean", axes="CYX", shape=(2, 32, 40), mask_types=("Cells",))


def _checked(clean):
    notebook = CellPatcherNotebook()
    for step in (notebook.welcome, notebook.run_step, notebook.settings_step, notebook.preview_step,
                 notebook.export_step):
        step()
    notebook.run_folder.value = str(clean.output)
    assert notebook.check_run() is not None, notebook.run_message.value
    return notebook


def test_checking_the_run_fills_the_settings_and_the_output_folder(clean):
    notebook = _checked(clean)
    assert "The preprocessing run is ready" in notebook.run_message.value
    assert notebook.form.mask_type.value == "Cells"
    assert notebook.image_choice.value == "ctrl/rep1/a.ome.tif"
    assert notebook.output_folder.value.endswith("experiment_CellPatcher")


def test_the_config_is_a_file_run_of_the_checked_run(clean):
    notebook = _checked(clean)
    notebook.form.mode.value = "grid"
    config = notebook.config()
    assert config.common.preprocessing_run == str(clean.output)
    assert (config.common.image_axes, config.common.mask_axes) == ("YX", "YX")
    assert config.matching.mask_type is None  # grid reads no masks


def test_preview_and_export_through_the_buttons(clean, tmp_path):
    notebook = _checked(clean)
    notebook.form.cell.size.value = "fixed"
    notebook.form.cell.fixed.value = 12
    notebook.form.cell.exclude_border.value = False
    preview = notebook.show_preview()
    assert preview is not None and preview.result.patches, notebook.preview_message.value
    notebook.output_folder.value = str(tmp_path / "out")
    results = notebook.export()
    assert results is not None, notebook.export_message.value
    assert "Exported" in notebook.export_message.value
    run_config = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["preprocessing"]["run"] == str(clean.output)


def test_a_dataset_patch_size_needs_the_scan_first(clean):
    notebook = _checked(clean)
    notebook.form.cell.size.value = "p95"
    assert notebook.show_preview() is None
    assert "Scan the object sizes of the dataset" in notebook.preview_message.value
    assert notebook.scan_objects() is not None
    assert notebook.config().cell.patch_size == notebook.scan.recommended_size(method="p95")


def test_changing_the_object_settings_drops_the_scan(clean):
    notebook = _checked(clean)
    notebook.form.cell.size.value = "max"
    notebook.scan_objects()
    notebook.form.cell.min_area.value = 5
    assert notebook.scan is None


def test_an_added_image_is_shown_as_unrecorded_with_its_action(clean):
    tifffile.imwrite(clean.experiment / "ctrl" / "rep1" / "new.tif", np.ones((2, 32, 40), np.uint16),
                     metadata={"axes": "CYX"})
    notebook = CellPatcherNotebook()
    notebook.run_folder.value = str(clean.output)
    assert notebook.check_run() is None
    assert "[unrecorded_files]" in notebook.run_message.value
    assert "ctrl/rep1/new.tif" in notebook.run_message.value
    assert "What to do:" in notebook.run_message.value


def test_steps_before_the_run_is_checked_say_so():
    notebook = CellPatcherNotebook()
    assert notebook.show_preview() is None
    assert "Check the preprocessing run in step 1 first" in notebook.preview_message.value


def test_an_unexpected_error_shows_its_traceback():
    try:
        raise KeyError("boom")
    except KeyError as error:
        card = issue_card(error)
    assert "Unexpected error: KeyError" in card and "boom" in card
    from morphobricks_preprocessing.validation import error as issue

    refusal = issue_card(PreprocessingCheckError([issue("some_code", "a message", ["a.tif"], "do this")]))
    assert "[some_code]" in refusal and "a.tif" in refusal and "do this" in refusal
    # A file refused while it is read (RunInputsError is a ValueError) is a refused run, not a setting.
    from morphobricks_preprocessing.run_inputs import RunInputsError

    changed = issue_card(RunInputsError("a.tif changed since the run"))
    assert "The preprocessing run cannot be used as it is" in changed and "a.tif changed" in changed
    assert "Please change the settings" not in changed


# CP-3b: the normalization parameters per group and the before/after views come back.


def _normalized(notebook, scope="image"):
    notebook.form.normalization.enabled.value = True
    notebook.form.normalization.scope.value = scope
    notebook.form.cell.size.value = "fixed"
    notebook.form.cell.fixed.value = 12
    notebook.form.cell.exclude_border.value = False


def test_show_parameters_per_group_lists_every_group(clean):
    notebook = _checked(clean)
    _normalized(notebook)
    notebook.parameters_button.click()
    text = notebook.parameters_message.value
    assert "Parameters per group (4 groups)" in text, text
    assert "drug/rep1/b.ome.tif" in text and "C1" in text
    assert "background included" in text  # the normalization warning is shown with the table


def test_show_parameters_per_group_says_what_is_missing(clean):
    notebook = _checked(clean)
    assert notebook.show_group_parameters() is None
    assert "Normalization is off" in notebook.parameters_message.value
    _normalized(notebook, scope="patch")
    assert notebook.show_group_parameters() is None
    assert "The patch scope has no parameters per group" in notebook.parameters_message.value


def test_a_group_that_cannot_be_normalized_is_shown_in_red():
    from cellpatcher.notebook_session import GroupParameters
    from cellpatcher.notebook_ui import parameters_html

    rows = (
        {"group": "rep1", "channel": 0, "t": 0, "planes": 2, "offset": 10.0, "scale": 90.5, "exact": True},
        {"group": "rep1", "channel": 1, "t": 0, "planes": 2, "problem": "zero intensity spread"},
    )
    text = parameters_html(GroupParameters("minmax", "biological_repeat", rows, ()))
    assert "Time point" in text and "90.5" in text
    assert "mb-error" in text and "rep1 (C1): zero intensity spread" in text


def test_the_normalized_preview_draws_the_patches_before_and_after(clean, monkeypatch):
    import cellpatcher.notebook_ui as ui

    drawn = []
    monkeypatch.setattr(ui, "draw_before_after", drawn.append)
    notebook = _checked(clean)
    _normalized(notebook)
    preview = notebook.show_preview()
    assert preview is not None, notebook.preview_message.value
    assert drawn == [preview.before_after] and preview.before_after
    assert "without and with normalization" in notebook.preview_message.value


def test_the_before_after_figure_has_a_patch_and_a_histogram_per_state(clean, monkeypatch):
    import matplotlib.pyplot as plt

    from cellpatcher.notebook_ui import draw_before_after

    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    notebook = _checked(clean)
    _normalized(notebook)
    pairs = notebook.show_preview().before_after
    figures.clear()
    draw_before_after(pairs)
    (figure,) = figures
    titles = [axis.get_title() for axis in figure.axes]
    assert len(titles) == 4 * len(pairs)
    assert titles[:4] == ["label 1, C0: without normalization", "intensities, without normalization",
                          "label 1, C0: normalized", "intensities, normalized"]


def test_a_patch_without_label_is_numbered_its_id_never_cut(monkeypatch):
    import matplotlib.pyplot as plt
    from morphobricks_normalization import compute_plane_histogram

    from cellpatcher.normalization_review import BeforeAfter
    from cellpatcher.notebook_ui import draw_before_after

    plane = np.arange(16, dtype=np.float32).reshape(4, 4)
    histogram = compute_plane_histogram(plane, 8)
    pair = BeforeAfter("fov/a/b/y0-4_x0-4", None, None, plane, plane, histogram, histogram)
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    draw_before_after((pair,))
    titles = [axis.get_title() for axis in figures[0].axes]
    assert titles[0] == "patch 1: without normalization"
    assert not any("y0-4_x0-4" in title for title in titles)


def test_the_plane_controls_have_no_depth_policy():
    """Issue #83: Z stacks are refused upstream, so the form offers the T policy only."""
    from cellpatcher.notebook_form import PlaneControls

    controls = PlaneControls.build()
    assert not hasattr(controls, "z_policy") and not hasattr(controls, "z_index")
    assert {value for _, value in controls.t_policy.options} == {"keep", "all", "first", "select"}
