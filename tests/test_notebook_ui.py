"""The notebook's pages (cellpatcher.notebook_ui, CP-3): its buttons call the session and show the result.

Skipped without the notebook extra (ipywidgets, matplotlib).
"""

import json

import numpy as np
import pytest
import tifffile

pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from cellpatcher.notebook_ui import CellPatcherNotebook, issue_card  # noqa: E402
from cellpatcher.statistics import PreprocessingCheckError  # noqa: E402


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
