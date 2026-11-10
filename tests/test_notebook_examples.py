"""The strategy examples notebook's pages (cellpatcher.notebook_examples, CP-3b).

Each example is the main notebook's preview of one image of the preprocessing run, with one
strategy and its settings: read through RunInputs, patched in memory, nothing written. Skipped
without the notebook extra (ipywidgets, matplotlib).
"""

import pytest

pytest.importorskip("ipywidgets", reason="the notebook extra installs ipywidgets")
pytest.importorskip("matplotlib", reason="the notebook extra installs matplotlib")

import matplotlib

matplotlib.use("Agg")

from cellpatcher.notebook_examples import STRATEGIES, StrategyExamplesNotebook
from cellpatcher.notebook_session import preview_image, summarize_run


@pytest.fixture
def clean(tmp_path):
    from tests.helpers import make_clean_run

    return make_clean_run(tmp_path / "clean", axes="CYX", shape=(2, 32, 40), mask_types=("Cells",))


def _checked(run):
    notebook = StrategyExamplesNotebook()
    notebook.welcome()
    notebook.run_step()
    for mode in STRATEGIES:
        notebook.strategy_step(mode)
    notebook.run_folder.value = str(run.output)
    assert notebook.check_run() is not None, notebook.run_message.value
    notebook.image_choice.value = "drug/rep1/b.ome.tif"
    notebook.form.grid_size.value = 16
    notebook.form.random_size.value, notebook.form.random_count.value = 8, 3
    notebook.form.cell.exclude_border.value = False
    return notebook


@pytest.mark.parametrize("mode", list(STRATEGIES))
def test_each_example_is_the_preview_with_its_strategy(clean, mode, tmp_path):
    notebook = _checked(clean)
    preview = notebook.show_example(mode)
    assert preview is not None, notebook.example_messages[mode].value
    assert preview.result.patches and {patch.mode for patch in preview.result.patches} == {mode}
    assert f"{STRATEGIES[mode][0]}: {len(preview.result.patches)} patches" in notebook.example_messages[mode].value
    # The same patches as the main notebook's preview with the same settings.
    config = notebook.config()
    same = preview_image(config, summarize_run(clean.output), "drug/rep1/b.ome.tif")
    assert [p.patch_id for p in preview.result.patches] == [p.patch_id for p in same.result.patches]
    assert not list(tmp_path.glob("**/patches.csv"))  # nothing written


def test_every_strategy_shows_its_own_settings_on_its_page(clean):
    notebook = _checked(clean)
    for mode in STRATEGIES:
        notebook.show_example(mode)  # the last example shown does not hide the other pages' settings
    from morphobricks_preprocessing import notebook_style as style

    for control in (notebook.form.grid_size, notebook.form.random_seed, notebook.form.cell.exclude_border,
                    notebook.form.mask_type):
        assert style.is_visible(control)
    assert not style.is_visible(notebook.form.cell.fixed)  # only for a fixed cell size
    notebook.form.cell.size.value = "fixed"
    assert style.is_visible(notebook.form.cell.fixed)


def test_the_cell_example_without_masks_says_why(tmp_path):
    from tests.helpers import make_clean_run

    run = make_clean_run(tmp_path / "clean", axes="CYX", shape=(2, 32, 40), mask_types=())
    notebook = StrategyExamplesNotebook()
    notebook.run_folder.value = str(run.output)
    assert notebook.check_run() is not None, notebook.run_message.value
    assert notebook.show_example("cell") is None
    assert "recorded none" in notebook.example_messages["cell"].value
    assert notebook.show_example("grid") is not None


def test_an_example_before_the_run_is_checked_says_so():
    notebook = StrategyExamplesNotebook()
    assert notebook.show_example("whole_fov") is None
    assert "Check the preprocessing run in step 1 first" in notebook.example_messages["whole_fov"].value

