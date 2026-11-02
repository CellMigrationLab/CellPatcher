"""The notebook's actions on a preprocessing run, without widgets (cellpatcher.notebook_session, CP-3).

The v0.2 notebook reads every file through the preprocessing run: it is given the run's output
folder and, only for a moved experiment, where the experiment and masks are now. Its preview
patches one image with the file-run code, so it gives the patches the export gives.
"""

import csv
import shutil

import numpy as np
import pytest
import tifffile

from cellpatcher import CellPatcherConfig, run_patching_condition_repeat_tree
from cellpatcher.notebook_session import (
    export_patches_of_run,
    preview_image,
    scan_object_sizes,
    summarize_run,
)
from cellpatcher.statistics import PreprocessingCheckError


@pytest.fixture
def clean(tmp_path):
    from morphobricks_preprocessing.testing import make_clean_run

    return make_clean_run(tmp_path / "clean", axes="CYX", shape=(2, 32, 40), mask_types=("Cells",))


def _cell_config():
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 12
    config.cell.exclude_border = False
    return config


def test_the_summary_is_what_the_run_recorded(clean):
    summary = summarize_run(clean.output)
    assert summary.images == tuple(f"{c}/rep1/{n}.ome.tif" for c in ("ctrl", "drug") for n in ("a", "b"))
    assert summary.groups == ((("ctrl", "rep1", "single"), 2), (("drug", "rep1", "single"), 2))
    assert (summary.layout, summary.mask_types, summary.channels) == ("CYX", ("Cells",), ("C0", "C1"))
    assert summary.experiment_root in {clean.experiment, clean.experiment.resolve()}
    assert not summary.moved_experiment and summary.masks_argument is None


def test_an_empty_folder_is_asked_for():
    with pytest.raises(ValueError, match="preprocessing run's output folder"):
        summarize_run("  ")


def test_the_preview_gives_the_patches_the_export_gives(clean, tmp_path):
    # The preview is the file-run code on one image: same patch ids, coordinates and pixels.
    summary = summarize_run(clean.output)
    preview = preview_image(_cell_config(), summary, "drug/rep1/b.ome.tif")
    assert preview.plane.shape == (2, 32, 40)
    results = export_patches_of_run(_cell_config(), summary, tmp_path / "out")
    exported = {patch.patch_id: patch for leaf in results for patch in leaf.result.patches}
    assert preview.result.patches
    for patch, array in zip(preview.result.patches, preview.result.patch_arrays, strict=True):
        assert patch.source_relative_path == "drug/rep1/b.ome.tif"
        twin = exported[patch.patch_id]
        assert (patch.y0, patch.y1, patch.x0, patch.x1) == (twin.y0, twin.y1, twin.x0, twin.x1)
        assert np.array_equal(array, tifffile.imread(tmp_path / "out" / twin.patch_file))


def test_the_export_writes_one_table_for_the_experiment(clean, tmp_path):
    summary = summarize_run(clean.output)
    export_patches_of_run(_cell_config(), summary, tmp_path / "out")
    assert {path.name for path in (tmp_path / "out").iterdir()} == {
        "ctrl", "drug", "patches.csv", "images.csv", "run_config.json",
    }
    with open(tmp_path / "out" / "images.csv", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4


def test_the_export_is_the_tree_run(clean, tmp_path):
    # Nothing of its own: the same patches as run_patching_condition_repeat_tree on the run.
    summary = summarize_run(clean.output)
    mine = export_patches_of_run(_cell_config(), summary, tmp_path / "a")
    config = _cell_config()
    config.common.preprocessing_run = str(clean.output)
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "b")
    tree = run_patching_condition_repeat_tree(clean.experiment, config=config, combine_metadata=True)
    assert [p.patch_id for leaf in mine for p in leaf.result.patches] == [
        p.patch_id for leaf in tree for p in leaf.result.patches
    ]


@pytest.mark.parametrize("inside", ["experiment", "masks", "output"])
def test_an_output_folder_inside_the_inputs_is_refused(clean, inside):
    summary = summarize_run(clean.output)
    with pytest.raises(ValueError, match="is inside the"):
        export_patches_of_run(_cell_config(), summary, getattr(clean, inside) / "patches")


def test_a_moved_experiment_is_read_where_it_is_now(clean, tmp_path):
    shutil.copytree(clean.experiment, tmp_path / "moved" / "experiment", copy_function=shutil.copy2)
    shutil.copytree(clean.masks, tmp_path / "moved" / "masks", copy_function=shutil.copy2)
    summary = summarize_run(clean.output, tmp_path / "moved" / "experiment", tmp_path / "moved" / "masks")
    assert summary.moved_experiment and summary.masks_argument == tmp_path / "moved" / "masks"
    results = export_patches_of_run(_cell_config(), summary, tmp_path / "out")
    assert all(leaf.image_folder.startswith(str(tmp_path / "moved")) for leaf in results)


def test_an_image_added_after_the_run_is_refused_when_the_run_is_checked(clean):
    tifffile.imwrite(clean.experiment / "ctrl" / "rep1" / "new.tif", np.ones((2, 32, 40), np.uint16),
                     metadata={"axes": "CYX"})
    with pytest.raises(PreprocessingCheckError) as refused:
        summarize_run(clean.output)
    assert [issue.code for issue in refused.value.issues] == ["unrecorded_files"]


def test_the_object_scan_measures_every_mask_of_the_run(clean):
    summary = summarize_run(clean.output)
    scan = scan_object_sizes(_cell_config(), summary)
    assert scan.n_images == 4
    assert scan.n_included == scan.n_objects_total > 0


def test_a_preview_of_an_image_not_in_the_run_is_refused(clean):
    summary = summarize_run(clean.output)
    with pytest.raises(ValueError, match="not one of them"):
        preview_image(_cell_config(), summary, "ctrl/rep1/other.tif")
