"""Files the preprocessing run did not record are refused, never left out (preprocessing v0.4.0rc2).

Since CP-2a a file run patches the run's images (``RunInputs.fovs()``) and discovers nothing
itself (design v0.4 §6). With preprocessing v0.4.0rc1 an image added to the experiment after
the run was therefore not patched and nothing said so (the documented known limitation).
Since rc2 ``RunInputs.open`` refuses an experiment holding images or masks the run did not
record (``UnrecordedFilesError``), and CellPatcher surfaces that as ``unrecorded_files``.
"""

import json

import numpy as np
import pytest
import tifffile

from cellpatcher import CellPatcherConfig, validate_image_and_mask_folders
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.statistics import PreprocessingCheckError


def _clean_run(tmp_path, **kwargs):
    from morphobricks_preprocessing.testing import make_clean_run

    return make_clean_run(tmp_path / "clean", axes="YX", shape=(16, 20), **kwargs)


def _config(run, mode="whole_fov"):
    config = CellPatcherConfig(mode=mode)
    config.cell.patch_size = 8
    config.cell.exclude_border = False
    config.common.preprocessing_run = str(run.output)
    return config


def _add_image(run, relative):
    path = run.experiment / relative
    tifffile.imwrite(path, np.ones((16, 20), np.uint16), metadata={"axes": "YX"})
    return path


def test_an_image_added_after_the_run_is_refused_not_skipped(tmp_path):
    # With rc1 the tree run patched the four recorded images and left new.tif out without a word.
    run = _clean_run(tmp_path)
    _add_image(run, "ctrl/rep1/new.tif")
    with pytest.raises(PreprocessingCheckError, match=r"1 image \(ctrl/rep1/new.tif\) that the preprocessing run "
                                                      r"did not record") as refused:
        run_patching_condition_repeat_tree(run.experiment, config=_config(run))
    (issue,) = refused.value.issues
    assert issue.code == "unrecorded_files"
    assert list(issue.paths) == ["ctrl/rep1/new.tif"]
    assert "never leaves one out" in issue.action


def test_an_image_added_after_the_run_is_refused_in_a_folder_run(tmp_path):
    # Also in a folder run, whose folder is one leaf of the experiment: the whole experiment is checked.
    run = _clean_run(tmp_path)
    _add_image(run, "drug/rep1/new.tif")
    with pytest.raises(PreprocessingCheckError, match=r"drug/rep1/new.tif") as refused:
        run_patching_pipeline(run.experiment, config=_config(run))
    assert [issue.code for issue in refused.value.issues] == ["unrecorded_files"]


def test_a_mask_added_after_the_run_is_refused(tmp_path):
    # A mask of a new type for a recorded image (masks without an image are the run's warning).
    run = _clean_run(tmp_path, mask_types=("Cells",))
    added = run.masks / "Nuclei" / "ctrl" / "rep1" / "a.tif"
    added.parent.mkdir(parents=True)
    added.write_bytes((run.masks / "Cells" / "ctrl" / "rep1" / "a.tif").read_bytes())
    with pytest.raises(PreprocessingCheckError, match="1 mask") as refused:
        run_patching_condition_repeat_tree(run.experiment, config=_config(run, "cell"), mask_root_path=run.masks)
    (issue,) = refused.value.issues
    assert issue.code == "unrecorded_files"
    assert list(issue.paths) == ["Nuclei/ctrl/rep1/a.tif"]


def test_validate_image_and_mask_folders_refuses_an_added_image(tmp_path):
    run = _clean_run(tmp_path, mask_types=("Cells",))
    _add_image(run, "ctrl/rep1/new.tif")
    with pytest.raises(PreprocessingCheckError, match="ctrl/rep1/new.tif"):
        validate_image_and_mask_folders(run.experiment, run.masks, preprocessing_run=run.output)


def test_a_run_with_ome_zarr_images_without_the_zarr_package_is_refused_with_the_extra(tmp_path, monkeypatch):
    # rc2's RunInputs.open refuses (never skips) OME-Zarr label images it cannot check without zarr.
    from morphobricks_preprocessing import run_inputs
    from morphobricks_preprocessing.zarr_input import ZarrUnavailableError

    def zarr_missing(*args, **kwargs):
        raise ZarrUnavailableError("zarr is not installed")

    run = _clean_run(tmp_path)
    monkeypatch.setattr(run_inputs, "unrecorded_files", zarr_missing)
    with pytest.raises(PreprocessingCheckError, match="zarr package is not installed") as refused:
        run_patching_condition_repeat_tree(run.experiment, config=_config(run))
    (issue,) = refused.value.issues
    assert issue.code == "zarr_extra_missing"
    assert "cellpatcher[zarr]" in issue.action


def test_a_run_recording_a_mask_without_axes_is_refused_when_opened(tmp_path):
    # A run before v0.4.0 (rc1) recorded a mask whose header put X before Y with no axes; rc2's
    # RunInputs.open refuses such a record. CellPatcher's own check (file_runs._record) stays as a
    # safety net behind it.
    run = _clean_run(tmp_path, mask_types=("Cells",))
    manifest_path = run.output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["masks"]["files"][0]["axes"] = ""
    manifest["masks"]["files"][0]["roles"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PreprocessingCheckError, match="recorded 1 mask without axes") as refused:
        run_patching_condition_repeat_tree(run.experiment, config=_config(run, "cell"), mask_root_path=run.masks)
    assert [issue.code for issue in refused.value.issues] == ["preprocessing_run_refused"]
