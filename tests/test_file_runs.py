"""CP-2a: folder and tree runs read files only through a clean preprocessing run (design v0.4 §6).

The images, masks and FOV identities come from ``RunInputs`` (``read_plane``, ``read_mask``,
``fovs``), opened with the experiment root and masks folder CellPatcher is given, so an
experiment moved or copied since its run is checked and read where it is now. The settings
that told CellPatcher how to read files are refused for file runs with a migration message.
"""

import hashlib
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from cellpatcher import CellPatcherConfig
from cellpatcher.orchestration import (
    run_patching_condition_repeat_tree,
    run_patching_pipeline,
)
from cellpatcher.statistics import PreprocessingCheckError


def _clean_run(tmp_path, **kwargs):
    from morphobricks_preprocessing.testing import make_clean_run

    return make_clean_run(tmp_path / "clean", **kwargs)


def _cell_config(run, **common):
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 8
    config.cell.exclude_border = False
    config.common.extract_arrays = True
    config.common.preprocessing_run = str(run.output)
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def test_a_tree_without_a_preprocessing_run_is_refused_with_the_command(tmp_path):
    # Former code: the tree was walked and read by CellPatcher itself.
    root = tmp_path / "raw_data"
    (root / "control").mkdir(parents=True)
    tifffile.imwrite(root / "control" / "c.tif", np.ones((8, 8), np.uint16))
    with pytest.raises(PreprocessingCheckError, match=r"(?s)preprocessing_run is not set.*morphobricks-preprocessing run") as refused:
        run_patching_condition_repeat_tree(root, config=CellPatcherConfig(mode="whole_fov"))
    assert [issue.code for issue in refused.value.issues] == ["no_preprocessing_run"]


def _patches(leaves):
    return {(p.patch_id, p.mask_file): array for leaf in leaves for p, array in zip(leaf.result.patches, leaf.result.patch_arrays)}


def _labels_moved(path):
    """Rewrite a mask with its two objects swapped in place, keeping its size: changed labels."""
    labels = tifffile.imread(path)
    tifffile.imwrite(path, np.where(labels > 0, 3 - labels, 0).astype(labels.dtype), metadata={"axes": "YX"})


def test_a_moved_experiment_is_opened_where_it_is_now(tmp_path):
    # CP-2a: RunInputs.open with the experiment root and masks folder CellPatcher is given.
    # Former code (CP-1): the run was opened where it recorded the experiment, so a moved
    # experiment was refused.
    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",))
    before = _patches(run_patching_condition_repeat_tree(run.experiment, config=_cell_config(run)))
    moved, moved_masks = tmp_path / "moved" / "experiment", tmp_path / "moved" / "masks"
    moved.parent.mkdir()
    shutil.move(run.experiment, moved)
    shutil.move(run.masks, moved_masks)
    after = _patches(run_patching_condition_repeat_tree(moved, mask_root_path=moved_masks, config=_cell_config(run)))
    assert after.keys() == before.keys() and len(after) == 8
    assert all(np.array_equal(after[key], before[key]) for key in before)
    # Without the masks folder's new place the run's masks are missing: refused, never guessed.
    with pytest.raises(PreprocessingCheckError, match="4 masks changed or missing since the run"):
        run_patching_condition_repeat_tree(moved, config=_cell_config(run))


def test_a_file_runs_patches_are_delivered_lazily_through_the_run_where_it_is_now(tmp_path):
    # CP-2b: lazy delivery from a file path read the file itself (with common.image_axes and its
    # own header check). A folder or tree run's patches are now read through their run.
    from dataclasses import replace

    from cellpatcher import iter_patch_arrays, iter_run_patch_arrays

    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",))
    config = _cell_config(run)
    config.extraction.apply_mask = True
    leaves = run_patching_condition_repeat_tree(run.experiment, config=config)
    eager = _patches(leaves)
    patches = [patch for leaf in leaves for patch in leaf.result.patches]
    with pytest.raises(TypeError, match=r"path .*a.ome.tif.*iter_run_patch_arrays\(patches, config\)"):
        next(iter_patch_arrays(leaves[0].result.patches, [run.experiment / "ctrl" / "rep1" / "a.ome.tif"], config))
    moved, moved_masks = tmp_path / "moved" / "experiment", tmp_path / "moved" / "masks"
    moved.parent.mkdir()
    shutil.move(run.experiment, moved)
    shutil.move(run.masks, moved_masks)
    lazy = dict(
        ((patch.patch_id, patch.mask_file), array)
        for patch, array in iter_run_patch_arrays(patches, config, experiment_root=moved, masks_root=moved_masks)
    )
    assert lazy.keys() == eager.keys() and len(lazy) == 8
    assert all(np.array_equal(lazy[key], eager[key]) for key in eager)
    with pytest.raises(ValueError, match="comes from 'ctrl/rep1/z.tif', which is not an image of the preprocessing run"):
        list(iter_run_patch_arrays([replace(patches[0], source_relative_path="ctrl/rep1/z.tif")], config,
                                   experiment_root=moved, masks_root=moved_masks))
    with pytest.raises(ValueError, match=r"has the FOV key \('drug',.*but the preprocessing run recorded \('ctrl'"):
        list(iter_run_patch_arrays([replace(patches[0], condition="drug")], config,
                                   experiment_root=moved, masks_root=moved_masks))


def test_a_copied_experiment_is_checked_where_it_is_read(tmp_path):
    # CP-2a closes a CP-1 gap: with a copy of the experiment (the original in place), the run
    # was opened at the original, and the copy's masks were read without any check. A mask
    # changed in the copy was used. Now the copy is opened and read through RunInputs.
    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",))
    copy, copy_masks = tmp_path / "copy" / "experiment", tmp_path / "copy" / "masks"
    shutil.copytree(run.experiment, copy)
    shutil.copytree(run.masks, copy_masks)
    config = _cell_config(run)
    config.normalization.enabled = True
    config.normalization.scope = "image"
    config.normalization.plane_histograms = str(run.output / "plane_histograms.csv")
    assert run_patching_condition_repeat_tree(copy, mask_root_path=copy_masks, config=config)
    _labels_moved(copy_masks / "Cells" / "ctrl" / "rep1" / "a.tif")
    with pytest.raises(PreprocessingCheckError, match=r"1 mask changed or missing since the run: Cells/ctrl/rep1/a.tif"):
        run_patching_condition_repeat_tree(copy, mask_root_path=copy_masks, config=config)


def test_an_opened_run_is_not_reused_once_a_mask_or_another_image_changed(tmp_path):
    # Former code (CP-1): cached_statistics kept the opened run for the session, keyed on the
    # table and manifest; a mask, or an image not patched, changed after the first open was
    # not checked again. Now every call opens the run again (cellpatcher.run_session).
    from cellpatcher.normalization import cached_statistics

    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",))
    table = run.output / "plane_histograms.csv"
    assert cached_statistics(table).inputs is not None
    _labels_moved(run.masks / "Cells" / "drug" / "rep1" / "b.tif")
    with pytest.raises(PreprocessingCheckError, match="1 mask changed or missing since the run"):
        cached_statistics(table)

    other = _clean_run(tmp_path / "other", axes="YX", shape=(16, 20))
    other_table = other.output / "plane_histograms.csv"
    assert cached_statistics(other_table).inputs is not None
    image = other.experiment / "drug" / "rep1" / "b.ome.tif"
    tifffile.imwrite(image, np.ones((16, 20), np.uint16), ome=True, metadata={"axes": "YX"})
    with pytest.raises(PreprocessingCheckError, match="1 image changed or missing since the run"):
        cached_statistics(other_table)


def test_within_one_call_the_run_is_opened_once(tmp_path, monkeypatch):
    # A tree run normalizes every leaf with one opened run (the reason the cache existed).
    from cellpatcher import file_runs

    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",), design=True)
    opened = []
    real = file_runs.open_run
    monkeypatch.setattr(file_runs, "open_run", lambda *args, **kwargs: opened.append(1) or real(*args, **kwargs))
    from cellpatcher import run_open

    monkeypatch.setattr(run_open, "open_run", lambda *args, **kwargs: pytest.fail("the run was opened again"))
    config = _cell_config(run)
    config.normalization.enabled, config.normalization.scope = True, "biological_repeat"
    assert len(run_patching_condition_repeat_tree(run.experiment, config=config)) == 2
    assert opened == [1]


def test_file_runs_read_through_run_inputs_only(tmp_path, monkeypatch):
    # No CellPatcher-side axis, pairing or file reading is used for a file run.
    import cellpatcher.io
    import cellpatcher.planes

    def refused(*args, **kwargs):
        raise AssertionError("CellPatcher read or paired a file itself")

    for name in ("read_array", "discover_images"):
        monkeypatch.setattr(cellpatcher.io, name, refused)
    # CellPatcher's own file readers and header checks are gone (CP-2b).
    for module, name in ((cellpatcher.io, "read_image"), (cellpatcher.io, "conform_axes"), (cellpatcher.io, "pair_masks"),
                         (cellpatcher.io, "refuse_image_header_axes"), (cellpatcher.io, "refuse_mask_header_axes"),
                         (cellpatcher.planes, "read_file_plane"), (cellpatcher.planes, "iter_file_planes")):
        assert not hasattr(module, name), name
    run = _clean_run(tmp_path, axes="ZCYX", shape=(3, 2, 16, 20), mask_types=())
    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 8
    config.common.preprocessing_run, config.common.extract_arrays = str(run.output), True
    for z_policy in ("keep", "all"):
        config.common.z_policy = z_policy
        leaves = run_patching_condition_repeat_tree(run.experiment, config=config)
        patch, array = leaves[0].result.patches[0], leaves[0].result.patch_arrays[0]
        source = tifffile.imread(run.experiment / patch.source_relative_path)  # Z, C, Y, X as written
        expected = source[:, :, :8, :8] if z_policy == "keep" else source[patch.z0, :, :8, :8]
        assert np.array_equal(array, expected)


@pytest.mark.parametrize(("setting", "value"), [
    ("image_axes", "ZCYX"), ("mask_axes", "ZYX"), ("mask_source", "ngff_labels"), ("mask_label_name", "Cells"),
    ("condition", "ctrl"), ("biological_repeat", "rep1"), ("technical_repeat", "B02"),
])
def test_the_former_file_settings_raise_a_migration_message(tmp_path, monkeypatch, setting, value):
    from cellpatcher import file_runs

    monkeypatch.setattr(file_runs, "open_run", lambda *args, **kwargs: pytest.fail("opened before the refusal"))
    config = _cell_config(SimpleNamespace(output=tmp_path / "run"), **{setting: value})
    with pytest.raises(ValueError, match=rf"(?s)since CellPatcher v0.2.0.*common.{setting}=.*in-memory arrays"):
        run_patching_pipeline(tmp_path, config=config)


def test_a_folder_run_patches_one_condition_and_repeat(tmp_path):
    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",))
    with pytest.raises(ValueError, match="2 conditions or biological repeats.*run_patching_condition_repeat_tree"):
        run_patching_pipeline(run.experiment, config=_cell_config(run))


def test_the_run_config_records_the_run_and_its_manifest_and_loads_back_for_a_file_run(tmp_path):
    # run_config.json version 2 (CP-2b). Old main recorded the run's layout as common.image_axes
    # ('CYX') and each leaf's labels, without a version or the manifest's SHA-256, so the file
    # loaded back was refused for a file run (a migration message about image_axes).
    from cellpatcher.orchestration import load_config

    run = _clean_run(tmp_path, axes="CYX", shape=(2, 16, 20), mask_types=("Cells",))
    config = _cell_config(run)
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    first = run_patching_condition_repeat_tree(run.experiment, config=config)
    leaf_file = tmp_path / "out" / first[0].relative_folder / "run_config.json"
    data = json.loads(leaf_file.read_text(encoding="utf-8"))
    assert data["run_config_version"] == 2
    assert data["preprocessing"] == {
        "run": str(run.output),
        "manifest_sha256": hashlib.sha256((run.output / "run_manifest.json").read_bytes()).hexdigest(),
        "layout": {"image": "CYX", "mask": "YX"},
    }
    assert not {"image_axes", "mask_axes", "condition", "biological_repeat", "technical_repeat"} & data["common"].keys()
    assert config.common.image_axes == "YX"  # the caller's config is not changed
    saved = load_config(leaf_file)
    assert saved.common.preprocessing_run == str(run.output)
    saved.common.output_dir = str(tmp_path / "again")
    again = run_patching_condition_repeat_tree(run.experiment, config=saved)
    assert [p.patch_id for leaf in again for p in leaf.result.patches] == [
        p.patch_id for leaf in first for p in leaf.result.patches
    ]


@pytest.mark.parametrize("version", [3, "2", None])
def test_a_run_config_of_an_unknown_version_is_refused_by_every_reader(tmp_path, version):
    from cellpatcher import iter_exported_patches
    from cellpatcher.orchestration import load_config

    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",))
    config = _cell_config(run)
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    config.normalization.enabled, config.normalization.scope = True, "image"
    config.normalization.export_values = "raw"
    run_patching_condition_repeat_tree(run.experiment, config=config, combine_metadata=True)
    path = tmp_path / "out" / "run_config.json"
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "run_config_version": version}))
    refusal = rf"run_config_version {version!r}; this CellPatcher reads versions 1, 2"
    with pytest.raises(ValueError, match=refusal):
        load_config(path)
    with pytest.raises(ValueError, match=refusal):
        list(iter_exported_patches(tmp_path / "out" / "patches.csv"))


def test_another_table_than_the_run_s_is_refused(tmp_path):
    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",))
    other = _clean_run(tmp_path / "other", axes="YX", shape=(16, 20))
    config = _cell_config(run)
    config.normalization.enabled, config.normalization.scope = True, "image"
    config.normalization.plane_histograms = str(other.output / "plane_histograms.csv")
    with pytest.raises(PreprocessingCheckError) as refused:
        run_patching_condition_repeat_tree(run.experiment, config=config)
    assert [issue.code for issue in refused.value.issues] == ["not_the_run_output"]


def test_masks_with_and_without_depth_share_the_run_s_layout(tmp_path):
    # A three-plane image with a ZYX mask beside a one-plane image with a 2-D mask: the run's
    # mask layout is ZYX, and the 2-D mask is its image's one plane (never a guessed plane).
    from tests.helpers import file_run_config, write_tiff

    root, masks = tmp_path / "exp", tmp_path / "masks"
    labels = np.zeros((3, 16, 16), np.uint16)
    for z in range(3):
        labels[z, 2 + 3 * z:5 + 3 * z, 2:5] = z + 1
    write_tiff(root / "ctrl" / "r1" / "a.tif", np.ones((3, 16, 16), np.uint16), "ZYX")
    write_tiff(masks / "cells" / "ctrl" / "r1" / "a.tif", labels, "ZYX")
    write_tiff(root / "ctrl" / "r1" / "b.tif", np.ones((16, 16), np.uint16), "YX")
    write_tiff(masks / "cells" / "ctrl" / "r1" / "b.tif", labels[1], "YX")
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size, config.cell.exclude_border, config.common.z_policy = 6, False, "all"
    file_run_config(config, root, "ZYX", masks=masks, missing_as_singleton=True)
    (leaf,) = run_patching_condition_repeat_tree(root, mask_root_path=masks, config=config)
    found = sorted((p.image_file, p.z0, p.label_id) for p in leaf.result.patches)
    assert found == [("a.tif", 0, 1), ("a.tif", 1, 2), ("a.tif", 2, 3), ("b.tif", 0, 2)]


def test_a_moved_experiment_s_combined_run_config_is_written_from_the_call_s_run(tmp_path):
    # Review of CP-2a: the combined run_config.json of a tree run was written after the call's
    # run session had closed, so its normalization_inputs opened the run again where it had
    # recorded the experiment: a moved experiment was refused after every leaf was patched.
    run = _clean_run(tmp_path, axes="YX", shape=(16, 20), mask_types=("Cells",))
    moved, moved_masks = tmp_path / "moved" / "experiment", tmp_path / "moved" / "masks"
    moved.parent.mkdir()
    shutil.move(run.experiment, moved)
    shutil.move(run.masks, moved_masks)
    config = _cell_config(run)
    config.normalization.enabled, config.normalization.scope = True, "image"
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    run_patching_condition_repeat_tree(moved, mask_root_path=moved_masks, config=config, combine_metadata=True)
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text())
    table_sha256 = hashlib.sha256((run.output / "plane_histograms.csv").read_bytes()).hexdigest()
    assert saved["normalization_inputs"]["plane_histograms_sha256"] == table_sha256
