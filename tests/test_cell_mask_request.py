"""Cell mode asks the resolver for its object set by mask ID (Stage 2 CP-2; Spec P27, P28, ADR 0002 AR5).

A file run in ``mode="cell"`` builds ``Request(channel_ids, (MaskRequest(<mask ID>, "objects",
cell.binary_mask_policy),))`` and reads through the selection:

- an ``instances`` mask is read as stored (D47: an ``instances`` mask holding 0 and 1 is one object);
- a ``binary`` mask is refused as an object set (``A_MASK_NOT_INSTANCES``) unless the request
  records ``binary_mask_policy="connected_components"``; it is then labelled with the shared
  labeller (:func:`cellpatcher.label_objects`), and the mask ID, policy and connectivity are
  recorded in ``run_config.json``, never in ``morphobricks.toml``;
- every refusal of the resolver (``A_MASK_NOT_INSTANCES``, ``A_MASK_UNCHECKED``,
  ``A_SCOPE_INCOMPLETE``) stops the run with its code before any pixel is read.
"""

import json

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.statistics import PreprocessingCheckError
from tests.helpers import describe_experiment, survey_experiment
from tests.test_described_ids import _described_run, _two_blob_folders


def _cell_config(output, out_dir=None, **matching):
    config = CellPatcherConfig(mode="cell")
    config.common.preprocessing_run = str(output)
    config.cell.exclude_border = False
    for name, value in matching.items():
        setattr(config.matching, name, value)
    if out_dir is not None:
        config.common.output_dir = str(out_dir)
        config.export.enabled = True
    return config


def _no_pixel_reads(monkeypatch):
    """Make every image or mask read fail, so a refusal is proven to come before any read."""
    from morphobricks_preprocessing.run_inputs import RunInputs

    def read(*args, **kwargs):
        raise AssertionError("a pixel was read before the refusal")

    for name in ("read_plane", "read_mask", "read_image"):
        monkeypatch.setattr(RunInputs, name, read)


def _codes(raised):
    return [issue.code for issue in raised.value.issues]


# --- v08 (P27 a/b/c) through the resolver -----------------------------------------------------


@pytest.fixture
def v08(tmp_path):
    return _described_run(tmp_path, "v08", masks=("Nuclei", "TumorRegion"))


def _tree(run, config):
    return [patch for leaf in run_patching_condition_repeat_tree(run.experiment, config=config) for patch in
            leaf.result.patches]


def test_v08_p27_a_b_c_binary_region_as_objects(v08, tmp_path, monkeypatch):
    description = (v08.output / "morphobricks.toml").read_bytes()

    # (a) TumorRegion plainly as an object set: A_MASK_NOT_INSTANCES, before any pixel is read.
    with monkeypatch.context() as patched:
        _no_pixel_reads(patched)
        with pytest.raises(PreprocessingCheckError, match="A_MASK_NOT_INSTANCES") as raised:
            _tree(v08, _cell_config(v08.output, tmp_path / "a", mask_id="TumorRegion"))
    assert _codes(raised) == ["A_MASK_NOT_INSTANCES"]
    assert "binary_mask_policy='connected_components'" in str(raised.value)
    assert not (tmp_path / "a").exists()

    # (b) with the per-request conversion: accepted, labelled by the shared labeller, recorded.
    config = _cell_config(v08.output, tmp_path / "b", mask_id="TumorRegion")
    config.cell.binary_mask_policy = "connected_components"
    with pytest.warns(UserWarning, match="'TumorRegion' is labelled by 8-connected components"):
        patches = _tree(v08, config)
    assert len(patches) == 4 and {p.label_id for p in patches} == {1}  # one region of 1s in each of 4 images
    assert {p.object_set for p in patches} == {"tumor_region"}  # the object set is still the mask type
    saved = json.loads((tmp_path / "b" / "CTRL" / "rep1" / "run_config.json").read_text("utf-8"))
    assert saved["matching"]["mask_id"] == "TumorRegion"
    assert (saved["cell"]["binary_mask_policy"], saved["cell"]["binary_mask_connectivity"]) == (
        "connected_components", 2
    )

    # (c) a later request without it is refused again: the conversion was for that request only.
    with pytest.raises(PreprocessingCheckError) as raised:
        _tree(v08, _cell_config(v08.output, tmp_path / "c", mask_id="TumorRegion"))
    assert _codes(raised) == ["A_MASK_NOT_INSTANCES"]
    # Nothing is written to the description (i112: binary_mask_policy is forbidden there).
    assert (v08.output / "morphobricks.toml").read_bytes() == description


def test_v08_an_instances_mask_is_read_as_stored(v08, tmp_path):
    patches = _tree(v08, _cell_config(v08.output, mask_id="Nuclei"))
    assert len(patches) == 8 and {p.label_id for p in patches} == {1, 2}  # two labelled squares per image
    # the mask type chooses the declared mask of that source type when no mask ID is given
    by_type = _tree(v08, _cell_config(v08.output, mask_type="nuclei"))
    assert [p.patch_id for p in by_type] == [p.patch_id for p in patches]


def test_v08_a_policy_on_an_instances_mask_is_refused_naming_the_setting(v08):
    config = _cell_config(v08.output, mask_id="Nuclei")
    config.cell.binary_mask_policy = "connected_components"
    with pytest.raises(ValueError, match="cell.binary_mask_policy='connected_components' converts a binary mask"):
        _tree(v08, config)


def test_v08_mask_id_and_mask_type_must_name_one_mask(v08):
    with pytest.raises(ValueError, match="mask_id='Nuclei' is the mask of source type 'nuclei'"):
        _tree(v08, _cell_config(v08.output, mask_id="Nuclei", mask_type="tumor_region"))


def test_an_undeclared_mask_id_is_present_in_0_of_n(v08, monkeypatch):
    _no_pixel_reads(monkeypatch)
    with pytest.raises(PreprocessingCheckError) as raised:
        _tree(v08, _cell_config(v08.output, mask_id="Cells"))
    assert _codes(raised) == ["A_SCOPE_INCOMPLETE"]
    assert "'Cells' is present in 0 of 4 acquisitions" in str(raised.value)


def test_p28_a_mask_the_mask_step_has_not_checked_is_refused_by_the_resolver(tmp_path, monkeypatch):
    run = _described_run(tmp_path, "v08", masks=("Nuclei",), mask_step=False)
    _no_pixel_reads(monkeypatch)
    with pytest.raises(PreprocessingCheckError) as raised:
        _tree(run, _cell_config(run.output, mask_id="Nuclei"))
    assert _codes(raised) == ["A_MASK_UNCHECKED"]
    assert "4 of 4 files" in str(raised.value)


def test_the_binary_mask_policy_has_no_default():
    from cellpatcher.config import CellConfig

    assert CellConfig().binary_mask_policy is None
    with pytest.raises(ValueError, match="'single_object' was removed"):
        CellConfig(binary_mask_policy="single_object")


# --- the 0/1 two-blob mask (the evidence report's fixture) ------------------------------------


def _two_blob_run(tmp_path, representation):
    """The 0/1 mask of two separate squares, described as *representation*, the mask step run."""
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.run.mask_step import describe_masks, prepare_mask_step

    images, masks = _two_blob_folders(tmp_path)
    output = tmp_path / representation
    output.mkdir()
    mapping = mbp.AxisMapping("YX", "YX")
    describe_experiment(output, survey_experiment(images, mapping, output, masks_folder=masks),
                        representations={"mask": representation})
    assert mbp.run_experiment(images, mapping, output_folder=output, masks_folder=masks).written
    # The mask step accepts a 0/1 instances mask (P05: pixels cannot prove instance semantics);
    # tests.helpers.check_masks would stop it, so it is called directly here.
    assert not describe_masks(prepare_mask_step(None, output)).issues
    return images, masks, output


def _patches(images, masks, config):
    return run_patching_pipeline(images, config=config, mask_folder_path=masks).patches


def test_the_two_blob_binary_mask_is_two_objects_with_the_policy(tmp_path):
    images, masks, output = _two_blob_run(tmp_path, "binary")
    config = _cell_config(output, mask_id="mask")
    config.cell.binary_mask_policy = "connected_components"
    config.common.extract_arrays = True
    config.extraction.apply_mask = True  # masks by the computed labels, not the file's 0/1
    with pytest.warns(UserWarning, match="8-connected"):
        result = run_patching_pipeline(images, config=config, mask_folder_path=masks)
    assert [(p.label_id, p.area_px, (p.y0, p.x0)) for p in result.patches] == [(1, 9, (2, 2)), (2, 9, (10, 10))]
    for array in result.patch_arrays:
        assert int(np.count_nonzero(array)) == 9  # each patch keeps its own object's pixels


def test_the_two_blob_binary_mask_is_refused_without_a_policy(tmp_path, monkeypatch):
    images, masks, output = _two_blob_run(tmp_path, "binary")
    _no_pixel_reads(monkeypatch)
    with pytest.raises(PreprocessingCheckError) as raised:
        _patches(images, masks, _cell_config(output, mask_id="mask"))
    assert _codes(raised) == ["A_MASK_NOT_INSTANCES"]


def test_the_two_blob_mask_declared_instances_is_one_object(tmp_path):
    # D47 stays as the instances rule: label 1 is one object, however many regions it has.
    images, masks, output = _two_blob_run(tmp_path, "instances")
    patches = _patches(images, masks, _cell_config(output, mask_id="mask"))
    assert [(p.label_id, p.area_px) for p in patches] == [(1, 18)]
