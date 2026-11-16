"""Cell mode asks the resolver for its object set by mask ID (CP-2; Spec P27, P28, ADR 0002, ADR 0005).

A file run in ``mode="cell"`` builds ``Request(channel_ids, (MaskRequest(<mask ID>, "objects",
cell.binary_mask_policy),))``, checks the resolver's contract version (exactly 3, Spec ADR 0005
BM10), and reads the object set's planes with preprocessing's ``RunInputs.read_objects``:

- an ``instances`` mask as stored (D47: an ``instances`` mask holding 0 and 1 is one object);
- a ``binary`` mask is refused as an object set (``A_MASK_NOT_INSTANCES``) unless the request
  carries ``binary_mask_policy="connected_components"``; ``read_objects`` then gives its
  8-connected components (BM1), and ``run_config.json`` copies the resolver's record under
  ``mask_conversions`` (BM2), never ``morphobricks.toml``; CellPatcher labels nothing;
- every refusal of the resolver (``A_MASK_NOT_INSTANCES``, ``A_MASK_UNCHECKED``,
  ``A_SCOPE_INCOMPLETE``, ``A_VOLUME_NOT_SUPPORTED``) stops the run with its code and its message
  unchanged, before any pixel is read.
"""

import json

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.statistics import PreprocessingCheckError
from tests.helpers import describe_experiment, survey_experiment
from tests.test_described_ids import _described_run


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


READS = ("read_plane", "read_mask", "read_mask_window", "read_objects", "read_image")


def _no_pixel_reads(monkeypatch):
    """Make every image or mask read fail, so a refusal is proven to come before any read."""
    from morphobricks_preprocessing.run_inputs import RunInputs

    def read(*args, **kwargs):
        raise AssertionError("a pixel was read before the refusal")

    for name in READS:
        monkeypatch.setattr(RunInputs, name, read)


def _count_reads(monkeypatch):
    """Count every image or mask read of ``RunInputs`` (a spy: the reads still happen)."""
    from morphobricks_preprocessing.run_inputs import RunInputs

    counts = dict.fromkeys(READS, 0)
    for name in READS:
        original = getattr(RunInputs, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(RunInputs, name, spy)
    return counts


def _codes(raised):
    return [issue.code for issue in raised.value.issues]


# --- v08 (P27 a/b/c) through the resolver -----------------------------------------------------


@pytest.fixture
def v08(tmp_path):
    return _described_run(tmp_path, "v08", masks=("Nuclei", "TumorRegion"))


def _tree(run, config):
    return [patch for leaf in run_patching_condition_repeat_tree(run.experiment, config=config) for patch in
            leaf.result.patches]


def _record(mask_id):
    return {"mask_id": mask_id, "conversion": "connected_components", "connectivity": 8,
            "labeller": "morphobricks_preprocessing.RunInputs.read_objects"}


def test_v08_p27_a_b_c_binary_region_as_objects(v08, tmp_path, monkeypatch):
    from morphobricks_preprocessing.mask_rules import refuse_not_instances

    description = (v08.output / "morphobricks.toml").read_bytes()

    # (a) TumorRegion plainly as an object set: A_MASK_NOT_INSTANCES, its message unchanged, no read.
    with monkeypatch.context() as patched:
        _no_pixel_reads(patched)
        with pytest.raises(PreprocessingCheckError, match="A_MASK_NOT_INSTANCES") as raised:
            _tree(v08, _cell_config(v08.output, tmp_path / "a", mask_id="TumorRegion"))
    assert _codes(raised) == ["A_MASK_NOT_INSTANCES"]
    assert raised.value.issues[0].message == str(refuse_not_instances("TumorRegion", declared="binary"))
    assert "binary_mask_policy='connected_components'" in raised.value.issues[0].action
    assert not (tmp_path / "a").exists()

    # (b) with the per-request conversion: read by RunInputs.read_objects, the record copied.
    config = _cell_config(v08.output, tmp_path / "b", mask_id="TumorRegion")
    config.cell.binary_mask_policy = "connected_components"
    with monkeypatch.context() as patched:
        reads = _count_reads(patched)
        with pytest.warns(UserWarning, match="'TumorRegion': the binary mask is converted to objects by "
                                             "connected_components"):
            patches = _tree(v08, config)
    assert reads["read_objects"] == 4 and reads["read_mask"] == 0  # one plane per image, never read raw
    assert len(patches) == 4 and {p.label_id for p in patches} == {1}  # one region of 1s in each of 4 images
    assert {p.object_set for p in patches} == {"tumor_region"}  # the object set is still the mask type
    saved = json.loads((tmp_path / "b" / "CTRL" / "rep1" / "run_config.json").read_text("utf-8"))
    assert saved["matching"]["mask_id"] == "TumorRegion"
    assert saved["cell"]["binary_mask_policy"] == "connected_components"  # the request field, as given
    assert "binary_mask_connectivity" not in saved["cell"]  # Spec ADR 0005 BM1: no setting
    assert saved["mask_conversions"] == [_record("TumorRegion")]  # BM2: the resolver's record, as is
    assert "mask_conversions" not in saved["preprocessing"]

    # (c) a later request without it is refused again: the conversion was for that request only.
    with pytest.raises(PreprocessingCheckError) as raised:
        _tree(v08, _cell_config(v08.output, tmp_path / "c", mask_id="TumorRegion"))
    assert _codes(raised) == ["A_MASK_NOT_INSTANCES"]
    # Nothing is written to the description (i112: binary_mask_policy is forbidden there).
    assert (v08.output / "morphobricks.toml").read_bytes() == description


def test_v08_an_instances_mask_is_read_as_stored(v08, tmp_path, monkeypatch):
    with monkeypatch.context() as patched:
        reads = _count_reads(patched)
        patches = _tree(v08, _cell_config(v08.output, tmp_path / "out", mask_id="Nuclei"))
    assert reads["read_objects"] == 4 and reads["read_mask"] == 0  # the objects use reads read_objects
    assert len(patches) == 8 and {p.label_id for p in patches} == {1, 2}  # two labelled squares per image
    saved = json.loads((tmp_path / "out" / "CTRL" / "rep1" / "run_config.json").read_text("utf-8"))
    assert saved["mask_conversions"] == []  # an instances mask has no record
    # the mask type chooses the declared mask of that source type when no mask ID is given
    by_type = _tree(v08, _cell_config(v08.output, mask_type="nuclei"))
    assert [p.patch_id for p in by_type] == [p.patch_id for p in patches]


def test_v08_a_policy_on_an_instances_mask_is_the_resolvers_error_unchanged(v08):
    # Spec ADR 0005: a malformed request stays the resolver's ValueError; CellPatcher adds nothing.
    from morphobricks_preprocessing import MaskRequest, Request
    from morphobricks_preprocessing.run_inputs import RunInputs

    config = _cell_config(v08.output, mask_id="Nuclei")
    config.cell.binary_mask_policy = "connected_components"
    with pytest.raises(ValueError) as raised:
        _tree(v08, config)
    with pytest.raises(ValueError) as direct:
        RunInputs.open(v08.output).resolve_request(
            Request(masks=(MaskRequest("Nuclei", "objects", "connected_components"),)))
    assert str(raised.value) == str(direct.value)
    assert "cell.binary_mask_policy" not in str(raised.value)


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
    with pytest.raises(ValueError, match=r"binary_mask_policy must be one of .*; got 'single_object'"):
        CellConfig(binary_mask_policy="single_object")


# --- the 0/1 two-blob mask (the evidence report's fixture) ------------------------------------


def _mask_run(tmp_path, representation, mask):
    """One image and its 0/1 *mask*, described as *representation*, the mask step run."""
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.run.mask_step import describe_masks, prepare_mask_step

    from cellpatcher.io import write_array

    images, masks = tmp_path / "images", tmp_path / "masks"
    write_array(images / "a.tif", (np.arange(mask.size, dtype=np.uint16) + 1).reshape(mask.shape))
    write_array(masks / "a_mask.tif", mask)
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


def _two_blob_run(tmp_path, representation):
    """The 0/1 mask of two separate squares (Spec ADR 0005 F02 at 16 x 16), the mask step run."""
    mask = np.zeros((16, 16), np.uint8)
    mask[2:5, 2:5] = mask[10:13, 10:13] = 1
    return _mask_run(tmp_path / "run", representation, mask)


def _patches(images, masks, config):
    return run_patching_pipeline(images, config=config, mask_folder_path=masks).patches


def test_the_two_blob_binary_mask_is_two_objects_with_the_policy(tmp_path):
    # F02 (b): labels 1 (top-left) and 2, read by read_objects; masking by those labels keeps each
    # object's own pixels (issue #44 is gone: the mask CellPatcher masks with is the objects).
    images, masks, output = _two_blob_run(tmp_path, "binary")
    config = _cell_config(output, tmp_path / "out", mask_id="mask")
    config.cell.binary_mask_policy = "connected_components"
    config.common.extract_arrays = True
    config.extraction.apply_mask = True
    with pytest.warns(UserWarning, match="8-connected"):
        result = run_patching_pipeline(images, config=config, mask_folder_path=masks)
    assert [(p.label_id, p.area_px, (p.y0, p.x0)) for p in result.patches] == [(1, 9, (2, 2)), (2, 9, (10, 10))]
    for array in result.patch_arrays:
        assert int(np.count_nonzero(array)) == 9  # each patch keeps its own object's pixels
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text("utf-8"))
    assert saved["mask_conversions"] == [_record("mask")]


@pytest.mark.parametrize("region", ["mask", "bbox"])
def test_patch_scope_normalization_selects_each_converted_objects_pixels(tmp_path, region):
    # Issue #44 (refused before ADR 0005): patch-scope normalization over the object's pixels now
    # reads the computed labels, so label 2 has its own pixels and gets parameters.
    images, masks, output = _two_blob_run(tmp_path, "binary")
    config = _cell_config(output, mask_id="mask")
    config.cell.binary_mask_policy = "connected_components"
    config.normalization.enabled, config.normalization.scope = True, "patch"
    config.normalization.patch_region = region
    with pytest.warns(UserWarning, match="8-connected"):
        patches = _patches(images, masks, config)
    assert [p.label_id for p in patches] == [1, 2]
    assert all(p.normalization_parameters is not None for p in patches)


def test_the_two_blob_binary_mask_is_refused_without_a_policy(tmp_path, monkeypatch):
    from morphobricks_preprocessing.mask_rules import refuse_not_instances

    images, masks, output = _two_blob_run(tmp_path, "binary")
    _no_pixel_reads(monkeypatch)
    with pytest.raises(PreprocessingCheckError) as raised:
        _patches(images, masks, _cell_config(output, tmp_path / "out", mask_id="mask"))
    assert _codes(raised) == ["A_MASK_NOT_INSTANCES"]
    assert raised.value.issues[0].message == str(refuse_not_instances("mask", declared="binary"))
    assert not (tmp_path / "out").exists()


def test_the_two_blob_mask_declared_instances_is_one_object(tmp_path):
    # D47 stays as the instances rule: label 1 is one object, however many regions it has.
    images, masks, output = _two_blob_run(tmp_path, "instances")
    patches = _patches(images, masks, _cell_config(output, mask_id="mask"))
    assert [(p.label_id, p.area_px) for p in patches] == [(1, 18)]


# --- Spec ADR 0005 F01: connectivity 8, fixed; no connectivity setting ------------------------


def test_f01_corner_touching_blobs_are_one_object_8_connected(tmp_path):
    mask = np.zeros((8, 8), np.uint8)
    mask[1:3, 1:3] = mask[3:5, 3:5] = 1
    images, masks, output = _mask_run(tmp_path, "binary", mask)
    config = _cell_config(output, tmp_path / "out", mask_id="mask")
    config.cell.binary_mask_policy = "connected_components"
    with pytest.warns(UserWarning, match="8-connected"):
        patches = _patches(images, masks, config)
    assert [(p.label_id, p.area_px) for p in patches] == [(1, 8)]
    saved = json.loads((tmp_path / "out" / "run_config.json").read_text("utf-8"))
    assert saved["mask_conversions"][0]["connectivity"] == 8


def test_f01_a_config_with_binary_mask_connectivity_is_refused_as_an_unknown_key(tmp_path):
    # AR10: removed keys are refused by name, never mapped (1 gave 4-connected objects before).
    from cellpatcher import load_config, save_default_config
    from cellpatcher.config import CellConfig

    with pytest.raises(TypeError, match="binary_mask_connectivity"):
        CellConfig(binary_mask_connectivity=2)
    with pytest.raises(AttributeError, match="binary_mask_connectivity"):
        CellConfig().binary_mask_connectivity = 1
    path = tmp_path / "config.json"
    save_default_config(path)
    data = json.loads(path.read_text("utf-8"))
    data.setdefault("cell", {})["binary_mask_connectivity"] = 2
    path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(TypeError, match="binary_mask_connectivity"):
        load_config(path)


# --- Spec ADR 0005 F13: the resolver's contract version ---------------------------------------


def _resolver_of_version(monkeypatch, version):
    """Make RunInputs.resolve_request answer as a resolver of contract *version* (else unchanged)."""
    from dataclasses import replace

    from morphobricks_preprocessing.run_inputs import RunInputs

    resolve = RunInputs.resolve_request

    def resolve_request(self, request):
        resolved = resolve(self, request)
        return replace(resolved, provenance=replace(resolved.provenance, resolver_version=version))

    monkeypatch.setattr(RunInputs, "resolve_request", resolve_request)


@pytest.mark.parametrize("version", [2, 4])
def test_f13_a_resolver_of_another_contract_version_is_refused_before_any_read(v08, tmp_path, monkeypatch, version):
    from morphobricks_normalization.contracts import refuse_resolver_version

    _resolver_of_version(monkeypatch, version)
    _no_pixel_reads(monkeypatch)
    for mask_id in ("Nuclei", "TumorRegion"):
        config = _cell_config(v08.output, tmp_path / "out", mask_id=mask_id)
        config.cell.binary_mask_policy = "connected_components" if mask_id == "TumorRegion" else None
        with pytest.raises(ValueError) as raised:
            _tree(v08, config)
        assert raised.value.code == "RESOLVER_VERSION_UNSUPPORTED"
        assert str(raised.value) == str(refuse_resolver_version(version, [3]))
        assert f"has contract version {version}; this consumer supports exactly [3]" in str(raised.value)
        assert not (tmp_path / "out").exists()  # no output, no run_config.json


def _mask_free_config(output, mode, out_dir=None):
    """A grid, whole-FOV or random run of *output* that names no channel, mask or region."""
    config = CellPatcherConfig(mode=mode)
    config.common.preprocessing_run = str(output)
    config.grid.tile_size = 8
    config.random.patch_size, config.random.n_patches = 8, 2
    if out_dir is not None:
        config.common.output_dir = str(out_dir)
        config.export.enabled = True
    return config


@pytest.mark.parametrize("mode", ["grid", "whole_fov", "random"])
@pytest.mark.parametrize("version", [2, 4])
def test_cp1_a_mask_free_run_without_channels_checks_the_resolver_version_before_any_read(
    v08, tmp_path, monkeypatch, mode, version
):
    """CP1 (v0.7.1): the version is checked for every described file run, not only when something is asked.

    v0.7.0 returned before ``resolve_request`` when the run named no channel ID, mask or region, so a
    grid, whole-FOV or mask-free random run of a resolver of another contract version read its pixels.
    """
    from morphobricks_normalization.contracts import refuse_resolver_version

    _resolver_of_version(monkeypatch, version)
    _no_pixel_reads(monkeypatch)
    config = _mask_free_config(v08.output, mode, tmp_path / "tree")
    assert config.extraction.split_channels is False and config.matching.mask_type is None
    with pytest.raises(ValueError) as raised:
        _tree(v08, config)
    assert raised.value.code == "RESOLVER_VERSION_UNSUPPORTED"
    assert str(raised.value) == str(refuse_resolver_version(version, [3]))
    assert not (tmp_path / "tree").exists()  # no output, no run_config.json
    with pytest.raises(ValueError) as raised:
        run_patching_pipeline(v08.experiment, config=_mask_free_config(v08.output, mode, tmp_path / "folder"))
    assert raised.value.code == "RESOLVER_VERSION_UNSUPPORTED"
    assert not (tmp_path / "folder").exists()


@pytest.mark.parametrize("mode", ["grid", "whole_fov", "random"])
def test_cp1_a_mask_free_run_asks_the_resolver_with_an_empty_request(v08, monkeypatch, mode):
    """The version comes from ``resolve_request`` with an empty request (preprocessing v0.6.0 has no other way)."""
    from morphobricks_preprocessing.run_inputs import RunInputs

    asked = []
    resolve = RunInputs.resolve_request

    def resolve_request(self, request):
        asked.append(request)
        return resolve(self, request)

    monkeypatch.setattr(RunInputs, "resolve_request", resolve_request)
    assert _tree(v08, _mask_free_config(v08.output, mode))
    assert asked and all((request.channel_ids, request.masks) == ((), ()) for request in asked)


def test_f13_contract_version_3_is_accepted(v08, monkeypatch):
    from cellpatcher.run_request import SUPPORTED_RESOLVER_VERSIONS

    assert SUPPORTED_RESOLVER_VERSIONS == frozenset({3})
    _resolver_of_version(monkeypatch, 3)
    assert len(_tree(v08, _cell_config(v08.output, mask_id="Nuclei"))) == 8


# --- propagation: the resolver's and read_objects' refusals reach the caller unchanged --------


def test_a_z_stack_conversion_refusal_is_raised_with_its_message_unchanged(v08, tmp_path, monkeypatch):
    # A_VOLUME_NOT_SUPPORTED (BM5): preprocessing refuses Z-stacks before CellPatcher reads one, so the
    # resolver's refusal is stubbed here, as preprocessing builds it, to check its propagation.
    from dataclasses import replace

    from morphobricks_preprocessing.run_inputs import RunInputs
    from morphobricks_preprocessing.run_inputs_request import Refusal

    message = ("'TumorRegion' would be converted by connected_components, which labels 2-D planes only; the images "
               "of CTRL/rep1/img0 have 2 Z planes. Converting a Z-stack is not supported (A_VOLUME_NOT_SUPPORTED).")
    resolve = RunInputs.resolve_request

    def resolve_request(self, request):
        resolved = resolve(self, request)
        refusal = Refusal("A_VOLUME_NOT_SUPPORTED", "TumorRegion", message)
        return replace(resolved, selections=(), refusals=(refusal,))

    monkeypatch.setattr(RunInputs, "resolve_request", resolve_request)
    _no_pixel_reads(monkeypatch)
    config = _cell_config(v08.output, tmp_path / "out", mask_id="TumorRegion")
    config.cell.binary_mask_policy = "connected_components"
    with pytest.raises(PreprocessingCheckError) as raised:
        _tree(v08, config)
    assert [(i.code, i.message) for i in raised.value.issues] == [("A_VOLUME_NOT_SUPPORTED", message)]
    assert not (tmp_path / "out").exists()


def test_the_read_guard_of_read_objects_is_raised_unchanged(tmp_path, monkeypatch):
    # R_MASK_CONTRADICTS_REPRESENTATION from label_binary (a plane holding {0, 3, 7}, stubbed: the
    # mask step refuses such a file before any run) reaches the caller as preprocessing raised it.
    import morphobricks_preprocessing.run_inputs_mask as run_inputs_mask
    from morphobricks_preprocessing.mask_rules import refuse_binary_values

    images, masks, output = _two_blob_run(tmp_path, "binary")
    plane = np.zeros((16, 16), np.uint8)
    plane[1, 1], plane[5, 5] = 3, 7
    monkeypatch.setattr(run_inputs_mask, "read_mask_plane", lambda *args, **kwargs: plane)
    config = _cell_config(output, tmp_path / "out", mask_id="mask")
    config.cell.binary_mask_policy = "connected_components"
    with pytest.raises(ValueError) as raised, pytest.warns(UserWarning, match="8-connected"):
        _patches(images, masks, config)
    assert raised.value.code == "R_MASK_CONTRADICTS_REPRESENTATION"
    expected = refuse_binary_values(raised.value.args[0].split(" is declared")[0].split(": ", 1)[1], values=[3, 7])
    assert str(raised.value) == str(expected)
    assert not (tmp_path / "out").exists()
