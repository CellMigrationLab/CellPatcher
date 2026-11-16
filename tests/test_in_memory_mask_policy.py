"""In memory an object set is an integer instance array; nothing is converted (Spec ADR 0005 BM4, F07).

``run_patching``, ``scan_cell_objects``/``scan_cell_patch_size`` and the cell strategy called
directly take a plain NumPy array of integer labels, used as stored (D47). A boolean or float
array, a list, None or a ``numpy.ma.MaskedArray`` is refused with ``A_MASK_NOT_INSTANCES`` and
the exact message of preprocessing's one constructor (``refuse_not_instances``), before any patch.
``cell.binary_mask_policy`` is refused in memory: it is only the request field of a described file
run (before ADR 0005 a boolean mask with the policy was labelled here by CellPatcher's labeller).
The same boolean array is accepted as a region (a random-mode sampling mask, a fraction mask).
"""

import numpy as np
import pytest

from cellpatcher import CellConfig, CellPatcherConfig, CellSegmentationStrategy, run_patching, scan_cell_objects
from cellpatcher.cell_scan import scan_cell_patch_size
from cellpatcher.mask_policy import IN_MEMORY_POLICY_MESSAGE
from tests.helpers import named


def _two_blobs(dtype):
    """The 16x16 0/1 mask of two separate 3x3 squares (the goldens' ``two_blob_0_1``)."""
    mask = np.zeros((16, 16), dtype)
    mask[2:5, 2:5] = mask[10:13, 10:13] = 1
    return mask


def _config(policy=None, mode="cell", **common):
    config = CellPatcherConfig(mode=mode)
    config.cell.exclude_border = False
    config.cell.binary_mask_policy = policy
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def _objects(mask, config):
    image = np.ones((16, 16), np.uint16)
    result = run_patching(images=image, masks=mask, config=config, image_metadata=named(image))
    return [(p.label_id, p.area_px) for p in result.patches]


def _expected(name, value):
    from morphobricks_preprocessing.mask_rules import refuse_not_instances

    return str(refuse_not_instances(name, value=value))


# F07: a bool and a float32 array (the array branch); a list, None and an integer MaskedArray (the
# non-array branch).
NOT_OBJECT_SETS = [
    pytest.param(_two_blobs(bool), "is a bool array", id="bool"),
    pytest.param(_two_blobs(np.float32), "is a float32 array", id="float32"),
    pytest.param(_two_blobs(np.uint8).tolist(), "is a list (not a plain NumPy array)", id="list"),
    pytest.param(np.ma.masked_array(_two_blobs(np.uint8)), "is a MaskedArray (not a plain NumPy array)",
                 id="masked_array"),
]


@pytest.mark.parametrize(("mask", "branch"), NOT_OBJECT_SETS)
def test_run_patching_refuses_what_is_not_an_object_set_with_the_shared_message(mask, branch, tmp_path):
    config = _config(output_dir=str(tmp_path / "out"))
    config.export.enabled = True
    with pytest.raises(ValueError) as raised:
        _objects(mask, config)
    assert raised.value.code == "A_MASK_NOT_INSTANCES"
    assert str(raised.value) == _expected("the mask of image_0000", mask)
    assert branch in str(raised.value)
    assert not (tmp_path / "out").exists()  # refused before anything is written


def test_a_none_mask_among_others_is_refused_with_the_non_array_branch():
    image = np.ones((16, 16), np.uint16)
    with pytest.raises(ValueError) as raised:
        run_patching(images=[image, image], masks=[_two_blobs(np.uint8), None], config=_config(),
                     image_metadata=named([image, image]))
    assert raised.value.code == "A_MASK_NOT_INSTANCES"
    assert str(raised.value) == _expected("the mask of image_0001", None)
    assert "is a NoneType (not a plain NumPy array)" in str(raised.value)


@pytest.mark.parametrize(("mask", "branch"), NOT_OBJECT_SETS)
def test_the_cell_strategy_and_the_object_scan_refuse_it_too(mask, branch):
    strategy = CellSegmentationStrategy(CellConfig(exclude_border=False))
    with pytest.raises(ValueError) as raised:
        strategy.generate_regions(image_shape=(16, 16), labels=mask, image_id="image_0000")
    assert str(raised.value) == _expected("the mask of image_0000", mask)
    with pytest.raises(ValueError) as scanned:
        scan_cell_objects(image_shape=(16, 16), labels=mask, config=CellConfig())
    assert str(scanned.value) == _expected("the mask of image_0000", mask)
    with pytest.raises(ValueError) as pooled:
        scan_cell_patch_size([(16, 16)], [mask], config=CellConfig())
    assert pooled.value.code == "A_MASK_NOT_INSTANCES"


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_an_integer_mask_is_used_as_stored(dtype):
    # D47: an instances mask holding 0 and 1 is one object, whatever its integer type.
    assert _objects(_two_blobs(dtype), _config()) == [(1, 18)]
    records = scan_cell_objects(image_shape=(16, 16), labels=_two_blobs(dtype), config=CellConfig(exclude_border=False))
    assert [(r.label_id, r.area_px) for r in records] == [(1, 18)]


@pytest.mark.parametrize("mask", [_two_blobs(bool), _two_blobs(np.uint8)], ids=["bool", "uint8"])
def test_the_policy_is_refused_in_memory_whatever_the_mask(mask, tmp_path):
    # BM4: no in-memory API converts or carries binary_mask_policy; the policy is never ignored.
    config = _config("connected_components", output_dir=str(tmp_path / "out"))
    config.export.enabled = True
    with pytest.raises(ValueError) as raised:
        _objects(mask, config)
    assert str(raised.value) == IN_MEMORY_POLICY_MESSAGE.format(policy="connected_components", where="run_patching")
    assert not (tmp_path / "out").exists()
    policy = CellConfig(binary_mask_policy="connected_components")
    with pytest.raises(ValueError, match="binary_mask_policy='connected_components' is set, but scan_cell_objects"):
        scan_cell_objects(image_shape=(16, 16), labels=mask, config=policy)


def test_the_policy_is_refused_when_planes_are_split_and_in_other_modes():
    masks = np.stack([_two_blobs(np.uint8)] * 2)
    image = np.ones((2, 16, 16), np.uint16)
    config = _config("connected_components", image_axes="TYX", mask_axes="TYX", t_policy="all")
    with pytest.raises(ValueError, match="binary_mask_policy='connected_components' is set, but run_patching"):
        run_patching(images=image, masks=masks, config=config, image_metadata=named(image))
    random = _config("connected_components", mode="random")
    with pytest.raises(ValueError, match="binary_mask_policy='connected_components' is set, but run_patching"):
        run_patching(images=np.ones((16, 16), np.uint16), config=random, image_metadata=named(np.ones((16, 16))))


def test_the_same_boolean_array_is_accepted_as_a_region():
    # F07's last case: a bool array is refused as an object set, accepted as a region.
    sampling = _two_blobs(bool)
    image = np.ones((16, 16), np.uint16)
    config = _config(mode="random")
    config.random.patch_size, config.random.n_patches, config.random.seed = 3, 2, 0
    result = run_patching(images=image, masks=sampling, config=config, image_metadata=named(image),
                          fraction_masks=[{"blobs": sampling}])
    assert len(result.patches) == 2
    assert all(p.mask_fractions["blobs"] > 0 for p in result.patches)
