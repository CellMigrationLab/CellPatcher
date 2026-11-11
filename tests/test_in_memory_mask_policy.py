"""In memory only a boolean mask is binary: ``cell.binary_mask_policy`` with an integer mask is refused.

The file route refuses a policy on a mask declared ``instances`` (nothing to convert; Spec ADR 0002
AR5/AR11 (4), ADR 0001 AQ9, D47: an instances mask holding 0 and 1 is one object). Before this fix
in-memory ``run_patching`` silently ignored the policy for an integer 0/1 mask and gave one object.
"""

import numpy as np
import pytest

from cellpatcher import CellPatcherConfig, run_patching
from tests.helpers import named

REFUSAL = "nothing to convert. Pass a boolean mask"


def _two_blobs(dtype):
    """The 16x16 0/1 mask of two separate 3x3 squares (the shared labeller's ``two_blob_0_1``)."""
    mask = np.zeros((16, 16), dtype)
    mask[2:5, 2:5] = mask[10:13, 10:13] = 1
    return mask


def _config(policy, **common):
    config = CellPatcherConfig(mode="cell")
    config.cell.exclude_border = False
    config.cell.binary_mask_policy = policy
    for name, value in common.items():
        setattr(config.common, name, value)
    return config


def _objects(mask, config):
    image = np.ones(mask.shape[-2:] if mask.ndim == 2 else mask.shape, np.uint16)
    result = run_patching(images=image, masks=mask, config=config, image_metadata=named(image))
    return [(p.label_id, p.area_px) for p in result.patches]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_a_policy_with_an_integer_mask_is_refused_naming_the_fix(dtype, tmp_path):
    config = _config("connected_components", output_dir=str(tmp_path / "out"))
    config.export.enabled = True
    with pytest.raises(ValueError, match=REFUSAL) as raised:
        _objects(_two_blobs(dtype), config)
    message = str(raised.value)
    assert "cell.binary_mask_policy='connected_components'" in message
    assert f"image 0 is integer ({np.dtype(dtype)})" in message
    assert "leave cell.binary_mask_policy None" in message
    assert not (tmp_path / "out").exists()  # refused before anything is written


def test_a_boolean_mask_with_the_policy_is_two_8_connected_objects():
    with pytest.warns(UserWarning, match="8-connected components"):
        assert _objects(_two_blobs(bool), _config("connected_components")) == [(1, 9), (2, 9)]


def test_an_integer_mask_without_a_policy_is_one_object_as_stored():
    assert _objects(_two_blobs(np.uint8), _config(None)) == [(1, 18)]


def test_one_integer_mask_among_boolean_ones_is_refused():
    images = [np.ones((16, 16), np.uint16)] * 2
    masks = [_two_blobs(bool), _two_blobs(np.uint8)]
    with pytest.raises(ValueError, match="the mask of image 1 is integer"):
        run_patching(images=images, masks=masks, config=_config("connected_components"),
                     image_metadata=named(images))


def test_the_policy_with_an_integer_mask_is_refused_when_planes_are_split():
    # The per-plane calls of a plane split are internal; the public call refuses first.
    mask = np.stack([_two_blobs(np.uint8)] * 2)
    config = _config("connected_components", image_axes="TYX", mask_axes="TYX", t_policy="all")
    with pytest.raises(ValueError, match=REFUSAL):
        _objects(mask, config)
    with pytest.warns(UserWarning, match="8-connected components"):
        assert _objects(mask.astype(bool), config) == [(1, 9), (2, 9)] * 2


# --- the object scan (scan_cell_objects / scan_cell_patch_size) takes the same rule ---------------


def _scan_objects(mask, policy):
    from cellpatcher import CellConfig, scan_cell_objects

    config = CellConfig(exclude_border=False, binary_mask_policy=policy)
    return [(r.label_id, r.area_px) for r in scan_cell_objects(image_shape=mask.shape, labels=mask, config=config)]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_the_object_scan_refuses_a_policy_with_an_integer_mask(dtype):
    from cellpatcher import CellConfig, scan_cell_patch_size

    with pytest.raises(ValueError, match=REFUSAL) as raised:
        _scan_objects(_two_blobs(dtype), "connected_components")
    assert f"the mask of image_0000 is integer ({np.dtype(dtype)})" in str(raised.value)
    masks = [_two_blobs(bool), _two_blobs(dtype)]
    config = CellConfig(exclude_border=False, binary_mask_policy="connected_components")
    with pytest.raises(ValueError, match="the mask of image_0001 is integer"):
        scan_cell_patch_size([(16, 16)] * 2, masks, config=config)


def test_the_object_scan_labels_a_boolean_mask_and_keeps_an_integer_one_as_stored():
    with pytest.warns(UserWarning, match="8-connected components"):
        assert _scan_objects(_two_blobs(bool), "connected_components") == [(1, 9), (2, 9)]
    assert _scan_objects(_two_blobs(np.uint8), None) == [(1, 18)]


def test_the_notebook_scan_of_a_file_run_labels_its_binary_mask_with_the_policy(tmp_path):
    # The notebook's file-mode scan gets the masks the resolver labelled (int32): not refused.
    from cellpatcher.notebook_session import scan_object_sizes, summarize_run
    from tests.test_cell_mask_request import _two_blob_run

    _, _, output = _two_blob_run(tmp_path, "binary")
    config = CellPatcherConfig(mode="cell")
    config.cell.exclude_border = False
    config.cell.binary_mask_policy = "connected_components"
    config.matching.mask_id = "mask"
    with pytest.warns(UserWarning, match="8-connected"):
        scan = scan_object_sizes(config, summarize_run(output))
    assert (scan.n_images, scan.n_objects_total) == (1, 2)


# --- the public cell strategy, called directly, takes the same rule ------------------------------


def _strategy_objects(mask, policy, **kwargs):
    from cellpatcher import CellConfig, CellSegmentationStrategy

    strategy = CellSegmentationStrategy(CellConfig(exclude_border=False, binary_mask_policy=policy))
    regions = strategy.generate_regions(image_shape=mask.shape, labels=mask, **kwargs)
    return [(r.label_id, r.area_px) for r in regions]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_the_cell_strategy_called_directly_refuses_a_policy_with_an_integer_mask(dtype):
    with pytest.raises(ValueError, match=REFUSAL) as raised:
        _strategy_objects(_two_blobs(dtype), "connected_components", image_id="image_0003")
    assert f"the mask of image_0003 is integer ({np.dtype(dtype)})" in str(raised.value)
    with pytest.raises(ValueError, match="the mask of the labels given to generate_regions is integer"):
        _strategy_objects(_two_blobs(dtype), "connected_components")


def test_the_cell_strategy_labels_a_boolean_mask_and_keeps_an_integer_one_as_stored():
    with pytest.warns(UserWarning, match="8-connected components"):
        assert _strategy_objects(_two_blobs(bool), "connected_components") == [(1, 9), (2, 9)]
    assert _strategy_objects(_two_blobs(np.uint8), None) == [(1, 18)]
