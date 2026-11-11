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
