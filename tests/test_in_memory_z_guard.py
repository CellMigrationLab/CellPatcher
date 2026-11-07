"""An in-memory array with more than one Z plane is refused under every ``z_policy`` (issue #83).

A file run already refuses such an image (``fov_layout``); the arrays passed in memory take the
same refusal, whatever the Z policy says, so a depth stack is never patched plane by plane,
kept whole, or silently reduced.
"""

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.errors import ZStackNotSupportedError
from cellpatcher.pipeline import run_patching
from cellpatcher.planes import iter_image_planes
from cellpatcher.preprocess import iter_image_planes_for_patching, prepare_image_for_patching
from tests.helpers import named

Z_POLICIES = ["keep", "all", "first", "select", "max", "mean", "sum"]


def _stack(depth, axes="ZYX"):
    shape = {"Z": depth, "Y": 16, "X": 16}
    return np.arange(depth * 256, dtype=np.float32).reshape([shape[axis] for axis in axes])


def _config(z_policy):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = "ZYX"
    config.common.z_policy = z_policy
    config.common.z_index = 0 if z_policy == "select" else None
    return config


@pytest.mark.parametrize("z_policy", Z_POLICIES)
def test_run_patching_refuses_a_z_stack_under_every_policy(z_policy):
    image = _stack(3)
    with pytest.raises(ZStackNotSupportedError, match="3 Z planes") as caught:
        run_patching(images=image, config=_config(z_policy), image_metadata=named(image))
    assert (caught.value.axis, caught.value.length) == ("Z", 3)


@pytest.mark.parametrize("z_policy", Z_POLICIES)
def test_run_patching_refuses_a_z_stack_mask_under_every_policy(z_policy):
    config = _config(z_policy)
    config.mode = "random"
    config.common.mask_axes = "ZYX"
    image = _stack(1)
    with pytest.raises(ZStackNotSupportedError, match="mask"):
        run_patching(images=image, masks=(_stack(2) > 0).astype(np.int32), config=config,
            image_metadata=named(image))


@pytest.mark.parametrize("z_policy", ["all", "first", "select"])
def test_iter_image_planes_refuses_a_z_stack(z_policy):
    config = _config(z_policy)
    with pytest.raises(ZStackNotSupportedError):
        list(iter_image_planes(_stack(2), "ZYX", None, None, config.common))


@pytest.mark.parametrize("z_policy", ["first", "select"])
def test_preprocess_helpers_refuse_a_z_stack(z_policy):
    with pytest.raises(ZStackNotSupportedError):
        prepare_image_for_patching(_stack(2), "ZYX", z_policy=z_policy, z_index=0)
    with pytest.raises(ZStackNotSupportedError):
        list(iter_image_planes_for_patching(_stack(2), "ZYX", z_policy=z_policy, z_index=0))


@pytest.mark.parametrize("z_policy", Z_POLICIES)
def test_one_z_plane_is_still_accepted(z_policy):
    image = _stack(1)
    result = run_patching(images=image, config=_config(z_policy), image_metadata=named(image))
    assert result.patches
