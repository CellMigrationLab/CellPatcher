"""The old normalizations are deprecated in favour of config.normalization (contract §15)."""

import importlib
import importlib.util
import sys
import types
import warnings

import numpy as np
import pytest

from cellpatcher.preprocess import (
    _minmax_normalize,
    _percentile_normalize,
    iter_cell_planes_for_patching,
    iter_image_planes_for_patching,
    minmax_normalize,
    percentile_normalize,
    prepare_image_for_patching,
)


def _deprecations(caught):
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def _record(action):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = action()
    return result, _deprecations(caught)


IMAGE = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)


def test_the_normalize_functions_warn_and_keep_their_values():
    plane = IMAGE[0]
    result, caught = _record(lambda: percentile_normalize(plane, 2.0, 98.0))
    np.testing.assert_array_equal(result, _percentile_normalize(plane, 2.0, 98.0))
    assert len(caught) == 1 and "config.normalization" in str(caught[0].message)

    result, caught = _record(lambda: minmax_normalize(plane))
    np.testing.assert_array_equal(result, _minmax_normalize(plane))
    assert len(caught) == 1 and "minmax_normalize is deprecated" in str(caught[0].message)
    # CP-5 (D86): the removal is announced.
    assert "removed in CellPatcher v0.2.0" in str(caught[0].message)


def test_prepare_image_warns_only_with_normalize():
    (_, axes), caught = _record(lambda: prepare_image_for_patching(IMAGE, "ZYX", z_policy="first"))
    assert axes == "YX" and caught == []
    (result, _), caught = _record(
        lambda: prepare_image_for_patching(IMAGE, "ZYX", z_policy="first", normalize="minmax")
    )
    assert len(caught) == 1 and "prepare_image_for_patching(normalize=...)" in str(caught[0].message)
    np.testing.assert_array_equal(result, _minmax_normalize(IMAGE[0]))


def test_the_plane_iterators_warn_once_per_call():
    planes, caught = _record(
        lambda: list(iter_image_planes_for_patching(IMAGE, "ZYX", z_policy="all", normalize="percentile"))
    )
    assert len(planes) == 3 and len(caught) == 1

    labels = np.ones((3, 8, 8), dtype=np.uint16)
    pairs, caught = _record(
        lambda: list(iter_cell_planes_for_patching(IMAGE, "ZYX", labels, "ZYX", z_policy="all", normalize="minmax"))
    )
    assert len(pairs) == 3 and len(caught) == 1

    _, caught = _record(lambda: list(iter_image_planes_for_patching(IMAGE, "ZYX", z_policy="all")))
    assert caught == []


@pytest.fixture
def torch_dataset(monkeypatch):
    """cellpatcher.torch_dataset, with a minimal stand-in for torch when it is not installed."""
    if importlib.util.find_spec("torch") is not None:
        yield importlib.import_module("cellpatcher.torch_dataset")
        return
    fake = types.ModuleType("torch")
    fake.from_numpy = np.asarray
    fake.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object))
    monkeypatch.setitem(sys.modules, "torch", fake)
    sys.modules.pop("cellpatcher.torch_dataset", None)
    yield importlib.import_module("cellpatcher.torch_dataset")
    # Never leave the module bound to the stand-in for later tests.
    sys.modules.pop("cellpatcher.torch_dataset", None)


def test_scale_policy_auto_warns_only_when_it_rescales_an_integer_patch(torch_dataset):
    PatchTensorConfig, patch_array_to_tensor = torch_dataset.PatchTensorConfig, torch_dataset.patch_array_to_tensor

    keep = PatchTensorConfig(channel_policy="keep")
    tensor, caught = _record(lambda: patch_array_to_tensor(IMAGE[0].astype(np.uint8), axes="YX", config=keep))
    assert len(caught) == 1 and float(tensor.max()) <= 1.0

    _, caught = _record(lambda: patch_array_to_tensor(IMAGE[0].astype(np.float32), axes="YX", config=keep))
    assert caught == []
    none = PatchTensorConfig(channel_policy="keep", scale_policy="none")
    _, caught = _record(lambda: patch_array_to_tensor(IMAGE[0].astype(np.uint8), axes="YX", config=none))
    assert caught == []
