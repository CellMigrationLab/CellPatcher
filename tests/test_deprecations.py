"""Deprecations still in place: the torch scale policy (the old normalizations were removed in v0.1.1)."""

import importlib
import importlib.util
import sys
import types
import warnings

import numpy as np
import pytest


def _deprecations(caught):
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def _record(action):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = action()
    return result, _deprecations(caught)


IMAGE = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)


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
