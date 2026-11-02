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


_DEFAULT_FILTERS_SCRIPT = """
import sys
from pathlib import Path

import numpy as np

from cellpatcher.torch_dataset import ExportedPatchTorchDataset, PatchTensorConfig

folder = Path(sys.argv[1])
np.save(folder / "p.npy", np.full((4, 4), 1000, dtype=np.uint16))
(folder / "patches.csv").write_text("patch_id,patch_file\\np1,p.npy\\n", encoding="utf-8")
for config in (None, PatchTensorConfig(channel_policy="keep", scale_policy="none")):
    item = ExportedPatchTorchDataset(folder / "patches.csv", tensor_config=config)[0]
    print(item["scale_divisor"], float(item["image"].max()))
"""


def test_the_default_scale_policy_rescaling_is_shown_and_recorded(tmp_path):
    # Python hides a DeprecationWarning raised inside a library, so the default
    # scale_policy='auto' divided integer patches by 65535 without a word (CP-3b review).
    pytest.importorskip("torch", reason="the torch extra installs torch")
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _DEFAULT_FILTERS_SCRIPT, str(tmp_path)],
        capture_output=True, text=True, check=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONWARNINGS"},
    )
    default, kept = result.stdout.strip().splitlines()
    divisor, maximum = map(float, default.split())
    assert (divisor, maximum) == (65535.0, pytest.approx(1000 / 65535))
    assert kept.split() == ["1.0", "1000.0"]  # scale_policy='none': unscaled, nothing to record
    assert "ScalePolicyDeprecationWarning" in result.stderr
    assert "divided an integer patch by its dtype maximum (65535)" in result.stderr
