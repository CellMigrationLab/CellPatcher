"""Deprecations still in place: the torch scale policy (the old normalizations were removed in v0.1.1).

Since v0.2.0 (CP-4) the torch adapters neither rescale nor repeat channels unless asked:
``scale_policy`` has no default and ``channel_policy`` defaults to ``"keep"``.
"""

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

    auto = PatchTensorConfig(scale_policy="auto")
    tensor, caught = _record(lambda: patch_array_to_tensor(IMAGE[0].astype(np.uint8), axes="YX", config=auto))
    assert len(caught) == 1 and float(tensor.max()) <= 1.0

    _, caught = _record(lambda: patch_array_to_tensor(IMAGE[0].astype(np.float32), axes="YX", config=auto))
    assert caught == []
    none = PatchTensorConfig(scale_policy="none")
    _, caught = _record(lambda: patch_array_to_tensor(IMAGE[0].astype(np.uint8), axes="YX", config=none))
    assert caught == []


def test_an_omitted_scale_policy_is_refused_naming_the_choices(torch_dataset):
    # CP-4: PatchTensorConfig() divided integer patches by their dtype maximum (scale_policy='auto').
    for omitted in (lambda: torch_dataset.PatchTensorConfig(),
                    lambda: torch_dataset.PatchTensorConfig(channel_policy="keep"),
                    lambda: torch_dataset.patch_array_to_tensor(IMAGE[0], axes="YX")):
        with pytest.raises(TypeError, match=r"explicit scale_policy.*'none' keeps the values.*'auto' \(deprecated\)"):
            omitted()


def test_the_default_channel_policy_keeps_one_channel_and_its_values(torch_dataset):
    # CP-4: a uint16 patch of 1000 came back as 0.0153 repeated to 3 channels by default.
    config = torch_dataset.PatchTensorConfig(scale_policy="none")
    assert config.channel_policy == "keep"
    tensor, caught = _record(lambda: torch_dataset.patch_array_to_tensor(
        np.full((4, 4), 1000, dtype=np.uint16), axes="YX", config=config))
    assert np.asarray(tensor).shape == (1, 4, 4) and np.asarray(tensor).dtype == np.float32
    assert set(np.asarray(tensor).ravel().tolist()) == {1000.0}
    assert caught == []


def _exported(folder, array):
    np.save(folder / "p.npy", array)
    (folder / "patches.csv").write_text("patch_id,patch_file\np1,p.npy\n", encoding="utf-8")
    return folder / "patches.csv"


def test_the_torch_datasets_need_a_tensor_config_with_a_scale_policy(torch_dataset, tmp_path):
    table = _exported(tmp_path, np.full((4, 4), 1000, dtype=np.uint16))
    with pytest.raises(TypeError, match="explicit scale_policy"):
        torch_dataset.ExportedPatchTorchDataset(table)
    with pytest.raises(TypeError, match="explicit scale_policy"):
        torch_dataset.PatchTorchDataset(np.zeros((4, 4)), [])


def test_the_old_defaults_are_reachable_by_name_and_recorded_per_item(torch_dataset, tmp_path):
    PatchTensorConfig = torch_dataset.PatchTensorConfig
    table = _exported(tmp_path, np.full((4, 4), 1000, dtype=np.uint16))

    old = PatchTensorConfig(channel_policy="repeat_to_rgb", scale_policy="auto")
    item, caught = _record(lambda: torch_dataset.ExportedPatchTorchDataset(table, tensor_config=old)[0])
    assert (item["scale_divisor"], item["channel_repeat"]) == (65535.0, 3)
    image = np.asarray(item["image"])
    assert image.shape == (3, 4, 4) and np.allclose(image, 1000 / 65535)
    assert [w.category for w in caught] == [torch_dataset.ScalePolicyDeprecationWarning]

    item, caught = _record(lambda: torch_dataset.ExportedPatchTorchDataset(
        table, tensor_config=PatchTensorConfig(scale_policy="none"))[0])
    assert (item["scale_divisor"], item["channel_repeat"]) == (1.0, 1)
    assert np.asarray(item["image"]).shape == (1, 4, 4) and float(np.asarray(item["image"]).max()) == 1000.0
    assert caught == []


_DEFAULT_FILTERS_SCRIPT = """
import sys
from pathlib import Path

import numpy as np

from cellpatcher.torch_dataset import ExportedPatchTorchDataset, PatchTensorConfig

folder = Path(sys.argv[1])
np.save(folder / "p.npy", np.full((4, 4), 1000, dtype=np.uint16))
(folder / "patches.csv").write_text("patch_id,patch_file\\np1,p.npy\\n", encoding="utf-8")
for config in (PatchTensorConfig(scale_policy="auto"), PatchTensorConfig(scale_policy="none")):
    item = ExportedPatchTorchDataset(folder / "patches.csv", tensor_config=config)[0]
    print(item["scale_divisor"], float(item["image"].max()))
"""


def test_the_asked_for_scale_policy_rescaling_is_shown_and_recorded(tmp_path):
    # Python hides a DeprecationWarning raised inside a library, so scale_policy='auto' (the
    # default before v0.2.0) divided integer patches by 65535 without a word (CP-3b review).
    pytest.importorskip("torch", reason="the torch extra installs torch")
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _DEFAULT_FILTERS_SCRIPT, str(tmp_path)],
        capture_output=True, text=True, check=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONWARNINGS"},
    )
    rescaled, kept = result.stdout.strip().splitlines()
    divisor, maximum = map(float, rescaled.split())
    assert (divisor, maximum) == (65535.0, pytest.approx(1000 / 65535))
    assert kept.split() == ["1.0", "1000.0"]  # scale_policy='none': unscaled, nothing to record
    assert "ScalePolicyDeprecationWarning" in result.stderr
    assert "divided an integer patch by its dtype maximum (65535)" in result.stderr


def test_the_live_dataset_records_channel_repeat_and_keeps_any_channel_count(torch_dataset):
    # CP-4 review: PatchTorchDataset records the repetition per item like the exported dataset, and
    # the default "keep" passes 1, 2 and 4 channels through with their values and no rescale.
    from cellpatcher.types import PatchRegion

    PatchTensorConfig, PatchTorchDataset = torch_dataset.PatchTensorConfig, torch_dataset.PatchTorchDataset
    patch = PatchRegion(patch_id="p", image_id="image_0000", y0=0, y1=4, x0=0, x1=4)
    for channels in (1, 2, 4):
        image = np.full((channels, 8, 8), 1000, dtype=np.uint16)
        item = PatchTorchDataset(image, [patch], image_axes="CYX",
                                 tensor_config=PatchTensorConfig(scale_policy="none"))[0]
        tensor = np.asarray(item["image"])
        assert tensor.shape == (channels, 4, 4) and set(tensor.ravel().tolist()) == {1000.0}
        assert (item["scale_divisor"], item["channel_repeat"]) == (1.0, 1)

    rgb = PatchTensorConfig(channel_policy="repeat_to_rgb", scale_policy="none")
    for shape, axes, repeat in (((8, 8), "YX", 3), ((1, 8, 8), "CYX", 3), ((3, 8, 8), "CYX", 1)):
        item = PatchTorchDataset(np.ones(shape, dtype=np.uint16), [patch], image_axes=axes, tensor_config=rgb)[0]
        assert np.asarray(item["image"]).shape == (3, 4, 4) and item["channel_repeat"] == repeat
