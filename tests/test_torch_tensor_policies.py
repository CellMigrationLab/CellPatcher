"""The torch adapters' tensor policies.

Since v0.2.0 (CP-4) the torch adapters neither rescale nor repeat channels unless asked:
``scale_policy`` has no default and ``channel_policy`` defaults to ``"keep"``. The deprecated
``scale_policy="auto"`` (divide an integer patch by its dtype maximum) was removed in the
compatibility sweep: ``"none"`` is the only choice.
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


def test_an_omitted_scale_policy_is_refused_naming_the_choices(torch_dataset):
    # CP-4: PatchTensorConfig() divided integer patches by their dtype maximum (scale_policy='auto').
    for omitted in (lambda: torch_dataset.PatchTensorConfig(),
                    lambda: torch_dataset.PatchTensorConfig(channel_policy="keep"),
                    lambda: torch_dataset.patch_array_to_tensor(IMAGE[0], axes="YX")):
        with pytest.raises(TypeError, match=r"explicit scale_policy.*'none' keeps the values"):
            omitted()


def test_scale_policy_auto_is_refused(torch_dataset):
    # Compatibility sweep CP-6: the deprecated divide-by-dtype-maximum is gone; normalize with
    # config.normalization.
    with pytest.raises(ValueError, match="scale_policy must be 'none'"):
        torch_dataset.PatchTensorConfig(scale_policy="auto")


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


def test_the_channel_repeat_is_reachable_by_name_and_recorded_per_item(torch_dataset, tmp_path):
    PatchTensorConfig = torch_dataset.PatchTensorConfig
    table = _exported(tmp_path, np.full((4, 4), 1000, dtype=np.uint16))

    rgb = PatchTensorConfig(channel_policy="repeat_to_rgb", scale_policy="none")
    item, caught = _record(lambda: torch_dataset.ExportedPatchTorchDataset(table, tensor_config=rgb)[0])
    assert item["channel_repeat"] == 3 and "scale_divisor" not in item
    image = np.asarray(item["image"])
    assert image.shape == (3, 4, 4) and set(image.ravel().tolist()) == {1000.0}
    assert caught == []

    item, caught = _record(lambda: torch_dataset.ExportedPatchTorchDataset(
        table, tensor_config=PatchTensorConfig(scale_policy="none"))[0])
    assert item["channel_repeat"] == 1
    assert np.asarray(item["image"]).shape == (1, 4, 4) and float(np.asarray(item["image"]).max()) == 1000.0
    assert caught == []


def test_the_exported_dataset_refuses_a_repeat_column(torch_dataset, tmp_path):
    # Compatibility sweep CP-1: an old patches.csv's `repeat` column is no longer read as biological_repeat.
    np.save(tmp_path / "p.npy", np.ones((4, 4), dtype=np.uint16))
    (tmp_path / "patches.csv").write_text("patch_id,repeat,patch_file\np1,r1,p.npy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'repeat' column"):
        torch_dataset.ExportedPatchTorchDataset(
            tmp_path / "patches.csv", tensor_config=torch_dataset.PatchTensorConfig(scale_policy="none")
        )


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
        assert item["channel_repeat"] == 1

    rgb = PatchTensorConfig(channel_policy="repeat_to_rgb", scale_policy="none")
    for shape, axes, repeat in (((8, 8), "YX", 3), ((1, 8, 8), "CYX", 3), ((3, 8, 8), "CYX", 1)):
        item = PatchTorchDataset(np.ones(shape, dtype=np.uint16), [patch], image_axes=axes, tensor_config=rgb)[0]
        assert np.asarray(item["image"]).shape == (3, 4, 4) and item["channel_repeat"] == repeat
