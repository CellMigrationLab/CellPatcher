"""Delivery modes give identical arrays (docs/normalization_contract.md §12).

The same configuration is delivered as eager arrays, normalized files, raw
files normalized on load, the lazy NumPy iterator (from arrays and from file
paths) and the torch datasets; the arrays must agree.
"""

import copy
import importlib.util
import json

import numpy as np
import pytest

from cellpatcher import iter_exported_patches, iter_patch_arrays, patch_array
from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig
from cellpatcher.io import list_supported_files, read_image
from cellpatcher.orchestration import run_patching_condition_repeat_tree

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs the 'preprocessing' extra"
    ),
    pytest.mark.filterwarnings("ignore::cellpatcher.normalization.NormalizationWarning"),
]

IMAGES = ("control/bio1/a.tif", "control/bio1/b.tif", "treated/bio1/c.tif")


def _write_tiff(path, data, axes):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": axes})


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    import morphobricks_preprocessing as mbp

    base = tmp_path_factory.mktemp("delivery")
    root, masks = base / "experiment", base / "masks"
    rng = np.random.default_rng(0)
    for index, path in enumerate(IMAGES):
        stack = rng.integers(0, 200, (2, 2, 24, 24), dtype=np.uint8)  # Z, C, Y, X
        stack[:, 1] += 40
        _write_tiff(root / path, stack, "ZCYX")
        labels = np.zeros((2, 24, 24), dtype=np.uint16)
        labels[0, 3:9, 3:11] = 1
        labels[1, 12:20, 10:17] = 2 + index
        _write_tiff(masks / path, labels, "ZYX")
    result = mbp.run_experiment(root, mbp.AxisMapping("ZCYX", "ZCYX"), output_folder=base / "preprocessing")
    assert result.written
    return root, masks, result.csv_path


def _grid_config(csv_path):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes, config.common.z_policy = "ZCYX", "all"
    config.normalization = NormalizationConfig(enabled=True, plane_histograms=str(csv_path), method="percentile")
    return config


def _cell_config(csv_path):
    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size, config.cell.exclude_border = 10, False
    config.common.image_axes, config.common.mask_axes, config.common.z_policy = "ZCYX", "ZYX", "all"
    config.extraction.apply_mask, config.extraction.mask_fill_value = True, float("nan")
    config.extraction.output_size, config.extraction.split_channels = 6, True
    config.normalization = NormalizationConfig(enabled=True, plane_histograms=str(csv_path), method="zscore")
    return config


def _selected_channel_config(csv_path):
    config = _cell_config(csv_path)
    config.extraction.split_channels = [1]  # one channel of two (CellPatcher follow-up C5)
    return config


def _clipped(make, clip):
    def config(csv_path):
        config = make(csv_path)
        config.normalization.clip = clip
        return config

    return config


CONFIGS = {
    "grid": _grid_config,
    "cell, masked, resized, split": _cell_config,
    "cell, masked, resized, channel 1 selected": _selected_channel_config,
    "grid, clipped": _clipped(_grid_config, (0.2, 0.8)),
    "cell, masked, resized, split, clipped": _clipped(_cell_config, (-1.0, 1.0)),
}


def _run(root, masks, config, **options):
    return run_patching_condition_repeat_tree(
        root, config=config, mask_root_path=masks if config.mode == "cell" else None, **options
    )


def _eager(root, masks, config):
    config = copy.deepcopy(config)
    config.common.extract_arrays = True
    arrays = {}
    for leaf in _run(root, masks, config):
        for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays):
            arrays[(patch.source_relative_path, patch.patch_id)] = array
    return arrays


def _exported(root, masks, config, out, export_values):
    config = copy.deepcopy(config)
    config.export.enabled, config.common.output_dir = True, str(out)
    config.normalization.export_values = export_values
    _run(root, masks, config, combine_metadata=True)
    return {(row["source_relative_path"], row["patch_id"]): array for row, array in iter_exported_patches(out / "patches.csv")}


def _same(actual, expected, *, exact=True):
    assert actual.keys() == expected.keys()
    for key, array in expected.items():
        assert actual[key].dtype == np.float32 and actual[key].shape == array.shape, key
        if exact:
            np.testing.assert_array_equal(actual[key], array, err_msg=str(key))
        else:
            np.testing.assert_allclose(actual[key], array, rtol=1e-5, atol=1e-5, err_msg=str(key))


@pytest.mark.parametrize("name", CONFIGS)
def test_files_raw_files_and_the_lazy_iterator_equal_eager_delivery(experiment, tmp_path, name):
    root, masks, csv_path = experiment
    config = CONFIGS[name](csv_path)
    eager = _eager(root, masks, config)
    assert eager and all(array.dtype == np.float32 for array in eager.values())

    _same(_exported(root, masks, config, tmp_path / "normalized", "normalized"), eager)
    # Raw files of masked/resized patches are float32 raw values (owner decision): equal up to rounding.
    _same(_exported(root, masks, config, tmp_path / "raw", "raw"), eager, exact=name.startswith("grid"))
    clip = config.normalization.clip
    if clip is not None:
        finite = [array[np.isfinite(array)] for array in eager.values()]  # NaN mask fill aside
        assert all(values.min() >= clip[0] and values.max() <= clip[1] for values in finite)

    lazy_from_paths, lazy_from_arrays = {}, {}
    for leaf in _run(root, masks, config):
        image_paths = list_supported_files(leaf.image_folder)
        mask_paths = list_supported_files(masks / leaf.relative_folder) if config.mode == "cell" else None
        for patch, array in iter_patch_arrays(leaf.result.patches, image_paths, config, masks=mask_paths):
            lazy_from_paths[(patch.source_relative_path, patch.patch_id)] = array
        images = [read_image(path, config.common.image_axes) for path in image_paths]
        mask_arrays = [read_image(path, config.common.mask_axes) for path in mask_paths] if mask_paths else None
        for patch in leaf.result.patches:
            mask = mask_arrays[int(patch.image_id[-4:])] if mask_arrays else None
            lazy_from_arrays[(patch.source_relative_path, patch.patch_id)] = patch_array(
                patch, images[int(patch.image_id[-4:])], config, mask=mask
            )
    _same(lazy_from_paths, eager)
    _same(lazy_from_arrays, eager)


def test_raw_files_keep_the_source_dtype_when_only_cropped(experiment, tmp_path):
    import tifffile

    root, masks, csv_path = experiment
    config = _grid_config(csv_path)
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    config.normalization.export_values = "raw"
    (leaf, *_) = _run(root, masks, config)
    patch = leaf.result.patches[0]
    stored = tifffile.imread(tmp_path / "out" / leaf.relative_folder / patch.patch_file)
    source = tifffile.imread(root / patch.source_relative_path)
    assert stored.dtype == np.uint8
    np.testing.assert_array_equal(stored, source[patch.z0, :, patch.y0:patch.y1, patch.x0:patch.x1])
    saved = json.loads((tmp_path / "out" / leaf.relative_folder / "run_config.json").read_text(encoding="utf-8"))
    assert saved["normalization"]["export_values"] == "raw"


def test_eager_delivery_stops_above_the_memory_limit(experiment, tmp_path):
    root, masks, csv_path = experiment
    config = _grid_config(csv_path)
    config.common.extract_arrays = True
    config.normalization.memory_limit = 4096
    with pytest.raises(ValueError, match="iter_patch_arrays"):
        _run(root, masks, config)

    # Export only: arrays are written to files, not kept, so the limit does not apply.
    config.common.extract_arrays = False
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    assert _run(root, masks, config)


def test_delivery_settings_are_validated():
    with pytest.raises(ValueError, match="export_values must be one of"):
        NormalizationConfig(export_values="both")
    with pytest.raises(ValueError, match="memory_limit must be a positive int"):
        NormalizationConfig(memory_limit=0)


# ---------------------------------------------------------------------------
# Torch adapters (skipped without torch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONFIGS)
def test_torch_datasets_give_the_same_arrays(experiment, tmp_path, name):
    pytest.importorskip("torch")
    from cellpatcher.torch_dataset import ExportedPatchTorchDataset, PatchTensorConfig, PatchTorchDataset

    root, masks, csv_path = experiment
    config = CONFIGS[name](csv_path)
    eager = _eager(root, masks, config)
    tensors = PatchTensorConfig(channel_policy="keep", dtype_policy="keep", scale_policy="none", return_metadata=True)

    from_images = {}
    for leaf in _run(root, masks, config):
        images = [read_image(p, config.common.image_axes) for p in list_supported_files(leaf.image_folder)]
        mask_list = (
            [read_image(p, config.common.mask_axes) for p in list_supported_files(masks / leaf.relative_folder)]
            if config.mode == "cell" else None
        )
        dataset = PatchTorchDataset(images, leaf.result.patches, masks=mask_list, config=config, tensor_config=tensors)
        for item in dataset:
            from_images[(item["patch"]["source_relative_path"], item["patch_id"])] = item["image"].numpy()
    # The tensors are channel-first; compare channel-first views of the eager arrays.
    _same({k: v.reshape(eager[k].shape) for k, v in from_images.items()}, eager)

    out = tmp_path / "raw"
    _exported(root, masks, config, out, "raw")
    axes = "YX" if config.extraction.split_channels else "CYX"  # the exported patches' axes
    files = ExportedPatchTorchDataset(out / "patches.csv", axes=axes, tensor_config=tensors)
    from_files = {(item["patch"]["source_relative_path"], item["patch_id"]): item["image"].numpy() for item in files}
    _same({k: v.reshape(eager[k].shape) for k, v in from_files.items()}, eager, exact=(name == "grid"))
