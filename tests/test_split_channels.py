import csv

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, ExtractionConfig, GridConfig
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.pipeline import run_patching


def _channel_image(n_channels=3, size=16, channels_last=False):
    # Channel k is filled with the value k + 1, so each split patch is easy to identify.
    stack = np.stack([np.full((size, size), channel + 1, dtype=np.uint16) for channel in range(n_channels)])
    return np.moveaxis(stack, 0, -1) if channels_last else stack


def _grid_config(image_axes="CYX", **extraction):
    return CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes=image_axes, extract_arrays=True),
        grid=GridConfig(tile_size=8),
        extraction=ExtractionConfig(split_channels=True, **extraction),
    )


def test_split_channels_creates_one_single_channel_patch_per_channel():
    result = run_patching(images=[_channel_image()], config=_grid_config())

    assert len(result.patches) == 4 * 3  # 4 tiles x 3 channels
    first_tile = result.patches[:3]
    assert [patch.patch_id for patch in first_tile] == [
        "image_0000_grid_00000_c0",
        "image_0000_grid_00000_c1",
        "image_0000_grid_00000_c2",
    ]
    assert [patch.c for patch in first_tile] == [0, 1, 2]
    assert len({(patch.y0, patch.y1, patch.x0, patch.x1) for patch in first_tile}) == 1
    for patch, array in zip(result.patches, result.patch_arrays):
        assert array.shape == (8, 8)
        assert np.all(array == patch.c + 1)
        assert (patch.output_height, patch.output_width) == (8, 8)


def test_split_channels_supports_channels_last_images():
    result = run_patching(images=[_channel_image(channels_last=True)], config=_grid_config("YXC"))

    assert len(result.patches) == 4 * 3
    assert all(np.all(array == patch.c + 1) for patch, array in zip(result.patches, result.patch_arrays))


def test_split_channels_resizes_before_splitting():
    result = run_patching(images=[_channel_image()], config=_grid_config(output_size=4))

    assert all(array.shape == (4, 4) for array in result.patch_arrays)
    assert all((patch.output_height, patch.output_width) == (4, 4) for patch in result.patches)


def test_split_channel_patches_share_a_default_position_id():
    config = _grid_config()
    config.common.condition, config.common.repeat = "treated", "2026_01_01"
    result = run_patching(
        images=[_channel_image(), _channel_image()],
        config=config,
        image_metadata=[{"image_file": "img0.tif"}, {"image_file": "img1.tif"}],
    )

    by_image = {}
    for patch in result.patches:
        by_image.setdefault(patch.image_file, set()).add(patch.position_id)
    assert by_image == {"img0.tif": {"treated/2026_01_01/img0"}, "img1.tif": {"treated/2026_01_01/img1"}}


def test_split_channels_keeps_a_supplied_position_id():
    result = run_patching(images=[_channel_image()], config=_grid_config(), image_metadata=[{"position_id": "P07"}])
    assert {patch.position_id for patch in result.patches} == {"P07"}


def test_split_channels_position_id_falls_back_to_image_id():
    result = run_patching(images=[_channel_image()], config=_grid_config())
    assert {patch.position_id for patch in result.patches} == {"image_0000"}


def test_without_split_channels_position_id_stays_empty_and_patches_stay_multichannel():
    config = _grid_config()
    config.extraction.split_channels = False
    result = run_patching(images=[_channel_image()], config=config)

    assert len(result.patches) == 4
    assert all(patch.position_id is None and patch.c is None for patch in result.patches)
    assert all(array.shape == (3, 8, 8) for array in result.patch_arrays)


def test_split_channels_without_extracting_arrays_still_splits_regions():
    config = _grid_config()
    config.common.extract_arrays = False
    result = run_patching(images=[_channel_image()], config=config)

    assert result.patch_arrays is None
    assert len(result.patches) == 4 * 3


def test_split_channels_requires_a_channel_axis():
    with pytest.raises(ValueError, match="split_channels=True needs a 'C' axis"):
        run_patching(images=[np.ones((16, 16), dtype=np.uint16)], config=_grid_config("YX"))


def test_split_channels_config_validation():
    with pytest.raises(ValueError, match="split_channels must be a bool"):
        ExtractionConfig(split_channels="yes")


def test_split_channels_with_cell_mode_and_apply_mask():
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[4:10, 4:10] = 1
    image = _channel_image(n_channels=2)
    config = CellPatcherConfig(
        mode="cell",
        common=CommonConfig(image_axes="CYX", mask_axes="YX", extract_arrays=True),
        extraction=ExtractionConfig(split_channels=True, apply_mask=True, mask_fill_value=0),
    )
    result = run_patching(images=[image], masks=[labels], config=config)

    assert [patch.c for patch in result.patches] == [0, 1]
    assert {patch.label_id for patch in result.patches} == {1}
    for patch, array in zip(result.patches, result.patch_arrays):
        assert array.ndim == 2
        assert set(np.unique(array)) <= {0, patch.c + 1}


def test_split_channels_export_writes_one_file_and_row_per_channel(tmp_path):
    config = _grid_config()
    config.export.enabled = True
    config.export.image_format = "npy"
    config.common.output_dir = str(tmp_path)
    result = run_patching(images=[_channel_image()], config=config, image_metadata=[{"image_file": "img0.tif"}])

    assert len(result.exported_files) == 12
    with (tmp_path / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {row["position_id"] for row in rows} == {"img0"}
    assert sorted({row["c"] for row in rows}) == ["0", "1", "2"]
    for row in rows:
        array = np.load(tmp_path / row["patch_file"])
        assert array.shape == (8, 8)
        assert np.all(array == int(row["c"]) + 1)


def test_split_channels_in_combined_tree_run(tmp_path):
    image_root = tmp_path / "raw_data"
    for condition in ("control", "treated"):
        path = image_root / condition / "img0.npy"
        path.parent.mkdir(parents=True)
        write_array(path, _channel_image(n_channels=2))

    config = _grid_config()
    config.export.enabled = True
    config.export.image_format = "npy"
    config.common.output_dir = str(tmp_path / "out")
    run_patching_condition_repeat_tree(image_root, config=config, combine_metadata=True)

    with (tmp_path / "out" / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 4 * 2  # conditions x tiles x channels
    assert {row["position_id"] for row in rows} == {"control/single/img0", "treated/single/img0"}
    assert all((tmp_path / "out" / row["patch_file"]).exists() for row in rows)
