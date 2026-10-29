import csv

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, ExtractionConfig, GridConfig
from cellpatcher.io import write_array
from cellpatcher.orchestration import load_config, run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.pipeline import run_patching
from tests.helpers import named


def _channel_image(n_channels=3, size=16, channels_last=False):
    # Channel k is filled with the value k + 1, so each split patch is easy to identify.
    stack = np.stack([np.full((size, size), channel + 1, dtype=np.uint16) for channel in range(n_channels)])
    return np.moveaxis(stack, 0, -1) if channels_last else stack


def _grid_config(image_axes="CYX", **extraction):
    return CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes=image_axes, extract_arrays=True),
        grid=GridConfig(tile_size=8),
        extraction=ExtractionConfig(**{"split_channels": True, **extraction}),
    )


def test_split_channels_creates_one_single_channel_patch_per_channel():
    result = run_patching(images=[_channel_image()], config=_grid_config(), image_metadata=[{"image_file": "img.tif"}])

    assert len(result.patches) == 4 * 3  # 4 tiles x 3 channels
    first_tile = result.patches[:3]
    assert [patch.patch_id for patch in first_tile] == [
        "single/single/single/img/c0/grid-y0-x0-h8-w8",
        "single/single/single/img/c1/grid-y0-x0-h8-w8",
        "single/single/single/img/c2/grid-y0-x0-h8-w8",
    ]
    assert [patch.c for patch in first_tile] == [0, 1, 2]
    assert len({(patch.y0, patch.y1, patch.x0, patch.x1) for patch in first_tile}) == 1
    for patch, array in zip(result.patches, result.patch_arrays):
        assert array.shape == (8, 8)
        assert np.all(array == patch.c + 1)
        assert (patch.output_height, patch.output_width) == (8, 8)


def test_split_channels_supports_channels_last_images():
    result = run_patching(images=[_channel_image(channels_last=True)], config=_grid_config("YXC"),
        image_metadata=named([_channel_image(channels_last=True)]))

    assert len(result.patches) == 4 * 3
    assert all(np.all(array == patch.c + 1) for patch, array in zip(result.patches, result.patch_arrays))


def test_split_channels_resizes_before_splitting():
    result = run_patching(images=[_channel_image()], config=_grid_config(output_size=4),
        image_metadata=named([_channel_image()]))

    assert all(array.shape == (4, 4) for array in result.patch_arrays)
    assert all((patch.output_height, patch.output_width) == (4, 4) for patch in result.patches)


def test_split_channel_patches_share_a_default_position_id():
    config = _grid_config()
    config.common.condition, config.common.biological_repeat = "treated", "2026_01_01"
    result = run_patching(
        images=[_channel_image(), _channel_image()],
        config=config,
        image_metadata=[{"image_file": "img0.tif"}, {"image_file": "img1.tif"}],
    )

    by_image = {}
    for patch in result.patches:
        by_image.setdefault(patch.image_file, set()).add(patch.position_id)
    assert by_image == {
        "img0.tif": {"img0"},
        "img1.tif": {"img1"},
    }


def test_split_channels_keeps_a_supplied_position_id():
    result = run_patching(images=[_channel_image()], config=_grid_config(), image_metadata=[{"position_id": "P07"}])
    assert {patch.position_id for patch in result.patches} == {"P07"}


def test_split_channels_refuse_an_image_without_position_id_or_file_name():
    # Never named by its place in the run (decision D45).
    with pytest.raises(ValueError, match="neither a position_id nor an image_file"):
        run_patching(images=[_channel_image()], config=_grid_config())


def test_without_split_channels_position_id_is_filled_and_patches_stay_multichannel():
    config = _grid_config()
    config.extraction.split_channels = False
    result = run_patching(images=[_channel_image()], config=config, image_metadata=[{"image_file": "img.tif"}])

    assert len(result.patches) == 4
    assert all(patch.position_id == "img" and patch.c is None for patch in result.patches)
    assert all(array.shape == (3, 8, 8) for array in result.patch_arrays)


def test_split_channels_without_extracting_arrays_still_splits_regions():
    config = _grid_config()
    config.common.extract_arrays = False
    result = run_patching(images=[_channel_image()], config=config, image_metadata=named([_channel_image()]))

    assert result.patch_arrays is None
    assert len(result.patches) == 4 * 3


def test_split_channels_requires_a_channel_axis():
    with pytest.raises(ValueError, match="split_channels=True needs a 'C' axis"):
        run_patching(images=[np.ones((16, 16), dtype=np.uint16)], config=_grid_config("YX"),
            image_metadata=named([np.ones((16, 16), dtype=np.uint16)]))


def test_split_channels_config_validation():
    with pytest.raises(ValueError, match="split_channels must be a bool"):
        ExtractionConfig(split_channels="yes")


# ---------------------------------------------------------------------------
# Channel selection (CellPatcher follow-up C5)
# ---------------------------------------------------------------------------


def test_a_channel_selection_gives_one_patch_per_selected_channel():
    image = _channel_image(n_channels=4)
    result = run_patching(images=[image], config=_grid_config(split_channels=[3, 1]),
                          image_metadata=[{"image_file": "img.tif"}])

    assert len(result.patches) == 4 * 2  # 4 tiles x 2 selected channels
    assert [patch.c for patch in result.patches[:2]] == [1, 3]  # in channel order
    assert [patch.patch_id for patch in result.patches[:2]] == [
        "single/single/single/img/c1/grid-y0-x0-h8-w8",
        "single/single/single/img/c3/grid-y0-x0-h8-w8",
    ]
    assert {patch.position_id for patch in result.patches} == {"img"}  # D85: the image name; the labels lead patch_id
    for patch, array in zip(result.patches, result.patch_arrays):
        assert array.shape == (8, 8) and np.all(array == patch.c + 1)
    assert [record.n_patches for record in result.image_records] == [8]


def test_selecting_every_channel_gives_the_patches_of_true():
    image = _channel_image()
    metadata = [{"image_file": "img.tif"}]
    every = run_patching(images=[image], config=_grid_config(), image_metadata=metadata)
    listed = run_patching(images=[image], config=_grid_config(split_channels=[0, 1, 2]), image_metadata=metadata)

    assert [patch.to_dict() for patch in listed.patches] == [patch.to_dict() for patch in every.patches]
    for left, right in zip(listed.patch_arrays, every.patch_arrays):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("selection", [[], [-1], [0, 0], ["0"], [True], 1])
def test_a_channel_selection_must_list_distinct_channel_indices(selection):
    with pytest.raises(ValueError, match="split_channels"):
        ExtractionConfig(split_channels=selection)


def test_numpy_channel_indices_are_accepted():
    config = ExtractionConfig(split_channels=[np.int64(2), np.int32(0)])
    assert config.split_channel_indices(3) == (0, 2)


def test_a_channel_the_images_lack_is_refused_before_patching(tmp_path):
    config = _grid_config(split_channels=[1, 3])
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    images = [_channel_image(n_channels=4), _channel_image(n_channels=3)]
    with pytest.raises(ValueError, match=r"Image 1 of the run: .*selects channel 3, but the image has 3 channels"):
        run_patching(images=images, config=config, image_metadata=named(images))
    assert not (tmp_path / "out").exists()


def test_a_channel_selection_set_after_construction_is_checked_too():
    config = _grid_config()
    config.extraction.split_channels = [5]
    with pytest.raises(ValueError, match="selects channel 5, but the image has 3 channels"):
        run_patching(images=[_channel_image()], config=config, image_metadata=named([_channel_image()]))


def test_a_folder_run_with_a_missing_channel_writes_nothing(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("a.npy", "b.npy"):
        write_array(folder / name, _channel_image(n_channels=2))
    config = _grid_config(split_channels=[2])
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    with pytest.raises(ValueError, match="has 2 channels"):
        run_patching_pipeline(folder, config=config)
    assert not (tmp_path / "out" / "patches.csv").exists()


def test_a_channel_selection_is_exported_and_reloaded(tmp_path):
    config = _grid_config(split_channels=[2])
    config.export.enabled, config.export.image_format = True, "npy"
    config.common.output_dir = str(tmp_path)
    run_patching(images=[_channel_image()], config=config, image_metadata=[{"image_file": "img0.tif"}])

    with (tmp_path / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and {row["c"] for row in rows} == {"2"}
    assert all(np.all(np.load(tmp_path / row["patch_file"]) == 3) for row in rows)
    assert load_config(tmp_path / "run_config.json").extraction.split_channels == [2]


def test_split_channels_with_cell_mode_and_apply_mask():
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[4:10, 4:10] = 1
    image = _channel_image(n_channels=2)
    config = CellPatcherConfig(
        mode="cell",
        common=CommonConfig(image_axes="CYX", mask_axes="YX", extract_arrays=True),
        extraction=ExtractionConfig(split_channels=True, apply_mask=True, mask_fill_value=0),
    )
    result = run_patching(images=[image], masks=[labels], config=config, image_metadata=named([image]))

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
    assert {row["position_id"] for row in rows} == {"img0"}
    assert all((tmp_path / "out" / row["patch_file"]).exists() for row in rows)


def test_same_file_name_in_two_wells_gives_two_positions(tmp_path):
    # The default position_id holds the technical repeat, so equal file names
    # in two wells of one biological repeat stay two positions (decision D02).
    image_root = tmp_path / "raw_data"
    for well in ("B02", "B03"):
        write_array(image_root / "control" / "2026_01_01" / well / "img0.npy", _channel_image(n_channels=2))

    config = _grid_config()
    config.export.enabled = True
    config.export.image_format = "npy"
    config.common.output_dir = str(tmp_path / "out")
    run_patching_condition_repeat_tree(image_root, config=config, combine_metadata=True)

    with (tmp_path / "out" / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # D85: one position_id (the image name); the wells are the technical_repeat field.
    assert {(row["technical_repeat"], row["position_id"]) for row in rows} == {("B02", "img0"), ("B03", "img0")}
    with (tmp_path / "out" / "images.csv").open("r", encoding="utf-8") as f:
        image_rows = list(csv.DictReader(f))
    assert sorted((row["technical_repeat"], row["position_id"]) for row in image_rows) == [
        ("B02", "img0"), ("B03", "img0"),
    ]


def test_default_position_id_takes_a_per_image_technical_repeat():
    result = run_patching(
        images=[_channel_image(), _channel_image()],
        config=_grid_config(),
        image_metadata=[
            {"image_file": "img0.tif", "technical_repeat": "wellA"},
            {"image_file": "img0.tif", "technical_repeat": "wellB"},
        ],
    )

    assert {patch.position_id for patch in result.patches} == {"img0"}
    assert [(record.technical_repeat, record.position_id) for record in result.image_records] == [
        ("wellA", "img0"), ("wellB", "img0"),
    ]


def test_channel_names_come_from_the_ome_header(tmp_path):
    pytest.importorskip("morphobricks_preprocessing")
    import tifffile

    from cellpatcher.io import channel_names

    tifffile.imwrite(tmp_path / "a.ome.tif", _channel_image(), ome=True,
                     metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "GFP", "Actin"]}})
    write_array(tmp_path / "b.npy", _channel_image())
    assert channel_names(tmp_path / "a.ome.tif") == ("DAPI", "GFP", "Actin")
    assert channel_names(tmp_path / "b.npy") == ()  # no header to read names from
