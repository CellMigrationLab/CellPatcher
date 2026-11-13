import csv

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, ExtractionConfig, GridConfig
from cellpatcher.io import write_array
from cellpatcher.orchestration import load_config
from cellpatcher.pipeline import run_patching
from tests.helpers import named, preprocessed_folder, preprocessed_tree, write_tiff


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
    with pytest.raises(ValueError, match="split_channels must be True"):
        ExtractionConfig(split_channels="yes")


# ---------------------------------------------------------------------------
# Channel selection: by canonical channel ID only, never by position (owner decision of 13 November 2026)
# ---------------------------------------------------------------------------

_IDS = ("DNA", "ACTIN", "TUB", "MITO")


def _named(images, name="img.tif"):
    """In-memory images' metadata naming each C index by a channel ID (``image_metadata["channel_ids"]``)."""
    return [{"image_file": name, "channel_ids": _IDS[: image.shape[0]]} for image in images]


def test_a_channel_selection_by_id_gives_one_patch_per_named_channel():
    image = _channel_image(n_channels=4)
    result = run_patching(images=[image], config=_grid_config(split_channels=["MITO", "ACTIN"]),
                          image_metadata=_named([image]))

    assert len(result.patches) == 4 * 2  # 4 tiles x 2 selected channels
    assert [(patch.c, patch.channel_id) for patch in result.patches[:2]] == [(1, "ACTIN"), (3, "MITO")]  # C order
    assert [patch.patch_id for patch in result.patches[:2]] == [
        "single/single/single/img/c1/grid-y0-x0-h8-w8",
        "single/single/single/img/c3/grid-y0-x0-h8-w8",
    ]
    assert {patch.position_id for patch in result.patches} == {"img"}  # D85: the image name; the labels lead patch_id
    for patch, array in zip(result.patches, result.patch_arrays, strict=True):
        assert array.shape == (8, 8) and np.all(array == patch.c + 1)
    assert [record.n_patches for record in result.image_records] == [8]


def test_one_id_is_the_c_index_it_has_in_each_image():
    first, second = _channel_image(n_channels=2), _channel_image(n_channels=2)
    metadata = [{"image_file": "a.tif", "channel_ids": ("DNA", "ACTIN")},
                {"image_file": "b.tif", "channel_ids": ("ACTIN", "DNA")}]
    config = _grid_config(split_channels=["ACTIN"])
    result = run_patching(images=[first, second], config=config, image_metadata=metadata)
    assert {(patch.position_id, patch.c, patch.channel_id) for patch in result.patches} == {
        ("a", 1, "ACTIN"), ("b", 0, "ACTIN")
    }


def test_selecting_every_channel_id_gives_the_patches_of_true():
    image = _channel_image()
    every = run_patching(images=[image], config=_grid_config(), image_metadata=_named([image]))
    listed = run_patching(images=[image], config=_grid_config(split_channels=["TUB", "DNA", "ACTIN"]),
                          image_metadata=_named([image]))

    assert [patch.to_dict() for patch in listed.patches] == [patch.to_dict() for patch in every.patches]
    for left, right in zip(listed.patch_arrays, every.patch_arrays, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("selection", [[], [0], [1, 3], [np.int64(2)], ["DNA", 1], [""], ["DNA", "DNA"], [True], 1,
                                       "DNA", ["my channel"]])
def test_a_channel_selection_is_true_false_or_distinct_channel_ids(selection):
    with pytest.raises(ValueError, match="split_channels"):
        ExtractionConfig(split_channels=selection)


@pytest.mark.parametrize("ids", [("DNA", "DNA", "TUB"), ("DNA", "my channel", "TUB"), ("DNA", 1, "TUB")])
def test_an_images_channel_ids_must_be_distinct_canonical_ids(ids):
    # Review of #109: ("DNA", "DNA") silently named C0 for DNA. Each image's IDs are checked with the
    # normalization brick's check_channel_ids, as the patch normalization does.
    image = _channel_image()
    with pytest.raises(ValueError, match=r'Image 0 of the run: image_metadata\["channel_ids"\]'):
        run_patching(images=[image], config=_grid_config(split_channels=["DNA"]),
                     image_metadata=[{"image_file": "img.tif", "channel_ids": ids}])


def test_metadata_of_another_length_is_named_before_the_channel_check():
    images = [_channel_image(), _channel_image()]
    with pytest.raises(ValueError, match="image_metadata must have exactly one entry per image"):
        run_patching(images=images, config=_grid_config(split_channels=["DNA"]), image_metadata=_named(images[:1]))


def test_an_in_memory_image_without_channel_ids_has_no_channel_to_name():
    image = _channel_image()
    with pytest.raises(ValueError, match=r"Image 0 of the run: .*names channel IDs \(DNA\).*image_metadata"):
        run_patching(images=[image], config=_grid_config(split_channels=["DNA"]), image_metadata=named([image]))


def test_a_channel_the_images_lack_is_refused_before_patching(tmp_path):
    config = _grid_config(split_channels=["ACTIN", "MITO"])
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    images = [_channel_image(n_channels=4), _channel_image(n_channels=3)]
    metadata = [{**meta, "image_file": f"img{index}.tif"} for index, meta in enumerate(_named(images))]
    with pytest.raises(ValueError, match=r"Image 1 of the run: .*names MITO, which the image .* does not have"):
        run_patching(images=images, config=config, image_metadata=metadata)
    assert not (tmp_path / "out").exists()


def test_a_channel_selection_set_after_construction_is_checked_too():
    config = _grid_config()
    config.extraction.split_channels = ["NUCLEUS"]
    with pytest.raises(ValueError, match="names NUCLEUS, which the image"):
        run_patching(images=[_channel_image()], config=config, image_metadata=_named([_channel_image()]))


def test_a_folder_run_naming_a_channel_no_image_has_writes_nothing(tmp_path):
    from cellpatcher.statistics import PreprocessingCheckError

    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("a.tif", "b.tif"):
        write_tiff(folder / name, _channel_image(n_channels=2), "CYX")
    config = _grid_config("YX", split_channels=["C2"])  # the test description names the channels C0, C1
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    with pytest.raises(PreprocessingCheckError, match="C2") as raised:
        preprocessed_folder(folder, config=config, axes="CYX")
    assert [issue.code for issue in raised.value.issues] == ["A_SCOPE_INCOMPLETE"]
    assert not (tmp_path / "out").exists()


def test_a_folder_run_selects_channels_by_the_ids_of_its_description(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    write_tiff(folder / "a.tif", _channel_image(n_channels=2), "CYX")
    config = _grid_config("YX", split_channels=["C1"])
    result = preprocessed_folder(folder, config=config, axes="CYX")
    assert {(patch.c, patch.channel_id) for patch in result.patches} == {(1, "C1")}
    assert all(np.all(array == 2) for array in result.patch_arrays)


def test_a_channel_selection_is_exported_and_reloaded(tmp_path):
    config = _grid_config(split_channels=["TUB"])
    config.export.enabled, config.export.image_format = True, "npy"
    config.common.output_dir = str(tmp_path)
    run_patching(images=[_channel_image()], config=config, image_metadata=_named([_channel_image()], "img0.tif"))

    with (tmp_path / "patches.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and {(row["c"], row["channel_id"]) for row in rows} == {("2", "TUB")}
    assert all(np.all(np.load(tmp_path / row["patch_file"]) == 3) for row in rows)
    assert load_config(tmp_path / "run_config.json").extraction.split_channels == ["TUB"]


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
        write_tiff(image_root / condition / "img0.tif", _channel_image(n_channels=2), "CYX")

    config = _grid_config("YX")
    config.export.enabled = True
    config.export.image_format = "npy"
    config.common.output_dir = str(tmp_path / "out")
    preprocessed_tree(image_root, config=config, axes="CYX", combine_metadata=True)

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
        write_tiff(image_root / "control" / "2026_01_01" / well / "img0.tif", _channel_image(n_channels=2), "CYX")

    config = _grid_config("YX")
    config.export.enabled = True
    config.export.image_format = "npy"
    config.common.output_dir = str(tmp_path / "out")
    preprocessed_tree(image_root, config=config, axes="CYX", combine_metadata=True)

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
