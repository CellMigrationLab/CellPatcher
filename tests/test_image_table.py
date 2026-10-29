"""The per-image table ``images.csv`` written beside every ``patches.csv``.

It lists every image a run considered, including those that gave no patch,
with the identity columns of ``patches.csv`` so the two tables join.
"""

import csv
import importlib.util
import warnings

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, ExportConfig, GridConfig, NormalizationConfig
from cellpatcher.export import IMAGE_TABLE_FILENAME, export_patches
from cellpatcher.io import write_array
from cellpatcher.orchestration import run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.pipeline import run_patching
from cellpatcher.types import IMAGE_TABLE_COLUMNS, ImageRecord, PatchRegion

from tests.helpers import make_synthetic_cell_image_and_labels
from tests.helpers import named

# image_id tells images apart in an in-memory run, which has no file names.
IDENTITY = ("condition", "biological_repeat", "technical_repeat", "image_id", "image_file")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _assert_tables_join(folder):
    """Every patches.csv row's image is in images.csv, which counts its rows."""
    _, patch_rows = _read_csv(folder / "patches.csv")
    header, image_rows = _read_csv(folder / IMAGE_TABLE_FILENAME)
    assert header == list(IMAGE_TABLE_COLUMNS)
    images = {tuple(row[column] for column in IDENTITY): row for row in image_rows}
    assert len(images) == len(image_rows), "the identity columns must name one image each"
    counts = {}
    for row in patch_rows:
        key = tuple(row[column] for column in IDENTITY)
        assert key in images, f"{row['patch_file']} comes from an image missing in images.csv: {key}"
        assert row["image_height"] == images[key]["image_height"]
        assert row["image_width"] == images[key]["image_width"]
        assert row["position_id"] == images[key]["position_id"]
        counts[key] = counts.get(key, 0) + 1
    for key, image_row in images.items():
        assert int(image_row["n_patches"]) == counts.get(key, 0)
        if counts.get(key):
            assert (image_row["status"], image_row["reason"]) == ("processed", "")
        else:
            assert image_row["status"] in ("no_patches", "skipped") and image_row["reason"]
    return patch_rows, image_rows


def _cell_folders(tmp_path, *, empty=("b",)):
    """An image folder with images a, b, c and their masks; the masks of *empty* hold no cell."""
    image, labels, _ = make_synthetic_cell_image_and_labels()
    for name in ("a", "b", "c"):
        write_array(tmp_path / "images" / f"{name}.tif", image)
        write_array(tmp_path / "masks" / f"{name}_mask.tif", np.zeros_like(labels) if name in empty else labels)
    return tmp_path / "images", tmp_path / "masks"


def _cell_config(output_dir, **common):
    config = CellPatcherConfig(mode="cell", common=CommonConfig(output_dir=str(output_dir), **common))
    config.cell.patch_size = 24
    config.export = ExportConfig(enabled=True, image_format="npy")
    return config


# ---------------------------------------------------------------------------
# Folder run (export entry point)
# ---------------------------------------------------------------------------


def test_folder_run_lists_images_with_and_without_patches(tmp_path):
    images, masks = _cell_folders(tmp_path)
    config = _cell_config(tmp_path / "out", condition="treated", biological_repeat="2026_01_01")
    result = run_patching_pipeline(images, mask_folder_path=masks, config=config)

    out = tmp_path / "out"
    assert result.image_table_file == str(out / IMAGE_TABLE_FILENAME)
    patch_rows, image_rows = _assert_tables_join(out)
    assert len(patch_rows) == len(result.patches) == 8
    assert [row["image_file"] for row in image_rows] == ["a.tif", "b.tif", "c.tif"]
    by_file = {row["image_file"]: row for row in image_rows}

    assert by_file["a.tif"] == {
        "condition": "treated", "biological_repeat": "2026_01_01", "technical_repeat": "single",
        "image_id": "image_0000", "image_file": "a.tif", "mask_file": "a_mask.tif",
        "source_relative_path": "a.tif", "image_height": "96", "image_width": "128",
        "n_patches": "4", "status": "processed", "reason": "",
        "position_id": "a", "n_regions": "4", "n_kept": "4", "n_normalized": "4",
    }
    # The image whose mask holds no cell is listed, not dropped.
    assert by_file["b.tif"]["n_patches"] == "0"
    assert by_file["b.tif"]["status"] == "no_patches"
    assert by_file["b.tif"]["reason"] == "the patch mode generated no region in this image"
    assert (by_file["b.tif"]["image_height"], by_file["b.tif"]["mask_file"]) == ("96", "b_mask.tif")
    assert [(r.image_file, r.status) for r in result.image_records] == [
        ("a.tif", "processed"), ("b.tif", "no_patches"), ("c.tif", "processed"),
    ]


def test_every_exported_patch_file_belongs_to_a_listed_image(tmp_path):
    images, masks = _cell_folders(tmp_path)
    run_patching_pipeline(images, mask_folder_path=masks, config=_cell_config(tmp_path / "out"))
    patch_rows, image_rows = _assert_tables_join(tmp_path / "out")
    listed = {row["image_file"] for row in image_rows}
    for row in patch_rows:
        assert (tmp_path / "out" / row["patch_file"]).is_file()
        assert row["image_file"] in listed


def test_a_run_where_no_image_gives_a_patch_still_lists_every_image(tmp_path):
    images, masks = _cell_folders(tmp_path, empty=("a", "b", "c"))
    result = run_patching_pipeline(images, mask_folder_path=masks, config=_cell_config(tmp_path / "out"))

    assert result.patches == []
    patch_rows, image_rows = _assert_tables_join(tmp_path / "out")
    assert patch_rows == []
    assert [(row["image_file"], row["status"]) for row in image_rows] == [
        ("a.tif", "no_patches"), ("b.tif", "no_patches"), ("c.tif", "no_patches"),
    ]


def test_filters_that_remove_every_region_are_the_reason(tmp_path):
    bright = np.full((32, 32), 100, np.uint16)
    dark = np.zeros((32, 32), np.uint16)
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(output_dir=str(tmp_path)),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True, image_format="npy"),
    )
    config.filters.min_mean_intensity = 50
    result = run_patching(images=[bright, dark], config=config, image_metadata=named([bright, dark]))

    _, image_rows = _assert_tables_join(tmp_path)
    assert [(row["image_id"], row["n_patches"], row["status"]) for row in image_rows] == [
        ("image_0000", "4", "processed"), ("image_0001", "0", "no_patches"),
    ]
    assert image_rows[1]["reason"] == "the filters removed all 4 region(s)"
    assert [(row["n_regions"], row["n_kept"], row["n_normalized"]) for row in image_rows] == [
        ("4", "4", "4"), ("4", "0", "0"),
    ]
    # A direct run_patching call has no labels or file names: "single" labels, empty file columns.
    assert (image_rows[1]["condition"], image_rows[1]["image_file"]) == ("single", "")
    assert result.image_records[1].n_regions == 4 and result.image_records[1].n_kept == 0


def test_split_channels_count_one_row_per_channel_patch(tmp_path):
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="CYX", output_dir=str(tmp_path)),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True, image_format="npy"),
    )
    config.extraction.split_channels = True
    run_patching(images=[np.ones((3, 32, 16), np.uint8)], config=config, image_metadata=[{"image_file": "img.tif"}])

    patch_rows, image_rows = _assert_tables_join(tmp_path)
    assert len(patch_rows) == 6
    assert [(row["n_patches"], row["image_height"], row["image_width"]) for row in image_rows] == [("6", "32", "16")]
    # Regions are counted before the split, patches after it.
    assert (image_rows[0]["n_regions"], image_rows[0]["position_id"]) == ("2", "img")


def test_the_supplied_position_id_is_listed_also_for_images_without_patches(tmp_path):
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(output_dir=str(tmp_path)),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True, image_format="npy"),
    )
    config.filters.min_mean_intensity = 50
    images = [np.full((32, 32), 100, np.uint16), np.zeros((32, 32), np.uint16), np.zeros((32, 32), np.uint16)]
    metadata = [{"position_id": "P1"}, {"position_id": "P2"}, {"image_file": "c.tif"}]
    result = run_patching(images=images, config=config, image_metadata=metadata)

    patch_rows, image_rows = _assert_tables_join(tmp_path)
    assert {row["position_id"] for row in patch_rows} == {"P1"}
    # Supplied, or from the file name (one file is one position), also without patches.
    assert [(row["position_id"], row["status"]) for row in image_rows] == [
        ("P1", "processed"), ("P2", "no_patches"), ("c", "no_patches"),
    ]
    assert [record.position_id for record in result.image_records] == ["P1", "P2", "c"]

    # An in-memory image with neither is refused, not named by its place in the run.
    with pytest.raises(ValueError, match="Image 2 of the run has neither a position_id nor an image_file"):
        run_patching(images=images, config=config, image_metadata=[*metadata[:2], {}])


def test_an_image_without_planes_is_skipped_with_a_reason(tmp_path):
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="ZYX", z_policy="all", output_dir=str(tmp_path)),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True, image_format="npy"),
    )
    stack = np.ones((2, 32, 32), np.uint8)
    empty = np.ones((0, 32, 32), np.uint8)
    result = run_patching(images=[stack, empty], config=config, image_metadata=named([stack, empty]))

    _, image_rows = _assert_tables_join(tmp_path)
    assert [(row["image_id"], row["n_patches"], row["status"]) for row in image_rows] == [
        ("image_0000", "8", "processed"), ("image_0001", "0", "skipped"),
    ]
    assert "no plane to patch" in image_rows[1]["reason"]
    assert [(row["n_regions"], row["n_kept"], row["n_normalized"]) for row in image_rows] == [
        ("8", "8", "8"), ("0", "0", "0"),
    ]
    # The planes of one image make one row, with the image's size.
    assert (image_rows[0]["image_height"], image_rows[0]["image_width"]) == ("32", "32")
    assert len(result.image_records) == 2


def test_no_table_without_the_metadata_csv(tmp_path):
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(output_dir=str(tmp_path)),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True, image_format="npy", save_metadata_csv=False),
    )
    result = run_patching(images=[np.ones((32, 32), np.uint8)], config=config,
        image_metadata=named([np.ones((32, 32), np.uint8)]))
    assert not (tmp_path / IMAGE_TABLE_FILENAME).exists() and not (tmp_path / "patches.csv").exists()
    assert result.image_table_file is None
    # The records are still returned in memory.
    assert [record.status for record in result.image_records] == ["processed"]


def test_in_memory_run_returns_records_without_writing_files(tmp_path):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=16))
    result = run_patching(images=[np.ones((32, 32), np.uint8), np.ones((8, 8), np.uint8)], config=config,
        image_metadata=named([np.ones((32, 32), np.uint8), np.ones((8, 8), np.uint8)]))
    assert [(r.image_id, r.n_patches, r.status) for r in result.image_records] == [
        ("image_0000", 4, "processed"), ("image_0001", 0, "no_patches"),
    ]
    assert result.image_table_file is None


# ---------------------------------------------------------------------------
# export_patches called directly
# ---------------------------------------------------------------------------


def test_export_patches_writes_the_table_beside_patches_csv(tmp_path):
    patches = [
        PatchRegion(
            patch_id=f"p{i}", image_id=f"image_000{i % 2}", y1=4, x1=4, condition="c", image_file=f"{i % 2}.tif"
        )
        for i in range(3)
    ]
    export_patches(patches, [np.zeros((4, 4))] * 3, tmp_path, image_format="npy")

    patch_rows, image_rows = _assert_tables_join(tmp_path)
    assert len(patch_rows) == 3
    assert [(row["image_file"], row["n_patches"], row["status"]) for row in image_rows] == [
        ("0.tif", "2", "processed"), ("1.tif", "1", "processed"),
    ]


def test_export_patches_writes_the_records_it_is_given(tmp_path):
    patches = [PatchRegion(patch_id="p0", image_id="image_0000", y1=4, x1=4, image_file="a.tif")]
    records = [
        ImageRecord(image_id="image_0000", image_file="a.tif", n_regions=1, n_kept=1, n_normalized=1, n_patches=1),
        ImageRecord(image_id="image_0001", image_file="b.tif"),
    ]
    export_patches(patches, [np.zeros((4, 4))], tmp_path, image_format="npy", image_records=records)

    _, image_rows = _assert_tables_join(tmp_path)
    assert [(row["image_file"], row["status"]) for row in image_rows] == [
        ("a.tif", "processed"), ("b.tif", "no_patches"),
    ]


# ---------------------------------------------------------------------------
# Tree run
# ---------------------------------------------------------------------------


def _tree(tmp_path):
    image, labels, _ = make_synthetic_cell_image_and_labels()
    empty = np.zeros_like(labels)
    layout = {
        "treated/2026_01_01/a.tif": labels,
        "treated/2026_01_01/b.tif": empty,
        "control/2026_01_01/B02/c.tif": labels,
        "control/2026_01_01/B03/c.tif": empty,
    }
    for relative, mask in layout.items():
        write_array(tmp_path / "raw" / relative, image)
        mask_path = (tmp_path / "masks" / relative).with_name(f"{relative.rsplit('/', 1)[1][:-4]}_mask.tif")
        write_array(mask_path, mask)
    return tmp_path / "raw", tmp_path / "masks"


EXPECTED_TREE_ROWS = [
    ("control", "2026_01_01", "B02", "c.tif", "control/2026_01_01/B02/c.tif", "4", "processed"),
    ("control", "2026_01_01", "B03", "c.tif", "control/2026_01_01/B03/c.tif", "0", "no_patches"),
    ("treated", "2026_01_01", "single", "a.tif", "treated/2026_01_01/a.tif", "4", "processed"),
    ("treated", "2026_01_01", "single", "b.tif", "treated/2026_01_01/b.tif", "0", "no_patches"),
]


def _summary(rows):
    return [
        (r["condition"], r["biological_repeat"], r["technical_repeat"], r["image_file"], r["source_relative_path"],
         r["n_patches"], r["status"])
        for r in rows
    ]


def test_tree_run_writes_a_table_in_every_leaf_folder(tmp_path):
    raw, masks = _tree(tmp_path)
    results = run_patching_condition_repeat_tree(raw, mask_root_path=masks, config=_cell_config(tmp_path / "out"))

    rows = []
    for leaf in results:
        folder = tmp_path / "out" / leaf.relative_folder
        assert leaf.result.image_table_file == str(folder / IMAGE_TABLE_FILENAME)
        rows.extend(_assert_tables_join(folder)[1])
    assert _summary(rows) == EXPECTED_TREE_ROWS


def test_combined_tree_run_writes_one_table_for_every_image(tmp_path):
    raw, masks = _tree(tmp_path)
    results = run_patching_condition_repeat_tree(
        raw, mask_root_path=masks, config=_cell_config(tmp_path / "out"), combine_metadata=True
    )

    out = tmp_path / "out"
    patch_rows, image_rows = _assert_tables_join(out)
    assert _summary(image_rows) == EXPECTED_TREE_ROWS
    assert {leaf.result.image_table_file for leaf in results} == {str(out / IMAGE_TABLE_FILENAME)}
    # No per-leaf tables, as for patches.csv.
    assert sorted(path.relative_to(out).as_posix() for path in out.rglob("*.csv")) == ["images.csv", "patches.csv"]
    for row in patch_rows:
        assert (out / row["patch_file"]).is_file()


# ---------------------------------------------------------------------------
# Normalization discards (decision D8)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs the 'preprocessing' extra"
)
def test_an_image_whose_patches_are_all_discarded_by_normalization(tmp_path):
    varied = np.arange(256, dtype=np.uint8).reshape(16, 16)
    constant = np.full((16, 16), 7, np.uint8)
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="YX", output_dir=str(tmp_path)),
        grid=GridConfig(tile_size=8),
        export=ExportConfig(enabled=True, image_format="npy"),
        normalization=NormalizationConfig(enabled=True, method="minmax", scope="patch"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_patching(images=[varied, constant], config=config, image_metadata=named([varied, constant]))

    _, image_rows = _assert_tables_join(tmp_path)
    assert [(row["n_patches"], row["status"]) for row in image_rows] == [("4", "processed"), ("0", "no_patches")]
    assert image_rows[1]["reason"].startswith("normalization discarded all 4 patch(es)")
    assert (image_rows[1]["n_regions"], image_rows[1]["n_kept"], image_rows[1]["n_normalized"]) == ("4", "4", "0")
