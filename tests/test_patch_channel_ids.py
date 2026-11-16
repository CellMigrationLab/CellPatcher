"""Every patch of a described run records its channel IDs in its C order (issue #125).

Found by the S3.4 adversarial suite (MorphoFeature draft PR #240, ``adversarial/``): without
normalization and without channel splitting, a patch recorded no channel ID at all (``c``,
``channel_id`` and ``normalization_channels`` empty), so two units storing DNA, ACTIN and ACTIN, DNA
gave patches nobody could tell apart without assuming the channel order (Spec S3.3, gate G3).

``PatchRegion.channel_ids`` (the ``channel_ids`` column of ``patches.csv``, ``;``-separated) is the
ID of every channel plane the patch holds, in its C order, from the run's description
(``Fov.channel_ids``, the order the resolver's selections index), whatever the normalization and
splitting settings. In memory it comes from ``image_metadata["channel_ids"]``; without them it is
empty, as before.

The pixels follow the G4 suite's rule (``adversarial/generate.py``): a channel's plane depends on
(image index, channel ID) and never on its C index, and every ID has its own intensity range, so a
patch whose IDs were in the wrong order fails the pixel check, not only the label check.
"""

import csv
import zlib
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig
from cellpatcher.export import read_patch_table
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.pipeline import run_patching

VALID = Path(__file__).parent / "spec_fixtures" / "fixtures" / "valid"
SIZE = 16
BASE = {"DNA": 1000, "ACTIN": 5000}
"""The lowest value of each channel ID; with :data:`SPREAD` the two ranges are disjoint."""
SPREAD = 2000
ORDERS = {"CTRL": ("DNA", "ACTIN"), "DrugA": ("ACTIN", "DNA")}
"""Spec fixture v04: each condition's one unit stores the same two channels in its own order."""


def channel_plane(index: int, channel_id: str) -> np.ndarray:
    """Image *index*'s plane of *channel_id*: the same wherever the channel is stored (G4 rule)."""
    rng = np.random.default_rng(zlib.crc32(f"{index}/{channel_id}".encode()))
    return (BASE[channel_id] + rng.integers(0, SPREAD, (SIZE, SIZE))).astype(np.uint16)


def _index(path: str) -> int:
    return int(Path(path).stem.removeprefix("img"))


def _v04_run(tmp_path):
    """Spec v04 with G4 pixels: CTRL stores DNA, ACTIN, DrugA ACTIN, DNA; one plane per (image, ID)."""
    from morphobricks_preprocessing.axes import prefill_mapping
    from morphobricks_preprocessing.testing_described import (
        description_of,
        publish_and_run,
        write_described_tree,
    )

    (fixture,) = VALID.glob("v04_*.toml")
    parsed, data = description_of(fixture.read_bytes())
    root = tmp_path / "run"
    images = write_described_tree(root / "experiment", None, parsed)
    for relative in images:
        order = ORDERS[relative.split("/")[0]]
        pixels = np.stack([channel_plane(_index(relative), channel) for channel in order])
        tifffile.imwrite(root / "experiment" / relative, pixels, photometric="minisblack", metadata={"axes": "CYX"})
    return publish_and_run(root, root / "experiment", None, prefill_mapping("CYX"), data)


def _config(run, output, *, split_channels=False, normalization=None):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.preprocessing_run, config.common.output_dir = str(run.output), str(output)
    config.common.extract_arrays = True
    config.export.enabled = True
    config.export.image_format = "npy"
    config.extraction.split_channels = split_channels
    if normalization is not None:
        config.normalization = normalization
    return config


def _patches(run, config):
    results = run_patching_condition_repeat_tree(run.experiment, config=config, combine_metadata=True)
    return [
        (patch, array) for leaf in results
        for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays, strict=True)
    ]


def _rows(output):
    with (output / "patches.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check_pixels_by_id(patch, array):
    """Each channel plane of *array* is the generator's plane of the ID the patch records for it."""
    array = array if array.ndim == 3 else array[np.newaxis]
    assert array.shape[0] == len(patch.channel_ids)
    for plane, channel in zip(array, patch.channel_ids, strict=True):
        assert BASE[channel] <= plane.min() and plane.max() < BASE[channel] + SPREAD, (patch.patch_id, channel)
        expected = channel_plane(_index(patch.source_relative_path), channel)[patch.y0:patch.y1, patch.x0:patch.x1]
        np.testing.assert_array_equal(plane, expected)


def test_v04_patches_without_normalization_or_splitting_record_their_own_channel_order(tmp_path):
    # The issue's case: normalization off, splitting off. The two orders give different, correct IDs on the
    # patch, in patches.csv and read back with read_patch_table; c, channel_id and normalization_channels stay
    # as documented (empty for a patch of every channel without normalization).
    run = _v04_run(tmp_path)
    output = tmp_path / "out"
    patches = _patches(run, _config(run, output))

    assert {patch.condition for patch, _ in patches} == {"CTRL", "DrugA"}
    assert len(patches) == 12 * 4
    for patch, array in patches:
        assert patch.channel_ids == ORDERS[patch.condition]
        assert (patch.c, patch.channel_id, patch.normalization_channels) == (None, None, None)
        _check_pixels_by_id(patch, array)

    rows = _rows(output)
    assert {(row["condition"], row["channel_ids"]) for row in rows} == {("CTRL", "DNA;ACTIN"), ("DrugA", "ACTIN;DNA")}
    assert {(row["c"], row["channel_id"]) for row in rows} == {("", "")}
    assert "normalization_channels" not in rows[0]  # no normalization: no normalization columns, as before
    read, _ = read_patch_table(output / "patches.csv")
    assert {(patch.condition, patch.channel_ids) for patch in read} == set(ORDERS.items())


def test_v04_split_patches_record_their_one_channel(tmp_path):
    run = _v04_run(tmp_path)
    output = tmp_path / "out"
    patches = _patches(run, _config(run, output, split_channels=True))

    assert len(patches) == 12 * 4 * 2
    for patch, array in patches:
        assert patch.channel_ids == (patch.channel_id,)
        assert patch.channel_id == ORDERS[patch.condition][patch.c]
        _check_pixels_by_id(patch, array)
    assert {row["channel_ids"] for row in _rows(output)} == {"DNA", "ACTIN"}


@pytest.mark.parametrize("split_channels", [False, True])
def test_v04_normalized_patches_record_the_same_order_as_normalization_channels(tmp_path, split_channels):
    run = _v04_run(tmp_path)
    output = tmp_path / "out"
    normalization = NormalizationConfig(enabled=True, method="minmax", scope="image")
    patches = _patches(run, _config(run, output, split_channels=split_channels, normalization=normalization))

    assert patches
    for patch, _ in patches:
        assert patch.channel_ids == patch.normalization_channels
        expected = ORDERS[patch.condition] if patch.c is None else (ORDERS[patch.condition][patch.c],)
        assert patch.channel_ids == expected
    for row in _rows(output):
        assert row["channel_ids"] == row["normalization_channels"]


@pytest.mark.parametrize("split_channels", [False, True])
def test_in_memory_images_given_channel_ids_record_them_and_images_without_record_none(split_channels):
    # In memory, two images of the two orders given their IDs record them as a file run does; an image
    # without IDs records none (nothing is guessed from the C index).
    images = [np.stack([channel_plane(0, channel) for channel in order]) for order in ORDERS.values()]
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = "CYX"
    config.extraction.split_channels = split_channels
    metadata = [
        {"position_id": f"p{i}", "channel_ids": order} for i, order in enumerate(ORDERS.values())
    ]
    result = run_patching(images=images, config=config, image_metadata=metadata)
    for patch in result.patches:
        order = metadata[int(patch.image_id.rsplit("_", 1)[1])]["channel_ids"]
        assert patch.channel_ids == (order if patch.c is None else (order[patch.c],))

    unnamed = run_patching(images=images, config=config, image_metadata=[{"position_id": f"p{i}"} for i in range(2)])
    assert unnamed.patches and {(patch.channel_id, patch.channel_ids) for patch in unnamed.patches} == {(None, None)}


def test_in_memory_single_plane_records_its_one_id_and_a_plane_of_several_ids_without_c_records_none():
    # An image without a C axis is one channel plane: its one ID is recorded; given the IDs of every
    # channel of its image but no "c", which one it is is not known, so none is recorded (never the first).
    plane = channel_plane(0, "DNA")
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    one = run_patching(images=[plane], config=config, image_metadata=[{"position_id": "p0", "channel_ids": ("DNA",)}])
    assert {patch.channel_ids for patch in one.patches} == {("DNA",)}
    named_c = run_patching(
        images=[plane], config=config, image_metadata=[{"position_id": "p0", "channel_ids": ("ACTIN", "DNA"), "c": 1}]
    )
    assert {(patch.c, patch.channel_id, patch.channel_ids) for patch in named_c.patches} == {(1, "DNA", ("DNA",))}
    several = run_patching(
        images=[plane], config=config, image_metadata=[{"position_id": "p0", "channel_ids": ("ACTIN", "DNA")}]
    )
    assert {patch.channel_ids for patch in several.patches} == {None}


@pytest.mark.parametrize("axes", ["CYX", "YXC"])
def test_in_memory_image_of_several_channels_given_c_is_refused_not_recorded_as_one_channel(axes):
    # Review of #127: an image with a C axis of two channels, given image_metadata["c"] without splitting, gave
    # patches holding both planes but recording channel_ids == (ids[c],). Its patches hold every channel, so a
    # "c" naming one of them is refused (as with extraction.split_channels), never recorded on them.
    image = np.stack([channel_plane(0, channel) for channel in ORDERS["CTRL"]], axis=0 if axes == "CYX" else -1)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = axes
    meta = {"position_id": "p0", "channel_ids": ORDERS["CTRL"]}
    with pytest.raises(ValueError, match=r'image_metadata\["c"\]=1 names one channel plane, but the image holds 2'):
        run_patching(images=[image], config=config, image_metadata=[{**meta, "c": 1}])
    result = run_patching(images=[image], config=config, image_metadata=[meta])
    assert result.patches and {patch.channel_ids for patch in result.patches} == {ORDERS["CTRL"]}


def test_a_table_without_channel_ids_is_refused_on_load_and_a_bad_cell_too(tmp_path):
    # A patches.csv written before issue #125 has no channel_ids column: read_patch_table refuses it like any
    # table missing a column CellPatcher writes (nothing is defaulted; run CellPatcher again). A cell that is not
    # canonical IDs, or names one twice, is refused with its line.
    from cellpatcher.export import PatchTableError

    run = _v04_run(tmp_path)
    output = tmp_path / "out"
    _patches(run, _config(run, output))
    table = output / "patches.csv"
    rows = _rows(output)

    def write(rows, columns):
        with table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    columns = list(rows[0])
    write(rows, [name for name in columns if name != "channel_ids"])
    with pytest.raises(PatchTableError, match=r"missing columns \['channel_ids'\]"):
        read_patch_table(table)
    for bad in ("DNA;DNA", "DNA;", "C 0"):
        write([{**rows[0], "channel_ids": bad}, *rows[1:]], columns)
        with pytest.raises(PatchTableError, match="line 2 .*channel_ids"):
            read_patch_table(table)
