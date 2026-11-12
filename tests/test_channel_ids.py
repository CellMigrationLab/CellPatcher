"""Channels by canonical ID on described runs (Stage 2 CP-3; Spec G2-id, G2-ref).

``extraction.split_channels`` may name channel IDs of the run's published description. A file
run resolves them **per image** through the run's resolver (``RunInputs.resolve_request``,
resolver contract v1), so one ID is whatever C index it has in each image. Every patch with a
``c`` gets the ``channel_id`` of that C index in its own image (``Fov.channel_ids``), in a column
appended to ``patches.csv``; ``patch_id`` is unchanged (it keeps ``c`` and is never parsed).

Runs are built with preprocessing's ``make_described_run`` from the vendored Spec fixtures.
"""

import csv
import dataclasses

import numpy as np
import pytest
import tifffile

from cellpatcher.config import CellPatcherConfig, GridConfig
from cellpatcher.orchestration import run_patching_condition_repeat_tree
from cellpatcher.pipeline import run_patching
from cellpatcher.statistics import PreprocessingCheckError
from tests.helpers import named
from tests.test_described_ids import _described_run


def _config(run, output, split_channels):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.preprocessing_run, config.common.output_dir = str(run.output), str(output)
    config.common.extract_arrays = True
    config.export.enabled = True
    config.extraction.split_channels = split_channels
    return config


def _tree(run, output, split_channels):
    """Patch *run*; return ``(rows of patches.csv, {patch_id: array})``."""
    results = run_patching_condition_repeat_tree(
        run.experiment, config=_config(run, output, split_channels), combine_metadata=True
    )
    arrays = {
        patch.patch_id: array for leaf in results
        for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays, strict=True)
    }
    with (output / "patches.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), arrays


def test_v13_channels_by_id_are_the_channels_by_index(tmp_path):
    # v13 declares DNA, ACTIN in C order for every unit: ACTIN by ID is C index 1 by index, the same
    # patches and patch_id bytes; only the appended channel_id column says which channel it is.
    run = _described_run(tmp_path, "v13")
    by_id, id_arrays = _tree(run, tmp_path / "by_id", ["ACTIN"])
    by_index, index_arrays = _tree(run, tmp_path / "by_index", [1])

    assert by_id and {(row["c"], row["channel_id"]) for row in by_id} == {("1", "ACTIN")}
    assert {row["channel_id"] for row in by_index} == {"ACTIN"}  # an index selection is named too
    assert [row["patch_id"] for row in by_id] == [row["patch_id"] for row in by_index]
    assert all(row["patch_id"].startswith("/".join(
        (row["condition"], row["biological_repeat"], row["technical_repeat"], row["position_id"], "c1/")
    )) for row in by_id)  # the unchanged FOV identity leads it
    for patch_id, array in id_arrays.items():
        np.testing.assert_array_equal(array, index_arrays[patch_id])

    every, _ = _tree(run, tmp_path / "every", True)
    assert {(row["c"], row["channel_id"]) for row in every} == {("0", "DNA"), ("1", "ACTIN")}


def test_a_multichannel_patch_has_no_channel_id(tmp_path):
    run = _described_run(tmp_path, "v13")
    rows, _ = _tree(run, tmp_path / "out", False)
    assert rows and {(row["c"], row["channel_id"]) for row in rows} == {("", "")}


@pytest.mark.parametrize("ids", [["NUCLEUS"], ["DNA", "NUCLEUS"]])
def test_an_unknown_channel_id_is_refused_by_the_resolver_before_anything_is_written(tmp_path, ids):
    # Resolver contract v1: an ID no image has is present in 0 of n acquisitions (A_SCOPE_INCOMPLETE);
    # nothing is selected, not even the known ID.
    run = _described_run(tmp_path, "v13")
    with pytest.raises(PreprocessingCheckError) as raised:
        _tree(run, tmp_path / "out", ids)
    assert [issue.code for issue in raised.value.issues] == ["A_SCOPE_INCOMPLETE"]
    assert "'NUCLEUS' is present in 0 of 90 acquisitions" in str(raised.value)
    assert not (tmp_path / "out").exists()


def test_v04_swapped_channel_order_is_patched_by_id(tmp_path):
    # v04: CTRL stores DNA, ACTIN and DrugA ACTIN, DNA. Since preprocessing v0.5.2 (Stage 3 S3.3) the run
    # exists: DNA by ID is C0 in CTRL and C1 in DrugA, and every DNA patch is cut from its own image's DNA plane.
    run = _described_run(tmp_path, "v04")
    rows, arrays = _tree(run, tmp_path / "out", ["DNA"])

    assert {row["channel_id"] for row in rows} == {"DNA"}
    assert {(row["condition"], row["c"]) for row in rows} == {("CTRL", "0"), ("DrugA", "1")}
    for row in rows:
        image = tifffile.imread(run.experiment / row["source_relative_path"])  # C, Y, X
        plane = image[int(row["c"])][int(row["y0"]):int(row["y1"]), int(row["x0"]):int(row["x1"])]
        np.testing.assert_array_equal(arrays[row["patch_id"]], plane)


def _normalized_v04(run, output, scope, export_values="normalized"):
    from cellpatcher.config import NormalizationConfig

    config = _config(run, output, False)
    config.export.image_format = "npy"
    config.normalization = NormalizationConfig(enabled=True, method="minmax", scope=scope, export_values=export_values)
    return run_patching_condition_repeat_tree(run.experiment, config=config, combine_metadata=True)


@pytest.mark.parametrize("scope", ["image", "patch"])
def test_v04_normalizes_each_channel_with_the_parameters_of_its_id(tmp_path, scope):
    # W3 (normalization v0.5.0, Spec ADR 0003 HV2): the parameter columns are named by channel ID, and a
    # multi-channel patch of DrugA (ACTIN, DNA) is normalized channel by channel with the parameters of its
    # own IDs, never by position. normalization_channels records each patch's own C order.
    run = _described_run(tmp_path, "v04")
    results = _normalized_v04(run, tmp_path / "out", scope)
    patches = [
        (patch, array) for leaf in results
        for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays, strict=True)
    ]
    assert {patch.condition for patch, _ in patches} == {"CTRL", "DrugA"}
    for patch, array in patches:
        order = ("DNA", "ACTIN") if patch.condition == "CTRL" else ("ACTIN", "DNA")
        assert patch.normalization_channels == order
        assert patch.normalization_parameters.channels == ("ACTIN", "DNA")
        image = tifffile.imread(run.experiment / patch.source_relative_path).astype(np.float64)
        for c, channel in enumerate(order):
            source = image[c] if scope == "image" else image[c][patch.y0:patch.y1, patch.x0:patch.x1]
            low, high = source.min(), source.max()
            index = patch.normalization_parameters.channels.index(channel)
            assert (patch.normalization_parameters.offset[index], patch.normalization_parameters.scale[index]) == (
                low, high - low
            )
            raw = image[c][patch.y0:patch.y1, patch.x0:patch.x1]
            np.testing.assert_array_equal(array[c], ((raw - low) / (high - low)).astype(np.float32))
    with (tmp_path / "out" / "patches.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["condition"], row["normalization_channels"]) for row in rows} == {
        ("CTRL", "DNA;ACTIN"), ("DrugA", "ACTIN;DNA")
    }
    assert {row["normalization_parameters_version"] for row in rows} == {"2"}
    assert not [name for name in rows[0] if name.startswith("Normalization_") and name[-2:] in ("C0", "C1")]


def test_v04_raw_patches_normalize_on_load_by_their_recorded_order(tmp_path):
    # A raw export of a multi-channel DrugA patch is read back with its row's normalization_channels, so
    # loading gives the arrays the run delivered (each channel by its ID).
    from cellpatcher.delivery import iter_exported_patches

    run = _described_run(tmp_path, "v04")
    results = _normalized_v04(run, tmp_path / "out", "image", export_values="raw")
    delivered = {
        patch.patch_id: array for leaf in results
        for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays, strict=True)
    }
    loaded = list(iter_exported_patches(tmp_path / "out" / "patches.csv"))
    assert {row["condition"] for row, _ in loaded} == {"CTRL", "DrugA"}
    for row, array in loaded:
        np.testing.assert_array_equal(array, delivered[row["patch_id"]])


def test_one_id_gives_the_same_channel_whatever_its_c_index_in_each_image(tmp_path, monkeypatch):
    # What v04 asks once preprocessing accepts it (Stage 3), on the v13 run: the DrugA images are
    # reported with their channel IDs in the opposite C order (as the v04 description declares
    # them), the resolver then gives DNA as C index 1 there. The DNA patches of DrugA are cut from
    # its C1 plane, those of CTRL from C0; every one says DNA. patch_id keeps each image's c.
    from morphobricks_preprocessing.run_inputs import RunInputs

    fovs = RunInputs.fovs

    def swapped(self):
        return tuple(
            dataclasses.replace(fov, channel_ids=fov.channel_ids[::-1]) if fov.key[0] == "DrugA" else fov
            for fov in fovs(self)
        )

    monkeypatch.setattr(RunInputs, "fovs", swapped)
    run = _described_run(tmp_path, "v13")
    rows, arrays = _tree(run, tmp_path / "out", ["DNA"])

    assert {row["channel_id"] for row in rows} == {"DNA"}
    assert {(row["condition"], row["c"]) for row in rows} == {("CTRL", "0"), ("DrugA", "1"), ("DrugB", "0")}
    for row in rows:
        image = tifffile.imread(run.experiment / row["source_relative_path"])  # C, Y, X
        plane = image[int(row["c"])][int(row["y0"]):int(row["y1"]), int(row["x0"]):int(row["x1"])]
        np.testing.assert_array_equal(arrays[row["patch_id"]], plane)
        assert f"/c{row['c']}/" in row["patch_id"]


def test_in_memory_images_have_no_channel_ids():
    image = np.zeros((2, 16, 16), dtype=np.uint16)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = "CYX"
    config.extraction.split_channels = ["DNA"]
    with pytest.raises(ValueError, match="names channel IDs, which only a described preprocessing run has"):
        run_patching(images=[image], config=config, image_metadata=named([image]))


def test_image_metadata_channel_ids_name_each_c():
    image = np.zeros((2, 16, 16), dtype=np.uint16)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = "CYX"
    config.extraction.split_channels = True
    metadata = [{"position_id": "p0", "channel_ids": ("DNA", "ACTIN")}]
    result = run_patching(images=[image], config=config, image_metadata=metadata)
    assert {(patch.c, patch.channel_id) for patch in result.patches} == {(0, "DNA"), (1, "ACTIN")}
    with pytest.raises(ValueError, match="names 1 channels, none for c=1"):
        run_patching(images=[image], config=config, image_metadata=[{"position_id": "p0", "channel_ids": ("DNA",)}])
