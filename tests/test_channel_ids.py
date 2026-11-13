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


def test_v13_a_channel_by_id_is_that_channel_of_every_channel_split(tmp_path):
    # v13 declares DNA, ACTIN in C order for every unit: ACTIN by ID gives the ACTIN patches of
    # split_channels=True, the same patch_id bytes (c1) and arrays. Channels are chosen by ID only.
    run = _described_run(tmp_path, "v13")
    by_id, id_arrays = _tree(run, tmp_path / "by_id", ["ACTIN"])
    every, every_arrays = _tree(run, tmp_path / "every", True)

    assert by_id and {(row["c"], row["channel_id"]) for row in by_id} == {("1", "ACTIN")}
    assert {(row["c"], row["channel_id"]) for row in every} == {("0", "DNA"), ("1", "ACTIN")}
    assert [row["patch_id"] for row in by_id] == [row["patch_id"] for row in every if row["channel_id"] == "ACTIN"]
    assert all(row["patch_id"].startswith("/".join(
        (row["condition"], row["biological_repeat"], row["technical_repeat"], row["position_id"], "c1/")
    )) for row in by_id)  # the unchanged FOV identity leads it
    for patch_id, array in id_arrays.items():
        np.testing.assert_array_equal(array, every_arrays[patch_id])


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


def test_in_memory_images_without_channel_ids_cannot_be_selected_by_id():
    image = np.zeros((2, 16, 16), dtype=np.uint16)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = "CYX"
    config.extraction.split_channels = ["DNA"]
    with pytest.raises(ValueError, match=r'not named by ID .*image_metadata\["channel_ids"\]'):
        run_patching(images=[image], config=config, image_metadata=named([image]))
    metadata = [{"position_id": "p0", "channel_ids": ("ACTIN", "DNA")}]  # named: selected by ID, never position
    result = run_patching(images=[image], config=config, image_metadata=metadata)
    assert {(patch.c, patch.channel_id) for patch in result.patches} == {(1, "DNA")}


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


def _as_pre_v050_table(csv_path):
    """Rewrite a normalized patches.csv as CellPatcher 0.4.0 wrote it: C-index columns, no version or C order."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rename = {}
    for row in rows:
        for index, channel in enumerate(row["normalization_channels"].split(";")):
            for kind in ("Offset", "Scale"):
                rename[f"Normalization_{kind}_{channel}"] = f"Normalization_{kind}_C{index}"
    dropped = ("normalization_parameters_version", "normalization_channels")
    header = [rename.get(name, name) for name in rows[0] if name not in dropped]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, header)
        writer.writeheader()
        for row in rows:
            writer.writerow({rename.get(name, name): value for name, value in row.items() if name not in dropped})


def test_a_table_written_before_normalization_v050_is_refused_before_any_pixel_is_read(tmp_path, monkeypatch):
    # Review of #104: loading raw exported patches read each file before refusing its row's old (C-index)
    # parameter columns. The row is now refused first, with what to do, and no patch file is opened.
    import cellpatcher.delivery as delivery
    from cellpatcher.delivery import iter_exported_patches, load_exported_patch

    run = _described_run(tmp_path, "v13")
    _normalized_v04(run, tmp_path / "out", "image", export_values="raw")
    csv_path = tmp_path / "out" / "patches.csv"
    _as_pre_v050_table(csv_path)

    def no_read(*args, **kwargs):
        raise AssertionError("a patch file was read before the row was refused")

    monkeypatch.setattr(delivery, "read_array", no_read)
    expected = "without normalization_parameters_version.*run CellPatcher again"
    with pytest.raises(ValueError, match=expected):
        next(iter_exported_patches(csv_path))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    run_config = __import__("json").loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match=expected):
        load_exported_patch(row, root_dir=tmp_path / "out", run_config=run_config)

    pytest.importorskip("torch")
    import cellpatcher.torch_dataset as torch_dataset

    monkeypatch.setattr(torch_dataset, "_load_exported_patch", no_read)
    with pytest.raises(ValueError, match=expected):
        tensors = torch_dataset.PatchTensorConfig(scale_policy="none")
        torch_dataset.ExportedPatchTorchDataset(csv_path, axes="CYX", tensor_config=tensors)[0]


@pytest.fixture
def v04_distinct(tmp_path, monkeypatch):
    """``v04`` with DNA pixels in 0..39 and ACTIN in 100..249, so parameters taken by position change the numbers.

    Preprocessing's test writer draws every channel from 0..199, so at a CSV scope both IDs get the same
    parameters and only the recorded order can tell a by-position mistake. Its image writer is replaced here
    (private to preprocessing at the pinned tag) before the run is made, so the run records these pixels.
    """
    import morphobricks_preprocessing.testing_described as described

    ranges = {"DNA": (0, 40), "ACTIN": (100, 250)}

    def write_image(path, unit, seed):
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        size = described.IMAGE_SIZE
        planes = [rng.integers(*ranges[channel], (size, size), dtype=np.uint8) for channel in unit.channels]
        tifffile.imwrite(path, np.stack(planes), photometric="minisblack", metadata={"axes": "CYX"})

    monkeypatch.setattr(described, "_write_image", write_image)
    return _described_run(tmp_path, "v04")


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("scope", ["biological_repeat", "image", "patch"])
def test_v04_every_scope_pools_and_applies_each_channel_by_its_id(v04_distinct, tmp_path, scope, split):
    # Review of #104: the parameters of each ID, pooled over the planes of that ID wherever its C index (CTRL
    # stores DNA, ACTIN; DrugA ACTIN, DNA), applied to the plane of that ID; expected values from the pixels.
    from cellpatcher.config import NormalizationConfig

    run = v04_distinct
    config = _config(run, tmp_path / "out", split)
    config.normalization = NormalizationConfig(enabled=True, method="minmax", scope=scope)
    results = run_patching_condition_repeat_tree(run.experiment, config=config, combine_metadata=True)
    pairs = [(p, a) for leaf in results for p, a in zip(leaf.result.patches, leaf.result.patch_arrays, strict=True)]
    images = {p.source_relative_path: tifffile.imread(run.experiment / p.source_relative_path).astype(np.float64)
              for p, _ in pairs}
    ids = {path: ("DNA", "ACTIN") if path.startswith("CTRL") else ("ACTIN", "DNA") for path in images}
    assert {p.condition for p, _ in pairs} == {"CTRL", "DrugA"}
    for patch, array in pairs:
        own = images[patch.source_relative_path]
        for position, c in enumerate([patch.c] if split else range(2)):
            channel = ids[patch.source_relative_path][c]
            if scope == "biological_repeat":  # one repeat group: every plane of this ID, in both conditions
                pool = np.concatenate([image[ids[path].index(channel)].ravel() for path, image in images.items()])
            elif scope == "image":
                pool = own[c]
            else:
                pool = own[c][patch.y0:patch.y1, patch.x0:patch.x1]
            raw = own[c][patch.y0:patch.y1, patch.x0:patch.x1]
            expected = ((raw - pool.min()) / (pool.max() - pool.min())).astype(np.float32)
            np.testing.assert_array_equal(array if split else array[position], expected, err_msg=patch.patch_id)
