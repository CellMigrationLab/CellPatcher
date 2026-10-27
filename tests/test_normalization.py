"""Normalization with the preprocessing brick's statistics (docs/normalization_contract.md §6–§11).

Fixtures are simulated experiments whose plane_histograms.csv and
run_manifest.json are written by the real ``morphobricks_preprocessing``; the
expected parameters are computed directly from the pixels.
"""

import csv
import importlib.util
import json
import os
import warnings

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig
from cellpatcher.orchestration import load_config, run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from cellpatcher.statistics import PreprocessingCheckError
from tests.helpers import named

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs the 'preprocessing' extra"
    ),
    pytest.mark.filterwarnings("ignore::cellpatcher.normalization.NormalizationWarning"),
]

NORMALIZATION_COLUMNS = [
    "normalization_method", "normalization_scope", "normalization_group",
    "normalization_offset", "normalization_scale", "normalization_exact",
]


def _write_tiff(path, data, axes):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": axes})


def _preprocess(root, output, axes):
    import morphobricks_preprocessing as mbp

    result = mbp.run_experiment(root, mbp.AxisMapping(axes, axes), output_folder=output)
    assert result.written, [str(issue) for issue in result.issues]
    return result.csv_path


def _config(csv_path, *, mode="grid", image_axes="ZCYX", scope="image", method="minmax", **normalization):
    config = CellPatcherConfig(mode=mode, grid=GridConfig(tile_size=8))
    config.common.image_axes = image_axes
    config.common.extract_arrays = True
    if "Z" in image_axes:
        config.common.z_policy = "all"
    config.normalization = NormalizationConfig(
        enabled=True, plane_histograms=None if csv_path is None else str(csv_path), method=method, scope=scope,
        **normalization,
    )
    return config


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


# Three images: two Z-stacks of two channels in control, one in treated.
IMAGES = {
    "control/bio1/a.tif": ("control", "bio1", "single"),
    "control/bio1/b.tif": ("control", "bio1", "single"),
    "treated/bio1/c.tif": ("treated", "bio1", "single"),
}


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "experiment"
    rng = np.random.default_rng(0)
    stacks = {}
    for index, path in enumerate(IMAGES):
        # Channel 1 is brighter, so channels get different parameters.
        stack = rng.integers(0, 100, (2, 2, 16, 16), dtype=np.uint8)  # Z, C, Y, X
        stack[:, 1] += 100 + 20 * index
        _write_tiff(root / path, stack, "ZCYX")
        stacks[path] = stack
    return root, stacks, _preprocess(root, tmp_path / "preprocessing", "ZCYX")


def _patches_by_image(results):
    return {patch.source_relative_path: patch for leaf in results for patch in leaf.result.patches}


def test_a_tree_of_links_matches_the_brick_statistics(experiment, tmp_path):
    # The same images reached through a tree of links (a file link, a linked
    # folder): the brick and CellPatcher record the same path for each image,
    # so every patched plane finds its row, with the manifest's file checks.
    stored, stacks, _ = experiment
    linked = tmp_path / "linked"
    (linked / "control" / "bio1").mkdir(parents=True)
    for name in ("a.tif", "b.tif"):
        (linked / "control" / "bio1" / name).symlink_to(stored / "control" / "bio1" / name)
    (linked / "treated").mkdir()
    (linked / "treated" / "bio1").symlink_to(stored / "treated" / "bio1", target_is_directory=True)
    csv_path = _preprocess(linked, tmp_path / "linked_preprocessing", "ZCYX")

    results = run_patching_condition_repeat_tree(linked, config=_config(csv_path))

    patches = [patch for leaf in results for patch in leaf.result.patches]
    assert {patch.source_relative_path for patch in patches} == set(IMAGES)
    assert all(patch.normalization_scope == "image" for patch in patches)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_defaults_and_validation():
    import cellpatcher

    assert cellpatcher.NormalizationConfig is NormalizationConfig
    settings = NormalizationConfig()
    assert (settings.enabled, settings.method, settings.scope, settings.percentile) == (False, "percentile", "image", (1.0, 99.8))
    with pytest.raises(ValueError, match="method must be one of"):
        NormalizationConfig(method="quantile")
    with pytest.raises(ValueError, match="scope must be one of"):
        NormalizationConfig(scope="well")
    with pytest.raises(ValueError, match="patch_region must be one of"):
        NormalizationConfig(patch_region="cell")
    with pytest.raises(ValueError, match="0 <= low < high <= 100"):
        NormalizationConfig(percentile=(99.0, 1.0))
    assert settings.clip is None
    assert NormalizationConfig(clip=[0, 1]).clip == (0.0, 1.0)  # as after a JSON round trip
    for bad in ((1.0, 0.0), (0.0, float("inf")), ("low", 1.0), (0.0,)):
        with pytest.raises(ValueError, match="normalization.clip must"):
            NormalizationConfig(clip=bad)


def test_a_run_config_without_normalization_loads_as_off(tmp_path):
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps({"mode": "grid"}), encoding="utf-8")
    assert load_config(path).normalization == NormalizationConfig()


def test_csv_scopes_need_the_table_and_regions_need_cell_mode():
    with pytest.raises(ValueError, match="needs the plane_histograms.csv"):
        run_patching(images=np.ones((16, 16), np.uint8), config=_config(None, image_axes="YX"),
            image_metadata=named(np.ones((16, 16), np.uint8)))
    with pytest.raises(ValueError, match="only mode='cell' has"):
        run_patching(
            images=np.ones((16, 16), np.uint8),
            config=_config(None, image_axes="YX", scope="patch", patch_region="mask"),
                image_metadata=named(np.ones((16, 16), np.uint8)),
        )


# ---------------------------------------------------------------------------
# Image and group scopes
# ---------------------------------------------------------------------------


def test_image_scope_pools_every_z_plane_of_the_image_per_channel(experiment, tmp_path):
    root, stacks, csv_path = experiment
    config = _config(csv_path, scope="image")
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")

    results = run_patching_condition_repeat_tree(root, config=config, combine_metadata=True)

    for leaf in results:
        for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays):
            stack = stacks[patch.source_relative_path]
            lows, highs = stack.min(axis=(0, 2, 3)), stack.max(axis=(0, 2, 3))  # per channel, all Z
            raw = stack[patch.z0, :, patch.y0:patch.y1, patch.x0:patch.x1].astype(np.float64)
            expected = (raw - lows[:, None, None]) / (highs - lows)[:, None, None]
            assert array.dtype == np.float32
            np.testing.assert_allclose(array, expected, rtol=1e-6)
            assert patch.normalization_offset == ";".join(repr(float(v)) for v in lows)
            assert (patch.normalization_method, patch.normalization_scope, patch.normalization_group) == (
                "min-max", "image", patch.source_relative_path,
            )
            assert patch.normalization_exact is True

    header, rows = _read_csv(tmp_path / "out" / "patches.csv")
    start = header.index("source_relative_path")
    assert header[start : start + 7] == ["source_relative_path", *NORMALIZATION_COLUMNS]
    assert {row["normalization_exact"] for row in rows} == {"true"}
    saved = json.loads((tmp_path / "out" / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["normalization"]["scope"] == "image"
    inputs = saved["normalization_inputs"]
    assert (inputs["schema_version"], inputs["manifest_used"], inputs["manifest_version"]) == ("3", True, "1")
    assert len(inputs["plane_histograms_sha256"]) == 64
    assert load_config(tmp_path / "out" / RUN_CONFIG_FILENAME).normalization == config.normalization


@pytest.mark.parametrize(
    ("scope", "members", "group"),
    [
        ("condition", ["control/bio1/a.tif", "control/bio1/b.tif"], "control"),
        ("biological_repeat", ["control/bio1/a.tif", "control/bio1/b.tif"], "control/bio1"),
        ("technical_repeat", ["control/bio1/a.tif", "control/bio1/b.tif"], "control/bio1/single"),
    ],
)
def test_group_scopes_pool_the_group_per_channel(experiment, scope, members, group):
    root, stacks, csv_path = experiment
    results = run_patching_condition_repeat_tree(root, config=_config(csv_path, scope=scope, method="zscore"))

    patch = _patches_by_image(results)["control/bio1/a.tif"]
    pooled = np.concatenate([stacks[m].transpose(1, 0, 2, 3).reshape(2, -1) for m in members], axis=1)
    np.testing.assert_allclose([float(v) for v in patch.normalization_offset.split(";")], pooled.mean(axis=1))
    np.testing.assert_allclose([float(v) for v in patch.normalization_scale.split(";")], pooled.std(axis=1))
    assert (patch.normalization_group, patch.normalization_method) == (group, "z-score")


def test_per_time_point_gives_each_time_point_its_own_parameters(tmp_path):
    root = tmp_path / "experiment"
    series = np.zeros((2, 8, 8), dtype=np.uint8)  # T, Y, X
    series[0] = np.arange(64).reshape(8, 8)
    series[1] = 100 + 2 * np.arange(64).reshape(8, 8)
    _write_tiff(root / "control" / "a.tif", series, "TYX")
    csv_path = _preprocess(root, tmp_path / "preprocessing", "TYX")
    config = _config(csv_path, image_axes="TYX", per_time_point=True)
    config.common.t_policy = "all"

    (leaf,) = run_patching_condition_repeat_tree(root, config=config)
    by_t = {patch.t: patch for patch in leaf.result.patches}
    assert (by_t[0].normalization_offset, by_t[1].normalization_offset) == ("0.0", "100.0")
    assert by_t[1].normalization_group == "control/a.tif::t=1"


def test_group_parameters_lists_every_group_for_review(experiment):
    from cellpatcher.normalization import group_parameters

    _, stacks, csv_path = experiment
    rows = group_parameters(_config(csv_path, scope="condition", method="minmax"))
    assert [(row["group"], row["channel"]) for row in rows] == [
        ("control", 0), ("control", 1), ("treated", 0), ("treated", 1)
    ]
    control = np.stack([stacks["control/bio1/a.tif"], stacks["control/bio1/b.tif"]])
    assert rows[1]["offset"] == float(control[:, :, 1].min()) and rows[1]["planes"] == 4
    assert rows[1]["exact"] is True
    with pytest.raises(ValueError, match="condition, repeat and image scopes"):
        group_parameters(_config(None, image_axes="YX", scope="patch"))


def test_split_channels_keep_their_own_parameters(experiment):
    root, _, csv_path = experiment
    config = _config(csv_path)
    config.extraction.split_channels = True
    results = run_patching_condition_repeat_tree(root, config=config)
    patches = [p for leaf in results for p in leaf.result.patches if p.source_relative_path == "control/bio1/a.tif"]
    offsets = {(p.c, p.normalization_offset) for p in patches}
    assert len(offsets) == 2 and all(";" not in offset for _, offset in offsets)


# ---------------------------------------------------------------------------
# Patch scope
# ---------------------------------------------------------------------------


def _cell_image():
    image = np.zeros((20, 20), dtype=np.uint8)
    labels = np.zeros((20, 20), dtype=np.uint16)
    labels[4:9, 4:13] = 1  # a 5 x 9 cell
    image[:] = 50
    image[4:9, 4:13] = np.arange(45).reshape(5, 9) + 100
    image[0, 0] = 255  # outside every cell
    return image, labels


@pytest.mark.parametrize(("region", "patch_size"), [("mask", 6), ("bbox", 6), ("bbox", 16), ("patch", 6)])
def test_patch_regions_use_exactly_their_pixels(region, patch_size):
    image, labels = _cell_image()
    config = _config(None, mode="cell", image_axes="YX", scope="patch", patch_region=region)
    config.common.mask_axes = "YX"
    config.cell.patch_size, config.cell.exclude_border = patch_size, False

    (patch,) = run_patching(images=image, masks=labels, config=config, image_metadata=named(image)).patches
    rows, cols = slice(max(patch.y0, 0), patch.y1), slice(max(patch.x0, 0), patch.x1)
    if region == "mask":
        pixels = image[rows, cols][labels[rows, cols] == 1]
    elif region == "bbox":
        pixels = image[4:9, 4:13]  # the whole box, whatever the patch size
    else:
        pixels = image[rows, cols]
    assert (float(patch.normalization_offset), float(patch.normalization_scale)) == (
        float(pixels.min()), float(pixels.max() - pixels.min()),
    )
    assert patch.normalization_group == f"{patch.image_id}::{patch.patch_id}"


def test_patch_scope_never_uses_padding():
    from cellpatcher.normalization import _region_selection
    from cellpatcher.types import PatchRegion

    # A region running off the top-left corner, which pad_mode="constant" pads.
    patch = PatchRegion(patch_id="p", y0=-3, y1=5, x0=-2, x1=6, label_id=1)
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[0:2, 0:3] = 1
    plane = np.arange(100).reshape(10, 10)
    assert plane[_region_selection(patch, "patch", None, plane.shape)].shape == (5, 6)
    np.testing.assert_array_equal(np.sort(plane[_region_selection(patch, "mask", labels, plane.shape)]), [0, 1, 2, 10, 11, 12])


def test_patches_without_spread_are_discarded_and_reported(tmp_path):
    # Decision D8: the two constant tiles cannot be normalized from their own pixels.
    image = np.full((16, 16), 7, np.uint8)
    image[:8] = np.arange(128).reshape(8, 16)
    config = _config(None, image_axes="YX", scope="patch")
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_patching(images=image, config=config, image_metadata=named(image))
    assert [(p.y0, p.x0) for p in result.patches] == [(0, 0), (0, 8)]
    assert len(result.patch_arrays) == len(result.exported_files) == 2
    (report,) = [w for w in caught if "discarded" in str(w.message)]
    assert "2 patch(es) were discarded" in str(report.message)
    assert "y=8:16, x=0:8" in str(report.message) and "zero intensity spread" in str(report.message)
    _, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert len(rows) == 2


def test_patches_without_finite_pixels_are_discarded_too():
    image = np.arange(256, dtype=np.float32).reshape(16, 16)
    image[8:, 8:] = np.nan
    result = run_patching(images=image, config=_config(None, image_axes="YX", scope="patch"),
        image_metadata=named(image))
    assert (8, 8) not in [(p.y0, p.x0) for p in result.patches] and len(result.patches) == 3


# ---------------------------------------------------------------------------
# Pipeline order and blocked policies
# ---------------------------------------------------------------------------


def test_filters_see_raw_values_and_mask_fill_is_in_normalized_units():
    image, labels = _cell_image()
    config = _config(None, mode="cell", image_axes="YX", scope="patch", patch_region="mask")
    config.common.mask_axes = "YX"
    config.cell.patch_size, config.cell.exclude_border = 12, False
    config.filters.min_mean_intensity = 60  # raw units: the patch mean is above 60, its normalized mean is not
    config.extraction.apply_mask, config.extraction.mask_fill_value = True, float("nan")
    config.extraction.output_size = 6

    result = run_patching(images=image, masks=labels, config=config, image_metadata=named(image))
    (patch,), (array,) = result.patches, result.patch_arrays
    assert array.dtype == np.float32 and array.shape == (6, 6)
    assert np.isnan(array).any() and np.nanmin(array) >= -0.01 and np.nanmax(array) <= 1.01


def test_keep_and_projections_are_refused_under_normalization(experiment):
    root, _, csv_path = experiment
    for policy in ("keep", "max"):
        config = _config(csv_path)
        config.common.z_policy = policy
        with pytest.raises(ValueError, match="Normalization needs one plane per patch"):
            run_patching_condition_repeat_tree(root, config=config)


def test_normalization_off_leaves_arrays_and_columns_unchanged(experiment):
    root, _, csv_path = experiment
    config = _config(csv_path)
    config.normalization.enabled = False
    results = run_patching_condition_repeat_tree(root, config=config)
    assert results[0].result.patch_arrays[0].dtype == np.uint8
    assert {getattr(p, column) for leaf in results for p in leaf.result.patches for column in NORMALIZATION_COLUMNS} == {None}


# ---------------------------------------------------------------------------
# Checks of the preprocessing outputs
# ---------------------------------------------------------------------------


def test_images_without_rows_are_reported_together(experiment):
    root, stacks, csv_path = experiment
    _write_tiff(root / "control" / "bio1" / "new.tif", stacks["control/bio1/a.tif"], "ZCYX")
    _write_tiff(root / "treated" / "bio1" / "later.tif", stacks["control/bio1/a.tif"], "ZCYX")
    with pytest.raises(PreprocessingCheckError) as error_info:
        run_patching_condition_repeat_tree(root, config=_config(csv_path))
    (issue,) = error_info.value.issues
    assert issue.paths == ("control/bio1/new.tif", "treated/bio1/later.tif")


def test_a_changed_image_is_detected_with_and_without_the_manifest(experiment):
    root, stacks, csv_path = experiment
    changed = stacks["treated/bio1/c.tif"].copy()
    changed[1, 0, 0, 0] = 255
    path = root / "treated" / "bio1" / "c.tif"
    stat = path.stat()
    _write_tiff(path, changed, "ZCYX")
    with pytest.raises(PreprocessingCheckError, match="changed after the preprocessing brick"):
        run_patching_condition_repeat_tree(root, config=_config(csv_path))

    # Without the manifest: same size and time, but other pixel values.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    alone = csv_path.parent.parent / "alone"
    alone.mkdir()
    (alone / csv_path.name).write_bytes(csv_path.read_bytes())
    with pytest.raises(PreprocessingCheckError, match="the images changed after the preprocessing run"):
        run_patching_condition_repeat_tree(root, config=_config(alone / csv_path.name))


def test_in_memory_runs_need_source_relative_path_for_csv_scopes(experiment):
    _, stacks, csv_path = experiment
    stack = stacks["control/bio1/a.tif"]
    config = _config(csv_path)
    with pytest.raises(ValueError, match="source_relative_path"):
        run_patching(images=stack, config=config, image_metadata=named(stack))

    # The labels must be the CSV's: run_patching does not read them from paths.
    config.common.condition, config.common.biological_repeat = "control", "bio1"
    metadata = [{"source_relative_path": "control/bio1/a.tif", "position_id": "a"}]
    result = run_patching(images=stack, config=config, image_metadata=metadata)
    assert {p.normalization_group for p in result.patches} == {"control/bio1/a.tif"}


def test_warnings_explain_the_scope(experiment):
    root, _, csv_path = experiment

    def messages(config):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_patching_condition_repeat_tree(root, config=config)
        return " ".join(str(w.message) for w in caught)

    assert "background" in messages(_config(csv_path, scope="image"))
    assert "removes real intensity differences" in messages(_config(csv_path, scope="condition"))
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_patching(images=image, config=_config(None, image_axes="YX", scope="patch"), image_metadata=named(image))
    assert not [w for w in caught if "background" in str(w.message)]


def test_a_flat_folder_run_normalizes_against_its_own_preprocessing(tmp_path):
    folder = tmp_path / "images"
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    _write_tiff(folder / "a.tif", image, "YX")
    csv_path = _preprocess(folder, tmp_path / "preprocessing", "YX")

    result = run_patching_pipeline(folder, config=_config(csv_path, image_axes="YX"))
    assert {(p.source_relative_path, p.normalization_offset, p.normalization_scale) for p in result.patches} == {
        ("a.tif", "0.0", "255.0")
    }


def test_clip_comes_after_normalizing_and_before_resizing():
    """Resizing interpolates clipped values, so no resized value leaves the clip range."""
    from cellpatcher.resize import resize_patch

    image, labels = _cell_image()
    config = _config(None, mode="cell", image_axes="YX", scope="patch", patch_region="bbox", clip=(0.25, 0.75))
    config.common.mask_axes = "YX"
    config.cell.patch_size, config.cell.exclude_border = 10, False
    (cropped,) = run_patching(images=image, masks=labels, config=config, image_metadata=named(image)).patch_arrays
    assert cropped.min() == 0.25 and cropped.max() == 0.75  # both limits are reached
    config.extraction.output_size, config.extraction.resize_order = 7, 1
    (resized,) = run_patching(images=image, masks=labels, config=config, image_metadata=named(image)).patch_arrays
    np.testing.assert_array_equal(resized, resize_patch(cropped, axes="YX", output_size=7, order=1))
    config.normalization.clip = None
    (unclipped,) = run_patching(images=image, masks=labels, config=config, image_metadata=named(image)).patch_arrays
    assert not np.array_equal(np.clip(unclipped, 0.25, 0.75), resized)  # clipping after resizing differs
