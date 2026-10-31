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
        importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs morphobricks-preprocessing (a core dependency)"
    ),
    pytest.mark.filterwarnings("ignore::cellpatcher.normalization.NormalizationWarning"),
]

NORMALIZATION_COLUMNS = [
    "normalization_method", "normalization_scope", "normalization_group", "normalization_exact",
]
# One pair per channel, grouped by channel (normalization brick N41, N42); the
# simulated experiment has two channels.
CHANNEL_COLUMNS = [
    "Normalization_Offset_C0", "Normalization_Scale_C0", "Normalization_Offset_C1", "Normalization_Scale_C1",
]


def _write_tiff(path, data, axes):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": axes})


def _confirmed_design(root, pairs=None):
    """The preprocessing brick's experiment design of *root*, every repeat group confirmed.

    *pairs* moves ``(condition, biological_repeat)`` folders into other repeat groups.
    """
    import morphobricks_preprocessing as mbp

    design = mbp.default_design(mbp.preflight(root).dataset.repeats)
    for key, group in (pairs or {}).items():
        design = design.with_group(key, group)
    for group in design.groups:
        design = design.with_confirmed(group, True, "2026-10-28")
    return design


def _preprocess(root, output, axes, design="confirmed", histogram_settings=None):
    """Run the preprocessing brick; by default with a confirmed design, written as experiment_design.csv."""
    import morphobricks_preprocessing as mbp

    if design == "confirmed":
        design = _confirmed_design(root)
    result = mbp.run_experiment(
        root, mbp.AxisMapping(axes, axes), output_folder=output, design=design, histogram_settings=histogram_settings
    )
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
    with pytest.raises(ValueError, match="Scope 'well' does not exist. Choose 'biological_repeat'"):
        NormalizationConfig(scope="well")
    for removed in ("condition", "technical_repeat"):
        with pytest.raises(ValueError, match=f"The '{removed}' scope was removed"):
            NormalizationConfig(scope=removed)
    assert (settings.pooling, settings.experiment_design) == ("pixels", None)
    with pytest.raises(ValueError, match="normalization.pooling must be"):
        NormalizationConfig(pooling="images")
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
            assert patch.normalization_parameters.offset == tuple(float(v) for v in lows)
            assert patch.normalization_parameters.channels == (0, 1)
            assert (patch.normalization_method, patch.normalization_scope, patch.normalization_group) == (
                "min-max", "image", patch.source_relative_path,
            )
            assert patch.normalization_exact is True

    header, rows = _read_csv(tmp_path / "out" / "patches.csv")
    start = header.index("source_relative_path")
    assert header[start : start + 9] == ["source_relative_path", *NORMALIZATION_COLUMNS, *CHANNEL_COLUMNS]
    assert {row["normalization_exact"] for row in rows} == {"true"}
    assert "normalization_offset" not in header and "normalization_parameters" not in header
    first = rows[0]
    lows = stacks[first["source_relative_path"]].min(axis=(0, 2, 3))
    assert (first["Normalization_Offset_C0"], first["Normalization_Offset_C1"]) == tuple(repr(float(v)) for v in lows)
    saved = json.loads((tmp_path / "out" / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["normalization"]["scope"] == "image"
    inputs = saved["normalization_inputs"]
    assert (inputs["schema_version"], inputs["manifest_used"], inputs["manifest_version"]) == ("4", True, "1")
    assert len(inputs["plane_histograms_sha256"]) == 64
    assert load_config(tmp_path / "out" / RUN_CONFIG_FILENAME).normalization == config.normalization
    from morphobricks_normalization import NormalizationSettings, __version__

    assert saved["normalization_settings"] == {
        "library_version": __version__, "image_type": "fluorescence", "method": "minmax", "scope": "image",
        "per_time_point": False,
    }
    assert NormalizationSettings.from_dict(saved["normalization_settings"]).method_name == "min-max"


def test_biological_repeat_pools_the_repeat_across_conditions_per_channel(experiment):
    root, stacks, csv_path = experiment
    config = _config(csv_path, scope="biological_repeat", method="zscore")
    results = run_patching_condition_repeat_tree(root, config=config)

    patches = _patches_by_image(results)
    pooled = np.concatenate([stacks[m].transpose(1, 0, 2, 3).reshape(2, -1) for m in IMAGES], axis=1)
    for path in IMAGES:  # one repeat group, bio1, in both conditions
        patch = patches[path]
        np.testing.assert_allclose(patch.normalization_parameters.offset, pooled.mean(axis=1))
        np.testing.assert_allclose(patch.normalization_parameters.scale, pooled.std(axis=1))
        assert (patch.normalization_scope, patch.normalization_group, patch.normalization_method) == (
            "biological_repeat", "bio1", "z-score",
        )


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
    assert (by_t[0].normalization_parameters.offset, by_t[1].normalization_parameters.offset) == ((0.0,), (100.0,))
    assert by_t[1].normalization_group == "control/a.tif::t=1"


def test_group_parameters_lists_every_group_for_review(experiment):
    from cellpatcher.normalization import group_parameters

    _, stacks, csv_path = experiment
    rows = group_parameters(_config(csv_path, scope="biological_repeat", method="minmax"))
    assert [(row["group"], row["channel"]) for row in rows] == [("bio1", 0), ("bio1", 1)]
    every = np.stack([stacks[path] for path in IMAGES])
    assert rows[1]["offset"] == float(every[:, :, 1].min()) and rows[1]["planes"] == 6
    assert rows[1]["exact"] is True
    rows = group_parameters(_config(csv_path, scope="image", method="minmax"))
    assert [row["group"] for row in rows[::2]] == sorted(IMAGES)
    with pytest.raises(ValueError, match="biological-repeat and image scopes"):
        group_parameters(_config(None, image_axes="YX", scope="patch"))


def test_split_channels_keep_their_own_parameters(experiment):
    root, _, csv_path = experiment
    config = _config(csv_path)
    config.extraction.split_channels = True
    results = run_patching_condition_repeat_tree(root, config=config)
    patches = [p for leaf in results for p in leaf.result.patches if p.source_relative_path == "control/bio1/a.tif"]
    offsets = {(p.c, p.normalization_parameters.channels, p.normalization_parameters.offset) for p in patches}
    assert len(offsets) == 2 and all(channels == (c,) and len(offset) == 1 for c, channels, offset in offsets)


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
    parameters = patch.normalization_parameters
    assert (parameters.offset, parameters.scale, parameters.channels) == (
        (float(pixels.min()),), (float(pixels.max() - pixels.min()),), (0,),
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
    columns = [*NORMALIZATION_COLUMNS, "normalization_parameters"]
    assert {getattr(p, column) for leaf in results for p in leaf.result.patches for column in columns} == {None}


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


def test_a_position_id_other_than_the_tables_is_refused(experiment):
    # Issue #46: the FOV key of a plane must be its row's (D85), like its labels.
    _, stacks, csv_path = experiment
    stack = stacks["control/bio1/a.tif"]
    config = _config(csv_path)
    config.common.condition, config.common.biological_repeat = "control", "bio1"
    metadata = [{"source_relative_path": "control/bio1/a.tif", "position_id": "OTHER_FOV"}]
    with pytest.raises(PreprocessingCheckError, match="another position_id"):
        run_patching(images=stack, config=config, image_metadata=metadata)


def test_warnings_explain_the_scope(experiment):
    root, _, csv_path = experiment

    def messages(config):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_patching_condition_repeat_tree(root, config=config)
        return " ".join(str(w.message) for w in caught)

    assert "background" in messages(_config(csv_path, scope="image"))
    assert "differences between conditions within a repeat are kept" in messages(
        _config(csv_path, scope="biological_repeat")
    )
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
    found = {
        (p.source_relative_path, p.normalization_parameters.offset, p.normalization_parameters.scale)
        for p in result.patches
    }
    assert found == {("a.tif", (0.0,), (255.0,))}


def test_a_table_with_the_robust_histogram_range_normalizes(tmp_path):
    """Preprocessing v0.3.0 writes schema version 4; with the robust range a hot pixel no longer
    squeezes the plane into a few bins (normalization brick Milestone 5, preprocessing P6)."""
    from morphobricks_normalization.ranges import HistogramSettings

    folder = tmp_path / "images"
    image = np.random.default_rng(3).normal(100.0, 10.0, (128, 128)).astype(np.float32)
    image[3, 5] = 1e9  # one hot pixel
    _write_tiff(folder / "a.tif", image, "YX")
    robust = _preprocess(
        folder, tmp_path / "robust", "YX", histogram_settings=HistogramSettings(histogram_range="robust")
    )

    config = _config(robust, image_axes="YX", method="percentile")
    result = run_patching_pipeline(folder, config=config)  # the checks of every plane pass
    parameters = {p.normalization_parameters for p in result.patches}
    assert len(parameters) == 1
    (only,) = parameters
    low, high = np.percentile(image, (1.0, 99.8), method="inverted_cdf")
    assert abs(only.offset[0] - low) < 0.05 and abs(only.offset[0] + only.scale[0] - high) < 0.05
    assert only.exact is False

    # With the default range the same image gives no intensity spread to normalize by.
    minmax = _preprocess(folder, tmp_path / "minmax", "YX")
    with pytest.raises(ValueError, match="zero intensity spread"):
        run_patching_pipeline(folder, config=_config(minmax, image_axes="YX", method="percentile"))


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


# ---------------------------------------------------------------------------
# The normalization brick's fitting and columns (CellPatcher follow-up C1)
# ---------------------------------------------------------------------------


def test_split_channel_rows_fill_only_their_own_channel_pair(experiment, tmp_path):
    root, stacks, csv_path = experiment
    config = _config(csv_path)
    config.extraction.split_channels = True
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    run_patching_condition_repeat_tree(root, config=config, combine_metadata=True)

    header, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert header[header.index("normalization_exact") + 1 :][:4] == CHANNEL_COLUMNS
    for row in rows:
        own, other = row["c"], "1" if row["c"] == "0" else "0"
        lows = stacks[row["source_relative_path"]].min(axis=(0, 2, 3))
        assert row[f"Normalization_Offset_C{own}"] == repr(float(lows[int(own)]))
        assert row[f"Normalization_Offset_C{other}"] == row[f"Normalization_Scale_C{other}"] == ""


def test_a_selected_channel_row_fills_only_its_channel_pair(experiment, tmp_path):
    # CellPatcher follow-up C5: the group's Parameters.select([i]) (N46).
    root, stacks, csv_path = experiment
    config = _config(csv_path)
    config.extraction.split_channels = [1]
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    run_patching_condition_repeat_tree(root, config=config, combine_metadata=True)

    header, rows = _read_csv(tmp_path / "out" / "patches.csv")
    assert rows and {row["c"] for row in rows} == {"1"}
    # Only the channels the table's patches carry get columns.
    assert [name for name in header if name.startswith("Normalization_")] == [
        "Normalization_Offset_C1", "Normalization_Scale_C1",
    ]
    for row in rows:
        lows = stacks[row["source_relative_path"]].min(axis=(0, 2, 3))
        assert row["Normalization_Offset_C1"] == repr(float(lows[1]))


def test_the_brick_gives_the_parameters_and_arrays_of_the_earlier_code(experiment):
    """Bit-identical to pooling each channel's histograms with the brick's method functions, clip included."""
    import morphobricks_normalization as mbn

    from cellpatcher.statistics import load_plane_statistics

    root, stacks, csv_path = experiment
    config = _config(
        csv_path, scope="biological_repeat", method="percentile", percentile=(2.0, 98.0), clip=(0.0, 1.0)
    )
    results = run_patching_condition_repeat_tree(root, config=config)
    records = load_plane_statistics(csv_path).records
    for leaf in results:
        for patch, array in zip(leaf.result.patches, leaf.result.patch_arrays):
            members = [r for r in records if r.biological_repeat == patch.biological_repeat]
            old = [
                mbn.percentile_range(mbn.PooledHistogram(r.channel_histogram(c) for r in members), 2.0, 98.0)
                for c in range(2)
            ]
            assert patch.normalization_parameters.offset == tuple(p.offset for p in old)
            assert patch.normalization_parameters.scale == tuple(p.scale for p in old)
            assert patch.normalization_method == old[0].method
            stack = stacks[patch.source_relative_path]
            raw = stack[patch.z0, :, patch.y0 : patch.y1, patch.x0 : patch.x1]
            expected = np.stack([p.apply(raw[c], clip=(0.0, 1.0), dtype=np.float32) for c, p in enumerate(old)])
            np.testing.assert_array_equal(array, expected)


def test_clipping_after_the_brick_equals_clipping_before_rounding():
    import morphobricks_normalization as mbn

    from cellpatcher.normalization import PatchNormalization

    rng = np.random.default_rng(1)
    pixels = rng.normal(0.5, 0.4, (3, 64, 64)) * 1000
    parameters = mbn.Parameters("min-max", "image", "g", True, offset=(1.0, -3.5, 0.25), scale=(999.7, 1000.3, 0.1))
    for clip in ((0.0, 1.0), (0.1, 0.30000001), (-1e-9, 1e-9)):
        new = PatchNormalization(parameters).apply(pixels, "CYX", clip=clip)
        old = np.stack([
            mbn.NormalizationParameters("min-max", o, s).apply(pixels[c], clip=clip, dtype=np.float32)
            for c, (o, s) in enumerate(zip(parameters.offset, parameters.scale))
        ])
        np.testing.assert_array_equal(new, old)


def test_rows_of_earlier_tables_still_normalize_on_load():
    """Normalization brick decision N47: CellPatcher converts the ';'-separated columns itself."""
    from cellpatcher.normalization import parameters_from_row

    settings = {"method": "minmax", "percentile": [1.0, 99.8]}
    old_row = {
        "patch_id": "p", "c": "", "normalization_method": "min-max", "normalization_scope": "image",
        "normalization_group": "a.tif", "normalization_offset": "1.5;2.0", "normalization_scale": "3.0;4.0",
        "normalization_exact": "true",
    }
    parameters = parameters_from_row(old_row, settings)
    assert (parameters.offset, parameters.scale, parameters.channels, parameters.exact) == (
        (1.5, 2.0), (3.0, 4.0), (0, 1), True,
    )
    split_row = {**old_row, "c": "1", "normalization_offset": "2.0", "normalization_scale": "4.0"}
    split = parameters_from_row(split_row, settings)
    assert (split.offset, split.channels) == ((2.0,), (1,))
    new_row = {**old_row, "normalization_offset": "", "normalization_scale": "", **parameters.to_columns()}
    assert parameters_from_row(new_row, settings) == parameters
    assert parameters_from_row({"patch_id": "p", "normalization_method": ""}, settings) is None


def test_images_with_different_channel_counts_are_refused():
    two, three = np.ones((2, 16, 16), np.uint8), np.ones((3, 16, 16), np.uint8)
    two[:, 0, 0] = three[:, 0, 0] = 9
    config = _config(None, image_axes="CYX", scope="patch")
    with pytest.raises(ValueError, match="different numbers of channels"):
        run_patching(images=[two, three], config=config, image_metadata=named([two, three]))


def test_run_config_records_the_brick_settings_and_the_design(experiment, tmp_path):
    import hashlib

    root, _, csv_path = experiment
    design_path = csv_path.parent / "experiment_design.csv"
    for scope in ("image", "biological_repeat"):
        config = _config(csv_path, scope=scope, pooling="conditions")
        config.export.enabled, config.common.output_dir = True, str(tmp_path / scope)
        run_patching_condition_repeat_tree(root, config=config, combine_metadata=True)
        saved = json.loads((tmp_path / scope / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
        assert saved["normalization_settings"]["scope"] == scope
        inputs = saved["normalization_inputs"]
        if scope == "biological_repeat":
            assert saved["normalization_settings"]["pooling"] == "conditions"
            assert inputs["experiment_design_file"] == "experiment_design.csv"
            assert inputs["experiment_design_sha256"] == hashlib.sha256(design_path.read_bytes()).hexdigest()
        else:
            assert "pooling" not in saved["normalization_settings"]  # no effect at the image scope
            assert "experiment_design_file" not in inputs


def test_the_old_offset_and_scale_attributes_are_deprecated(experiment):
    root, _, csv_path = experiment
    (leaf, *_) = run_patching_condition_repeat_tree(root, config=_config(csv_path))
    patch = leaf.result.patches[0]
    with pytest.warns(DeprecationWarning, match="normalization_parameters"):
        offset = patch.normalization_offset
    assert offset == ";".join(repr(value) for value in patch.normalization_parameters.offset)


# ---------------------------------------------------------------------------
# The biological-repeat scope and the experiment design (follow-up C2)
# ---------------------------------------------------------------------------


def _unpaired_experiment(tmp_path, pairs=None, design="confirmed"):
    """control/rep_a and treated/rep_b, a repeat named differently in each condition."""
    root = tmp_path / "unpaired"
    images = {}
    for index, path in enumerate(("control/rep_a/a.tif", "treated/rep_b/b.tif")):
        image = (np.arange(64, dtype=np.uint8).reshape(8, 8) + 50 * index).astype(np.uint8)
        _write_tiff(root / path, image, "YX")
        images[path] = image
    if design == "confirmed":
        design = _confirmed_design(root, pairs)
    return root, images, _preprocess(root, tmp_path / "unpaired_out", "YX", design=design)


def test_the_design_pairs_repeats_with_different_names(tmp_path):
    root, images, csv_path = _unpaired_experiment(tmp_path)
    config = _config(csv_path, image_axes="YX", scope="biological_repeat")
    patches = _patches_by_image(run_patching_condition_repeat_tree(root, config=config))
    assert {p.normalization_group for p in patches.values()} == {"rep_a", "rep_b"}  # unpaired: two groups

    root, images, csv_path = _unpaired_experiment(tmp_path / "paired", pairs={("treated", "rep_b"): "rep_a"})
    config = _config(csv_path, image_axes="YX", scope="biological_repeat")
    patches = _patches_by_image(run_patching_condition_repeat_tree(root, config=config))
    every = np.concatenate([image.ravel() for image in images.values()])
    assert {p.normalization_group for p in patches.values()} == {"rep_a"}
    assert {p.normalization_parameters.offset for p in patches.values()} == {(float(every.min()),)}


def test_pooling_by_conditions_weighs_each_condition_equally(tmp_path):
    root = tmp_path / "weights"
    small = np.zeros((4, 4), dtype=np.uint8)
    large = np.full((16, 16), 100, dtype=np.uint8)
    small[0, 0], large[0, 0] = 10, 110
    _write_tiff(root / "control" / "rep1" / "a.tif", small, "YX")
    _write_tiff(root / "treated" / "rep1" / "b.tif", large, "YX")
    csv_path = _preprocess(root, tmp_path / "weights_out", "YX")

    def offsets(pooling):
        config = _config(csv_path, image_axes="YX", scope="biological_repeat", method="zscore", pooling=pooling)
        config.grid.tile_size = 4
        patches = _patches_by_image(run_patching_condition_repeat_tree(root, config=config))
        return patches["control/rep1/a.tif"].normalization_parameters.offset[0]

    by_pixels = np.concatenate([small.ravel(), large.ravel()]).astype(np.float64).mean()
    by_conditions = (small.astype(np.float64).mean() + large.astype(np.float64).mean()) / 2
    assert offsets("pixels") == pytest.approx(by_pixels)
    assert offsets("conditions") == pytest.approx(by_conditions)


def test_a_missing_or_unconfirmed_design_stops_the_run_before_any_image_is_read(tmp_path, monkeypatch):
    import cellpatcher.io

    def no_reading(*args, **kwargs):
        raise AssertionError("an image was read")

    root, _, csv_path = _unpaired_experiment(tmp_path, design=None)  # no experiment_design.csv
    monkeypatch.setattr(cellpatcher.io, "read_image", no_reading)
    config = _config(csv_path, image_axes="YX", scope="biological_repeat")
    with pytest.raises(PreprocessingCheckError) as missing:
        run_patching_condition_repeat_tree(root, config=config)
    assert [issue.code for issue in missing.value.issues] == ["missing_experiment_design"]
    assert "step 1 of the preprocessing brick" in str(missing.value)

    import morphobricks_preprocessing as mbp

    unconfirmed = mbp.default_design(mbp.preflight(root).dataset.repeats)
    mbp.write_experiment_design(unconfirmed, tmp_path / "unconfirmed.csv")
    config.normalization.experiment_design = str(tmp_path / "unconfirmed.csv")
    with pytest.raises(PreprocessingCheckError) as not_confirmed:
        run_patching_condition_repeat_tree(root, config=config)
    assert "design_not_confirmed" in [issue.code for issue in not_confirmed.value.issues]


def test_a_design_changed_after_preprocessing_is_refused(tmp_path):
    import morphobricks_preprocessing as mbp

    root, _, csv_path = _unpaired_experiment(tmp_path)
    design_path = csv_path.parent / "experiment_design.csv"
    changed = _confirmed_design(root, pairs={("treated", "rep_b"): "rep_a"})
    mbp.write_experiment_design(changed, design_path)
    config = _config(csv_path, image_axes="YX", scope="biological_repeat")
    with pytest.raises(PreprocessingCheckError) as caught:
        run_patching_condition_repeat_tree(root, config=config)
    assert [issue.code for issue in caught.value.issues] == ["changed_experiment_design"]


def _design_edited(tmp_path, edit):
    """An unpaired experiment whose experiment_design.csv is edited by *edit* (text -> text).

    The manifest's design hash is removed, so only the reader can refuse the edit.
    """
    root, _, csv_path = _unpaired_experiment(tmp_path)
    manifest_path = csv_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("experiment_design_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    design_path = csv_path.parent / "experiment_design.csv"
    design_path.write_bytes(edit(design_path.read_bytes().decode("utf-8")).encode("utf-8"))
    return root, csv_path


@pytest.mark.parametrize("edit, match", [
    # Before v0.1.2 CellPatcher read the design leniently: each of these was accepted.
    (lambda text: text.replace(",true,", ",yes,"), r"line 2.*confirmed"),
    (lambda text: text.replace("\ncontrol,", "\n control,", 1), r"line 2.*condition"),
    (lambda text: "\ufeff" + text, "byte-order mark"),
    (lambda text: text.replace("\ntreated,", "\n\ntreated,", 1), r"line 3 has 0 values"),
], ids=["confirmed-yes", "padded-condition", "byte-order-mark", "blank-line"])
def test_the_design_is_read_with_the_shared_strict_reader(tmp_path, edit, match):
    root, csv_path = _design_edited(tmp_path, edit)
    config = _config(csv_path, image_axes="YX", scope="biological_repeat")
    with pytest.raises(PreprocessingCheckError, match=match) as caught:
        run_patching_condition_repeat_tree(root, config=config)
    assert [issue.code for issue in caught.value.issues] == ["unreadable_experiment_design"]


def test_the_design_reader_is_the_preprocessing_brick_s(tmp_path, monkeypatch):
    import morphobricks_preprocessing.design as design_module

    calls = []
    reader = design_module.read_confirmed_design

    def recording(path, *, expected_sha256=None):
        calls.append(expected_sha256)
        return reader(path, expected_sha256=expected_sha256)

    monkeypatch.setattr(design_module, "read_confirmed_design", recording)
    root, _, csv_path = _unpaired_experiment(tmp_path)
    manifest = json.loads((csv_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    run_patching_condition_repeat_tree(root, config=_config(csv_path, image_axes="YX", scope="biological_repeat"))
    assert calls and set(calls) == {manifest["experiment_design_sha256"]}


def test_a_missing_design_says_where_cellpatcher_looks(tmp_path):
    root, _, csv_path = _unpaired_experiment(tmp_path, design=None)
    with pytest.raises(PreprocessingCheckError) as caught:
        run_patching_condition_repeat_tree(root, config=_config(csv_path, image_axes="YX", scope="biological_repeat"))
    (issue,) = caught.value.issues
    assert issue.code == "missing_experiment_design"
    assert "normalization.experiment_design" in issue.action


def test_repeat_groups_missing_a_condition_are_reported(tmp_path):
    root = tmp_path / "incomplete"
    for path in ("control/rep1/a.tif", "control/rep2/b.tif", "treated/rep1/c.tif"):
        _write_tiff(root / path, np.arange(64, dtype=np.uint8).reshape(8, 8), "YX")
    csv_path = _preprocess(root, tmp_path / "incomplete_out", "YX")
    config = _config(csv_path, image_axes="YX", scope="biological_repeat")
    with pytest.warns(UserWarning, match="do not have images from every condition"):
        run_patching_condition_repeat_tree(root, config=config)


def test_raw_patches_of_a_removed_scope_still_normalize_on_load(experiment, tmp_path):
    from cellpatcher import iter_exported_patches

    root, _, csv_path = experiment
    config = _config(csv_path)
    config.export.enabled, config.common.output_dir = True, str(tmp_path / "out")
    config.normalization.export_values = "raw"
    results = run_patching_condition_repeat_tree(root, config=config, combine_metadata=True)
    delivered = {p.patch_id: a for leaf in results for p, a in zip(leaf.result.patches, leaf.result.patch_arrays)}

    # The same files as a run at the removed condition scope would have left them.
    run_config = tmp_path / "out" / RUN_CONFIG_FILENAME
    saved = json.loads(run_config.read_text(encoding="utf-8"))
    saved["normalization"]["scope"] = "condition"
    run_config.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="The 'condition' scope was removed"):
        load_config(run_config)
    loaded = list(iter_exported_patches(tmp_path / "out" / "patches.csv"))
    assert len(loaded) == len(delivered)
    for row, array in loaded:
        np.testing.assert_array_equal(array, delivered[row["patch_id"]])


def _messages(run):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run()
    return " ".join(str(w.message) for w in caught)


def test_csv_scopes_warn_when_the_inputs_cannot_be_checked(experiment):
    # Issue #46: without run_manifest.json, the table, size/time and axis checks
    # are skipped; at the biological_repeat scope a design change goes unseen.
    root, _, csv_path = experiment
    with_manifest = _messages(lambda: run_patching_condition_repeat_tree(
        root, config=_config(csv_path, scope="biological_repeat")
    ))
    assert "No run_manifest.json" not in with_manifest
    assert "experiment_design_sha256" not in with_manifest

    alone = csv_path.parent.parent / "alone"
    alone.mkdir()
    for name in (csv_path.name, "experiment_design.csv"):
        (alone / name).write_bytes((csv_path.parent / name).read_bytes())
    image_scope = _messages(lambda: run_patching_condition_repeat_tree(root, config=_config(alone / csv_path.name)))
    assert "No run_manifest.json was found" in image_scope
    assert "experiment_design_sha256" not in image_scope
    repeat_scope = _messages(lambda: run_patching_condition_repeat_tree(
        root, config=_config(alone / csv_path.name, scope="biological_repeat")
    ))
    assert "No run_manifest.json was found" in repeat_scope
    assert "records no experiment_design_sha256 (there is no manifest)" in repeat_scope


def test_a_manifest_without_the_design_hash_warns_at_the_repeat_scope(experiment):
    root, _, csv_path = experiment
    manifest_path = csv_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("experiment_design_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    messages = _messages(lambda: run_patching_condition_repeat_tree(
        root, config=_config(csv_path, scope="biological_repeat")
    ))
    assert "The run manifest records no experiment_design_sha256, so a change" in messages
    assert "No run_manifest.json" not in messages


def _two_object_binary_mask():
    image = np.full((20, 20), 50, dtype=np.uint16)
    image[2:7, 2:7] = 200
    image[12:17, 12:17] = 100
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:7, 2:7] = True
    mask[12:17, 12:17] = True
    return image, mask


@pytest.mark.parametrize("region", ["mask", "bbox", None])
def test_patch_scope_object_regions_refuse_computed_component_labels(region):
    # Issue #44: the computed component labels are not the binary mask's 0/1
    # values, so label 2 would have "no finite pixels" and label 1 would take
    # in every foreground pixel of its rectangle.
    image, mask = _two_object_binary_mask()
    config = _config(None, mode="cell", image_axes="YX", scope="patch", patch_region=region)
    config.common.mask_axes = "YX"
    config.cell.patch_size, config.cell.exclude_border = 6, False
    config.cell.binary_mask_policy = "connected_components"
    with pytest.raises(ValueError, match="binary_mask_policy='connected_components' computes"):
        run_patching(images=image, masks=mask, config=config, image_metadata=named(image))


def test_patch_scope_patch_region_accepts_computed_component_labels():
    image, mask = _two_object_binary_mask()
    config = _config(None, mode="cell", image_axes="YX", scope="patch", patch_region="patch")
    config.common.mask_axes = "YX"
    config.cell.patch_size, config.cell.exclude_border = 6, False
    config.cell.binary_mask_policy = "connected_components"
    with pytest.warns(UserWarning, match="connected components"):
        result = run_patching(images=image, masks=mask, config=config, image_metadata=named(image))
    assert sorted(p.label_id for p in result.patches) == [1, 2]
    assert all(p.normalization_parameters is not None for p in result.patches)
