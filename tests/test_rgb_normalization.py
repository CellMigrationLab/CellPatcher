"""RGB normalization: StarDist's percentile method over R, G and B together (follow-up C3).

The reference is ``csbdeep.utils.normalize(x, 1, 99.8, axis=(0, 1, 2))``,
repeated here in NumPy (CSBDeep is not a dependency): linear percentiles of
the three channels' pixels together, then ``(x - mi) / (ma - mi + 1e-20)`` in
float32.
"""

import importlib.util

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, GridConfig, NormalizationConfig
from cellpatcher.delivery import iter_exported_patches, iter_patch_arrays
from cellpatcher.orchestration import load_config, run_patching_condition_repeat_tree, run_patching_pipeline
from cellpatcher.pipeline import RUN_CONFIG_FILENAME, run_patching
from cellpatcher.statistics import PreprocessingCheckError, recorded_image_type
from tests.helpers import as_file_run, run_described

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs morphobricks-preprocessing (a core dependency)"
    ),
    pytest.mark.filterwarnings("ignore::cellpatcher.normalization.NormalizationWarning"),
]

# RGB parameters are recorded for the channel IDs R, G and B (normalization v0.5.0), sorted.
SHARED_COLUMNS = [
    "Normalization_Offset_B", "Normalization_Scale_B", "Normalization_Offset_G", "Normalization_Scale_G",
    "Normalization_Offset_R", "Normalization_Scale_R",
]


def _stardist_normalize(pixels, low=1.0, high=99.8):
    """csbdeep.utils.normalize(pixels, low, high, axis=None) with its float32 arithmetic."""
    mi = np.float32(np.percentile(pixels, low))
    ma = np.float32(np.percentile(pixels, high))
    return (pixels.astype(np.float32) - mi) / (ma - mi + np.float32(1e-20))


def _rgb_images(count=2, seed=0, size=16):
    rng = np.random.default_rng(seed)
    images = []
    for index in range(count):
        image = rng.integers(0, 200, (3, size, size), dtype=np.uint8)  # C, Y, X
        image[1] += 20 + 10 * index  # the channels differ, but share one offset and scale
        images.append(image)
    return images


def _config(csv_path=None, *, scope="patch", image_axes="CYX", **normalization):
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=8))
    config.common.image_axes = image_axes
    config.common.extract_arrays = True
    config.normalization = NormalizationConfig(
        enabled=True, plane_histograms=None if csv_path is None else str(csv_path), method="percentile",
        scope=scope, image_type="rgb", **normalization,
    )
    return config


def _run(images, config):
    """run_patching with an image_file per image, which gives its patches their identity."""
    metadata = [{"image_file": f"img{index}.tif"} for index in range(len(images))]
    return run_patching(images, config=config, image_metadata=metadata)


def _write_tiff(path, data, rgb=True):
    """Write a (C, Y, X) image: an RGB colour TIFF (``YXS``), or a three-channel ``CYX`` stack.

    A described run's ``rgb`` unit must be an RGB image by its header (preprocessing v0.5.0).
    """
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb:
        tifffile.imwrite(path, np.moveaxis(data, 0, -1), photometric="rgb", metadata={"axes": "YXS"})
    else:
        tifffile.imwrite(path, data, imagej=True, metadata={"axes": "CYX"})


def _preprocess(root, output, image_type="rgb"):
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.axes import prefill_mapping

    mapping = prefill_mapping("YXS") if image_type == "rgb" else mbp.AxisMapping("CYX", "CYX")
    result = run_described(root, mapping, output_folder=output, image_type=image_type)
    assert result.written, [str(issue) for issue in result.issues]
    return result.csv_path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_image_type_defaults_to_fluorescence_and_round_trips(tmp_path):
    assert NormalizationConfig().image_type == "fluorescence"
    config = _config()
    config.export.enabled = True
    config.common.output_dir = str(tmp_path / "out")
    _run(_rgb_images(1), config)
    assert load_config(tmp_path / "out" / RUN_CONFIG_FILENAME).normalization.image_type == "rgb"


def test_an_unknown_image_type_is_refused():
    with pytest.raises(ValueError, match="image_type must be 'fluorescence'"):
        NormalizationConfig(image_type="brightfield")


@pytest.mark.parametrize("method", ["minmax", "zscore", "robust_zscore"])
def test_rgb_allows_only_the_percentile_method(method):
    with pytest.raises(ValueError, match="StarDist's percentile method only"):
        NormalizationConfig(image_type="rgb", method=method)


@pytest.mark.parametrize("split_channels", [True, [0], [0, 1, 2]])
def test_rgb_with_split_channels_is_refused_before_patching(split_channels):
    config = _config()
    config.extraction.split_channels = split_channels
    with pytest.raises(ValueError, match="split_channels must be False"):
        _run(_rgb_images(1), config)


def test_rgb_needs_three_integer_channels():
    two_channels = [image[:2] for image in _rgb_images(1)]
    with pytest.raises(ValueError, match="exactly three 8- or 16-bit colour channels"):
        _run(two_channels, _config())
    floats = [image.astype(np.float32) for image in _rgb_images(1)]
    with pytest.raises(ValueError, match="exactly three 8- or 16-bit colour channels"):
        _run(floats, _config())


# ---------------------------------------------------------------------------
# Patch scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_axes", ["CYX", "YXC"])
def test_patch_scope_matches_stardist_normalization(image_axes):
    images = _rgb_images(1)
    if image_axes == "YXC":
        images = [np.moveaxis(image, 0, -1) for image in images]
    result = _run(images, _config(image_axes=image_axes, patch_region="patch"))

    assert len(result.patches) == 4
    for patch, array in zip(result.patches, result.patch_arrays):
        crop = images[0][:, patch.y0:patch.y1, patch.x0:patch.x1] if image_axes == "CYX" else \
            images[0][patch.y0:patch.y1, patch.x0:patch.x1]
        np.testing.assert_array_equal(array, _stardist_normalize(crop))
        parameters = patch.normalization_parameters
        assert parameters.method == "rgb percentile 1-99.8"
        assert parameters.channels == ("B", "G", "R")  # sorted IDs; the patch keeps its own C order
        assert patch.normalization_channels == ("R", "G", "B")
        assert len(set(parameters.offset)) == 1 and len(set(parameters.scale)) == 1
        row = patch.to_dict()
        assert [row[name] for name in SHARED_COLUMNS[0::2]] == [row["Normalization_Offset_R"]] * 3
        assert [row[name] for name in SHARED_COLUMNS[1::2]] == [row["Normalization_Scale_R"]] * 3


def test_an_rgb_patch_without_spread_becomes_zeros_and_is_kept():
    image = _rgb_images(1)[0]
    image[:, :8, :8] = 77  # the first tile has one value in every channel
    result = _run([image], _config(patch_region="patch"))

    assert len(result.patches) == 4  # not discarded, unlike a fluorescence patch (decision D8)
    flat = next(array for patch, array in zip(result.patches, result.patch_arrays) if (patch.y0, patch.x0) == (0, 0))
    np.testing.assert_array_equal(flat, np.zeros_like(flat))


def test_rgb_raw_export_and_lazy_delivery_give_the_eager_arrays(tmp_path):
    images = _rgb_images(2)
    eager = _run(images, _config(patch_region="patch"))

    lazy_config = _config(patch_region="patch")
    lazy_config.common.extract_arrays = False
    lazy = _run(images, lazy_config)
    for (patch, array), expected in zip(iter_patch_arrays(lazy.patches, images, lazy_config), eager.patch_arrays):
        np.testing.assert_array_equal(array, expected)

    raw = _config(patch_region="patch", export_values="raw")
    raw.common.extract_arrays = False
    raw.export.enabled = True
    raw.common.output_dir = str(tmp_path / "raw")
    _run(images, raw)
    loaded = [array for _, array in iter_exported_patches(tmp_path / "raw" / "patches.csv")]
    assert len(loaded) == len(eager.patch_arrays)
    for array, expected in zip(loaded, eager.patch_arrays):
        np.testing.assert_array_equal(array, expected)


# ---------------------------------------------------------------------------
# Scopes with the preprocessing brick's statistics
# ---------------------------------------------------------------------------


def test_image_scope_pools_r_g_and_b_of_the_whole_image(tmp_path):
    folder = tmp_path / "images"
    images = _rgb_images(2)
    for index, image in enumerate(images):
        _write_tiff(folder / f"img{index}.tif", image)
    csv_path = _preprocess(folder, tmp_path / "preprocessing")

    result = run_patching_pipeline(folder, config=as_file_run(_config(csv_path, scope="image")))

    by_image = {}
    for patch, array in zip(result.patches, result.patch_arrays):
        by_image.setdefault(patch.image_file, []).append((patch, array))
    assert set(by_image) == {"img0.tif", "img1.tif"}
    for index, image in enumerate(images):
        low, high = np.percentile(image, (1.0, 99.8))
        for patch, array in by_image[f"img{index}.tif"]:
            parameters = patch.normalization_parameters
            assert parameters.offset == (low,) * 3
            assert parameters.scale == pytest.approx((high - low,) * 3)
            crop = image[:, patch.y0:patch.y1, patch.x0:patch.x1]
            expected = (crop.astype(np.float32) - np.float32(low)) / (np.float32(high) - np.float32(low))
            np.testing.assert_allclose(array, expected, rtol=1e-6)


def test_biological_repeat_scope_pools_rgb_across_conditions(tmp_path):
    root = tmp_path / "experiment"
    images = dict(zip(("control/bio1/a.tif", "treated/bio1/b.tif"), _rgb_images(2, seed=4)))
    for path, image in images.items():
        _write_tiff(root / path, image)
    csv_path = _preprocess(root, tmp_path / "preprocessing")

    results = run_patching_condition_repeat_tree(root, config=as_file_run(_config(csv_path, scope="biological_repeat")))

    pooled = np.concatenate([image.ravel() for image in images.values()])
    low, high = np.percentile(pooled, (1.0, 99.8))
    patches = [patch for leaf in results for patch in leaf.result.patches]
    assert {patch.normalization_parameters for patch in patches} == {patches[0].normalization_parameters}
    assert patches[0].normalization_parameters.offset == (low,) * 3
    assert patches[0].normalization_parameters.scale == pytest.approx((high - low,) * 3)


@pytest.mark.parametrize("scope", ["image", "patch"])
def test_the_declared_image_type_wins_and_a_contradicting_setting_is_refused(tmp_path, scope):
    # CP-4 (P24): the image type is the unit's, declared in the description (Fov.image_type), at every
    # scope of a file run. A setting that contradicts it is refused, naming both; it never overrides it.
    # Before, the manifest's run-wide type was compared at the CSV scopes only, and a run without one
    # let the setting decide.
    folder = tmp_path / "images"
    _write_tiff(folder / "img0.tif", _rgb_images(1)[0])
    rgb_csv = _preprocess(folder, tmp_path / "rgb")

    fluorescence = _config(rgb_csv, scope=scope)
    fluorescence.normalization.image_type = "fluorescence"
    if scope == "patch":
        fluorescence.normalization.plane_histograms = None
        fluorescence = as_file_run(fluorescence, rgb_csv.parent)
    else:
        fluorescence = as_file_run(fluorescence)
    with pytest.raises(PreprocessingCheckError) as raised:
        run_patching_pipeline(folder, config=fluorescence)
    assert [issue.code for issue in raised.value.issues] == ["image_type_differs"]
    text = str(raised.value)
    assert "declares the image type 'rgb'" in text and "normalization.image_type is 'fluorescence'" in text
    assert "Set normalization.image_type='rgb'" in text and "img0.tif" in text

    stack = tmp_path / "stack"  # the same pixels as a three-channel stack: an ordinary channel unit
    _write_tiff(stack / "img0.tif", _rgb_images(1)[0], rgb=False)
    channels_csv = _preprocess(stack, tmp_path / "channels", image_type=None)
    rgb = _config(channels_csv, scope=scope)  # image_type="rgb"
    if scope == "patch":
        rgb.normalization.plane_histograms = None
        rgb = as_file_run(rgb, channels_csv.parent)
    else:
        rgb = as_file_run(rgb)
    with pytest.raises(PreprocessingCheckError) as raised:
        run_patching_pipeline(stack, config=rgb)
    text = str(raised.value)
    assert "declares the image type 'channels' (normalized as 'fluorescence')" in text
    assert "normalization.image_type is 'rgb'" in text


def test_the_declared_rgb_type_of_v28_is_read_through_run_inputs(tmp_path):
    # CP-4: Spec v28 declares two rgb units; every position gives Fov.image_type "rgb", and an rgb
    # setting normalizes them as RGB (one offset and scale for R, G and B).
    from morphobricks_preprocessing.testing_described import make_described_run

    from tests.test_described_ids import _fixture

    run = make_described_run(tmp_path / "v28", _fixture("v28"))
    csv_path = run.output / "plane_histograms.csv"
    assert recorded_image_type(csv_path) == "rgb"
    results = run_patching_condition_repeat_tree(run.experiment, config=as_file_run(_config(csv_path, scope="image")))
    patches = [patch for leaf in results for patch in leaf.result.patches]
    assert patches and {patch.normalization_parameters.method for patch in patches} == {"rgb percentile 1-99.8"}

    fluorescence = as_file_run(_config(csv_path, scope="image"))
    fluorescence.normalization.image_type = "fluorescence"
    with pytest.raises(PreprocessingCheckError, match="declares the image type 'rgb'"):
        run_patching_condition_repeat_tree(run.experiment, config=fluorescence)


def test_recorded_image_type_is_read_through_run_inputs(tmp_path):
    # CP-1: the image type is the run's, read through RunInputs. Before, any manifest could be
    # named for any table, and a table without a manifest gave None. CP-4: it is the type the
    # description declares (Fov.image_type), an ordinary channel unit giving "fluorescence".
    folder = tmp_path / "images"
    _write_tiff(folder / "img0.tif", _rgb_images(1)[0])
    rgb_csv = _preprocess(folder, tmp_path / "rgb")
    stack = tmp_path / "stack"  # the same pixels as a three-channel stack, an ordinary channel unit
    _write_tiff(stack / "img0.tif", _rgb_images(1)[0], rgb=False)
    channels_csv = _preprocess(stack, tmp_path / "channels", image_type=None)

    assert recorded_image_type(rgb_csv) == "rgb"
    assert recorded_image_type(rgb_csv, rgb_csv.parent / "run_manifest.json") == "rgb"
    assert recorded_image_type(channels_csv) == "fluorescence"
    with pytest.raises(PreprocessingCheckError, match="normalization.manifest names"):
        recorded_image_type(channels_csv, rgb_csv.parent / "run_manifest.json")
    lone = tmp_path / "lone" / "plane_histograms.csv"
    lone.parent.mkdir()
    lone.write_bytes(rgb_csv.read_bytes())
    with pytest.raises(PreprocessingCheckError, match="holds no run_manifest.json"):
        recorded_image_type(lone)
