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
from cellpatcher.statistics import PreprocessingCheckError

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("morphobricks_preprocessing") is None, reason="needs the 'preprocessing' extra"
    ),
    pytest.mark.filterwarnings("ignore::cellpatcher.normalization.NormalizationWarning"),
]

SHARED_COLUMNS = [
    "Normalization_Offset_C0", "Normalization_Scale_C0", "Normalization_Offset_C1", "Normalization_Scale_C1",
    "Normalization_Offset_C2", "Normalization_Scale_C2",
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


def _write_tiff(path, data):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "CYX"})


def _preprocess(root, output, image_type="rgb"):
    import morphobricks_preprocessing as mbp

    design = mbp.default_design(mbp.preflight(root).dataset.repeats)
    for group in design.groups:
        design = design.with_confirmed(group, True, "2026-10-28")
    result = mbp.run_experiment(
        root, mbp.AxisMapping("CYX", "CYX"), output_folder=output, design=design, image_type=image_type
    )
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


def test_rgb_with_split_channels_is_refused_before_patching():
    config = _config()
    config.extraction.split_channels = True
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
        assert parameters.channels == (0, 1, 2)
        assert len(set(parameters.offset)) == 1 and len(set(parameters.scale)) == 1
        row = patch.to_dict()
        assert [row[name] for name in SHARED_COLUMNS[0::2]] == [row["Normalization_Offset_C0"]] * 3
        assert [row[name] for name in SHARED_COLUMNS[1::2]] == [row["Normalization_Scale_C0"]] * 3


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

    result = run_patching_pipeline(folder, config=_config(csv_path, scope="image"))

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

    results = run_patching_condition_repeat_tree(root, config=_config(csv_path, scope="biological_repeat"))

    pooled = np.concatenate([image.ravel() for image in images.values()])
    low, high = np.percentile(pooled, (1.0, 99.8))
    patches = [patch for leaf in results for patch in leaf.result.patches]
    assert {patch.normalization_parameters for patch in patches} == {patches[0].normalization_parameters}
    assert patches[0].normalization_parameters.offset == (low,) * 3
    assert patches[0].normalization_parameters.scale == pytest.approx((high - low,) * 3)


def test_the_manifest_image_type_must_match_the_configuration(tmp_path):
    folder = tmp_path / "images"
    _write_tiff(folder / "img0.tif", _rgb_images(1)[0])
    rgb_csv = _preprocess(folder, tmp_path / "rgb")

    fluorescence = _config(rgb_csv, scope="image")
    fluorescence.normalization.image_type = "fluorescence"
    with pytest.raises(PreprocessingCheckError, match="confirmed for 'rgb' images") as raised:
        run_patching_pipeline(folder, config=fluorescence)
    assert raised.value.issues[0].code == "image_type_differs"

    fluorescence_csv = _preprocess(folder, tmp_path / "fluorescence", image_type="fluorescence")
    with pytest.raises(PreprocessingCheckError, match="confirmed for 'fluorescence' images"):
        run_patching_pipeline(folder, config=_config(fluorescence_csv, scope="image"))


def test_without_an_image_type_in_the_manifest_the_configuration_decides(tmp_path):
    folder = tmp_path / "images"
    _write_tiff(folder / "img0.tif", _rgb_images(1)[0])
    csv_path = _preprocess(folder, tmp_path / "preprocessing", image_type=None)

    result = run_patching_pipeline(folder, config=_config(csv_path, scope="image"))
    assert {patch.normalization_parameters.method for patch in result.patches} == {"rgb percentile 1-99.8"}
