import numpy as np
import pytest

from cellpatcher.config import FilterConfig
from cellpatcher.filters import compute_threshold, filter_regions, foreground_fraction
from cellpatcher.types import PatchRegion


def test_compute_threshold_none():
    assert compute_threshold(np.arange(5), None) is None


def test_compute_threshold_numeric():
    assert compute_threshold(np.arange(5), 3) == 3.0


def test_compute_threshold_mean_median():
    arr = np.array([0, 1, 2, 3])
    assert compute_threshold(arr, "mean") == float(np.mean(arr))
    assert compute_threshold(arr, "median") == float(np.median(arr))


def test_compute_threshold_percentile():
    arr = np.arange(100)
    assert compute_threshold(arr, "percentile:90") == float(np.percentile(arr, 90))


def test_compute_threshold_invalid_percentile_raises():
    with pytest.raises(ValueError):
        compute_threshold(np.arange(10), "percentile:abc")


def test_compute_threshold_unknown_string_raises():
    with pytest.raises(ValueError):
        compute_threshold(np.arange(10), "unknown")


def test_compute_threshold_otsu_optional_dependency_behavior():
    arr = np.array([0, 0, 1, 1], dtype=float)
    try:
        val = compute_threshold(arr, "otsu")
        assert isinstance(val, float)
    except ImportError as exc:
        assert "scikit-image" in str(exc)


def test_foreground_fraction_expected():
    patch = np.array([[0, 2], [3, 4]])
    assert foreground_fraction(patch, 2) == 0.5


def test_foreground_fraction_none_threshold_raises():
    with pytest.raises(ValueError):
        foreground_fraction(np.zeros((2, 2)), None)


def test_filter_regions_no_active_filters_returns_same():
    img = np.arange(16).reshape(4, 4)
    regions = [PatchRegion(patch_id="p", y0=0, y1=2, x0=0, x1=2)]
    cfg = FilterConfig()
    out = filter_regions(img, regions, cfg)
    assert out == regions


def test_filter_regions_min_mean_intensity_filters():
    img = np.array([[0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5]])
    regions = [
        PatchRegion(patch_id="a", y0=0, y1=2, x0=0, x1=2),
        PatchRegion(patch_id="b", y0=0, y1=2, x0=2, x1=4),
    ]
    cfg = FilterConfig(min_mean_intensity=1)
    out = filter_regions(img, regions, cfg)
    assert [r.patch_id for r in out] == ["b"]


def test_filter_regions_min_foreground_fraction_numeric_threshold():
    img = np.array([[0, 0, 2, 2], [0, 1, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]])
    regions = [PatchRegion(patch_id="r", y0=0, y1=2, x0=0, x1=2)]
    cfg = FilterConfig(min_foreground_fraction=0.2, foreground_threshold=0.5)
    out = filter_regions(img, regions, cfg)
    assert len(out) == 1


def test_filter_regions_min_foreground_fraction_default_threshold_uses_image_mean():
    img = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]])
    regions = [PatchRegion(patch_id="r", y0=0, y1=2, x0=0, x1=2)]
    cfg = FilterConfig(min_foreground_fraction=0.1, foreground_threshold=None)
    out = filter_regions(img, regions, cfg)
    assert len(out) == 0


def test_filter_regions_invalid_min_foreground_fraction_raises():
    with pytest.raises(ValueError):
        FilterConfig(min_foreground_fraction=1.5)


def test_filter_regions_cyx_axes():
    img = np.zeros((2, 4, 4), dtype=float)
    img[:, :, 2:] = 5
    regions = [PatchRegion(patch_id="r", y0=0, y1=2, x0=2, x1=4)]
    cfg = FilterConfig(min_mean_intensity=1.0)
    out = filter_regions(img, regions, cfg, axes="CYX")
    assert len(out) == 1
