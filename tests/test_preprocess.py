import numpy as np
import pytest

from cellpatcher import CellConfig, CellPatcherConfig, CommonConfig, run_patching

# These tests keep covering the deprecated normalizations (docs/normalization_contract.md §15).
pytestmark = pytest.mark.filterwarnings("ignore:.*config.normalization:DeprecationWarning")
from cellpatcher.preprocess import (
    PreparedPlane,
    PreparedPlanePair,
    iter_cell_planes_for_patching,
    iter_image_planes_for_patching,
    minmax_normalize,
    percentile_normalize,
    prepare_image_for_patching,
    prepare_labels_for_cell_patching,
    project_axis,
    repeat_to_rgb,
    select_axis_index,
)


def test_percentile_normalize_scales_finite_values():
    arr = np.array([0, 5, 10], dtype=np.float32)
    out = percentile_normalize(arr, p_low=0, p_high=100)
    assert out.dtype == np.float32
    assert np.all(out >= 0) and np.all(out <= 1)
    assert out[0] == 0 and out[-1] == 1


def test_percentile_normalize_constant_returns_zeros():
    out = percentile_normalize(np.ones((3, 3)))
    assert np.all(out == 0)


def test_percentile_normalize_all_nonfinite_returns_zeros():
    out = percentile_normalize(np.array([np.nan, np.inf, -np.inf]))
    assert np.all(out == 0)


def test_percentile_normalize_nonfinite_becomes_zero():
    arr = np.array([0.0, 1.0, np.nan, np.inf])
    out = percentile_normalize(arr, p_low=0, p_high=100)
    assert out[2] == 0 and out[3] == 0


def test_percentile_normalize_invalid_percentiles_raise():
    with pytest.raises(ValueError):
        percentile_normalize(np.arange(5), p_low=90, p_high=10)


def test_minmax_normalize_scales_finite_values():
    out = minmax_normalize(np.array([2.0, 4.0, 6.0]))
    assert np.all(out >= 0) and np.all(out <= 1)
    assert out[0] == 0 and out[-1] == 1


def test_minmax_constant_returns_zeros():
    out = minmax_normalize(np.full((2, 2), 7.0))
    assert np.all(out == 0)


def test_minmax_all_nonfinite_returns_zeros():
    out = minmax_normalize(np.array([np.nan, np.inf]))
    assert np.all(out == 0)


def test_minmax_nonfinite_becomes_zero():
    out = minmax_normalize(np.array([1.0, 3.0, np.nan]))
    assert out[-1] == 0


def test_select_axis_index_zyx_select_z():
    arr = np.zeros((3, 4, 5))
    out, ax = select_axis_index(arr, "ZYX", "Z", 1)
    assert out.shape == (4, 5)
    assert ax == "YX"


def test_select_axis_index_cyx_select_c():
    arr = np.zeros((2, 4, 5))
    out, ax = select_axis_index(arr, "CYX", "C", 0)
    assert out.shape == (4, 5)
    assert ax == "YX"


def test_select_axis_index_yxc_select_c_negative_index():
    arr = np.zeros((4, 5, 3))
    out, ax = select_axis_index(arr, "YXC", "C", -1)
    assert out.shape == (4, 5)
    assert ax == "YX"


def test_select_axis_index_invalid_axis_raises():
    with pytest.raises(ValueError):
        select_axis_index(np.zeros((3, 4, 5)), "ZYX", "C", 0)


def test_select_axis_index_out_of_range_raises():
    with pytest.raises(ValueError):
        select_axis_index(np.zeros((3, 4, 5)), "ZYX", "Z", 4)


def test_project_axis_zyx_methods():
    arr = np.arange(24).reshape(2, 3, 4)
    out_max, ax_max = project_axis(arr, "ZYX", "Z", "max")
    out_mean, ax_mean = project_axis(arr, "ZYX", "Z", "mean")
    out_sum, ax_sum = project_axis(arr, "ZYX", "Z", "sum")
    out_first, ax_first = project_axis(arr, "ZYX", "Z", "first")
    assert ax_max == ax_mean == ax_sum == ax_first == "YX"
    assert out_max.shape == (3, 4)
    assert np.allclose(out_first, arr[0])


def test_project_axis_cyx_mean_over_c():
    arr = np.ones((2, 3, 4))
    out, ax = project_axis(arr, "CYX", "C", "mean")
    assert out.shape == (3, 4)
    assert ax == "YX"


def test_project_axis_invalid_method_raises():
    with pytest.raises(ValueError):
        project_axis(np.zeros((2, 3, 4)), "ZYX", "Z", "median")


def test_project_axis_invalid_axis_raises():
    with pytest.raises(ValueError):
        project_axis(np.zeros((2, 3, 4)), "ZYX", "C", "max")


def test_repeat_to_rgb_paths():
    yx = np.zeros((4, 5))
    out, ax = repeat_to_rgb(yx, "YX")
    assert out.shape == (4, 5, 3) and ax == "YXC"

    cyx = np.zeros((1, 4, 5))
    out, ax = repeat_to_rgb(cyx, "CYX")
    assert out.shape == (3, 4, 5) and ax == "CYX"

    yxc = np.zeros((4, 5, 1))
    out, ax = repeat_to_rgb(yxc, "YXC")
    assert out.shape == (4, 5, 3) and ax == "YXC"


def test_repeat_to_rgb_c3_valid_and_c2_raises_and_no_yx_raises():
    out1, ax1 = repeat_to_rgb(np.zeros((3, 4, 5)), "CYX")
    assert out1.shape == (3, 4, 5) and ax1 == "CYX"
    out2, ax2 = repeat_to_rgb(np.zeros((4, 5, 3)), "YXC")
    assert out2.shape == (4, 5, 3) and ax2 == "YXC"

    with pytest.raises(ValueError):
        repeat_to_rgb(np.zeros((2, 4, 5)), "CYX")
    with pytest.raises(ValueError):
        repeat_to_rgb(np.zeros((2, 3)), "CZ")


def test_prepare_image_for_patching_core_behaviors():
    arr = np.zeros((6, 7))
    out, ax = prepare_image_for_patching(arr, "YX")
    assert out.shape == (6, 7) and ax == "YX"

    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "ZYX")

    out, ax = prepare_image_for_patching(np.arange(2 * 6 * 7).reshape(2, 6, 7), "ZYX", z_policy="max")
    assert out.shape == (6, 7) and ax == "YX"

    out, ax = prepare_image_for_patching(np.zeros((3, 6, 7)), "ZYX", z_policy="select", z_index=1)
    assert out.shape == (6, 7) and ax == "YX"

    out, ax = prepare_image_for_patching(np.zeros((1, 6, 7)), "ZYX")
    assert out.shape == (6, 7) and ax == "YX"


def test_prepare_image_for_patching_t_and_c_behaviors():
    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 3, 6, 7)), "TZYX")

    out, ax = prepare_image_for_patching(np.zeros((1, 3, 6, 7)), "TZYX", z_policy="first")
    assert out.shape == (6, 7) and ax == "YX"

    out, ax = prepare_image_for_patching(np.zeros((2, 4, 3, 6, 7)), "TCZYX", t_policy="select", t_index=0, z_policy="max")
    assert out.shape == (4, 6, 7) and ax == "CYX"

    out, ax = prepare_image_for_patching(np.zeros((4, 6, 7)), "CYX", channel_policy="select", channel_index=2)
    assert out.shape == (6, 7) and ax == "YX"

    out, ax = prepare_image_for_patching(np.zeros((4, 6, 7)), "CYX", channel_policy="first")
    assert out.shape == (6, 7) and ax == "YX"

    out, ax = prepare_image_for_patching(np.zeros((4, 6, 7)), "CYX", channel_policy="mean")
    assert out.shape == (6, 7) and ax == "YX"


def test_prepare_image_for_patching_repeat_to_rgb_and_normalize_and_errors():
    out, ax = prepare_image_for_patching(np.zeros((6, 7)), "YX", channel_policy="repeat_to_rgb")
    assert out.shape == (6, 7, 3) and ax == "YXC"

    out, ax = prepare_image_for_patching(np.zeros((6, 7, 3)), "YXC", channel_policy="repeat_to_rgb")
    assert out.shape == (6, 7, 3) and ax == "YXC"

    out, _ = prepare_image_for_patching(np.array([[0.0, 1.0, 2.0]]), "YX", normalize="minmax")
    assert out.dtype == np.float32

    out, _ = prepare_image_for_patching(np.array([[0.0, 1.0, 2.0]]), "YX", normalize="percentile", percentile=(0, 100))
    assert out.dtype == np.float32

    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "ZYX", z_policy="all")
    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "ZYX", z_policy="select")
    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "TYX", t_policy="select")
    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "CYX", channel_policy="select")
    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "CYX", channel_policy="bad")
    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "ZYX", z_policy="bad")
    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "TYX", t_policy="bad")
    with pytest.raises(ValueError):
        prepare_image_for_patching(np.zeros((6, 7)), "YX", normalize="bad")


def test_prepare_labels_for_cell_patching_behaviors():
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)
    out, ax = prepare_labels_for_cell_patching(labels, "YX")
    assert out.shape == (2, 2) and ax == "YX"

    out, ax = prepare_labels_for_cell_patching(np.zeros((2, 4, 5), dtype=np.int32), "CYX", channel_policy="select", channel_index=0)
    assert out.shape == (4, 5) and ax == "YX"

    out, ax = prepare_labels_for_cell_patching(np.zeros((4, 5, 2), dtype=np.int32), "YXC", channel_policy="select", channel_index=1)
    assert out.shape == (4, 5) and ax == "YX"

    with pytest.raises(ValueError):
        prepare_labels_for_cell_patching(np.zeros((2, 3, 4, 5), dtype=np.int32), "CZYX")

    with pytest.raises(ValueError):
        prepare_labels_for_cell_patching(np.zeros((2, 4, 5), dtype=np.int32), "ZYX")

    out, ax = prepare_labels_for_cell_patching(np.zeros((2, 4, 5), dtype=np.int32), "ZYX", z_policy="select", z_index=1)
    assert out.shape == (4, 5) and ax == "YX"

    out, ax = prepare_labels_for_cell_patching(np.zeros((2, 4, 5), dtype=np.int32), "ZYX", z_policy="max")
    assert out.shape == (4, 5) and ax == "YX"

    out, ax = prepare_labels_for_cell_patching(np.zeros((1, 1, 4, 5), dtype=np.int32), "TZYX")
    assert out.shape == (4, 5) and ax == "YX"

    with pytest.raises(ValueError):
        prepare_labels_for_cell_patching(np.array([[0.0, 0.5], [1.0, 0.0]], dtype=float), "YX")

    out, ax = prepare_labels_for_cell_patching(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float), "YX")
    assert out.shape == (2, 2) and ax == "YX"

    with pytest.raises(ValueError):
        prepare_labels_for_cell_patching(np.array([[0, -1], [1, 0]], dtype=np.int32), "YX")

    with pytest.raises(ValueError):
        prepare_labels_for_cell_patching(np.zeros((2, 4, 5), dtype=np.int32), "ZYX", z_policy="mean")
    with pytest.raises(ValueError):
        prepare_labels_for_cell_patching(np.zeros((2, 4, 5), dtype=np.int32), "ZYX", z_policy="sum")
    with pytest.raises(ValueError):
        prepare_labels_for_cell_patching(np.zeros((2, 4, 5), dtype=np.int32), "ZYX", z_policy="all")
    with pytest.raises(ValueError):
        prepare_labels_for_cell_patching(np.zeros((2, 4, 5), dtype=np.int32), "TYX", t_policy="all")


def test_iter_image_planes_yx_single_plane_id():
    planes = list(iter_image_planes_for_patching(np.zeros((5, 6)), "YX"))
    assert len(planes) == 1
    assert isinstance(planes[0], PreparedPlane)
    assert planes[0].plane_id == "plane000"


def test_iter_image_planes_all_variants_and_order():
    z_planes = list(iter_image_planes_for_patching(np.zeros((3, 5, 6)), "ZYX", z_policy="all"))
    assert len(z_planes) == 3
    assert [p.metadata for p in z_planes] == [{"z": 0}, {"z": 1}, {"z": 2}]

    t_planes = list(iter_image_planes_for_patching(np.zeros((2, 5, 6)), "TYX", t_policy="all"))
    assert len(t_planes) == 2
    assert [p.metadata for p in t_planes] == [{"t": 0}, {"t": 1}]

    tz = list(iter_image_planes_for_patching(np.zeros((2, 3, 5, 6)), "TZYX", t_policy="all", z_policy="all"))
    assert len(tz) == 6
    assert [p.plane_id for p in tz][:3] == ["t000_z000", "t000_z001", "t000_z002"]
    assert [p.plane_id for p in tz][3:] == ["t001_z000", "t001_z001", "t001_z002"]


def test_iter_image_planes_projection_and_channels_and_errors():
    planes = list(iter_image_planes_for_patching(np.zeros((2, 3, 5, 6)), "TZYX", t_policy="all", z_policy="max"))
    assert len(planes) == 2 and all(p.axes == "YX" for p in planes)

    planes = list(iter_image_planes_for_patching(np.zeros((4, 3, 5, 6)), "CZYX", z_policy="all", channel_policy="keep"))
    assert len(planes) == 3 and all(p.axes == "CYX" for p in planes)

    planes = list(iter_image_planes_for_patching(np.zeros((5, 6)), "YX", channel_policy="repeat_to_rgb"))
    assert planes[0].axes == "YXC"

    with pytest.raises(ValueError):
        list(iter_image_planes_for_patching(np.zeros((3, 5, 6)), "ZYX"))
    with pytest.raises(ValueError):
        list(iter_image_planes_for_patching(np.zeros((3, 5, 6)), "ZYX", z_policy="bogus"))
    with pytest.raises(ValueError):
        list(iter_image_planes_for_patching(np.zeros((2, 5, 6)), "TYX", t_policy="bogus"))


def test_iter_cell_planes_all_and_axes_and_errors():
    pairs = list(iter_cell_planes_for_patching(np.zeros((2, 5, 6)), "ZYX", np.zeros((2, 5, 6), dtype=int), "ZYX", z_policy="all"))
    assert len(pairs) == 2
    assert isinstance(pairs[0], PreparedPlanePair)

    pairs = list(iter_cell_planes_for_patching(
        np.zeros((2, 3, 5, 6)), "TZYX", np.zeros((2, 3, 5, 6), dtype=int), "TZYX", t_policy="all", z_policy="all"
    ))
    assert len(pairs) == 6
    assert pairs[0].plane_id == "t000_z000"
    assert pairs[-1].plane_id == "t001_z002"

    pairs = list(iter_cell_planes_for_patching(
        np.zeros((4, 3, 5, 6)), "CZYX", np.zeros((3, 5, 6), dtype=int), "ZYX", z_policy="all", image_channel_policy="keep"
    ))
    assert all(p.image_axes == "CYX" and p.label_axes == "YX" for p in pairs)

    with pytest.raises(ValueError, match="z_policy='all' requires both image and labels to contain a Z axis"):
        list(iter_cell_planes_for_patching(np.zeros((2, 5, 6)), "ZYX", np.zeros((2, 5, 6), dtype=int), "CYX", z_policy="all"))
    with pytest.raises(ValueError, match="Z sizes must match"):
        list(iter_cell_planes_for_patching(np.zeros((2, 5, 6)), "ZYX", np.zeros((3, 5, 6), dtype=int), "ZYX", z_policy="all"))
    with pytest.raises(ValueError, match="T sizes must match"):
        list(iter_cell_planes_for_patching(np.zeros((2, 3, 5, 6)), "TZYX", np.zeros((3, 3, 5, 6), dtype=int), "TZYX", t_policy="all", z_policy="all"))
    with pytest.raises(ValueError, match="Spatial shape mismatch"):
        list(iter_cell_planes_for_patching(np.zeros((2, 5, 6)), "ZYX", np.zeros((2, 4, 6), dtype=int), "ZYX", z_policy="all"))


def test_iter_cell_planes_integration_run_patching_cell_mode():
    image = np.zeros((2, 16, 16), dtype=np.uint16)
    labels = np.zeros((2, 16, 16), dtype=np.uint16)
    labels[:, 4:8, 4:8] = 1
    pairs = list(iter_cell_planes_for_patching(image, "ZYX", labels, "ZYX", z_policy="all"))
    images = [p.image for p in pairs]
    labels_2d = [p.labels for p in pairs]

    assert len({p.image_axes for p in pairs}) == 1

    cfg = CellPatcherConfig(mode="cell")
    cfg.common = CommonConfig(image_axes=pairs[0].image_axes, mask_axes="YX", extract_arrays=True)
    cfg.cell = CellConfig(patch_size="auto", exclude_border=False)

    # The planes' metadata gives each plane's z, which tells their patches apart.
    metadata = [{**p.metadata, "position_id": "P"} for p in pairs]
    result = run_patching(images=images, masks=labels_2d, config=cfg, image_metadata=metadata)
    assert len(result.patches) > 0
    assert result.patch_arrays is not None
    assert [patch.z0 for patch in result.patches] == [0, 1]


def test_percentile_and_minmax_dtype_override():
    arr = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    p_out = percentile_normalize(arr, p_low=0, p_high=100, dtype=np.float64)
    m_out = minmax_normalize(arr, dtype=np.float64)
    assert p_out.dtype == np.float64
    assert m_out.dtype == np.float64


def test_select_axis_index_does_not_squeeze_other_axes():
    arr = np.zeros((2, 1, 5, 6))  # CZYX
    out, ax = select_axis_index(arr, "CZYX", "C", 1)
    assert out.shape == (1, 5, 6)
    assert ax == "ZYX"


def test_iter_image_plane_id_stability_for_single_axis_all():
    z_planes = list(iter_image_planes_for_patching(np.zeros((2, 5, 6)), "ZYX", z_policy="all"))
    assert [p.plane_id for p in z_planes] == ["z000", "z001"]

    t_planes = list(iter_image_planes_for_patching(np.zeros((2, 5, 6)), "TYX", t_policy="all"))
    assert [p.plane_id for p in t_planes] == ["t000", "t001"]


def test_iter_image_mixed_policy_metadata_semantics():
    planes = list(iter_image_planes_for_patching(np.zeros((2, 3, 5, 6)), "TZYX", t_policy="all", z_policy="max"))
    assert [p.metadata for p in planes] == [{"t": 0}, {"t": 1}]


def test_prepare_labels_channel_error_message_clear():
    with pytest.raises(ValueError, match=r"channel_policy='error'"):
        prepare_labels_for_cell_patching(np.zeros((2, 5, 6), dtype=np.int32), "CYX", channel_policy="error")


def test_prepare_image_channel_select_missing_index_error_message():
    with pytest.raises(ValueError, match=r"channel_policy='select' requires channel_index"):
        prepare_image_for_patching(np.zeros((2, 6, 7)), "CYX", channel_policy="select")


def test_prepare_image_all_policy_error_message_points_to_iterator():
    with pytest.raises(ValueError, match="iter_image_planes_for_patching"):
        prepare_image_for_patching(np.zeros((2, 5, 6)), "ZYX", z_policy="all")


def test_prepare_labels_all_policy_error_message_points_to_iterator():
    with pytest.raises(ValueError, match="iter_cell_planes_for_patching"):
        prepare_labels_for_cell_patching(np.zeros((2, 5, 6), dtype=np.int32), "ZYX", z_policy="all")


def test_iter_image_planes_tzyx_t_all_z_first_yields_t_planes():
    """TZYX + t_policy='all' + z_policy='first' yields T planes; only t in metadata (no z)."""
    planes = list(
        iter_image_planes_for_patching(np.zeros((2, 3, 5, 6)), "TZYX", t_policy="all", z_policy="first")
    )
    assert len(planes) == 2
    assert all(p.axes == "YX" for p in planes)
    assert [p.metadata for p in planes] == [{"t": 0}, {"t": 1}]
    assert [p.plane_id for p in planes] == ["t000", "t001"]


def test_iter_cell_planes_t_all_only_yields_t_planes():
    """TZYX + t_policy='all', z_policy='first' yields T paired planes with t metadata only."""
    image = np.zeros((2, 3, 5, 6))
    labels = np.zeros((2, 3, 5, 6), dtype=np.int32)
    pairs = list(iter_cell_planes_for_patching(image, "TZYX", labels, "TZYX", t_policy="all", z_policy="first"))
    assert len(pairs) == 2
    assert [p.plane_id for p in pairs] == ["t000", "t001"]
    assert all(p.image_axes == "YX" and p.label_axes == "YX" for p in pairs)
    assert [p.metadata for p in pairs] == [{"t": 0}, {"t": 1}]
