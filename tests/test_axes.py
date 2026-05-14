import numpy as np
import pytest

from cellpatcher.axes import axis_index, axes_dict, mapping_to_axes, move_axes, normalize_axes, require_spatial_axes


def test_normalize_axes_uppercasses_axes():
    assert normalize_axes(" yx ") == "YX"


def test_normalize_axes_rejects_duplicate_axes():
    with pytest.raises(ValueError):
        normalize_axes("YYX")


def test_normalize_axes_rejects_unsupported_axes():
    with pytest.raises(ValueError):
        normalize_axes("AB")


def test_normalize_axes_validates_ndim():
    with pytest.raises(ValueError):
        normalize_axes("YX", ndim=3)


def test_axis_index_returns_expected_index():
    assert axis_index("TCYX", "Y") == 2


def test_axis_index_returns_none_for_missing_axis():
    assert axis_index("YX", "C") is None


def test_axes_dict_returns_all_keys():
    d = axes_dict("CYX")
    assert set(d.keys()) == {"T", "C", "Z", "Y", "X"}
    assert d["C"] == 0
    assert d["T"] is None


def test_require_spatial_axes_returns_indices():
    assert require_spatial_axes("CYX") == (1, 2)


def test_require_spatial_axes_rejects_missing_spatial_axes():
    with pytest.raises(ValueError):
        require_spatial_axes("TCZ")


def test_move_axes_reorders_array_from_cyx_to_yxc():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    moved = move_axes(arr, "CYX", "YXC")
    assert moved.shape == (3, 4, 2)
    assert np.array_equal(moved, np.transpose(arr, (1, 2, 0)))



def test_mapping_to_axes_cyx():
    assert mapping_to_axes({0: "C", 1: "Y", 2: "X"}) == "CYX"


def test_mapping_to_axes_tczyx():
    assert mapping_to_axes({0: "T", 1: "C", 2: "Z", 3: "Y", 4: "X"}) == "TCZYX"


def test_mapping_to_axes_ignores_other_and_none_and_missing_indices():
    assert mapping_to_axes({0: "Other", 1: "Y", 2: "X"}) == "YX"
    assert mapping_to_axes({0: None, 1: "Y", 2: "X"}) == "YX"
    assert mapping_to_axes({1: "Y", 2: "X"}) == "YX"


def test_mapping_to_axes_rejects_duplicate_labels():
    with pytest.raises(ValueError):
        mapping_to_axes({0: "Y", 1: "Y"})


def test_mapping_to_axes_rejects_unsupported_label():
    with pytest.raises(ValueError):
        mapping_to_axes({0: "A", 1: "X"})


def test_mapping_to_axes_rejects_out_of_range_indices_with_ndim():
    with pytest.raises(ValueError):
        mapping_to_axes({0: "Y", 2: "X"}, ndim=2)


def test_mapping_to_axes_raises_when_empty():
    with pytest.raises(ValueError):
        mapping_to_axes({0: "Other", 1: None})
