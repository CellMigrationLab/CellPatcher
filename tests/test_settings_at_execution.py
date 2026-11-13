"""Every scientific setting is re-checked where a run executes, without coercion (release blocker 38).

``int(True)`` is 1, ``int("1")`` is 1 and ``int(1.9)`` is 1, so an index read through ``int()``
could pick the wrong plane, and the normalization settings were checked at construction only, so
``cfg.normalization.per_time_point = 1`` (set later) went through as a truthy flag. The config
dataclasses are mutable and documented as edited in place, so each setting below is mutated after
construction and must be refused when the run executes.
"""

import dataclasses

import numpy as np
import pytest

from cellpatcher import (
    CellConfig,
    CellPatcherConfig,
    NormalizationConfig,
    compute_threshold,
    iter_patch_arrays,
    patch_array,
    run_patching,
    scan_cell_objects,
)
from cellpatcher.config import (
    CommonConfig,
    ExportConfig,
    ExtractionConfig,
    FilterConfig,
    GridConfig,
    MatchConfig,
    RandomConfig,
    WholeFOVConfig,
)
from cellpatcher.strategies.cell import CellSegmentationStrategy

IMAGE = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
META = [{"position_id": "fov"}]


def _two_layer_labels():
    """Layer 0 has one object (label 7), layer 1 has two (labels 3 and 4)."""
    labels = np.zeros((2, 64, 64), dtype=np.int32)
    labels[0, 10:20, 10:20] = 7
    labels[1, 10:20, 10:20] = 3
    labels[1, 30:40, 30:40] = 4
    return labels


def _cell_config():
    config = CellPatcherConfig(mode="cell", cell=CellConfig(patch_size=8, exclude_border=False))
    config.common.mask_axes = "CYX"
    return config


def test_a_label_mask_of_several_layers_is_refused_wherever_it_is_read():
    # Owner decision of 13 November 2026: a mask layer is never chosen by its position (cell.mask_channel was
    # removed); a label mask has one layer.
    with pytest.raises(ValueError, match="single-layer label mask"):
        run_patching(images=[IMAGE], masks=[_two_layer_labels()], config=_cell_config(), image_metadata=META)
    cell = CellConfig(patch_size=8)
    with pytest.raises(ValueError, match="single-layer label mask"):
        CellSegmentationStrategy(cell, mask_axes="CYX").generate_regions(
            image_shape=(64, 64), labels=_two_layer_labels(), image_id="image_0000"
        )
    with pytest.raises(ValueError, match="single-layer label mask"):
        scan_cell_objects(image_shape=(64, 64), labels=_two_layer_labels(), config=cell, mask_axes="CYX")


def test_a_label_mask_of_one_layer_along_c_is_its_labels():
    labels = _two_layer_labels()[1:]
    result = run_patching(images=[IMAGE], masks=[labels], config=_cell_config(), image_metadata=META)
    assert {patch.label_id for patch in result.patches} == {3, 4}


def _normalization_config(setting, value, **fixed):
    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 16
    for name, fixed_value in fixed.items():
        setattr(config.normalization, name, fixed_value)
    setattr(config.normalization, setting, value)
    return config


@pytest.mark.parametrize("bad", [1, 0, "true", "False", 1.0, None, np.True_, [True]], ids=repr)
@pytest.mark.parametrize("enabled", [False, True])
def test_per_time_point_edited_after_construction_is_refused_at_execution(bad, enabled):
    # Former code: a truthy 1 or "false" was a flag when the run used it, with or without normalization.
    config = _normalization_config("per_time_point", bad, enabled=enabled, scope="patch")
    with pytest.raises(ValueError, match="per_time_point"):
        run_patching(images=[IMAGE], config=config, image_metadata=META)


@pytest.mark.parametrize("bad", [1, "true", None, np.True_], ids=repr)
def test_the_normalization_flag_edited_after_construction_is_refused_at_execution(bad):
    with pytest.raises(ValueError, match="enabled"):
        run_patching(images=[IMAGE], config=_normalization_config("enabled", bad), image_metadata=META)


@pytest.mark.parametrize("bad", [True, 1.5, "1", 0, -5, 2.0**30, None, np.float64(4.0)], ids=repr)
def test_the_normalization_memory_limit_edited_after_construction_is_refused(bad):
    with pytest.raises(ValueError, match="memory_limit"):
        run_patching(images=[IMAGE], config=_normalization_config("memory_limit", bad), image_metadata=META)


@pytest.mark.parametrize(
    ("setting", "bad"),
    [
        ("percentile", (True, 99.8)), ("percentile", ("1", 99.8)), ("percentile", [1.0]), ("percentile", "1,99"),
        ("clip", (0, "1")), ("clip", (False, 1)), ("clip", (1, 0)), ("clip", "01"),
        ("method", 1), ("method", ["percentile"]), ("method", None),
        ("scope", 1), ("scope", ["image"]), ("scope", None),
        ("pooling", 1), ("image_type", 1), ("patch_region", 1), ("export_values", 1),
    ],
    ids=repr,
)
def test_the_other_normalization_settings_edited_after_construction_are_refused(setting, bad):
    with pytest.raises(ValueError, match=setting.split("_")[0]):
        run_patching(images=[IMAGE], config=_normalization_config(setting, bad), image_metadata=META)


def test_valid_normalization_settings_are_kept_as_given():
    settings = NormalizationConfig(percentile=[1, 99], clip=[0, 1], memory_limit=np.int64(10**9), per_time_point=True)
    assert settings.percentile == (1, 99)
    assert settings.clip == (0, 1)
    assert settings.memory_limit == 10**9


# Every field of every section and a value of the wrong type for it, assigned after construction. The
# test below fails when a field is added to a section without a line here, so a new setting cannot
# skip the execution-time check.
SECTIONS = {
    "common": CommonConfig,
    "cell": CellConfig,
    "random": RandomConfig,
    "whole_fov": WholeFOVConfig,
    "grid": GridConfig,
    "filters": FilterConfig,
    "export": ExportConfig,
    "extraction": ExtractionConfig,
    "matching": MatchConfig,
    "normalization": NormalizationConfig,
}
FLAG = [1, 0, "True", None, np.True_]
CHOICE = [1, ["x"], 1.5, True]
TEXT = [5, ["x"], True, 1.5]
INTEGER = [True, 3.0, "3", 2.5, np.float64(3.0)]
NUMBER = [True, "0.5", [1], np.True_]
BAD_VALUES = {
    "mode": CHOICE + [None],
    "common.image_axes": TEXT + [None],
    "common.mask_axes": TEXT + [None],
    "common.output_dir": [5, True, ["x"]],
    "common.extract_arrays": FLAG,
    "common.preprocessing_run": [5, True, ["x"]],
    "common.condition": TEXT,
    "common.biological_repeat": TEXT,
    "common.technical_repeat": TEXT,
    "common.z_policy": CHOICE + [None],
    "common.z_index": INTEGER + [-1],
    "common.t_policy": CHOICE + [None],
    "common.t_index": INTEGER + [-1],
    "common.object_set": TEXT,
    "cell.patch_size": [True, 8.0, "8px", 0, -4, None],
    "cell.center_on": CHOICE + [None],
    "cell.exclude_border": FLAG,
    "cell.min_area": INTEGER + [0, None],
    "cell.max_area": INTEGER,
    "cell.binary_mask_policy": CHOICE + ["single_object"],  # None is no policy; single_object was removed
    "cell.binary_mask_connectivity": INTEGER + [0, 3, None],
    "random.patch_size": INTEGER + [0, None],
    "random.n_patches": INTEGER + [0, None],
    "random.seed": INTEGER,
    "random.seed_scope": CHOICE + [None],
    "random.min_mask_fraction": NUMBER + [1.5, float("nan")],
    "whole_fov.square_cover": FLAG,
    "grid.tile_size": INTEGER + [0, None],
    "grid.overlap": NUMBER + [1, -0.1, None],
    "grid.include_partial": FLAG,
    "grid.partial_mode": CHOICE + [None],
    "filters.min_foreground_fraction": NUMBER + [1.5],
    "filters.foreground_threshold": NUMBER + [float("nan")],
    "filters.min_mean_intensity": NUMBER + [float("inf")],
    "export.enabled": FLAG,
    "export.save_metadata_csv": FLAG,
    "export.image_format": CHOICE + [None, "png"],
    "export.save_run_config": FLAG,
    "extraction.pad_mode": CHOICE + [None],
    "extraction.constant_values": NUMBER + [None],
    "extraction.output_size": INTEGER + [(8, True), (8, "8"), (8,), 0, ["8", 8]],
    "extraction.resize_order": INTEGER + [-1, None],
    "extraction.apply_mask": FLAG,
    "extraction.mask_fill_value": NUMBER + [None],
    "extraction.split_channels": [1, "True", None, [], [True], [1.0], ["1", 1], [""], [0, 0], [-1]],
    "matching.mask_type": [5, True, "", "  ", ["cells"]],
    "matching.mask_id": [5, True, "", "  ", ["Nuclei"]],
    "normalization.enabled": FLAG,
    "normalization.plane_histograms": [5, True, ["x"]],
    "normalization.manifest": [5, True, ["x"]],
    "normalization.experiment_root": [5, True, ["x"]],
    "normalization.masks_root": [5, True, ["x"]],
    "normalization.method": CHOICE + [None],
    "normalization.percentile": [(True, 99.8), ("1", 99.8), [1.0], "1,99", None],
    "normalization.clip": [(0, "1"), (False, 1), (1, 0), "01", [1]],
    "normalization.scope": CHOICE + [None],
    "normalization.per_time_point": FLAG,
    "normalization.patch_region": CHOICE,
    "normalization.export_values": CHOICE + [None],
    "normalization.memory_limit": INTEGER + [0, None],
    "normalization.pooling": CHOICE + [None],
    "normalization.image_type": CHOICE + [None],
}
CASES = [(name, value) for name, values in BAD_VALUES.items() for value in values]


def _valid_config():
    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 16
    return config


def _with_setting(name, value):
    """A valid grid config with the setting assigned after construction (``__post_init__`` never sees it)."""
    config = _valid_config()
    section, _, setting = name.rpartition(".")
    setattr(getattr(config, section) if section else config, setting, value)
    return config


def test_every_setting_of_every_section_is_in_the_table():
    fields = {"mode"} | {f"{section}.{f.name}" for section, cls in SECTIONS.items() for f in dataclasses.fields(cls)}
    assert fields == set(BAD_VALUES)
    assert {section for section in SECTIONS} == {f.name for f in dataclasses.fields(CellPatcherConfig)} - {"mode"}


@pytest.mark.parametrize(("name", "value"), CASES, ids=repr)
def test_a_setting_of_the_wrong_type_edited_after_construction_is_refused_by_run_patching(name, value):
    with pytest.raises(ValueError, match=name.rpartition(".")[2].split("_")[0]):
        run_patching(images=[IMAGE], config=_with_setting(name, value), image_metadata=META)


# Identity labels are stamped by a run (_run_labels), never read by delivery.
LABELS = ("common.condition", "common.biological_repeat", "common.technical_repeat")
DELIVERY_CASES = [case for case in CASES if case[0] not in LABELS]


@pytest.mark.parametrize(("name", "value"), DELIVERY_CASES, ids=repr)
def test_a_setting_of_the_wrong_type_edited_after_construction_is_refused_by_lazy_delivery(name, value):
    patches = run_patching(images=[IMAGE], config=_valid_config(), image_metadata=META).patches
    with pytest.raises(ValueError, match=name.rpartition(".")[2].split("_")[0]):
        next(iter_patch_arrays(patches, [IMAGE], _with_setting(name, value)))


@pytest.mark.parametrize(("name", "value"), DELIVERY_CASES, ids=repr)
def test_a_setting_of_the_wrong_type_edited_after_construction_is_refused_by_patch_array(name, value):
    # Former code: patch_array (one patch, lazily) used the edited setting without a check.
    patches = run_patching(images=[IMAGE], config=_valid_config(), image_metadata=META).patches
    with pytest.raises(ValueError, match=name.rpartition(".")[2].split("_")[0]):
        patch_array(patches[0], IMAGE, _with_setting(name, value))


def _region_values(config, masks=None):
    patches = run_patching(images=[IMAGE], masks=masks, config=config, image_metadata=META).patches
    return [[(name, type(value), value) for name, value in sorted(vars(patch).items())] for patch in patches]


def test_numpy_integer_sizes_give_the_patches_of_python_integers():
    # Former code of this change: an np.int64 patch_size made np.int64 coordinates and np.bool_ flags.
    labels = _two_layer_labels()[1]
    for section, settings in (("cell", {"patch_size": 8}), ("random", {"patch_size": 8, "n_patches": 5, "seed": 2})):
        python = CellPatcherConfig(mode=section)
        numpy = CellPatcherConfig(mode=section)
        for name, value in settings.items():
            setattr(getattr(python, section), name, value)
            setattr(getattr(numpy, section), name, np.int64(value))
        masks = [labels] if section == "cell" else None
        assert _region_values(numpy, masks) == _region_values(python, masks)


@pytest.mark.parametrize(
    ("percentile", "clip"),
    [
        ((1, 99), (0, 1)),
        ((np.float32(1.0), np.int64(99)), (np.float32(0.0), np.float64(1.0))),
        ([1.0, 99.0], [0.0, 1.0]),
    ],
    ids=repr,
)
def test_percentile_and_clip_are_recorded_as_python_floats(percentile, clip):
    # run_config.json is written with json.dump(default=str): an np.float32 became the text "1.0".
    normalization = NormalizationConfig(percentile=percentile, clip=clip)
    assert normalization.percentile == (1.0, 99.0) and normalization.clip == (0.0, 1.0)
    assert all(type(value) is float for value in normalization.percentile + normalization.clip)


@pytest.mark.parametrize("threshold", [100, np.int64(100), np.float32(100.0), 100.0], ids=repr)
def test_a_numeric_threshold_is_returned_as_a_python_float(threshold):
    value = compute_threshold(IMAGE, threshold)
    assert type(value) is float and value == 100.0


def test_a_valid_config_is_unchanged_by_the_execution_time_checks():
    patches = run_patching(images=[IMAGE], config=_valid_config(), image_metadata=META).patches
    assert len(patches) == 16
    assert len(list(iter_patch_arrays(patches, [IMAGE], _valid_config()))) == 16


@pytest.mark.parametrize(
    ("name", "value"),
    [("normalization.per_time_point", 1), ("normalization.memory_limit", "4"), ("cell.patch_size", 8.0),
     ("random.min_mask_fraction", True), ("extraction.pad_mode", 1)],
    ids=repr,
)
def test_the_folder_route_refuses_an_edited_setting_before_any_file_is_read(tmp_path, name, value):
    from cellpatcher.orchestration import run_patching_pipeline

    config = _with_setting(name, value)
    config.common.preprocessing_run = str(tmp_path / "no_such_run")
    with pytest.raises(ValueError, match=name.rpartition(".")[2].split("_")[0]):
        run_patching_pipeline(tmp_path, config=config)
