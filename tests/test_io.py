import re

import numpy as np
import pytest

from cellpatcher.io import (
    list_supported_files,
    read_array,
    validate_image_and_mask_folders,
    write_array,
)
from tests.helpers import write_tiff


def test_list_supported_files_sorted_and_filtered(tmp_path):
    (tmp_path / "b.TIFF").write_bytes(b"x")
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("x")
    files = list_supported_files(tmp_path)
    assert [p.name for p in files] == ["a.tif", "b.TIFF"]


def test_list_supported_files_refuses_the_image_formats_no_longer_read(tmp_path):
    # Owner decision of 2026-10-30 (#45): images are TIFF or OME-Zarr, as the preprocessing
    # brick discovers them. The former formats are refused by name, never skipped.
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "b.npy").write_bytes(b"x")
    (tmp_path / "c.PNG").write_bytes(b"x")
    with pytest.raises(ValueError, match=r"2 image\(s\) in a format CellPatcher no longer reads: b.npy, c.PNG"):
        list_supported_files(tmp_path)


def test_list_supported_files_rejects_missing_folder(tmp_path):
    with pytest.raises(ValueError):
        list_supported_files(tmp_path / "missing")


def test_read_write_array_npy_roundtrip(tmp_path):
    arr = np.arange(6).reshape(2, 3)
    p = tmp_path / "arr.npy"
    write_array(p, arr)
    out = read_array(p)
    assert np.array_equal(out, arr)


def test_read_array_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "arr.xyz"
    p.write_text("x")
    with pytest.raises(ValueError):
        read_array(p)


def test_write_array_rejects_unsupported_extension(tmp_path):
    with pytest.raises(ValueError):
        write_array(tmp_path / "arr.xyz", np.zeros((2, 2)))


def test_validate_image_and_mask_folders_empty_image_folder_raises(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(mask_dir / "a_mask.tif", np.ones((4, 4), np.uint8))
    with pytest.raises(ValueError, match="image folder"):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_validate_image_and_mask_folders_empty_mask_folder_raises(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    (img_dir / "a.tif").write_bytes(b"x")
    with pytest.raises(ValueError, match="No mask in .* pairs with an image"):
        validate_image_and_mask_folders(img_dir, mask_dir)


# ---------------------------------------------------------------------------
# Additional IO edge cases
# ---------------------------------------------------------------------------

from cellpatcher.io import read_array_shape


def test_read_array_rejects_nonexistent_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        read_array(tmp_path / "missing.npy")


def test_read_array_shape_returns_correct_shape(tmp_path):
    arr = np.arange(12).reshape(3, 4)
    p = tmp_path / "a.npy"
    write_array(p, arr)
    shape = read_array_shape(p)
    assert shape == (3, 4)


def test_write_read_tiff_roundtrip(tmp_path):
    """Write and read back a TIFF file using tifffile."""
    arr = np.arange(9).reshape(3, 3).astype(np.uint16)
    p = tmp_path / "img.tif"
    try:
        write_array(p, arr)
        loaded = read_array(p)
        assert np.array_equal(loaded, arr)
    except ImportError:
        pytest.skip("tifffile not available")


def test_list_supported_files_with_custom_extensions(tmp_path):
    (tmp_path / "a.npy").write_bytes(b"x")
    (tmp_path / "b.csv").write_text("x")
    files = list_supported_files(tmp_path, supported_extensions={".csv"})
    assert [p.name for p in files] == ["b.csv"]


from cellpatcher import CellPatcherConfig
from cellpatcher.config import CommonConfig, GridConfig
from cellpatcher.io import run_patching_from_folders


def test_run_patching_from_folders_grid_mode(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.arange(64).reshape(8, 8))

    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 4

    result = run_patching_from_folders(img_dir, config=config)
    assert len(result.patches) == 4


def test_run_patching_from_folders_cell_mode(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((16, 16), dtype=np.float32)
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:12, 5:13] = 1

    write_array(img_dir / "sample.tif", image)
    write_array(mask_dir / "sample_mask.tif", labels)

    config = CellPatcherConfig(mode="cell")
    config.cell.patch_size = 8

    result = run_patching_from_folders(img_dir, mask_folder_path=mask_dir, config=config)
    assert len(result.patches) == 1


def test_run_patching_from_folders_cell_requires_mask_folder(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.zeros((8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="mask_folder_path"):
        run_patching_from_folders(img_dir, config=CellPatcherConfig(mode="cell"))

from cellpatcher.orchestration import (
    load_config,
    run_patching_condition_repeat_tree,
    run_patching_pipeline,
    save_default_config,
)


def test_run_patching_pipeline_alias_grid_mode(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.arange(64).reshape(8, 8))

    config = CellPatcherConfig(mode="grid")
    config.grid.tile_size = 4
    result = run_patching_pipeline(img_dir, config=config)
    assert len(result.patches) == 4


def test_run_patching_pipeline_grid_raises_if_mask_folder_provided(tmp_path):
    """grid mode must not silently ignore a provided mask folder."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(img_dir / "a.tif", np.zeros((8, 8)))
    write_array(mask_dir / "a_mask.tif", np.zeros((8, 8), dtype=np.int32))

    config = CellPatcherConfig(mode="grid")
    with pytest.raises(ValueError, match="mask_folder_path"):
        run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)


def test_run_patching_pipeline_whole_fov_raises_if_mask_folder_provided(tmp_path):
    """whole_fov mode must not silently ignore a provided mask folder."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(img_dir / "a.tif", np.zeros((8, 8)))
    write_array(mask_dir / "a_mask.tif", np.zeros((8, 8), dtype=np.int32))

    config = CellPatcherConfig(mode="whole_fov")
    with pytest.raises(ValueError, match="mask_folder_path"):
        run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)


def test_run_patching_pipeline_random_mode_with_optional_mask(tmp_path):
    """random mode accepts an optional mask folder to restrict sampling region."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    img = np.zeros((32, 32), dtype=np.float32)
    # mask restricts sampling to top half
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[:16, :] = 1
    write_array(img_dir / "a.tif", img)
    write_array(mask_dir / "a_mask.tif", mask)

    config = CellPatcherConfig(mode="random")
    config.random.patch_size = 4
    config.random.n_patches = 5
    config.random.seed = 0
    result = run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)
    assert len(result.patches) == 5


def test_run_patching_pipeline_random_mode_without_mask(tmp_path):
    """random mode works without a mask folder."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.zeros((32, 32), dtype=np.float32))

    config = CellPatcherConfig(mode="random")
    config.random.patch_size = 4
    config.random.n_patches = 3
    result = run_patching_pipeline(img_dir, config=config)
    assert len(result.patches) == 3


def test_save_and_load_default_config(tmp_path):
    config_path = tmp_path / "cellpatcher_config.json"
    save_default_config(config_path)
    loaded = load_config(config_path)
    assert isinstance(loaded, CellPatcherConfig)
    assert loaded.mode == "cell"


def test_load_config_nested_types_are_dataclasses(tmp_path):
    """load_config must reconstruct all nested sub-configs as dataclass instances."""
    config_path = tmp_path / "cfg.json"
    save_default_config(config_path)
    loaded = load_config(config_path)

    assert isinstance(loaded.common, CommonConfig)
    assert isinstance(loaded.grid, GridConfig)
    # Nested field access must work without AttributeError
    assert loaded.common.image_axes == "YX"
    assert loaded.grid.tile_size == 128


def test_load_config_is_usable_for_patching(tmp_path):
    """A config saved and loaded must work end-to-end with run_patching_pipeline."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    write_array(img_dir / "a.tif", np.arange(64).reshape(8, 8))

    config_path = tmp_path / "cfg.json"
    save_default_config(config_path)
    loaded = load_config(config_path)
    # Switch to grid so no mask folder is needed
    loaded.mode = "grid"
    loaded.grid.tile_size = 4

    result = run_patching_pipeline(img_dir, config=loaded)
    assert len(result.patches) == 4


# ---------------------------------------------------------------------------
# Issue #45/#46: ambiguous pairs are refused; system files are skipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["._a.tif", ".hidden.tif"])
def test_list_supported_files_skips_system_files(tmp_path, name):
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_supported_files(tmp_path)] == ["a.tif"]


def _folders(tmp_path, images, masks):
    """An image folder and a mask folder beside it, with real (tiny) files."""
    img_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    for name in images:
        write_array(img_dir / name, np.zeros((4, 4), np.uint8))
    for name in masks:
        (mask_dir / name).parent.mkdir(parents=True, exist_ok=True)
        write_array(mask_dir / name, np.ones((4, 4), np.uint16))
    return img_dir, mask_dir


def _preprocessing_pairs(img_dir, mask_dir):
    """The preprocessing brick's own pairing of the same folders, run as the brick runs it.

    ``discover_experiment`` + ``inspect_images`` + ``find_masks`` on TIFF
    images: ``{image name: (mask type, mask path)}``, or the error codes.
    """
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.metadata import inspect_images

    discovery = mbp.discover_experiment(img_dir)
    images = inspect_images(img_dir, discovery.images).files
    inventory = mbp.find_masks(mask_dir, discovery, images, ome_zarr_labels=False)
    errors = sorted(issue.code for issue in inventory.issues if issue.is_error)
    if errors:
        return errors
    return {f.image: (f.mask_type, f.path) for f in inventory.files}


# ---------------------------------------------------------------------------
# Issue #45: CellPatcher pairs images and masks with the preprocessing brick's
# rule (find_masks), so both bricks pair the same files. Each case below was
# paired differently by CellPatcher's former rule (exact_or_suffix with the
# suffixes _mask/_labels/_label/_seg/_segmentation), shown in the comment.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("masks", "mask_type", "expected"),
    [
        # Any mask type after "<image>_": former rule: "Images with no mask match".
        (["a_cells.tif"], None, {"a.tif": ("cells", "a_cells.tif")}),
        # The <mask type>/ folder layout: former rule: no mask files found (no recursion).
        (["nuclei/a.tif"], None, {"a.tif": ("nuclei", "nuclei/a.tif")}),
        # Several types, one chosen: former rule: refused unless ignore_unrequested_masks.
        (["a_cells.tif", "a_nuclei.tif"], "nuclei", {"a.tif": ("nuclei", "a_nuclei.tif")}),
    ],
)
def test_pairing_is_the_preprocessing_bricks(tmp_path, masks, mask_type, expected):
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], masks)
    preprocessing = _preprocessing_pairs(img_dir, mask_dir)
    assert {image: pair for image, pair in preprocessing.items() if mask_type in (None, pair[0])} == expected
    options = {} if mask_type is None else {"mask_type": mask_type}
    out = validate_image_and_mask_folders(img_dir, mask_dir, **options)
    assert {image.name: mask.relative_to(mask_dir).as_posix() for image, mask in out.items()} == {
        image: path for image, (_, path) in expected.items()
    }


def test_a_mask_name_starting_with_another_images_name_is_refused_like_preprocessing(tmp_path):
    # a_seg_mask.tif is "a" + "_seg_mask" and "a_seg" + "_mask". The former rule
    # paired it with a_seg (and a_mask.tif with a); preprocessing refuses it as
    # ambiguous, and so does CellPatcher now.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif", "a_seg.tif"], ["a_mask.tif", "a_seg_mask.tif"])
    assert _preprocessing_pairs(img_dir, mask_dir) == ["ambiguous_mask"]
    with pytest.raises(ValueError, match=r"\[ambiguous_mask\].*a_seg_mask.tif"):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_a_mask_named_exactly_like_its_image_no_longer_pairs(tmp_path):
    # Former rule ("exact" and the default "exact_or_suffix"): masks/a.tif paired with a.tif.
    # Preprocessing reads it as no image's mask (a mask has a type), so it is refused, never dropped.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a.tif"])
    assert _preprocessing_pairs(img_dir, mask_dir) == {}
    with pytest.raises(ValueError, match=r"\[mask_without_image\].*\(a.tif\)"):
        validate_image_and_mask_folders(img_dir, mask_dir)


def test_masks_that_are_not_tiff_files_are_refused_not_dropped(tmp_path):
    # Former rule: a_mask.npy paired with a.tif. Preprocessing pairs TIFF masks only.
    img_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    write_array(img_dir / "a.tif", np.zeros((4, 4), np.float32))
    write_array(mask_dir / "a_mask.npy", np.ones((4, 4), np.uint16))
    with pytest.raises(ValueError, match=r"\[mask_outside_type_folder\].*a_mask.npy"):
        validate_image_and_mask_folders(img_dir, mask_dir)


@pytest.mark.parametrize("name", ["b.png", "b.jpg", "b.jpeg", "b.bmp", "b.gif", "b.npy", "B.PNG"])
def test_images_in_formats_other_than_tiff_and_ome_zarr_are_refused_by_name(tmp_path, name):
    # Owner decision of 2026-10-30 (#45): CellPatcher finds images with the preprocessing
    # brick's discover_experiment, which reads TIFF and OME-Zarr only. The former formats
    # (read before as images) are refused, naming the files; never skipped. Former code:
    # paired b.<ext> with b_cells.tif and patched it.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif", "b_cells.tif"])
    write_array(img_dir / name, np.zeros((4, 4), np.uint8))
    refusal = rf"1 image\(s\) in a format CellPatcher no longer reads: {name}\..*TIFF files .* OME-Zarr"
    with pytest.raises(ValueError, match=refusal):
        validate_image_and_mask_folders(img_dir, mask_dir)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=4))
    with pytest.raises(ValueError, match=refusal):
        run_patching_pipeline(img_dir, config=config)



@pytest.mark.parametrize("name", ["b.czi", "b.nd2", "b.lif", "b.LSM", "b.webp"])
def test_microscope_and_other_image_formats_are_refused_by_name_not_skipped(tmp_path, name):
    # Owner decision of 2026-10-30: only TIFF and OME-Zarr images are found; other image
    # formats are refused by file name, never skipped. Former code (this branch before the
    # review): a .czi beside the TIFF images was silently left out of the run.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif"])
    (img_dir / name).write_bytes(b"not decoded")
    refusal = rf"1 image\(s\) in a format CellPatcher no longer reads: {name}\..*Convert them to TIFF or OME-Zarr"
    with pytest.raises(ValueError, match=refusal):
        validate_image_and_mask_folders(img_dir, mask_dir)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=4))
    with pytest.raises(ValueError, match=refusal):
        run_patching_pipeline(img_dir, config=config)
    (img_dir / name).unlink()
    (img_dir / "notes.txt").write_text("not an image")  # other files stay ignored
    assert validate_image_and_mask_folders(img_dir, mask_dir)

def test_a_folder_run_refuses_images_beside_subfolders(tmp_path):
    # Owner decision of 2026-10-30: a folder holding images and subfolders is refused in
    # every brick (the preprocessing brick's "tiff_outside_leaf"). Former code: a folder run
    # patched a.tif and ignored the subfolder.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif", "more/b.tif"], ["a_cells.tif"])
    with pytest.raises(ValueError, match=r"\[tiff_outside_leaf\].*\(a.tif\)"):
        validate_image_and_mask_folders(img_dir, mask_dir)
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=4))
    with pytest.raises(ValueError, match=r"\[tiff_outside_leaf\].*\(a.tif\)"):
        run_patching_pipeline(img_dir, config=config)


def test_a_folder_run_refuses_a_folder_of_image_subfolders(tmp_path):
    img_dir, _ = _folders(tmp_path, ["ctrl/a.tif", "treated/b.tif"], [])
    config = CellPatcherConfig(mode="grid", grid=GridConfig(tile_size=4))
    with pytest.raises(ValueError, match=r"holds its images in subfolders \(ctrl, treated\).*run_patching_condition_repeat_tree"):
        run_patching_pipeline(img_dir, config=config)


def test_several_mask_types_need_a_choice(tmp_path):
    img_dir, mask_dir = _folders(tmp_path, ["a.tif", "b.tif"], ["a_cells.tif", "b_cells.tif", "a_nuclei.tif"])
    with pytest.raises(ValueError, match=r"2 mask types \(cells, nuclei\); set matching.mask_type"):
        validate_image_and_mask_folders(img_dir, mask_dir)
    out = validate_image_and_mask_folders(img_dir, mask_dir, mask_type="cells")
    assert {image.name: mask.name for image, mask in out.items()} == {"a.tif": "a_cells.tif", "b.tif": "b_cells.tif"}
    # An image without a mask of the chosen type is refused, not left out.
    with pytest.raises(ValueError, match=r"Images with no 'nuclei' mask: b.tif"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_type="nuclei")
    with pytest.raises(ValueError, match=r"No mask of type 'Cells'.*cells, nuclei"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_type="Cells")


def test_a_mask_folder_inside_the_image_folder_is_refused_like_preprocessing(tmp_path):
    # Former rule: accepted (the image folder listing skips subfolders).
    img_dir = tmp_path / "images"
    write_array(img_dir / "a.tif", np.zeros((4, 4), np.float32))
    write_array(img_dir / "masks" / "a_mask.tif", np.ones((4, 4), np.uint16))
    with pytest.raises(ValueError, match=r"\[masks_inside_experiment\]"):
        validate_image_and_mask_folders(img_dir, img_dir / "masks")


def test_the_former_matching_options_are_refused_by_name(tmp_path):
    # Removed in v0.1.1 (D86, owner decision): refused whatever their value, old defaults included.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_mask.tif"])
    with pytest.raises(ValueError, match="strategy were removed in CellPatcher v0.1.1.*mask_type"):
        validate_image_and_mask_folders(img_dir, mask_dir, strategy="exact_or_suffix")
    with pytest.raises(ValueError, match="set matching.mask_type='cells'"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_suffixes=("_cells",), ignore_unrequested_masks=True)
    with pytest.raises(ValueError, match="case_sensitive were removed"):
        run_patching_pipeline(img_dir, mask_folder_path=mask_dir, case_sensitive=False)
    with pytest.raises(ValueError, match="mask_suffixes were removed"):
        run_patching_condition_repeat_tree(img_dir, mask_root_path=mask_dir, mask_suffixes=("_mask",))
    with pytest.raises(TypeError, match=r"unknown option\(s\) \['colour'\]"):
        validate_image_and_mask_folders(img_dir, mask_dir, colour="red")
    assert [mask.name for mask in validate_image_and_mask_folders(img_dir, mask_dir).values()] == ["a_mask.tif"]


def test_the_former_matchers_are_gone():
    import cellpatcher
    from cellpatcher import io

    for name in ("find_candidate_matches", "filter_shared_matches", "select_requested_masks", "image_stem",
                 "LEGACY_MATCHING_DEFAULTS", "check_legacy_matching", "warn_legacy_matching_removed"):
        assert not hasattr(io, name), name
        assert name not in cellpatcher.__all__


def test_validate_ignores_macos_resource_forks_in_both_folders(tmp_path):
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif"])
    for folder in (img_dir, mask_dir):
        (folder / "._a.tif").write_bytes(b"x")
        (folder / "._a_cells.tif").write_bytes(b"x")
    (mask_dir / "__MACOSX").mkdir()
    (mask_dir / "__MACOSX" / "._a_cells.tif").write_bytes(b"x")
    out = validate_image_and_mask_folders(img_dir, mask_dir)
    assert {image.name: mask.name for image, mask in out.items()} == {"a.tif": "a_cells.tif"}


# ---------------------------------------------------------------------------
# find_masks' header checks (preprocessing v0.3.2): refused for the mask type
# the run uses, ignored for the types it does not use (owner decision).
# ---------------------------------------------------------------------------


def _header_codes(img_dir, mask_dir):
    import morphobricks_preprocessing as mbp
    from morphobricks_preprocessing.metadata import inspect_images

    discovery = mbp.discover_experiment(img_dir)
    images = inspect_images(img_dir, discovery.images).files
    inventory = mbp.find_masks(mask_dir, discovery, images, ome_zarr_labels=False)
    return {issue.code for issue in inventory.issues if issue.is_error}


def test_differing_planes_of_an_unused_mask_type_do_not_block_the_type_used(tmp_path):
    # a.tif has a 2D cells mask and a nuclei mask of 3 depth slices (named Z axis):
    # find_masks reports mask_planes_differ for a.tif, which concerns nuclei only.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif"])
    write_tiff(mask_dir / "a_nuclei.tif", np.ones((3, 4, 4), np.uint16), "ZYX")
    assert _header_codes(img_dir, mask_dir) == {"mask_planes_differ"}
    out = validate_image_and_mask_folders(img_dir, mask_dir, mask_type="cells")
    assert [mask.name for mask in out.values()] == ["a_cells.tif"]
    out = validate_image_and_mask_folders(img_dir, mask_dir, mask_type="nuclei")
    assert [mask.name for mask in out.values()] == ["a_nuclei.tif"]


def test_a_header_problem_is_refused_for_the_type_used_only(tmp_path):
    # The nuclei mask has the wrong size and the mito mask is not integer: refused when
    # used, ignored when another type is used. Former code (v0.3.1 pairing): never refused.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif"])
    write_array(mask_dir / "a_nuclei.tif", np.ones((5, 5), np.uint16))
    write_array(mask_dir / "a_mito.tif", np.ones((4, 4), np.float32))
    assert _header_codes(img_dir, mask_dir) == {"mask_size_mismatch", "mask_pixel_type"}
    out = validate_image_and_mask_folders(img_dir, mask_dir, mask_type="cells")
    assert [mask.name for mask in out.values()] == ["a_cells.tif"]
    with pytest.raises(ValueError, match=r"'nuclei' masks cannot be used: \[mask_size_mismatch\].*\(a_nuclei.tif\)"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_type="nuclei")
    with pytest.raises(ValueError, match=r"'mito' masks cannot be used: \[mask_pixel_type\].*\(a_mito.tif\)"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_type="mito")


def test_a_mask_with_unnamed_extra_axes_is_refused_when_used(tmp_path):
    # Three planes with no axes in the header: find_masks' mask_not_2d. Former code: read with
    # common.mask_axes. Owner decision (v0.3.2): refused for the type used.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif"])
    write_array(mask_dir / "a_stack.tif", np.ones((3, 4, 4), np.uint16))
    assert _header_codes(img_dir, mask_dir) == {"mask_not_2d"}
    assert validate_image_and_mask_folders(img_dir, mask_dir, mask_type="cells")
    with pytest.raises(ValueError, match=r"'stack' masks cannot be used: \[mask_not_2d\].*\(a_stack.tif\)"):
        validate_image_and_mask_folders(img_dir, mask_dir, mask_type="stack")



def test_pairing_uses_no_deprecated_preprocessing_name(tmp_path):
    # preprocessing v0.3.2 deprecates morphobricks_preprocessing.SINGLE (moved to the
    # normalization brick) and will remove it in its next version.
    import warnings

    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], ["a_cells.tif"])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*morphobricks_preprocessing", category=DeprecationWarning)
        assert validate_image_and_mask_folders(img_dir, mask_dir)

_MASK_CHANNEL_ADVICE = r"Save each label channel as its own mask type, for example <image>_cells\.tif and <image>_nuclei\.tif"


@pytest.mark.parametrize("axes", ["CYX", None], ids=["channel_axis", "no_axes"])
def test_a_refused_multi_axis_mask_says_how_to_save_it(tmp_path, axes):
    # Owner decision (2026-10-30): masks with a C axis or without axis metadata are refused,
    # and the refusal says what to do: one mask type per file, or named Z/T axes.
    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], [])
    data = np.ones((2, 4, 4), np.uint16)
    if axes is None:
        write_array(mask_dir / "a_cells.tif", data)
    else:
        write_tiff(mask_dir / "a_cells.tif", data, axes)
    assert _header_codes(img_dir, mask_dir) == {"mask_not_2d"}
    with pytest.raises(ValueError, match=r"\[mask_not_2d\]") as refused:
        validate_image_and_mask_folders(img_dir, mask_dir)
    message = str(refused.value)
    assert re.search(_MASK_CHANNEL_ADVICE, message)
    assert "Z/T axes named" in message


@pytest.mark.parametrize("mask_axes", ["CYX", "YXC"])
def test_mask_axes_cannot_read_a_named_z_stack_as_label_channels(tmp_path, mask_axes):
    # A mask whose header names a Z axis passes find_masks. common.mask_axes with a C axis
    # used to override that header and read the Z slices as label channels (with
    # cell.mask_channel); a C axis is now refused for mask files (owner decision 2026-10-30).
    from cellpatcher import CellPatcherConfig

    img_dir, mask_dir = _folders(tmp_path, ["a.tif"], [])
    labels = np.zeros((4, 4), np.uint16)
    labels[1:3, 1:3] = 1
    write_tiff(mask_dir / "a_cells.tif", np.stack([labels, labels]), "ZYX")
    assert _header_codes(img_dir, mask_dir) == set()
    config = CellPatcherConfig(mode="cell")
    config.common.mask_axes = mask_axes
    config.cell.mask_channel = 0
    config.cell.exclude_border = False
    with pytest.raises(ValueError, match=r"common\.mask_axes=.*channel \(C\) axis") as refused:
        run_patching_pipeline(img_dir, config=config, mask_folder_path=mask_dir)
    assert re.search(_MASK_CHANNEL_ADVICE, str(refused.value))
