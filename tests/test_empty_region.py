"""An empty region gives no patch and never a refusal (Spec ADR 0005 BM8, issue #118).

A patch set sampled in an empty region (``mask != 0`` holds no pixel) has no patch on that plane;
the image is listed with the reason ``empty_region`` in ``images.csv`` and ``positions.csv``, and
the other images of the run keep their patches, unchanged. A region that is not empty but cannot
give ``n_patches`` draws reaching ``min_mask_fraction`` (too small or too sparse for the patch) is
not an empty region: it stays refused with ``SamplingExhaustedError``. The fraction of a patch
inside an empty region (``fraction_masks``) is defined, ``0.0``, in every mode.
"""

import csv
import warnings

import numpy as np
import pytest

from cellpatcher.config import CellPatcherConfig, CommonConfig, RandomConfig
from cellpatcher.errors import SamplingExhaustedError
from cellpatcher.orchestration import run_patching_pipeline
from cellpatcher.pipeline import run_patching
from cellpatcher.strategies.random import RandomPatchStrategy
from cellpatcher.types import EMPTY_REGION
from tests.helpers import preprocess, write_tiff

SHAPE = (48, 64)
IMAGE = np.arange(SHAPE[0] * SHAPE[1], dtype=np.uint16).reshape(SHAPE)
EMPTY = np.zeros(SHAPE, dtype=bool)
BLOCK = np.zeros(SHAPE, dtype=bool)
BLOCK[5:30, 20:50] = True
OBJECTS = np.zeros(SHAPE, dtype=np.int32)
OBJECTS[10:20, 10:20], OBJECTS[25:35, 30:42] = 1, 2
METADATA = [{"position_id": "empty"}, {"position_id": "full"}]
# min_mask_fraction values that filter draws: None (any overlap, the default) and > 0.
FILTERING = [None, 1 / 64, 0.5, 1.0]


def _config(mode, fraction=None, output=None):
    config = CellPatcherConfig(mode=mode)
    config.random = RandomConfig(patch_size=8, n_patches=5, seed=3, min_mask_fraction=fraction)
    config.grid.tile_size = 16
    config.cell.patch_size = 8
    config.cell.exclude_border = False
    if output is not None:
        config.export.enabled = True
        config.common.output_dir = str(output)
    return config


def _rectangles(patches):
    return [(p.patch_id, p.y0, p.y1, p.x0, p.x1) for p in patches]


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- the strategy -----------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int32])
@pytest.mark.parametrize("fraction", FILTERING)
def test_an_empty_sampling_mask_gives_no_region(dtype, fraction):
    config = RandomConfig(patch_size=8, n_patches=5, seed=0, min_mask_fraction=fraction)
    regions = RandomPatchStrategy(config).generate_regions(image_shape=SHAPE, labels=np.zeros(SHAPE, dtype=dtype))
    assert regions == []


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_a_region_too_small_for_the_patch_is_still_refused(fraction):
    """Not an empty region: a non-empty region no draw can reach keeps its refusal."""
    small = np.zeros(SHAPE, dtype=bool)
    small[40, 3] = True  # one pixel: a share of 1/64 at most
    config = RandomConfig(patch_size=8, n_patches=5, seed=0, min_mask_fraction=fraction)
    with pytest.raises(SamplingExhaustedError) as info:
        RandomPatchStrategy(config).generate_regions(image_shape=SHAPE, labels=small)
    assert (info.value.requested, info.value.collected, info.value.attempts) == (5, 0, 1000)
    with pytest.raises(SamplingExhaustedError):
        run_patching(images=[IMAGE], masks=[small], config=_config("random", fraction), image_metadata=METADATA[:1])


def test_an_invalid_empty_mask_is_still_refused():
    """The region is checked first (BM7): an all-zero float mask is MASK_REGION_INVALID, not empty_region."""
    with pytest.raises(ValueError, match=r"MASK_REGION_INVALID \(dtype\)"):
        RandomPatchStrategy(RandomConfig(patch_size=8, n_patches=5, seed=0)).generate_regions(
            image_shape=SHAPE, labels=np.zeros(SHAPE, dtype=np.float32)
        )


# --- random mode in memory ----------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", FILTERING)
def test_random_mode_in_memory_lists_the_empty_region_and_patches_the_other_images(tmp_path, fraction):
    config = _config("random", fraction, output=tmp_path)
    result = run_patching(images=[IMAGE, IMAGE], masks=[EMPTY, BLOCK], config=config, image_metadata=METADATA)
    alone = run_patching(images=[IMAGE], masks=[BLOCK], config=_config("random", fraction), image_metadata=METADATA[1:])
    assert _rectangles(result.patches) == _rectangles(alone.patches) and len(alone.patches) == 5
    empty, full = result.image_records
    assert (empty.status, empty.reason, empty.n_regions, empty.n_patches) == ("no_patches", EMPTY_REGION, 0, 0)
    assert (full.status, full.reason, full.n_patches) == ("processed", None, 5)
    images = _rows(tmp_path / "images.csv")
    assert [(row["status"], row["reason"]) for row in images] == [("no_patches", EMPTY_REGION), ("processed", "")]
    positions = _rows(tmp_path / "positions.csv")
    assert [(row["status"], row["reason"]) for row in positions] == [
        ("analysed_no_regions", EMPTY_REGION),
        ("analysed", ""),
    ]


def test_random_mode_at_fraction_zero_samples_an_empty_mask_as_no_mask():
    """An explicit 0 excludes nothing (F8): an empty mask gives the rectangles drawn without one."""
    config = _config("random", 0)
    with pytest.warns(UserWarning, match="excludes no rectangle"):
        masked = run_patching(images=[IMAGE], masks=[EMPTY], config=config, image_metadata=METADATA[:1])
    unmasked = run_patching(images=[IMAGE], config=config, image_metadata=METADATA[:1])
    assert _rectangles(masked.patches) == _rectangles(unmasked.patches) and len(unmasked.patches) == 5
    assert masked.image_records[0].reason is None


def test_random_mode_with_a_plane_split_lists_an_image_whose_planes_are_all_empty():
    """Each T plane is sampled alone; the empty one has no patch, the other plane keeps its own."""
    config = _config("random")
    config.common.image_axes, config.common.mask_axes, config.common.t_policy = "TYX", "TYX", "all"
    images = [np.stack([IMAGE, IMAGE]), np.stack([IMAGE, IMAGE])]
    masks = [np.stack([EMPTY, EMPTY]), np.stack([EMPTY, BLOCK])]
    result = run_patching(images=images, masks=masks, config=config, image_metadata=METADATA)
    empty, mixed = result.image_records
    assert (empty.status, empty.reason, empty.n_empty_region_planes) == ("no_patches", EMPTY_REGION, 2)
    assert (mixed.status, mixed.n_patches, mixed.n_empty_region_planes) == ("processed", 5, 1)
    assert {p.t for p in result.patches} == {1}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_random_mode_with_normalization_patches_the_other_images():
    config = _config("random")
    config.normalization.enabled = True
    config.normalization.scope = "patch"
    result = run_patching(images=[IMAGE, IMAGE], masks=[EMPTY, BLOCK], config=config, image_metadata=METADATA)
    assert [r.reason for r in result.image_records] == [EMPTY_REGION, None]
    assert result.image_records[1].n_patches == 5


# --- every mode in memory: the fraction of a patch in an empty region is 0.0 ------------------------


@pytest.mark.parametrize("mode", ["whole_fov", "grid", "cell", "random"])
def test_every_mode_gives_a_zero_fraction_in_an_empty_region_and_the_same_patches(mode):
    masks = {"cell": [OBJECTS, OBJECTS], "random": [BLOCK, BLOCK]}.get(mode)
    config = _config(mode, 0.5)
    plain = run_patching(images=[IMAGE, IMAGE], masks=masks, config=config, image_metadata=METADATA)
    regions = [{"region": EMPTY}, {"region": BLOCK}]
    result = run_patching(
        images=[IMAGE, IMAGE], masks=masks, config=config, image_metadata=METADATA, fraction_masks=regions
    )
    assert _rectangles(result.patches) == _rectangles(plain.patches) and result.patches
    for patch in result.patches:
        if patch.image_id == "image_0000":
            assert patch.mask_fractions["region"] == 0.0
    assert [r.reason for r in result.image_records] == [None, None]


def test_cell_mode_with_an_empty_object_set_is_unchanged():
    """An all-zero object set has zero objects: no region, the reason it had before (not empty_region)."""
    result = run_patching(
        images=[IMAGE, IMAGE], masks=[np.zeros(SHAPE, np.int32), OBJECTS], config=_config("cell"),
        image_metadata=METADATA,
    )
    assert [r.reason for r in result.image_records] == ["the patch mode generated no region in this image", None]
    assert result.image_records[1].n_patches == 2


# --- file runs ------------------------------------------------------------------------------------------


@pytest.fixture
def run_with_an_empty_mask(tmp_path):
    """Two YX images; the region mask of ``a`` is all zero, ``b``'s holds a block."""
    root, masks = tmp_path / "experiment", tmp_path / "masks"
    for name, mask in (("a", EMPTY), ("b", BLOCK)):
        write_tiff(root / "ctrl" / "rep1" / f"{name}.tif", IMAGE, "YX")
        write_tiff(masks / "tissue" / "ctrl" / "rep1" / f"{name}.tif", mask.astype(np.uint8), "YX")
    run = preprocess(root, masks=masks, output=tmp_path / "run", representations={"tissue": "binary"})
    return root, masks, run


def _file_run(root, masks, run, config):
    config.common.preprocessing_run = str(run)
    if config.mode == "random":
        config.matching.mask_type = "tissue"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return run_patching_pipeline(root, config=config, mask_folder_path=masks if config.mode == "random" else None)


@pytest.mark.parametrize("fraction", FILTERING)
def test_random_file_run_lists_the_empty_region_and_patches_the_other_image(run_with_an_empty_mask, tmp_path, fraction):
    root, masks, run = run_with_an_empty_mask
    result = _file_run(root, masks, run, _config("random", fraction, output=tmp_path / "out"))
    assert [(r.image_file, r.status, r.reason) for r in result.image_records] == [
        ("a.tif", "no_patches", EMPTY_REGION),
        ("b.tif", "processed", None),
    ]
    assert len(result.patches) == 5 and all(p.image_file == "b.tif" for p in result.patches)
    # b's rectangles do not depend on a: those of b sampled alone, under b's FOV key.
    config = _config("random", fraction)
    config.common = CommonConfig(
        image_axes="YX", mask_axes="YX", condition="ctrl", biological_repeat="rep1", technical_repeat="single"
    )
    alone = run_patching(images=[IMAGE], masks=[BLOCK], config=config, image_metadata=[{"position_id": "b"}])
    assert _rectangles(result.patches) == _rectangles(alone.patches)
    rows = _rows(tmp_path / "out" / "images.csv")
    assert [(row["image_file"], row["status"], row["reason"]) for row in rows] == [
        ("a.tif", "no_patches", EMPTY_REGION),
        ("b.tif", "processed", ""),
    ]
    assert [row["reason"] for row in _rows(tmp_path / "out" / "positions.csv")] == [EMPTY_REGION, ""]


@pytest.mark.parametrize("mode", ["whole_fov", "grid", "random"])
def test_file_runs_of_the_other_modes_are_unchanged_by_an_empty_mask(run_with_an_empty_mask, mode):
    """grid and whole_fov read no mask; random at an explicit 0 excludes nothing: every image is patched."""
    root, masks, run = run_with_an_empty_mask
    result = _file_run(root, masks, run, _config(mode, 0))
    assert [r.status for r in result.image_records] == ["processed", "processed"]
    first = sorted((p.y0, p.x0) for p in result.patches if p.image_file == "a.tif")
    second = sorted((p.y0, p.x0) for p in result.patches if p.image_file == "b.tif")
    assert first and (mode == "random" or first == second)
