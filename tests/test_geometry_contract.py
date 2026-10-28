"""The patch-geometry contract shared with the MorphoBricks bricks (issue #33).

Seed scopes, coverage-aware random sampling, the whole-FOV square cover, and
the descriptor fields every region carries: requested (unclipped) bounds, the
patch and object border flags, the square's parent, and mask fractions.
"""

import csv
import hashlib

import numpy as np
import pytest

from cellpatcher.config import CellConfig, CellPatcherConfig, CommonConfig, ExportConfig, GridConfig, RandomConfig
from cellpatcher.config import WholeFOVConfig
from cellpatcher.ids import build_patch_id, random_seed
from cellpatcher.pipeline import run_patching
from cellpatcher.strategies.whole_fov import square_cover_starts


def _run(mode, image, masks=None, metadata=None, fraction_masks=None, **sections):
    config = CellPatcherConfig(mode=mode, **sections)
    metadata = metadata or [{"position_id": "pos0"}]
    return run_patching(
        images=[image], masks=masks, config=config, image_metadata=metadata, fraction_masks=fraction_masks
    ).patches


def _rectangles(patches):
    return [(p.y0, p.y1, p.x0, p.x1) for p in patches]


# --- seed scope ---------------------------------------------------------------------------------


def test_default_seed_is_the_seed_before_the_option():
    meta = {"position_id": "pos0", "t": 3, "z": 1, "c": 2}
    digest = hashlib.sha256(b"pos0|t=3|z=1|c=2").digest()
    words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, len(digest), 4)]
    before = int(np.random.SeedSequence([7, *words]).generate_state(1, dtype=np.uint64)[0])
    assert random_seed(7, meta) == random_seed(7, meta, "plane_channel") == before


@pytest.mark.parametrize(
    ("scope", "same_channel", "same_time"),
    [("plane_channel", False, False), ("plane", True, False), ("position", True, True)],
)
def test_seed_scope_decides_which_planes_share_rectangles(scope, same_channel, same_time):
    image = np.zeros((64, 64), dtype=np.uint16)
    random = RandomConfig(patch_size=8, n_patches=6, seed=11, seed_scope=scope)

    def draw(t, c):
        return _rectangles(_run("random", image, metadata=[{"position_id": "pos0", "t": t, "c": c}], random=random))

    assert (draw(0, 0) == draw(0, 1)) is same_channel
    assert (draw(0, 0) == draw(1, 0)) is same_time
    assert draw(0, 0) == draw(0, 0)


def test_unknown_seed_scope_is_refused():
    with pytest.raises(ValueError, match="seed_scope"):
        RandomConfig(seed_scope="channel")


# --- coverage-aware random sampling -------------------------------------------------------------


def _tissue():
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:50, 20:70] = True
    return mask


@pytest.mark.parametrize("fraction", [0.25, 0.5, 1.0])
def test_every_draw_reaches_the_minimum_mask_fraction(fraction):
    tissue = _tissue()
    random = RandomConfig(patch_size=16, n_patches=40, seed=3, min_mask_fraction=fraction)
    patches = _run("random", np.zeros((80, 80)), masks=tissue, random=random)
    assert len(patches) == 40
    assert all(tissue[p.y0:p.y1, p.x0:p.x1].mean() >= fraction for p in patches)


def test_zero_minimum_keeps_the_any_overlap_rule():
    tissue = _tissue()
    before = _run("random", np.zeros((80, 80)), masks=tissue, random=RandomConfig(patch_size=16, seed=3))
    after = _run(
        "random", np.zeros((80, 80)), masks=tissue, random=RandomConfig(patch_size=16, seed=3, min_mask_fraction=0)
    )
    assert _rectangles(before) == _rectangles(after)
    assert min(tissue[p.y0:p.y1, p.x0:p.x1].mean() for p in before) < 0.25  # partial overlaps exist


def test_minimum_mask_fraction_needs_a_mask_and_a_valid_value():
    with pytest.raises(ValueError, match="min_mask_fraction"):
        RandomConfig(min_mask_fraction=1.5)
    with pytest.raises(ValueError, match="sampling mask"):
        _run("random", np.zeros((40, 40)), random=RandomConfig(patch_size=8, min_mask_fraction=0.5))


# --- whole-FOV square cover ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("long", "short", "starts"),
    [(20, 15, [0, 5]), (2048, 1536, [0, 512]), (15, 15, [0]), (40, 15, [0, 12, 25]), (16, 15, [0, 1])],
)
def test_square_cover_starts(long, short, starts):
    assert square_cover_starts(long, short) == starts


@pytest.mark.parametrize("shape", [(15, 20), (20, 15), (15, 40), (16, 16)])
def test_square_cover_regions_cover_the_image_with_squares(shape):
    patches = _run("whole_fov", np.zeros(shape), whole_fov=WholeFOVConfig(square_cover=True))
    parent, squares = patches[0], patches[1:]
    side = min(shape)
    assert parent.mode == "whole_fov" and parent.patch_id == "pos0/whole_fov"
    covered = np.zeros(shape, dtype=bool)
    for index, square in enumerate(squares):
        assert square.mode == "whole_fov_square"
        assert (square.y1 - square.y0, square.x1 - square.x0) == (side, side)
        assert square.patch_id == f"pos0/whole_fov/sq{index}" == build_patch_id(square)
        assert square.parent_id == parent.patch_id and square.square_index == index
        covered[square.y0:square.y1, square.x0:square.x1] = True
    assert covered.all()
    assert len(squares) == -(-max(shape) // side)


def test_without_square_cover_whole_fov_is_one_region():
    patches = _run("whole_fov", np.zeros((15, 20)))
    assert [p.mode for p in patches] == ["whole_fov"]
    assert patches[0].parent_id is None


# --- requested bounds and border flags ----------------------------------------------------------


def test_partial_grid_tiles_record_their_full_tile():
    patches = _run("grid", np.zeros((40, 50)), grid=GridConfig(tile_size=16, include_partial=True))
    for patch in patches:
        requested = (patch.requested_y0, patch.requested_y1, patch.requested_x0, patch.requested_x1)
        assert requested == (patch.y0, patch.y0 + 16, patch.x0, patch.x0 + 16)
        clipped = requested != (patch.y0, patch.y1, patch.x0, patch.x1)
        assert clipped == (patch.y1 - patch.y0 < 16 or patch.x1 - patch.x0 < 16)
        assert patch.patch_touches_image_border == (
            patch.y0 == 0 or patch.x0 == 0 or patch.y1 == 40 or patch.x1 == 50
        )
        assert patch.object_touches_image_border is None
    assert any(p.requested_y1 != p.y1 or p.requested_x1 != p.x1 for p in patches)


def _cells():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[18:22, 18:22] = 1  # centre: square fits
    labels[2:6, 3:7] = 2  # near the edge, not touching it: its square is clipped
    labels[30:40, 30:34] = 3  # touches the bottom edge
    return labels


def test_cell_patches_record_the_unclipped_square_and_both_border_flags():
    labels = _cells()
    patches = _run(
        "cell", np.zeros((40, 40)), masks=labels, cell=CellConfig(patch_size=12, exclude_border=False)
    )
    by_label = {p.label_id: p for p in patches}
    centre, edge, border = by_label[1], by_label[2], by_label[3]
    assert (centre.requested_y0, centre.requested_y1) == (centre.y0, centre.y1) == (14, 26)
    assert (edge.requested_y0, edge.requested_x0) == (-2, -1) and (edge.y0, edge.x0) == (0, 0)
    assert edge.patch_touches_image_border and not edge.object_touches_image_border
    assert border.object_touches_image_border and border.touches_border
    assert not centre.patch_touches_image_border and not centre.object_touches_image_border
    for patch in patches:
        window = labels[patch.y0:patch.y1, patch.x0:patch.x1]
        assert patch.mask_fractions["own_object"] == (window == patch.label_id).mean()


def test_existing_touches_border_keeps_its_meaning():
    labels = _cells()
    patches = _run("cell", np.zeros((40, 40)), masks=labels, cell=CellConfig(patch_size=12, exclude_border=False))
    assert {p.label_id: p.touches_border for p in patches} == {1: False, 2: False, 3: True}


# --- mask fractions -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["grid", "random", "cell", "whole_fov"])
def test_mask_fractions_equal_the_share_of_mask_pixels(mode):
    rng = np.random.default_rng(5)
    tissue = rng.random((40, 40)) < 0.4
    nuclei = (rng.integers(0, 4, (40, 40)) * (rng.random((40, 40)) < 0.2)).astype(np.uint16)
    sections = {
        "grid": {"grid": GridConfig(tile_size=12, include_partial=True)},
        "random": {"random": RandomConfig(patch_size=10, n_patches=15, seed=1)},
        "cell": {"cell": CellConfig(patch_size=12, exclude_border=False)},
        "whole_fov": {"whole_fov": WholeFOVConfig(square_cover=True)},
    }[mode]
    masks = _cells() if mode == "cell" else None
    patches = _run(
        mode, np.zeros((40, 40)), masks=masks, fraction_masks=[{"tissue": tissue, "nuclei": nuclei}], **sections
    )
    assert patches
    for patch in patches:
        for name, mask in (("tissue", tissue), ("nuclei", nuclei != 0)):
            assert patch.mask_fractions[name] == mask[patch.y0:patch.y1, patch.x0:patch.x1].mean()


def test_mask_fractions_become_csv_columns(tmp_path):
    tissue = np.zeros((32, 32), dtype=bool)
    tissue[:16] = True
    config = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(output_dir=str(tmp_path)),
        grid=GridConfig(tile_size=16),
        export=ExportConfig(enabled=True, image_format="npy"),
    )
    run_patching(
        images=[np.zeros((32, 32))], config=config, image_metadata=[{"position_id": "p"}],
        fraction_masks=[{"tissue": tissue}],
    )
    with (tmp_path / "patches.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames[-1] == "mask_fraction_tissue"
    assert sorted(float(row["mask_fraction_tissue"]) for row in rows) == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ("fraction_masks", "message"),
    [
        ([{"own_object": np.zeros((20, 20))}], "own_object"),
        ([{"bad/name": np.zeros((20, 20))}], "name"),
        ([{"tissue": np.zeros((10, 20))}], "shape"),
        ([{}, {}], "one mapping per image"),
    ],
)
def test_malformed_fraction_masks_are_refused(fraction_masks, message):
    with pytest.raises(ValueError, match=message):
        _run("grid", np.zeros((20, 20)), fraction_masks=fraction_masks, grid=GridConfig(tile_size=10))


def test_fraction_masks_are_refused_when_planes_are_split():
    config = CellPatcherConfig(mode="grid", common=CommonConfig(image_axes="ZYX", z_policy="all"))
    with pytest.raises(ValueError, match="split planes"):
        run_patching(
            images=[np.zeros((2, 20, 20))], config=config, image_metadata=[{"position_id": "p"}],
            fraction_masks=[{"tissue": np.zeros((20, 20))}],
        )


def test_new_options_survive_a_saved_run_config(tmp_path):
    from cellpatcher.orchestration import load_config
    from cellpatcher.pipeline import run_config_data
    import json

    config = CellPatcherConfig(
        mode="random",
        random=RandomConfig(seed_scope="position", min_mask_fraction=0.5),
        whole_fov=WholeFOVConfig(square_cover=True),
    )
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(run_config_data(config), default=str))
    loaded = load_config(path)
    assert loaded.random == config.random and loaded.whole_fov == config.whole_fov
