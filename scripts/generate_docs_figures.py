from __future__ import annotations

from pathlib import Path
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from cellpatcher import (
    CellConfig,
    CellPatcherConfig,
    CommonConfig,
    ExtractionConfig,
    GridConfig,
    RandomConfig,
    read_array,
    run_patching,
    scan_cell_patch_size,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_IMAGE = REPO_ROOT / "tests/fixtures/cellpatcher_demo_image.tif"
FIXTURE_LABELS = REPO_ROOT / "tests/fixtures/cellpatcher_demo_labels.tif"
OUTPUT_DIR = REPO_ROOT / "docs/assets"


# Curated high-contrast colors. This is a deterministic Glasbey-style fallback when
# a true Glasbey LUT package is unavailable in the runtime.
FALLBACK_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def normalize_percentile(image, p_low=1, p_high=99.8):
    """Normalize image for display only using percentile clipping to [0, 1]."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo = np.percentile(finite, p_low)
    hi = np.percentile(finite, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    norm = np.clip(arr, lo, hi)
    norm = (norm - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _label_color(label: int) -> tuple[float, float, float]:
    idx = label - 1
    if idx < len(FALLBACK_COLORS):
        r, g, b = FALLBACK_COLORS[idx]
        return (r / 255.0, g / 255.0, b / 255.0)

    # Deterministic HSV fallback for labels beyond curated palette.
    hue = (idx * 0.6180339887498949) % 1.0
    sat = 0.72
    val = 0.95
    return colorsys.hsv_to_rgb(hue, sat, val)


def labels_to_glasbey_rgb(labels, background=(0, 0, 0)):
    """Map integer labels to deterministic distinct RGB colors.

    Uses a Glasbey-style fallback palette (curated + HSV extension) so no extra
    dependencies are required for docs generation.
    """
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError("labels_to_glasbey_rgb expects a 2D label array.")

    rgb = np.zeros((*lab.shape, 3), dtype=np.float32)
    bg = np.asarray(background, dtype=np.float32)
    if bg.max() > 1:
        bg = bg / 255.0
    rgb[:] = bg

    nonzero = np.unique(lab)
    nonzero = nonzero[nonzero != 0]
    for v in nonzero:
        rgb[lab == v] = _label_color(int(v))
    return rgb


def draw_patch_boxes(ax, patches_in, color="yellow", max_boxes=12, linewidth=1.5, label=None):
    shown = 0
    for patch in patches_in[:max_boxes]:
        rect = patches.Rectangle(
            (patch.x0, patch.y0),
            patch.width,
            patch.height,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)
        shown += 1
    if label:
        ax.text(0.01, 0.99, f"{label} (showing {shown})", transform=ax.transAxes,
                ha="left", va="top", color=color, fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))


def _render_patch_gallery(ax, arrays, n=6, title="Patches"):
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if not arrays:
        ax.text(0.5, 0.5, "No patches", ha="center", va="center")
        return

    m = min(n, len(arrays))
    cols = min(3, m)
    rows = int(np.ceil(m / cols))
    h, w = arrays[0].shape[-2], arrays[0].shape[-1]
    canvas = np.zeros((rows * h, cols * w), dtype=np.float32)

    for i in range(m):
        r = i // cols
        c = i % cols
        patch_arr = np.asarray(arrays[i])
        if patch_arr.ndim > 2:
            patch_arr = np.squeeze(patch_arr)
        patch_disp = normalize_percentile(patch_arr)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = patch_disp

    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)


def save_fixture_overview(image_n, labels_rgb):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
    axs[0].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Image (percentile normalized)", fontsize=9)
    axs[1].imshow(labels_rgb)
    axs[1].set_title("Labels (Glasbey-style LUT)", fontsize=9)
    axs[2].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    alpha = (np.any(labels_rgb > 0, axis=-1)).astype(np.float32) * 0.42
    axs[2].imshow(labels_rgb, alpha=alpha)
    axs[2].set_title("Overlay", fontsize=9)
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    p = OUTPUT_DIR / "fixture_overview.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def save_cell_mode(image_n, labels_rgb, image, labels):
    summary = scan_cell_patch_size(image_shapes=image.shape, labels_list=labels, image_axes="YX", mask_axes="YX")
    p95 = summary.recommended_size("p95")
    pmax = summary.recommended_size("max")
    chosen = p95 if p95 > 0 else pmax
    if chosen <= 0:
        chosen = 64

    cfg = CellPatcherConfig(
        mode="cell",
        common=CommonConfig(image_axes="YX", mask_axes="YX", extract_arrays=True),
        cell=CellConfig(patch_size=chosen, exclude_border=True, min_area=5),
        extraction=ExtractionConfig(output_size=64, apply_mask=True, mask_fill_value=0),
    )
    result = run_patching(images=image, masks=labels, config=cfg)

    fig = plt.figure(figsize=(9, 5), dpi=150)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    ax0.imshow(image_n, cmap="gray", vmin=0, vmax=1)
    alpha = (labels > 0).astype(np.float32) * 0.35
    ax0.imshow(labels_rgb, alpha=alpha)
    draw_patch_boxes(ax0, result.patches, color="yellow", max_boxes=8, label="Cell patches")
    ax0.set_title("Cell mode boxes on image", fontsize=10)
    ax0.axis("off")

    ax1.imshow(labels_rgb)
    ax1.set_title("Label map", fontsize=10)
    ax1.axis("off")

    _render_patch_gallery(ax2, result.patch_arrays or [], n=6, title="Masked cell patches (64x64)")

    fig.tight_layout()
    p = OUTPUT_DIR / "cell_mode_apply_mask.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, result


def save_grid_mode(image_n, image):
    cfg = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="YX", extract_arrays=True),
        grid=GridConfig(tile_size=128, overlap=0.25, include_partial=True),
        extraction=ExtractionConfig(output_size=64),
    )
    result = run_patching(images=image, config=cfg)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    axs[0].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    draw_patch_boxes(axs[0], result.patches, color="cyan", max_boxes=20, label="Grid patches")
    axs[0].set_title("Grid mode (overlap=0.25)", fontsize=10)
    axs[0].axis("off")

    _render_patch_gallery(axs[1], result.patch_arrays or [], n=6, title="Grid patch samples (64x64)")

    fig.tight_layout()
    p = OUTPUT_DIR / "grid_mode_example.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, result


def save_random_mode(image_n, image, labels):
    cfg = CellPatcherConfig(
        mode="random",
        common=CommonConfig(image_axes="YX", mask_axes="YX", extract_arrays=True),
        random=RandomConfig(patch_size=128, n_patches=6, seed=42),
        extraction=ExtractionConfig(output_size=64),
    )
    mask = labels > 0
    try:
        result = run_patching(images=image, masks=mask, config=cfg)
    except Exception:
        result = run_patching(images=image, config=cfg)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    axs[0].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    draw_patch_boxes(axs[0], result.patches, color="magenta", max_boxes=6, label="Random patches (seed=42)")
    axs[0].set_title("Random mode", fontsize=10)
    axs[0].axis("off")

    _render_patch_gallery(axs[1], result.patch_arrays or [], n=6, title="Random patch samples (64x64)")

    fig.tight_layout()
    p = OUTPUT_DIR / "random_mode_example.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, result


def save_whole_fov_mode(image_n, image):
    cfg = CellPatcherConfig(
        mode="whole_fov",
        common=CommonConfig(image_axes="YX", extract_arrays=True),
        extraction=ExtractionConfig(output_size=128),
    )
    result = run_patching(images=image, config=cfg)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    axs[0].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    draw_patch_boxes(axs[0], result.patches, color="lime", max_boxes=1, label="Whole FOV")
    axs[0].set_title("Whole FOV mode", fontsize=10)
    axs[0].axis("off")

    if result.patch_arrays:
        axs[1].imshow(normalize_percentile(np.asarray(result.patch_arrays[0])), cmap="gray", vmin=0, vmax=1)
        axs[1].set_title("Resized full-image patch (128x128)", fontsize=10)
    axs[1].axis("off")

    fig.tight_layout()
    p = OUTPUT_DIR / "whole_fov_example.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, result


def save_modes_overview(image_n, cell_result, rand_result, grid_result, whole_result):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), dpi=150)
    panels = [
        (axs[0, 0], "Cell", cell_result.patches, "yellow", 8),
        (axs[0, 1], "Random", rand_result.patches, "magenta", 6),
        (axs[1, 0], "Grid", grid_result.patches, "cyan", 20),
        (axs[1, 1], "Whole FOV", whole_result.patches, "lime", 1),
    ]
    for ax, title, patch_list, color, maxb in panels:
        ax.imshow(image_n, cmap="gray", vmin=0, vmax=1)
        draw_patch_boxes(ax, patch_list, color=color, max_boxes=maxb)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    p = OUTPUT_DIR / "patching_modes_overview.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = read_array(FIXTURE_IMAGE)
    labels = read_array(FIXTURE_LABELS)
    image_n = normalize_percentile(image)
    labels_rgb = labels_to_glasbey_rgb(labels)

    generated = []
    generated.append(save_fixture_overview(image_n, labels_rgb))
    p_cell, res_cell = save_cell_mode(image_n, labels_rgb, image, labels)
    generated.append(p_cell)
    p_grid, res_grid = save_grid_mode(image_n, image)
    generated.append(p_grid)
    p_rand, res_rand = save_random_mode(image_n, image, labels)
    generated.append(p_rand)
    p_whole, res_whole = save_whole_fov_mode(image_n, image)
    generated.append(p_whole)
    generated.append(save_modes_overview(image_n, res_cell, res_rand, res_grid, res_whole))

    for p in generated:
        print(p.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
