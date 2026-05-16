from __future__ import annotations

from pathlib import Path
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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

# Global figure style — white background, clean look.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 10,
})

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


def _add_info_box(ax, lines, color="white", fontsize=8.5):
    """Add a semi-transparent annotation box in the top-left of an axes."""
    text = "\n".join(lines)
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        color="white",
        fontsize=fontsize,
        linespacing=1.5,
        bbox=dict(facecolor="#111111", alpha=0.72, edgecolor="none", pad=4),
    )


def draw_patch_boxes(
    ax,
    patches_in,
    color="yellow",
    max_boxes=None,
    linewidth=2.0,
    fill_alpha=0.08,
    label_lines=None,
    show_numbers=False,
):
    """Draw patch bounding boxes with optional semi-transparent fill.

    The edge is always fully opaque at *linewidth*. Fill transparency is
    controlled via *fill_alpha* (0 = no fill, 1 = solid).  Using separate
    RGBA facecolor / opaque edgecolor avoids the problem where setting
    ``alpha`` on the whole Rectangle makes both fill AND edge invisible.

    Parameters
    ----------
    label_lines:
        List of strings shown in an info box in the top-left corner.
    show_numbers:
        Number each box with its 1-based index.
    """
    rgb = mcolors.to_rgb(color)
    facecolor_rgba = (*rgb, fill_alpha)

    display = patches_in if max_boxes is None else patches_in[:max_boxes]
    for i, patch in enumerate(display):
        rect = mpatches.Rectangle(
            (patch.x0, patch.y0),
            patch.width,
            patch.height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=facecolor_rgba,
        )
        ax.add_patch(rect)
        if show_numbers:
            cx = patch.x0 + patch.width / 2
            cy = patch.y0 + patch.height / 2
            ax.text(
                cx, cy, str(i + 1),
                ha="center", va="center",
                color="white", fontsize=7, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.55, edgecolor="none", pad=1),
            )
    if label_lines:
        _add_info_box(ax, label_lines)


def _render_patch_gallery(ax, arrays, n=6, title="Patches", gap=3):
    """Render up to *n* patches as a grid with a thin gap between cells.

    Each patch is contrast-stretched individually so fine details are visible
    even in very dark patches.  A 1-pixel white gap separates adjacent cells.
    """
    if not arrays:
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        ax.text(0.5, 0.5, "No patches", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        return

    m = min(n, len(arrays))
    cols = min(3, m)
    rows = int(np.ceil(m / cols))

    # Normalise each patch individually for the best possible contrast.
    patches_disp = []
    for arr in arrays[:m]:
        a = np.asarray(arr)
        if a.ndim > 2:
            a = a.reshape(a.shape[-2], a.shape[-1])
        patches_disp.append(normalize_percentile(a))

    h, w = patches_disp[0].shape

    canvas_h = rows * h + (rows - 1) * gap
    canvas_w = cols * w + (cols - 1) * gap
    # Use a mid-gray background so the gap is visually distinct from the image.
    canvas = np.full((canvas_h, canvas_w), 0.35, dtype=np.float32)

    for i, patch_disp in enumerate(patches_disp):
        r = i // cols
        c = i % cols
        y0 = r * (h + gap)
        x0 = c * (w + gap)
        canvas[y0:y0 + h, x0:x0 + w] = patch_disp

    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{title}  ({m} of {len(arrays)} shown)", fontsize=10)
    ax.axis("off")


def save_fixture_overview(image_n, labels_rgb, labels):
    """Three-panel overview: raw image | labeled cells | overlay."""
    n_cells = int(np.max(labels))
    fig, axs = plt.subplots(1, 3, figsize=(11, 4), dpi=150)

    axs[0].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Raw image\n(percentile-normalised for display)", fontsize=9)

    axs[1].imshow(labels_rgb)
    axs[1].set_title(f"Cell labels — {n_cells} objects\n(Glasbey-style colour map)", fontsize=9)
    # Annotate each cell with its label ID.
    nonzero = np.unique(labels)
    nonzero = nonzero[nonzero != 0]
    for v in nonzero:
        ys, xs = np.where(labels == v)
        cy, cx = ys.mean(), xs.mean()
        axs[1].text(cx, cy, str(v), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1))

    axs[2].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    alpha = (np.any(labels_rgb > 0, axis=-1)).astype(np.float32) * 0.50
    axs[2].imshow(labels_rgb, alpha=alpha)
    axs[2].set_title("Image + label overlay\n(cells shown semi-transparent)", fontsize=9)

    for ax in axs:
        ax.axis("off")
    fig.suptitle("Demo fixture — source data for all documentation figures",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    p = OUTPUT_DIR / "fixture_overview.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def save_cell_mode(image_n, labels_rgb, image, labels):
    """Cell-mode figure: patch boxes on overlay | label map with IDs | gallery."""
    summary = scan_cell_patch_size(
        image_shapes=image.shape, labels_list=labels,
        image_axes="YX", mask_axes="YX",
    )
    p95 = summary.recommended_size("p95")
    pmax = summary.recommended_size("max")
    chosen = p95 if p95 > 0 else pmax
    if chosen <= 0:
        chosen = 64

    # exclude_border=False so border-touching cells are included and the figure
    # shows a representative number of patches.
    cfg = CellPatcherConfig(
        mode="cell",
        common=CommonConfig(image_axes="YX", mask_axes="YX", extract_arrays=True),
        cell=CellConfig(patch_size=chosen, exclude_border=False, min_area=5),
        extraction=ExtractionConfig(output_size=64, apply_mask=True, mask_fill_value=0),
    )
    result = run_patching(images=image, masks=labels, config=cfg)
    n_patches = len(result.patches)

    fig = plt.figure(figsize=(11, 6), dpi=150)
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], hspace=0.35, wspace=0.15)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_labels = fig.add_subplot(gs[0, 1])
    ax_gallery = fig.add_subplot(gs[0, 2])
    ax_desc = fig.add_subplot(gs[1, 1:])

    # --- Left: image overlay with patch boxes and cell IDs ---
    ax_main.imshow(image_n, cmap="gray", vmin=0, vmax=1)
    alpha = (labels > 0).astype(np.float32) * 0.40
    ax_main.imshow(labels_rgb, alpha=alpha)
    draw_patch_boxes(
        ax_main, result.patches, color="#FFE033",
        max_boxes=n_patches, linewidth=2.0, fill_alpha=0.0,
        show_numbers=True,
        label_lines=[f"Cell mode  ·  patch_size={chosen}", f"{n_patches} patches"],
    )
    ax_main.set_title("Patch boxes on image + labels overlay", fontsize=10)
    ax_main.axis("off")

    # --- Top-right: label map with cell IDs ---
    ax_labels.imshow(labels_rgb)
    ax_labels.set_title("Label map (cell IDs)", fontsize=10)
    ax_labels.axis("off")
    nonzero = np.unique(labels)
    nonzero = nonzero[nonzero != 0]
    for v in nonzero:
        ys, xs = np.where(labels == v)
        cy, cx = ys.mean(), xs.mean()
        ax_labels.text(cx, cy, str(v), ha="center", va="center",
                       fontsize=7, fontweight="bold", color="white",
                       bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1))

    # --- Middle-right: gallery of extracted masked patches ---
    _render_patch_gallery(
        ax_gallery, result.patch_arrays or [], n=6,
        title=f"Masked patches (64 × 64)",
    )

    # --- Bottom text: key parameters ---
    ax_desc.axis("off")
    param_text = (
        "patch_size = p95 scan recommendation   ·   exclude_border = False   ·   "
        "min_area = 5 px\n"
        "apply_mask = True  →  pixels outside the selected cell are set to 0 before resizing"
    )
    ax_desc.text(0.5, 0.6, param_text, ha="center", va="center",
                 fontsize=8.5, color="#333333", transform=ax_desc.transAxes,
                 bbox=dict(facecolor="#F5F5F5", edgecolor="#CCCCCC", pad=6, boxstyle="round,pad=0.4"))

    fig.suptitle("Cell mode — one patch per segmented object", fontsize=11, fontweight="bold", y=1.01)
    p = OUTPUT_DIR / "cell_mode_apply_mask.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, result


def save_grid_mode(image_n, image):
    """Grid-mode figure: full image with all tiles | patch gallery."""
    cfg = CellPatcherConfig(
        mode="grid",
        common=CommonConfig(image_axes="YX", extract_arrays=True),
        grid=GridConfig(tile_size=128, overlap=0.25, include_partial=True),
        extraction=ExtractionConfig(output_size=64),
    )
    result = run_patching(images=image, config=cfg)
    n_patches = len(result.patches)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), dpi=150)

    axs[0].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    # Show ALL tiles; no fill so the underlying image stays readable.
    # linewidth=2.5 keeps each tile boundary visible on dark backgrounds.
    draw_patch_boxes(
        axs[0], result.patches, color="#00E5FF",
        max_boxes=None, linewidth=2.5, fill_alpha=0.0,
        label_lines=[
            "Grid mode",
            f"tile_size=128 px  ·  overlap=25 %",
            f"{n_patches} tiles  (include_partial=True)",
        ],
    )
    axs[0].set_title("All grid tiles overlaid on image", fontsize=10)
    axs[0].axis("off")

    _render_patch_gallery(axs[1], result.patch_arrays or [], n=9,
                          title="Sample tiles (64 × 64, individually normalised)")

    fig.suptitle("Grid mode — deterministic tiling across the full image",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = OUTPUT_DIR / "grid_mode_example.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, result


def save_random_mode(image_n, image, labels):
    """Random-mode figure: patch boxes on image | individual patch gallery."""
    # Run without a mask so patches are not constrained to foreground and
    # the figure avoids all-black (empty) patch thumbnails.
    cfg = CellPatcherConfig(
        mode="random",
        common=CommonConfig(image_axes="YX", extract_arrays=True),
        random=RandomConfig(patch_size=128, n_patches=8, seed=7),
        extraction=ExtractionConfig(output_size=64),
    )
    result = run_patching(images=image, config=cfg)
    n_patches = len(result.patches)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), dpi=150)

    axs[0].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    draw_patch_boxes(
        axs[0], result.patches, color="#FF44CC",
        max_boxes=None, linewidth=2.0, fill_alpha=0.12,
        show_numbers=True,
        label_lines=[
            "Random mode",
            f"patch_size=128 px  ·  n_patches={n_patches}",
            "seed=7  (reproducible)",
        ],
    )
    axs[0].set_title("Random patch locations (numbered)", fontsize=10)
    axs[0].axis("off")

    _render_patch_gallery(axs[1], result.patch_arrays or [], n=8,
                          title="Extracted patches (64 × 64, individually normalised)")

    fig.suptitle("Random mode — fixed-size patches at random locations",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = OUTPUT_DIR / "random_mode_example.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, result


def save_whole_fov_mode(image_n, image):
    """Whole-FOV figure: border box on image | resized output patch."""
    output_sz = 256
    cfg = CellPatcherConfig(
        mode="whole_fov",
        common=CommonConfig(image_axes="YX", extract_arrays=True),
        extraction=ExtractionConfig(output_size=output_sz),
    )
    result = run_patching(images=image, config=cfg)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    axs[0].imshow(image_n, cmap="gray", vmin=0, vmax=1)
    draw_patch_boxes(
        axs[0], result.patches, color="#66FF66",
        max_boxes=None, linewidth=4.0, fill_alpha=0.0,
        label_lines=["Whole FOV mode", "1 patch = entire image"],
    )
    axs[0].set_title("Single patch covering the full field of view", fontsize=10)
    axs[0].axis("off")

    if result.patch_arrays:
        arr = np.asarray(result.patch_arrays[0])
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        axs[1].imshow(normalize_percentile(arr), cmap="gray", vmin=0, vmax=1)
        axs[1].set_title(
            f"Resized output  ({output_sz} × {output_sz} px)\n"
            "ready for downstream model input",
            fontsize=10,
        )
    axs[1].axis("off")

    fig.suptitle("Whole-FOV mode — one standardised patch per field of view",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = OUTPUT_DIR / "whole_fov_example.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, result


def save_modes_overview(image_n, cell_result, rand_result, grid_result, whole_result):
    """2 × 2 overview comparing all four patching modes on the same image."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
    panels = [
        (axs[0, 0], "Cell mode",      cell_result.patches,  "#FFE033", None,  2.0, 0.0,  True),
        (axs[0, 1], "Random mode",    rand_result.patches,  "#FF44CC", None,  2.0, 0.08, True),
        (axs[1, 0], "Grid mode",      grid_result.patches,  "#00E5FF", None,  2.0, 0.0,  False),
        (axs[1, 1], "Whole-FOV mode", whole_result.patches, "#66FF66", None,  5.0, 0.0,  False),
    ]
    for ax, title, patch_list, color, maxb, lw, falpha, nums in panels:
        ax.imshow(image_n, cmap="gray", vmin=0, vmax=1)
        n = len(patch_list)
        draw_patch_boxes(
            ax, patch_list, color=color,
            max_boxes=maxb, linewidth=lw, fill_alpha=falpha,
            show_numbers=nums,
            label_lines=[title, f"{n} patch{'es' if n != 1 else ''}"],
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

    fig.suptitle(
        "CellPatcher — four patching modes on the same microscopy image",
        fontsize=12, fontweight="bold", y=1.01,
    )
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
    generated.append(save_fixture_overview(image_n, labels_rgb, labels))
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
