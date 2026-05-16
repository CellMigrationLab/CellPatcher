from __future__ import annotations

from pathlib import Path

import numpy as np

from cellpatcher.io import read_array

FIXTURE_DIR = Path(__file__).resolve().parent
REAL_TIFF_IMAGE_PATH = FIXTURE_DIR / "cellpatcher_demo_image.tif"
REAL_TIFF_LABELS_PATH = FIXTURE_DIR / "cellpatcher_demo_labels.tif"


def load_real_tiff_fixture() -> tuple[np.ndarray, np.ndarray]:
    image = read_array(REAL_TIFF_IMAGE_PATH)
    labels = read_array(REAL_TIFF_LABELS_PATH)
    return image, labels


def summarize_real_tiff_fixture() -> dict[str, object]:
    image, labels = load_real_tiff_fixture()
    unique_labels = np.unique(labels)
    foreground_labels = unique_labels[unique_labels > 0]
    return {
        "image_shape": image.shape,
        "label_shape": labels.shape,
        "image_dtype": str(image.dtype),
        "labels_dtype": str(labels.dtype),
        "unique_labels": unique_labels.tolist(),
        "n_foreground_labels": int(foreground_labels.size),
        "foreground_pixel_count": int(np.sum(labels > 0)),
        "image_min": int(np.min(image)),
        "image_max": int(np.max(image)),
    }
