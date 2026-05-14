import numpy as np


def make_synthetic_cell_image_and_labels(shape=(96, 128), seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape

    image = rng.normal(loc=0.1, scale=0.02, size=(h, w)).astype(np.float32)
    labels = np.zeros((h, w), dtype=np.int32)

    cells = {
        1: (20, 35, 20, 38),   # interior
        2: (45, 63, 55, 78),   # interior
        3: (65, 82, 90, 112),  # interior
        4: (0, 12, 8, 22),     # border-touching
        5: (88, 90, 5, 7),     # tiny (area 4)
    }

    for label_id, (y0, y1, x0, x1) in cells.items():
        labels[y0:y1, x0:x1] = label_id
        image[y0:y1, x0:x1] += 0.5 + 0.05 * label_id

    metadata = {
        "expected_valid_labels": [1, 2, 3],
        "expected_border_labels": [4],
        "expected_tiny_labels": [5],
        "tiny_label_area": 4,
    }
    return image.astype(np.float32), labels.astype(np.int32), metadata
