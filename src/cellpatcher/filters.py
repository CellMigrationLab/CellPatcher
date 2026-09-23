import numpy as np

from cellpatcher.config import FilterConfig
from cellpatcher.extract import extract_patch


def compute_threshold(image, threshold):
    """Resolve a scalar foreground threshold from an image and threshold spec.

    Note:
        The threshold is computed over *all* values in ``image`` after ``np.asarray``.
        For multidimensional arrays (e.g. ``CYX``, ``TCZYX``), this includes all channels
        and non-spatial axes. No per-channel projection/selection is performed here.
        If channel-specific filtering is needed, pass a preselected/projection image.
    """
    arr = np.asarray(image)
    if threshold is None:
        return None
    if isinstance(threshold, (int, float)):
        return float(threshold)
    if isinstance(threshold, str):
        # String thresholds are resolved against `image`'s own statistics
        # rather than being fixed numbers, so the same config works across
        # images with different intensity ranges.
        t = threshold.strip().lower()
        if t == "mean":
            return float(np.mean(arr))
        if t == "median":
            return float(np.median(arr))
        if t == "otsu":
            try:
                from skimage.filters import threshold_otsu
            except ImportError as exc:
                raise ImportError("foreground_threshold='otsu' requires scikit-image.") from exc
            return float(threshold_otsu(arr))
        if t.startswith("percentile:"):
            try:
                value = float(t.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError("Invalid percentile threshold format. Use 'percentile:<0-100>'.") from exc
            if value < 0 or value > 100:
                raise ValueError("Percentile threshold must be in [0, 100].")
            return float(np.percentile(arr, value))
        raise ValueError(f"Unsupported threshold string: {threshold}")
    raise ValueError("threshold must be None, a scalar number, or a supported string.")


def foreground_fraction(patch, threshold) -> float:
    """Return fraction of values in ``patch`` above ``threshold``.

    Note:
        The fraction is computed over *all* values in the extracted patch (including
        channel/z/time axes when present). No per-channel reduction policy exists yet.
    """
    if threshold is None:
        raise ValueError("threshold is required for foreground_fraction.")
    arr = np.asarray(patch)
    return float(np.mean(arr > threshold))


def filter_regions(image, regions, config: FilterConfig, *, axes: str = "YX", pad_mode: str = "clip", constant_values=0):
    """Filter regions by intensity and/or foreground fraction.

    Current semantics are intentionally simple:
    - threshold statistics are computed over the full ``image`` array;
    - foreground fraction is computed over all values of each extracted patch.
    Users needing channel-specific behavior should pass a projected/selected image.
    """
    # No filters configured: skip extracting every patch just to keep all of
    # them.
    if (
        config.min_foreground_fraction is None
        and config.foreground_threshold is None
        and config.min_mean_intensity is None
    ):
        return list(regions)

    if config.min_foreground_fraction is not None:
        frac = float(config.min_foreground_fraction)
        if frac < 0 or frac > 1:
            raise ValueError("min_foreground_fraction must be in [0, 1].")

    # Resolve the threshold once against the full image, rather than
    # per-patch, so every region is judged against the same reference.
    threshold = None
    if config.foreground_threshold is not None:
        threshold = compute_threshold(image, config.foreground_threshold)

    kept = []
    for region in regions:
        patch = extract_patch(image, region, axes=axes, pad_mode=pad_mode, constant_values=constant_values)

        if config.min_mean_intensity is not None:
            if float(np.mean(patch)) < float(config.min_mean_intensity):
                continue

        if config.min_foreground_fraction is not None:
            # No explicit threshold was given: fall back to the image's own
            # mean intensity as the foreground/background split point.
            effective_threshold = threshold
            if effective_threshold is None:
                effective_threshold = float(np.mean(np.asarray(image)))
            if foreground_fraction(patch, effective_threshold) < float(config.min_foreground_fraction):
                continue

        kept.append(region)

    return kept
