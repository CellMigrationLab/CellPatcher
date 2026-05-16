# Documentation figures

These figures are generated from the bundled real TIFF fixture:

- `tests/fixtures/cellpatcher_demo_image.tif`
- `tests/fixtures/cellpatcher_demo_labels.tif`

Regenerate with:

```bash
PYTHONPATH=src python scripts/generate_docs_figures.py
```

Notes:

- Figures are for documentation/visual explanation only.
- Microscopy image display uses percentile normalization (ImageJ-style contrast for visualization).
- Label display uses a deterministic Glasbey-style categorical LUT fallback.

## Asset usage map

All PNG assets in this folder are used by repository documentation:

| Asset | Used in |
|---|---|
| `patching_modes_overview.png` | `README.md`, `docs/patching_modes.md` |
| `cell_mode_apply_mask.png` | `docs/patching_modes.md` |
| `random_mode_example.png` | `docs/patching_modes.md` |
| `grid_mode_example.png` | `docs/patching_modes.md` |
| `whole_fov_example.png` | `docs/patching_modes.md` |
| `fixture_overview.png` | `docs/patching_modes.md` |
