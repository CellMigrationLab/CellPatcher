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
