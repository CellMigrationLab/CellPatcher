# AGENTS.md

## Project goal

CellPatcher is a reusable backend Python package (under `src/cellpatcher`) plus notebooks for microscopy patch generation workflows.

Current notebook status:
- Legacy notebook is preserved at `notebooks/CellPatcher/CellPatcher.ipynb` (reference/provenance; not the preferred path).
- Notebook v2 is available at `notebooks/CellPatcher/CellPatcher_v2.ipynb` (package-backed staged workflow).
- Notebook v3 is available at `notebooks/CellPatcher/CellPatcher_v3.ipynb` (app-style/tabbed workflow and preferred manual-testing path when present).

## Core backend design

CellPatcher is **coordinate-first**:
1. generate patch regions (`PatchRegion` coordinates + metadata)
2. optionally filter regions
3. optionally extract arrays
4. optionally apply label mask
5. optionally resize extracted arrays
6. optionally export files

Strategies generate patch coordinates; downstream extraction/export is handled by the pipeline.

Keep Torch/DINO/GPU/model-specific logic out of the core package.

## Supported patch modes

- `cell`
- `random`
- `grid`
- `whole_fov`

## Important modules

- `src/cellpatcher/config.py`
- `src/cellpatcher/types.py`
- `src/cellpatcher/axes.py`
- `src/cellpatcher/io.py`
- `src/cellpatcher/cell_scan.py`
- `src/cellpatcher/pipeline.py`
- `src/cellpatcher/filters.py`
- `src/cellpatcher/extract.py`
- `src/cellpatcher/masking.py`
- `src/cellpatcher/resize.py`
- `src/cellpatcher/export.py`
- `src/cellpatcher/strategies/`

## Important behavior

- `image_axes` and `mask_axes` are explicit and axis-aware.
- Image/mask matching supports:
  - `exact`
  - `exact_or_suffix` (default)
  - `substring` (legacy, permissive)
- `scan_cell_patch_size` computes dataset-level patch-size recommendations.
- `ExtractionConfig.apply_mask` is cell-mode oriented and clears pixels outside selected labels before resize/export.
- Export can write patch files and metadata CSV.
- Files in `references/legacy_patch_generation/` are reference-only and not production code.

## Testing

Primary command:

```bash
PYTHONPATH=src python -m pytest -q
```

Current health snapshot (latest audit):
- 353 tests passing.

## Development guidance

- Backend remains package-backed and coordinate-first.
- Do not copy backend logic into notebooks.
- Notebooks should call package APIs (`run_patching`, config dataclasses, utility helpers).
- Notebook code should not duplicate backend patching logic.
- Keep Torch/DINO/model-specific logic out of core package.
- If optional ML support is added later, keep it optional and model-agnostic.
- Do not import code from `references/legacy_patch_generation/`.

