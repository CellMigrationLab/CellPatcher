# Notebooks

Notebook status and conventions:

| Notebook | Status | Description |
|----------|--------|-------------|
| `CellPatcher/CellPatcher_v3.ipynb` | Active (preferred) | App-style notebook: a `CellPatcherApp` widget wraps configuration tabs; the full run/QC is an explicit, separate final cell for visibility. Preferred UI for current manual testing. |
| `CellPatcher/CellPatcher_v2.ipynb` | Active (fallback/reference) | Package-backed, step-by-step staged workflow notebook. Useful as a guided fallback and reference workflow. |
| `CellPatcher/CellPatcher.ipynb` | Legacy — preserved | Original notebook kept for provenance/reference; not the preferred path for current manual testing. |

## Design convention

Both v2 and v3 call `run_patching(images=…, masks=…, config=…)` directly with pre-loaded arrays.  
The notebooks handle folder scanning, image↔mask file matching, and array loading themselves (with progress feedback), so there is no benefit to delegating that to `run_patching_pipeline` — which would redo the same work silently.

Scripts and one-off runs that don't need interactive UI should use `run_patching_pipeline(folder, …)` directly.

## Runtime compatibility expectations

- **V3 (`CellPatcher_v3.ipynb`)** is expected to work in both local JupyterLab/LabConstrictor-style environments and Google Colab.
- Colab-specific installation/mount behavior must stay guarded so local Jupyter usage is not affected.
- Notebook UI should remain package-backed (no duplication of backend patching logic in notebook cells).

## Bundled real TIFF fixture

Bundled fixture files:
- `tests/fixtures/cellpatcher_demo_image.tif`
- `tests/fixtures/cellpatcher_demo_labels.tif`

Intended use:
- pytest smoke/integration tests
- quick notebook smoke/manual testing

Not intended for benchmarking or biological interpretation.

Notes:
- `CellPatcher_strategy_examples.ipynb` can optionally load the fixture (default is synthetic data).
- `CellPatcher_v3.ipynb` provides a "Load bundled TIFF demo fixture" button for quick app setup.
