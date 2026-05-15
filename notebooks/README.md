# Notebooks

Notebook status and conventions:

| Notebook | Status | Description |
|----------|--------|-------------|
| `CellPatcher/CellPatcher.ipynb` | Legacy — preserved | Original notebook; kept as reference only. |
| `CellPatcher/CellPatcher_v2.ipynb` | Active | Step-by-step widget notebook backed by the `cellpatcher` package. Each workflow stage (folder validation, axes, mode settings, scan, preview, export) is its own titled cell. |
| `CellPatcher/CellPatcher_v3.ipynb` | Active (latest) | App-style notebook: a single `CellPatcherApp` widget wraps all configuration tabs; the full run/QC is an explicit, separate cell at the end for visibility. |

## Design convention

Both v2 and v3 call `run_patching(images=…, masks=…, config=…)` directly with pre-loaded arrays.  
The notebooks handle folder scanning, image↔mask file matching, and array loading themselves (with progress feedback), so there is no benefit to delegating that to `run_patching_pipeline` — which would redo the same work silently.

Scripts and one-off runs that don't need interactive UI should use `run_patching_pipeline(folder, …)` directly.
