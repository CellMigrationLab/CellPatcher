# Notebooks

This folder contains multiple notebooks because they serve different purposes.

## Quick recommendation

- **Use `CellPatcher_v3.ipynb` for normal use and manual testing.**
- **Use `CellPatcher_strategy_examples.ipynb` to learn the Python API fastest.**
- Use `CellPatcher_v2.ipynb` as a staged fallback/reference workflow.
- Use `CellPatcher.ipynb` only as historical reference.

## Which notebook should I open?

| I want to... | Open this notebook |
|---|---|
| use the app-style interface | `CellPatcher/CellPatcher_v3.ipynb` |
| learn the API with minimal examples | `CellPatcher/CellPatcher_strategy_examples.ipynb` |
| follow every step manually | `CellPatcher/CellPatcher_v2.ipynb` |
| inspect the old workflow | `CellPatcher/CellPatcher.ipynb` |

## Notebook status

| Notebook | Status | Description |
|---|---|---|
| `CellPatcher/CellPatcher_v3.ipynb` | recommended | App-style configuration tabs (`CellPatcherApp`) plus explicit run/QC cells. Preferred manual-testing notebook. |
| `CellPatcher/CellPatcher_strategy_examples.ipynb` | recommended examples | Minimal API-focused examples covering `cell`, `random`, `grid`, and `whole_fov` modes. |
| `CellPatcher/CellPatcher_v2.ipynb` | active fallback/reference | Guided, staged package-backed workflow for users who want a step-by-step path. |
| `CellPatcher/CellPatcher.ipynb` | legacy/reference only | Historical notebook preserved for provenance; not recommended for new work. |
| `CellPatcher/CellPatcher_torch_colab_validation.ipynb` | developer validation | Colab-first validation notebook for optional Torch adapter: installs Torch, runs full pytest, Torch-specific tests, heavy real-fixture E2E checks, and tensor parity checks; not a normal user-facing workflow. |

## Runtime notes

- **Local JupyterLab / LabConstrictor:** V3 is expected to run directly in local notebook environments.
- **Google Colab:** V3 and strategy examples are designed to work in Colab when Colab-specific setup is guarded.
- **Private repository install in Colab (if needed):** use a `GITHUB_TOKEN` flow for authenticated installs.
- **Colab Drive paths:** paths often look like `/content/gdrive/MyDrive/...`; verify image/mask paths before running.

## Bundled demo fixture

Bundled real TIFF fixture files:

- `tests/fixtures/cellpatcher_demo_image.tif`
- `tests/fixtures/cellpatcher_demo_labels.tif`

You can use these for quick smoke tests in notebooks and for reproducible demos.

## Notebook hygiene

- Keep interactive code cells titled with `# @title` where applicable.
- Strip notebook outputs before committing.
- Do not duplicate backend patching logic in notebooks.
- Use package APIs (`run_patching`, config dataclasses, I/O helpers).
