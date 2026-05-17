# CellPatcher

CellPatcher is a package-backed microscopy patch generation project for scientists, notebook users, and Python users who want reproducible patch extraction workflows. It creates patch coordinates (and optional extracted arrays) from microscopy images using multiple strategies, including segmentation-based cell patches, random patches, grid tiles, and whole-field patches. You can run it through interactive notebooks or directly through the Python API, depending on your workflow.

## What CellPatcher does

## Start here in 3 steps

1. **Choose your path**
   - App-style notebook: `notebooks/CellPatcher/CellPatcher_v3.ipynb`
   - Minimal API learning: `notebooks/CellPatcher/CellPatcher_strategy_examples.ipynb`
   - Python-first docs: `docs/api_examples.md`
2. **Choose a patching mode** (`cell`, `random`, `grid`, `whole_fov`) in `docs/patching_modes.md`.
3. **If your data has Z/T/C axes**, read `docs/preprocessing.md` first. Core patching modes generate 2D `Y/X` regions; multidimensional handling is explicit preprocessing.

- Matches image files and mask/label files with configurable matching strategies.
- Generates patch regions (`PatchRegion`) for `cell`, `random`, `grid`, and `whole_fov` modes.
- Optionally filters regions, extracts arrays, resizes outputs, and exports patch files.
- Optionally clears pixels outside the selected segmented object in cell mode (`apply_mask=True`).
- Writes patch metadata (CSV) alongside exported patch files when enabled.

## Which patching mode should I use?

| Mode | Use it when... | Needs masks? | Typical output |
|---|---|---|---|
| `cell` | You have segmentation labels and want one patch per object/cell. | Yes | One patch per labeled object/cell |
| `random` | You want random fixed-size sampling for controls or coverage checks. | Optional | Random fixed-size patches |
| `grid` | You want deterministic tiling over the image. | No | Regular tiles (with optional overlap/partials) |
| `whole_fov` | You want one patch representing the entire field of view. | No | One patch covering the full image |

## Workflow at a glance

```text
images/masks → CellPatcherConfig → run_patching → PatchRegion metadata + patch arrays and/or exported files
```

All four modes use the same orchestration function: `run_patching`. The notebooks build `CellPatcherConfig` for you and call the same backend API.


## Patching modes at a glance

![CellPatcher patching modes](docs/assets/patching_modes_overview.png)

## Quick start: notebooks

| Notebook | Best for | Status |
|---|---|---|
| `notebooks/CellPatcher/CellPatcher_v3.ipynb` | app-style UI, manual testing, most users | recommended |
| `notebooks/CellPatcher/CellPatcher_strategy_examples.ipynb` | learning the Python API and modes | recommended examples |
| `notebooks/CellPatcher/CellPatcher_v2.ipynb` | step-by-step guided workflow | fallback/reference |
| `notebooks/CellPatcher/CellPatcher.ipynb` | historical notebook | legacy/reference only |

V3 and strategy examples are the best starting points for most users. The legacy notebook is preserved for provenance and is not recommended for new work.

## Quick start: Python API

For the most readable getting-started examples (cell mode and grid mode), see:

- `docs/quick_start.md`

For more scenario-based snippets:

- `docs/api_examples.md`

## Real TIFF demo fixture

A small real TIFF fixture is bundled for smoke tests and notebook demos:

- `tests/fixtures/cellpatcher_demo_image.tif`
- `tests/fixtures/cellpatcher_demo_labels.tif`

Use it for validation and examples, not for benchmarking or biological conclusions.

## Installation / local development

**Local / JupyterLab:**

- From the repository root, use `PYTHONPATH=src` for development commands.
- If needed, install editable from source: `pip install -e .`.
- Notebook/runtime dependencies are pinned in `requirements.txt` and notebook-specific requirements files.

**Google Colab (private repo):**

While the repository is private, install from GitHub using a personal access token stored as a Colab Secret (`GITHUB_TOKEN`):

```python
import subprocess, sys
from urllib.parse import quote
from google.colab import userdata

token = userdata.get("GITHUB_TOKEN")
encoded = quote(token, safe="")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    f"cellpatcher @ git+https://x-access-token:{encoded}@github.com/guijacquemet/CellPatcher.git",
], check=True)
```

The notebooks (`CellPatcher_v3.ipynb`, `CellPatcher_v2.ipynb`) contain a ready-to-run version of this install cell — run it first when opening in Colab.

## Running tests

```bash
PYTHONPATH=src python -m pytest -q
```

Run the command above to verify the current test suite in your environment.


## Optional preprocessing for multidimensional data

CellPatcher generates 2D Y/X patch regions. If your image has Z or T axes, you can either select/project a single 2D plane before patching, or expand the stack into multiple 2D planes and run the same patching mode on each plane.

For a human-friendly guide to preprocessing helpers and plane iterators, see:

- `docs/preprocessing.md`

Note:

- Plane iterator metadata (`metadata`, `plane_id`) is returned by helper objects and is not automatically injected into `PatchRegion` or export CSV metadata.

## Limitations / non-goals

- CellPatcher does **not** segment cells; you must provide masks/labels for cell mode.
- CellPatcher does **not** run DINO/Torch inference in the core package.
- Demo fixtures are for smoke testing, not biological benchmarking.
- Users must provide valid images and, for cell mode, masks/labels.

## Development notes

- Package code lives in `src/cellpatcher`.
- Notebooks should call package APIs and avoid duplicating backend patching logic.
- See `AGENTS.md` for contributor/agent guidance.
- See `notebooks/README.md` for notebook-specific guidance.


## More documentation

- [Patching mode guide](docs/patching_modes.md)
- [API examples](docs/api_examples.md)
- [Preprocessing guide](docs/preprocessing.md)
- [Notebook guide](notebooks/README.md)
- [Fixture guide](tests/fixtures/readme.md)
- [Quick start](docs/quick_start.md)

