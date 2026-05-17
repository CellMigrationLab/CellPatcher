# CellPatcher

CellPatcher helps you turn microscopy images into reproducible patch datasets without wrestling with custom one-off scripts. It generates patch coordinates (and, if you want, extracted arrays) using segmentation-based cell patches, random patches, grid tiles, and whole-field patches. You can drive everything from interactive notebooks or the Python API.

## Start here in 3 steps

1. **Choose your path**
   - App-style notebook: `notebooks/CellPatcher/CellPatcher_v3.ipynb`
   - Minimal API learning: `notebooks/CellPatcher/CellPatcher_strategy_examples.ipynb`
   - Python-first docs: `docs/quick_start.md`
2. **Pick a patching mode** (`cell`, `random`, `grid`, `whole_fov`) below.
3. **Working with Z/T/C axes?** Patching modes are 2D Y/X; see `docs/preprocessing.md`.

- Matches image files and mask/label files with configurable matching strategies.
- Generates patch regions (`PatchRegion`) for `cell`, `random`, `grid`, and `whole_fov` modes.
- Optionally filters regions, extracts arrays, resizes outputs, and exports patch files.
- Optionally clears pixels outside the selected segmented object in cell mode (`apply_mask=True`).
- Writes patch metadata (CSV) alongside exported patch files when enabled.

## Patching modes at a glance

![CellPatcher patching modes](docs/assets/patching_modes_overview.png)

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

## Quick start: notebooks

- **Recommended:** `notebooks/CellPatcher/CellPatcher_v3.ipynb` (app-style workflow and preferred manual testing path)
- **API learning examples:** `notebooks/CellPatcher/CellPatcher_strategy_examples.ipynb`
- **Guided fallback:** `notebooks/CellPatcher/CellPatcher_v2.ipynb`
- **Legacy reference only:** `notebooks/CellPatcher/CellPatcher.ipynb`

Open the v3 notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellPatcher/blob/main/notebooks/CellPatcher/CellPatcher_v3.ipynb)

## Quick start: Python API

- Start here: [Quick start](docs/quick_start.md)
- More usage patterns: [API examples](docs/api_examples.md)
- Full options and fields: [Configuration reference](docs/configuration_reference.md)

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

## Optional preprocessing for multidimensional data

Patching modes are 2D Y/X; for C/Z/T data, see `docs/preprocessing.md`.

Note: plane iterator metadata (`metadata`, `plane_id`) is available on helper outputs, but it is not automatically copied into `PatchRegion` rows or export CSV metadata.

## Limitations / non-goals

- CellPatcher does **not** segment cells; you must provide masks/labels for cell mode.
- CellPatcher does **not** run DINO/Torch inference in the core package.
- Users must provide valid images and, for cell mode, masks/labels.

## More documentation

- [Documentation map](docs/README.md)
- [Quick start](docs/quick_start.md)
- [Patching mode guide](docs/patching_modes.md)
- [API examples](docs/api_examples.md)
- [Preprocessing guide](docs/preprocessing.md)
- [Configuration reference](docs/configuration_reference.md)
- [Developer notes](docs/developer_notes.md)
- [Notebook guide](notebooks/README.md)
- [Fixture guide](tests/fixtures/readme.md)


- Optional Torch dataset adapter is available via `cellpatcher.torch_dataset`; Torch remains optional and DINO inference is not included in core (see `docs/api_examples.md`).
