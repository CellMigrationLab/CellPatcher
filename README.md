# CellPatcher

CellPatcher helps you turn microscopy images into reproducible patch datasets without wrestling with custom one-off scripts. It generates patch coordinates (and, if you want, extracted arrays) using segmentation-based cell patches, random patches, grid tiles, and whole-field patches. You can drive everything from interactive notebooks or the Python API.

## Start here

- **Normal user / manual workflow:** open `notebooks/CellPatcher/CellPatcher.ipynb`.
- **Python learner:** open `notebooks/CellPatcher_Strategy_Examples/CellPatcher_Strategy_Examples.ipynb`.
- **API copy-paste examples:** read `docs/api_examples.md`.
- **Multidimensional C/Z/T data:** read `docs/preprocessing.md`.

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

- **Recommended:** `notebooks/CellPatcher/CellPatcher.ipynb` (app-style workflow and preferred manual testing path)
- **API learning examples:** `notebooks/CellPatcher_Strategy_Examples/CellPatcher_Strategy_Examples.ipynb`

Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CellMigrationLab/CellPatcher/blob/main/notebooks/CellPatcher/CellPatcher.ipynb)

## Common user-facing APIs vs advanced helpers

**Common user-facing APIs**
- `CellPatcherConfig` and config dataclasses
- `run_patching` and `run_patching_pipeline`
- `read_array` and `validate_image_and_mask_folders`
- `scan_cell_patch_size`
- `prepare_image_for_patching` and `iter_image_planes_for_patching`

**Advanced helpers**
- low-level strategy classes
- axis utilities
- extract/resize/export helper functions
- other lower-level utility APIs

## Quick start: Python API

- Start here: [Quick start](docs/quick_start.md)
- More usage patterns: [API examples](docs/api_examples.md)
- Full options and fields: [Configuration reference](docs/configuration_reference.md)

## Installation / local development

**Local / JupyterLab:**

- From the repository root, install the package: `python -m pip install -e .`
- Add extras as needed:
  - tests: `python -m pip install -e .[test]`
  - notebooks: `python -m pip install -e .[notebook]`
  - optional Torch adapter: `python -m pip install -e .[torch]`
- `requirements.txt` remains a pinned notebook/tooling environment snapshot.

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
    f"cellpatcher @ git+https://x-access-token:{encoded}@github.com/CellMigrationLab/CellPatcher.git",
], check=True)
```

The notebooks `CellPatcher.ipynb` contain a ready-to-run version of this install cell — run it first when opening in Colab.

## Optional preprocessing for multidimensional data

Patching modes are 2D Y/X; for C/Z/T data, see `docs/preprocessing.md`.

Note: plane iterator `metadata` (the `t`/`z` indices) reaches `PatchRegion` rows and the export CSV when passed through as `run_patching(..., image_metadata=...)`; the `plane_id` string itself is not mirrored onto `PatchRegion` and still needs an explicit join if you need it. See `docs/preprocessing.md`.

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
