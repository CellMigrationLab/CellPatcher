# Test fixtures

This folder contains bundled, deterministic test fixtures used by CellPatcher tests and notebooks.

## Real TIFF smoke fixture

- `cellpatcher_demo_image.tif`
- `cellpatcher_demo_labels.tif`

### Purpose

Use this pair for:
- pytest smoke/integration coverage (`tests/test_real_tiff_fixture_modes.py`)
- quick notebook/manual validation (`CellPatcher_strategy_examples.ipynb`, `CellPatcher_v3.ipynb`)

Do **not** use this fixture for benchmarking or biological conclusions.

### Required properties

- single-channel real TIFF image
- associated label TIFF with matching `YX` shape
- stable small size suitable for CI/repo test runs
- no external downloads required

### Loading

Use helper utilities:
- `tests/fixtures/helpers.py::load_real_tiff_fixture()`
- `tests/fixtures/helpers.py::summarize_real_tiff_fixture()`

Both loaders use package I/O (`cellpatcher.read_array`) so fixture behavior matches runtime loading paths.

### Updating the fixture

If this fixture is intentionally replaced/cropped:
1. Recompute fixture summary constants in `tests/test_real_tiff_fixture_modes.py`.
2. Re-validate expected patch counts used in real-TIFF tests.
3. Run full tests: `PYTHONPATH=src python -m pytest -q`.
4. Keep fixture files small and deterministic.
