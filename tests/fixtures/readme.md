# CellPatcher demo fixtures

This folder contains small, deterministic TIFF fixtures used for smoke testing and notebook demos.

## Files in this folder

- `cellpatcher_demo_image.tif`
- `cellpatcher_demo_labels.tif`

## Technical details

| File | Shape | Dtype | File size |
|---|---:|---:|---:|
| `cellpatcher_demo_image.tif` | `(1024, 1024)` | `uint16` | `2,117,933` bytes (~2.02 MiB) |
| `cellpatcher_demo_labels.tif` | `(1024, 1024)` | `uint8` | `1,070,953` bytes (~1.02 MiB) |

Label-mask summary (`cellpatcher_demo_labels.tif`):

- Foreground labels (non-zero): `10`
- Unique label values including background: `11`
- Foreground pixel count (`labels > 0`): `686,973`

## What this fixture is for

Use this pair for:

- pytest smoke/integration checks (especially `tests/test_real_tiff_fixture_modes.py`)
- quick notebook smoke tests
- reproducible small examples

## What this fixture is not for

Do **not** use this fixture for:

- benchmarking
- biological conclusions
- performance claims

## Update policy

If this fixture pair is replaced or edited:

1. Keep files small and deterministic.
2. Update test expectations in `tests/test_real_tiff_fixture_modes.py` (shape/dtype/count constants and any expected mode outputs).
3. Re-run the test suite:
   - `PYTHONPATH=src python -m pytest -q`
