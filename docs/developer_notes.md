# Developer notes

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

## Architecture

CellPatcher is coordinate-first: strategies generate `PatchRegion` metadata, then downstream steps optionally filter, extract arrays, apply label masks, resize, and export. Keep mode logic in the package backend and keep notebooks as API consumers.

## Pipeline order

`generate regions → filter → extract → apply mask → resize → export`

## Preprocessing boundary

C/Z/T handling is explicit preprocessing (`prepare_*` helpers and iterators). Core patching modes remain 2D `Y/X` and do not perform true 3D patching or automatic model prep.

## Notebook hygiene

- Use package APIs (`run_patching`, config dataclasses, helpers).
- Keep `# @title` UI cells where used.
- Strip outputs before committing.
- Avoid duplicating backend patching logic in notebooks.

## Regenerate docs figures

```bash
PYTHONPATH=src python scripts/generate_docs_figures.py
```


## Validation and tests

Primary test command:

```bash
PYTHONPATH=src python -m pytest -q
```

Current health snapshot (latest audit): 417 tests passing.

## Demo fixture (for smoke tests and docs demos)

A small real TIFF fixture is bundled for smoke tests and notebook demos:

- `tests/fixtures/cellpatcher_demo_image.tif`
- `tests/fixtures/cellpatcher_demo_labels.tif`

Use it for validation and examples, not for benchmarking or biological conclusions.

## Repository development notes

- Package code lives in `src/cellpatcher`.
- Notebooks should call package APIs and avoid duplicating backend patching logic.
- See `AGENTS.md` for contributor/agent guidance.
- See `notebooks/README.md` for notebook-specific guidance.
