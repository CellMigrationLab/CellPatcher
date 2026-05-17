# Developer notes

## Architecture

CellPatcher is coordinate-first: strategies generate `PatchRegion` metadata, then downstream steps optionally filter, extract arrays, apply label masks, resize, and export. Keep mode logic in the package backend and keep notebooks as API consumers.

## Pipeline order

`generate regions → filter → extract → apply mask → resize → export`

## Preprocessing boundary

C/Z/T handling is explicit preprocessing (`prepare_*` helpers and iterators). Core patching modes remain 2D `Y/X` and do not perform true 3D patching or automatic model prep.

## Test command

```bash
PYTHONPATH=src python -m pytest -q
```

## Notebook hygiene

- Use package APIs (`run_patching`, config dataclasses, helpers).
- Keep `# @title` UI cells where used.
- Strip outputs before committing.
- Avoid duplicating backend patching logic in notebooks.

## Regenerate docs figures

```bash
PYTHONPATH=src python scripts/generate_docs_figures.py
```
