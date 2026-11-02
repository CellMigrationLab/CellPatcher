# Developer notes

> See also: [Docs map](README.md) · [Quick start](quick_start.md) · [Patching modes](patching_modes.md) · [API examples](api_examples.md) · [Preprocessing](preprocessing.md) · [Config reference](configuration_reference.md)

## Architecture

CellPatcher is coordinate-first: strategies generate `PatchRegion` metadata, then downstream steps optionally filter, extract arrays, apply label masks, resize, and export. Keep mode logic in the package backend and keep notebooks as API consumers.

## Pipeline order

`generate regions → filter → extract → apply mask → resize → export`

## Preprocessing boundary

C/Z/T handling is explicit preprocessing (`prepare_*` helpers and iterators). Core patching modes remain 2D `Y/X` and do not perform true 3D patching or automatic model prep.

## Notebook hygiene

- The CellPatcher notebook only creates `cellpatcher.notebook_ui.CellPatcherNotebook` and calls
  one step per cell; its actions are `cellpatcher.notebook_session` (no widgets, tested
  without them) and its pages `cellpatcher.notebook_ui`/`notebook_form`.
- LabConstrictor portability (`.tools/docs/notebook_portability.md`): `# @title` first on every
  code cell; Colab-only code (Drive, the widget manager, pip through `subprocess`) only inside
  `if "google.colab" in sys.modules:`; no magics and no `/content` path outside it; the GitHub
  install pinned to `RELEASE_TAG`; `current_version`/`notebook_name` with the
  `notebooks/notebook_latest_versions.yaml` entry. `tests/test_notebook_file.py` checks these
  and runs the notebook in a local kernel (nbclient).
- Before tagging a release: set `RELEASE_TAG` and `current_version` in the notebook's first
  cell, `version` in `setup.py` and `__version__` to the new version.
- Strip outputs before committing; avoid duplicating backend patching logic in notebooks.

## Regenerate docs figures

```bash
PYTHONPATH=src python .tools/python/generate_docs_figures.py
```


## Validation and tests

Before a pull request:

```bash
python tools/check.py           # pytest, Ruff and the code rules
python tools/check.py --rules   # only Ruff and the code rules (CI)
```

The code rules are the MorphoBricks standards (design v0.4 §9): the shared
`tools/code_rules.py` (identical in every MorphoBricks repository, checked by its SHA-256),
configured in `pyproject.toml`. Existing violations, and Ruff's findings counted per file,
function and rule, are in `tools/code_rules_baseline.json`, which may only shrink
(`--update-baseline` lowers it; it never adds an entry), with a ratchet on `# noqa` counts. New
and moved code must comply.

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
