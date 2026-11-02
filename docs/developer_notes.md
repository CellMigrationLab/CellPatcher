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
  without them) and its pages `cellpatcher.notebook_ui`/`notebook_form`. The strategy examples
  notebook is `cellpatcher.notebook_examples.StrategyExamplesNotebook` (the same actions, one
  page per strategy) and the PyTorch notebook `cellpatcher.notebook_torch.TorchNotebook`
  (its action is `cellpatcher.torch_session.load_exported_batch`). The before/after views of the
  preview are `cellpatcher.normalization_review` (pure; histograms from the normalization brick).
- Every notebook's first cell is the same (`tests/test_notebook_file.py` checks it), except its
  title, `notebook_name`, `EXTRAS` and `INTERFACE` (`"module:class"` of its page); change all
  three together.
- LabConstrictor portability (`.tools/docs/notebook_portability.md`): `# @title` first on every
  code cell; Colab-only code (Drive, the widget manager, pip through `subprocess`) only inside
  `if "google.colab" in sys.modules:`; no magics and no `/content` path outside it; the GitHub
  install pinned to `RELEASE_TAG`; `current_version`/`notebook_name` with the
  `notebooks/notebook_latest_versions.yaml` entry. `tests/test_notebook_file.py` checks these
  for every notebook and runs each in a local kernel (nbclient; the strategy examples and the
  PyTorch notebook with a test run or export filled in; the PyTorch notebook only with `torch`
  installed, which no CI job has today).
- Before tagging a release: set `RELEASE_TAG` and `current_version` in every notebook's first
  cell and in `notebooks/notebook_latest_versions.yaml`, `version` in `setup.py` and
  `__version__` to the new version.
- **No installer for a release candidate** (owner decision of 2026-11-02): an `rc` tag
  (`v0.2.0rc1`) is installed with pip or on Colab only. The release workflow's
  `.tools/python/bump_version.py` refuses a pre-release version with that reason before it
  changes anything, so no installer is built; the installers are built at the final release
  (`v0.2.0`). `tests/test_installers.py` checks the refusal (the `.tools` folder is synced from
  the LabConstrictor template, which would drop it).
- **Installer package layout (fixed in CP-3b, 2026-11-02).** This repository's copy of the
  template's `.tools/python/bump_constructor.py` (synced before LabConstrictor fixed it in
  `a76be61`, 2026-07-06) maps `src/cellpatcher/x.py` to `CellPatcher/src/cellpatcher/cellpatcher/x.py`,
  where `find_packages(where="src")` found no package: the installed app had no `cellpatcher`
  module. The current template maps it to `CellPatcher/src/cellpatcher/x.py`. `setup.py` now
  reads the package from `src/cellpatcher/` when it finds `src/cellpatcher/cellpatcher/__init__.py`
  (the old layout) and from `src/` otherwise, so both install the whole package;
  `tests/test_installers.py` copies the files as each layout lists them, pip-installs them and
  imports `cellpatcher`. `construct.yaml` is left as the release workflow writes it. Syncing the
  template (`.github` and `.tools`) would also replace `tests.yml` (the rules job and the extras
  matrix) and `bump_version.py` (the release-candidate refusal): the owner decided (2026-11-02)
  that one PR before the final `v0.2.0` syncs the template, re-applies those files' CellPatcher
  changes (proposing them upstream) and adds a CPU-torch CI job (`docs/plan.md`). Until then CI
  skips the PyTorch notebook's kernel run and the torch tests; run them locally with CPU `torch`
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
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
