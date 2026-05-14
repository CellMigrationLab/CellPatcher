
# AGENTS.md

## Project goal

CellPatcher is being refactored from a notebook-driven workflow into a clean reusable Python package for generating microscopy image patches.

The package should support:
- cell/object-centered patches from segmentation labels
- random patches
- whole-FOV patches
- grid/quadrant patches with optional overlap
- shared background/quality filters
- optional patch extraction/export

The core design should be coordinate-first:
1. generate patch regions
2. optionally filter regions
3. optionally extract arrays
4. optionally export files

Do not make Torch, DINO, or GPU processing part of the core package yet. Torch support may be added later as an optional adapter.

## Architecture preference

Use a simple strategy architecture:

- `src/cellpatcher/types.py`
- `src/cellpatcher/config.py`
- `src/cellpatcher/pipeline.py`
- `src/cellpatcher/axes.py`
- `src/cellpatcher/extract.py`
- `src/cellpatcher/filters.py`
- `src/cellpatcher/strategies/base.py`
- `src/cellpatcher/strategies/cell.py`
- `src/cellpatcher/strategies/random.py`
- `src/cellpatcher/strategies/whole_fov.py`
- `src/cellpatcher/strategies/grid.py`

Each strategy should generate `PatchRegion` coordinates and metadata. Strategies should not save files or run downstream analysis.

## Important design rules

- Keep implementation simple and explicit.
- Avoid clever abstractions.
- Do not rewrite the whole notebook at once.
- Do not import code from `references/legacy_patch_generation/`.
- Use reference files only as design inspiration.
- Do not add heavy dependencies unless specifically requested.
- Do not make PyTorch a required dependency.
- Keep functions testable outside the notebook.
- Prefer NumPy arrays and plain dataclasses for the core package.
- Keep image export optional.

## Reference files

Reference-only files may exist in:

`references/legacy_patch_generation/`

They contain useful ideas from CSBDeep-style patch generation:
- explicit axes handling
- random patch-center sampling
- background patch filtering
- percentile normalization
- memory checks
- transform-style named operations

Do not copy them directly. Reimplement only the relevant ideas in simple CellPatcher-specific code.

## Testing

Use pytest.

Before finishing a coding task, run:

```bash
python -m pytest
````

If a task changes packaging, also run:

```bash
python -m pip install -e .
python -m pytest
```

If notebook execution is later added to CI, use:

```bash
jupyter nbconvert --execute notebooks/CellPatcher/CellPatcher.ipynb --to notebook
```

Do not run notebook execution unless explicitly requested.

## Style

* Python 3.10+ or 3.11+ compatible, depending on the current package config.
* Use dataclasses for configuration.
* Add clear error messages.
* Keep public API stable and minimal.
* Add tests for every new strategy or utility.
* Avoid broad try/except blocks unless there is a clear fallback.

````

## Also add a reference README

```text
references/legacy_patch_generation/README.md
````

```markdown
# Legacy patch generation references

These files are reference material for the CellPatcher refactor.

They contain useful ideas for:
- axes validation and permutation
- random patch sampling
- background patch filtering
- percentile normalization
- memory checks
- transform-style named operations

Do not import these modules directly into CellPatcher.
Do not preserve their old package dependencies.
Do not copy large sections directly.

Instead, reimplement only the useful concepts in a simple, CellPatcher-specific way.

That gives Codex enough context to work effectively without overcomplicating the environment.

[1]: https://arxiv.org/abs/2601.20404?utm_source=chatgpt.com "On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents"
