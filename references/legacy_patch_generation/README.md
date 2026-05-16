# Legacy patch generation references

This folder contains **reference-only** legacy files used as inspiration during the CellPatcher refactor.

## Usage policy

- These files are **not imported** by the current CellPatcher package.
- Do **not** copy code directly from these files into production modules.
- Reimplement ideas in a simple CellPatcher-specific form when needed.

## Useful ideas already incorporated (conceptually)

- axis handling concepts
- random sampling concepts
- background filtering concepts
- memory/checking inspiration

## Intentionally not incorporated into core package

- CSBDeep source/target training abstractions
- generator transform stack design
- legacy `six`/old import style
- anisotropic distortion simulation workflows

## Lifecycle note

These files are historical reference only.
Useful ideas have already been reimplemented where appropriate in the package-backed workflow.
Revisit archive/deletion decisions after Notebook v3 app-style workflow has been manually tested.
