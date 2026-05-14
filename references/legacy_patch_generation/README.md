# Legacy patch generation references

These files are provided as reference material for the CellPatcher refactor.

They contain useful ideas for:
- axis handling
- random patch sampling
- background patch filtering
- transform-style named operations

Do not import these modules directly into CellPatcher.
Do not preserve their old package dependencies.
Reimplement only the useful ideas in a simple, CellPatcher-specific way.
