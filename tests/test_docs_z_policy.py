"""Docs, README and docstrings never offer ``z_policy="all"`` as a way to patch several Z planes (issue #94).

CellPatcher refuses Z>1 on every route (issue #83). ``z_policy="all"`` still works for the one plane
of Z=1 (it names it ``z000``), but prose that mentions ``z_policy`` and ``"all"`` must say, in the same
paragraph, that Z>1 is refused, so no text reads as if ``"all"`` patched every Z plane of a stack.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ``z_policy`` and the value "all" close together (either order), in the same paragraph.
Z_POLICY_ALL = re.compile(r"""z_policy.{0,100}?["'`]all["'`]|["'`]all["'`].{0,100}?z_policy""", re.DOTALL)
# "Z>1 is refused", "refused for Z>1", "more than one Z plane ... refused" and the like.
REFUSED = re.compile(
    r"Z\s*>\s*1.{0,100}?refus|refus.{0,100}?(?:Z\s*>\s*1|more than one Z plane)"
    r"|more than one Z plane.{0,100}?refus",
    re.IGNORECASE | re.DOTALL,
)


def _paragraphs_of_text(text: str) -> list[str]:
    """Blank-line separated blocks; each row of a Markdown table is its own paragraph."""
    blocks = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    return [
        piece
        for block in blocks
        for piece in (block.splitlines() if block.lstrip().startswith("|") else [block])
    ]


def _python_prose(path: Path) -> list[str]:
    """Docstrings and comment blocks of one module."""
    source = path.read_text(encoding="utf-8")
    prose: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                prose.extend(_paragraphs_of_text(doc))
    block: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            block.append(token.string.lstrip("# "))
        elif token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT) and block:
            prose.append(" ".join(block))
            block = []
    return prose + ([" ".join(block)] if block else [])


def _notebook_prose(path: Path) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    return [
        paragraph
        for cell in cells
        if cell["cell_type"] == "markdown"
        for paragraph in _paragraphs_of_text("".join(cell["source"]))
    ]


def _sources() -> list[tuple[Path, list[str]]]:
    markdown = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    found = [(p, _paragraphs_of_text(p.read_text(encoding="utf-8"))) for p in markdown]
    found += [(p, _python_prose(p)) for p in sorted((ROOT / "src").rglob("*.py"))]
    found += [(p, _notebook_prose(p)) for p in sorted((ROOT / "notebooks").rglob("*.ipynb"))]
    return found


def offending_paragraphs(paragraphs: list[str]) -> list[str]:
    return [p for p in paragraphs if Z_POLICY_ALL.search(p) and not REFUSED.search(p)]


def test_z_policy_all_prose_says_z_above_one_is_refused():
    offenders = [
        f"{path.relative_to(ROOT)}: {paragraph.strip()[:160]!r}"
        for path, paragraphs in _sources()
        for paragraph in offending_paragraphs(paragraphs)
    ]
    assert not offenders, "z_policy 'all' prose must say Z>1 is refused:\n" + "\n".join(offenders)


def test_the_scan_flags_the_old_wording_and_accepts_the_new():
    old = 'set `z_policy`/`t_policy`: `"all"` patches every plane, `"first"` one plane.'
    new = 'set `z_policy`/`t_policy`: `"all"` patches every T frame; Z>1 is refused.'
    assert offending_paragraphs([old]) == [old]
    assert offending_paragraphs([new]) == []
    assert offending_paragraphs(['`t_policy="all"` expands every frame.']) == []
