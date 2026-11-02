"""The MorphoBricks pins in setup.py agree with each other (owner decision D-h).

CellPatcher imports the normalization brick directly, so it declares it, at the
tag the preprocessing tag it pins requires. pip refuses two different direct
URLs for one package, so the two requirement strings must be identical.
"""

import ast
import importlib.metadata
import json
import re
from pathlib import Path

SETUP_PY = Path(__file__).resolve().parents[1] / "setup.py"


def _setup_requirements():
    """install_requires and extras_require of setup.py, read without running it."""
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    )
    keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
                if keyword.arg in ("install_requires", "extras_require")}
    return keywords["install_requires"], keywords["extras_require"]


def _by_name(requirements):
    pins = {}
    for requirement in requirements:
        name = re.split(r"[\s\[@;<>=!~]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        pins[name] = requirement
    return pins


def _url(requirement):
    return requirement.split("@", 1)[1].split(";", 1)[0].strip()


def test_normalization_is_pinned_where_preprocessing_pins_it():
    core, _ = _setup_requirements()
    pins = _by_name(core)
    assert "morphobricks-preprocessing" in pins
    assert "morphobricks-normalization" in pins, "CellPatcher imports normalization, so it declares it (D-h)"
    brick = _by_name(importlib.metadata.requires("morphobricks-preprocessing"))
    assert _url(pins["morphobricks-normalization"]) == _url(brick["morphobricks-normalization"])


def _installed_revision(name):
    """The git tag *name* was installed from (PEP 610 ``direct_url.json``), or its version as ``v<version>``."""
    distribution = importlib.metadata.distribution(name)
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        revision = json.loads(direct_url).get("vcs_info", {}).get("requested_revision")
        if revision:
            return revision
    return f"v{distribution.version}"


def test_the_installed_preprocessing_is_the_pinned_tag():
    # The tag an install came from, not the version it reports: preprocessing's v0.4.0rc1
    # tag still reports version 0.3.4 in its package metadata.
    core, _ = _setup_requirements()
    pins = _by_name(core)
    for name in ("morphobricks-preprocessing", "morphobricks-normalization"):
        tag = _url(pins[name]).rsplit("@", 1)[1]
        assert _installed_revision(name) == tag, name


def test_the_zarr_extra_installs_the_core_preprocessing_tag():
    # CP-6: the preprocessing brick is a core dependency; only its zarr reader is optional.
    core, extras = _setup_requirements()
    assert "preprocessing" not in extras
    (zarr,) = extras["zarr"]
    assert zarr.startswith("morphobricks-preprocessing[zarr]")
    assert _url(zarr) == _url(_by_name(core)["morphobricks-preprocessing"])
