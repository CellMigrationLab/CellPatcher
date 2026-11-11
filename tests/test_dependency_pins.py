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

import pytest

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


def _git_url(url):
    """A git URL without ``git+``, a trailing ``.git`` or ``/``, to compare two spellings of one repository."""
    url = url.removeprefix("git+").rstrip("/")
    return url.removesuffix(".git")


def _installed_from(name):
    """The repository URL and git tag *name* was installed from (PEP 610 ``direct_url.json``).

    A tag is not a version: preprocessing's v0.4.0rc1 and v0.4.0rc2 tags still report version 0.3.4, so
    only a git install can show which tag it came from. Any other install (from a local
    folder, editable or not, an archive or an index) fails the test with the reason, never
    passes on a matching version number.
    """
    distribution = importlib.metadata.distribution(name)
    text = distribution.read_text("direct_url.json")
    if not text:
        pytest.fail(
            f"{name} {distribution.version} was not installed from a URL (no direct_url.json), so the "
            "tag it came from cannot be checked. Install CellPatcher with pip, which installs the "
            "pinned tags."
        )
    direct_url = json.loads(text)
    revision = direct_url.get("vcs_info", {}).get("requested_revision")
    if revision is None:
        editable = direct_url.get("dir_info", {}).get("editable", False)
        pytest.fail(
            f"{name} was installed from {direct_url.get('url')}"
            f"{' (editable)' if editable else ''}, not from a git tag, so the tag it came from cannot "
            "be checked. Install CellPatcher with pip, which installs the pinned tags."
        )
    return _git_url(direct_url["url"]), revision


def test_the_installed_preprocessing_is_the_pinned_tag():
    # The repository and tag an install came from, not the version it reports.
    core, _ = _setup_requirements()
    pins = _by_name(core)
    for name in ("morphobricks-preprocessing", "morphobricks-normalization"):
        repository, tag = _url(pins[name]).rsplit("@", 1)
        assert _installed_from(name) == (_git_url(repository), tag), name


class _Installed:
    """A distribution with the given ``direct_url.json`` text (None: no file)."""

    version = "0.4.0rc5"

    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, filename):
        return self._direct_url if filename == "direct_url.json" else None


@pytest.mark.parametrize("direct_url, reason", [
    (None, "no direct_url.json"),
    ('{"url": "file:///src/morphobricks-preprocessing", "dir_info": {"editable": true}}', r"\(editable\), not from a git tag"),
    ('{"url": "file:///src/morphobricks-preprocessing", "dir_info": {}}', "preprocessing, not from a git tag"),
])
def test_an_install_not_from_a_git_tag_fails_with_its_reason(monkeypatch, direct_url, reason):
    # Even when the version it reports matches the tag: only a git install names its tag.
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _Installed(direct_url))
    with pytest.raises(pytest.fail.Exception, match=reason):
        _installed_from("morphobricks-preprocessing")


def test_the_zarr_extra_installs_the_core_preprocessing_tag():
    # CP-6: the preprocessing brick is a core dependency; only its zarr reader is optional.
    core, extras = _setup_requirements()
    assert "preprocessing" not in extras
    (zarr,) = extras["zarr"]
    assert zarr.startswith("morphobricks-preprocessing[zarr]")
    assert _url(zarr) == _url(_by_name(core)["morphobricks-preprocessing"])


# A merged release commit the owner has not tagged yet, and the tag it becomes (Stage 2 W2: preprocessing
# v0.5.1 is merged at this commit). A draft may pin it; the pin switches to the tag when the owner tags
# it, and this entry goes with it. Any other commit is refused.
UNTAGGED_RELEASES = {"8726cb45acac4b16121001fc2c93863479ce3788": "v0.5.1"}


def test_preprocessing_is_pinned_to_a_release_tag_not_a_commit():
    # Stage 2 CP-1 pinned the v0.5.0 release-candidate commit until the tag existed; a commit pin
    # is not a release, and every brick must spell the same URL (pip refuses two for one package).
    core, extras = _setup_requirements()
    for requirement in (_by_name(core)["morphobricks-preprocessing"], *extras["zarr"]):
        repository, ref = _url(requirement).rsplit("@", 1)
        assert repository == "git+https://github.com/CellMigrationLab/morphobricks-preprocessing", requirement
        ref = UNTAGGED_RELEASES.get(ref, ref)
        assert re.fullmatch(r"v\d+\.\d+\.\d+", ref), f"{requirement}: pin a release tag, not {ref!r}"


def test_normalization_is_pinned_to_a_release_tag():
    core, _ = _setup_requirements()
    ref = _url(_by_name(core)["morphobricks-normalization"]).rsplit("@", 1)[1]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", ref), f"normalization: pin a release tag, not {ref!r}"
