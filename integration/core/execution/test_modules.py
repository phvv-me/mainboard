"""`importable` keeps the modules that import and tells a missing optional extra from a defect."""

from pathlib import Path

import pytest

from mainboard.core.execution.modules import importable

_EXTRAS = [
    pytest.param("import a_library_nobody_installed\n", id="a-library-that-is-not-installed"),
    pytest.param("from json import a_name_json_lacks\n", id="an-installed-library-lacking-a-name"),
]

_DEFECTS = [
    pytest.param("from . import a_sibling_nobody_wrote\n", id="a-sibling-of-its-own-package"),
    pytest.param("from .fine import a_name_fine_lacks\n", id="a-name-its-own-sibling-lacks"),
    pytest.param("raise ImportError('raised by hand')\n", id="an-import-error-naming-no-module"),
]


def _package(root: Path, monkeypatch: pytest.MonkeyPatch, broken: str) -> list[str]:
    """The names of `fine` and `broken`, two modules of a package unique to this test."""
    package = root / f"pkg_{root.name}"
    package.mkdir()
    for name, text in (("__init__", ""), ("fine", ""), ("broken", broken)):
        (package / f"{name}.py").write_text(text, encoding="utf-8")
    monkeypatch.syspath_prepend(root)
    return [f"{package.name}.fine", f"{package.name}.broken"]


@pytest.mark.parametrize("broken", _EXTRAS)
def test_a_module_missing_an_optional_extra_is_left_out_and_named_in_the_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    broken: str,
) -> None:
    fine, extra = _package(tmp_path, monkeypatch, broken)

    kept = importable([fine, extra])

    assert kept == [fine]
    assert f"{extra} is skipped" in capsys.readouterr().err


@pytest.mark.parametrize("broken", _DEFECTS)
def test_a_module_failing_inside_its_own_package_raises_instead_of_vanishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    broken: str,
) -> None:
    names = _package(tmp_path, monkeypatch, broken)

    with pytest.raises(ImportError):
        importable(names)

    assert capsys.readouterr().err == ""
