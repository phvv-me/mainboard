"""A namespace package in an import root is first-party code like a regular one: the closure names
it, and the guard refuses a module of it the dispatch did not ship, while a portion an installed
distribution adds to the same namespace still imports."""

import sysconfig
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from mainboard.jobs import Closure, Target
from mainboard.jobs.call import Guard


def _closure(root: Path) -> Closure:
    """The closure of a job importing one module of the namespace `ns`, beside one it does not."""
    (root / "mainboard.toml").write_text('[workspace]\nname = "w"\n', encoding="utf-8")
    (root / "ns").mkdir()
    (root / "ns" / "shipped.py").write_text("", encoding="utf-8")
    (root / "ns" / "unshipped.py").write_text("", encoding="utf-8")
    (root / "job.py").write_text("import ns.shipped\n\n\ndef main() -> None: ...\n")
    return Closure.of(
        Target(file="job.py", name="main"), root=root, distributions=(), environment=root / "env"
    )


def test_a_namespace_portion_is_first_party_code(tmp_path: Path) -> None:
    closure = _closure(tmp_path)

    assert "ns" in closure.first_party
    assert "ns/shipped.py" in closure.files
    assert "ns/unshipped.py" not in closure.files


def test_the_guard_refuses_an_unshipped_portion_module_and_passes_an_installed_one(
    tmp_path: Path,
) -> None:
    closure = _closure(tmp_path)
    guard = Guard(closure.first_party, shipped=closure.files, root=tmp_path)
    portion = [str(tmp_path / "ns")]
    installed = Path(sysconfig.get_paths()["purelib"]) / "ns" / "lib.py"

    assert guard.find_spec("ns.shipped", portion) is not None
    with pytest.raises(ModuleNotFoundError, match="outside this job's closure"):
        guard.find_spec("ns.unshipped", portion)
    assert guard.holds(ModuleSpec("ns.lib", None, origin=str(installed)))
