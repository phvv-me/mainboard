# A pytest target runs through native pytest, and the closure carries the harness the walk
# cannot see. Every proof here runs the runner as its own process over a committed workspace,
# the guard armed from the staged listing, so what passes is what a dispatch would run.

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.dispatch.provenance import Repositories, listing
from mainboard.dispatch.shared import CLOSURE_VAR, DEFERRED_VAR, FIRST_PARTY_VAR
from mainboard.jobs.closure import Closure
from mainboard.jobs.target import Target

from ..support import Lab

# The packaged shape mainboard's own test tree has: `__init__.py` up the chain, a conftest
# importing a helper the test modules never name, and pytest configuration at the root.
_CONTEST = """import pytest

from tests.helpers import FIELD, mark


@pytest.fixture
def field():
    mark("fixture")
    return FIELD
"""

_SAMPLE = """import pytest

from tests.helpers import mark


@pytest.mark.parametrize("number", [1, 2])
def test_runs(field, number):
    mark(f"{number}:{field}")


def test_imports_a_stray_module(field):
    import importlib

    importlib.import_module("tests.test_" + "stray")


def test_fails():
    value = 41
    assert value == 42
"""

# A module carrying its metadata where the file declares it, on a parametrized function and on
# a method inside a class, the two node-id spellings a dispatch must see through.
_DECLARED = """import pytest

from mainboard.jobs import job

from tests.helpers import mark


@job(needs=("data/corpus",), resources=("templates/t.j2",), fetch="evidence")
@pytest.mark.parametrize("number", [1])
def test_declared(number):
    mark(f"{number}")


class TestCases:
    @job(needs=("data/other",))
    @pytest.mark.parametrize("number", [1])
    def test_method(self, number):
        mark(f"class{number}")


class TestOther:
    @job(needs=("data/distinct",))
    def test_method(self):
        mark("other")
"""

_GROUP = """from tests.helpers import mark


class TestGroup:
    def test_method(self):
        mark("group")
"""

_HELPERS = '''from pathlib import Path

FIELD = "injected"


def mark(line: str) -> None:
    """Leave one line beside the workspace root, the side channel a subprocess test reads."""
    with Path("marks.txt").open("a", encoding="utf-8") as told:
        told.write(f"{line}\\n")
'''


@pytest.fixture
def pytest_lab(tmp_path: Path) -> Lab:
    """A committed workspace whose test tree needs its conftest, helper and config shipped."""
    lab = Lab(tmp_path / "projects")
    lab.root.mkdir(parents=True)
    lab.git("init")
    lab.write("mainboard.toml", "")
    lab.write("tests/__init__.py", "")
    lab.write("tests/helpers.py", _HELPERS)
    lab.write("tests/conftest.py", _CONTEST)
    lab.write("tests/jobs/__init__.py", "")
    lab.write("tests/jobs/test_sample.py", _SAMPLE)
    lab.write("tests/jobs/test_group.py", _GROUP)
    lab.write("tests/jobs/test_declared.py", _DECLARED)
    lab.write("tests/test_stray.py", "STRAY = 1\n")
    lab.write("templates/t.j2", "t\n")
    lab.write("pytest.ini", "[pytest]\ntestpaths = tests\n")
    lab.commit("a test tree")
    return lab


def sealed(lab: Lab, spelling: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the runner as its own process over `spelling`, the guard armed from the listing."""
    target = Target.spelled([spelling], lab.root)
    assert target is not None
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=(),
        environment=lab.root / Lab.ENVIRONMENT,
    )
    _, rows = Repositories(lab.root).seal(closure.owner, closure.files, built=closure.built)
    written = lab.root / ".mainboard/closure.tsv"
    written.parent.mkdir(exist_ok=True)
    written.write_text(listing(rows), encoding="utf-8")
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in {CLOSURE_VAR, FIRST_PARTY_VAR, DEFERRED_VAR}
    }
    env.update(
        {
            "PYTHONPATH": ":".join(str(lab.root / place) for place in closure.roots),
            CLOSURE_VAR: str(written),
            FIRST_PARTY_VAR: ":".join(closure.first_party),
        }
    )
    return subprocess.run(
        [sys.executable, "-m", "mainboard.jobs.call", spelling, *args],
        cwd=lab.root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def closure_of(lab: Lab, spelling: str) -> Closure:
    target = Target.spelled([spelling], lab.root)
    assert target is not None
    return Closure.of(
        target, root=lab.root, distributions=(), environment=lab.root / Lab.ENVIRONMENT
    )


def test_a_parametrized_test_runs_with_its_fixture_through_native_pytest(
    pytest_lab: Lab,
) -> None:
    """The node id selects the one parameter, the fixture injects, and pytest answers 0."""
    done = sealed(pytest_lab, "tests/jobs/test_sample.py::test_runs[2]")
    assert done.returncode == 0, done.stderr
    assert (pytest_lab.root / "marks.txt").read_text(encoding="utf-8") == "fixture\n2:injected\n"


def test_a_class_node_id_fits_the_target_spelling_and_runs(pytest_lab: Lab) -> None:
    """`file.py::Class::test_method` is one spelling: file, then the whole node id."""
    target = Target.spelled(["tests/jobs/test_group.py::TestGroup::test_method"], pytest_lab.root)
    assert target is not None
    assert (target.file, target.name, target.test) == (
        "tests/jobs/test_group.py",
        "TestGroup::test_method",
        True,
    )
    done = sealed(pytest_lab, "tests/jobs/test_group.py::TestGroup::test_method")
    assert done.returncode == 0, done.stderr
    assert (pytest_lab.root / "marks.txt").read_text(encoding="utf-8") == "group\n"


def test_same_named_methods_keep_their_own_declarations(pytest_lab: Lab) -> None:
    """Class qualification is part of the identity, not discarded before reading metadata."""
    for owner, needs in (("TestCases", "data/other"), ("TestOther", "data/distinct")):
        target = Target.spelled(
            [f"tests/jobs/test_declared.py::{owner}::test_method"], pytest_lab.root
        )
        assert target is not None
        assert target.declaration(pytest_lab.root).needs == (needs,)


def test_a_bare_test_file_spelling_means_the_whole_file(pytest_lab: Lab) -> None:
    """No `::name` on a `test_` file is the file's own tests, not a missing `app` or `main`."""
    target = Target.spelled(["tests/jobs/test_group.py"], pytest_lab.root)
    assert target is not None
    assert (target.name, target.test) == ("", True)
    done = sealed(pytest_lab, "tests/jobs/test_group.py")
    assert done.returncode == 0, done.stderr
    assert (pytest_lab.root / "marks.txt").read_text(encoding="utf-8") == "group\n"


def test_collection_only_runs_neither_body_nor_fixture(pytest_lab: Lab) -> None:
    """Arguments ride through to pytest, and a collection answers without executing anything."""
    done = sealed(pytest_lab, "tests/jobs/test_sample.py", "--collect-only", "-q")
    assert done.returncode == 0, done.stderr
    assert "test_runs" in done.stdout
    assert not (pytest_lab.root / "marks.txt").exists()


def test_the_closure_carries_the_harness_the_walk_cannot_see(pytest_lab: Lab) -> None:
    """Conftest, the helper only it imports, and the config all ship beside the node."""
    closure = closure_of(pytest_lab, "tests/jobs/test_sample.py::test_runs[2]")
    for carried in (
        "tests/jobs/test_sample.py",
        "tests/conftest.py",
        "tests/helpers.py",
        "pytest.ini",
        "mainboard.toml",
    ):
        assert carried in closure.files
    assert closure.first_party == ("tests",)


def test_the_config_and_conftests_are_found_without_importing_anything(pytest_lab: Lab) -> None:
    """A conftest whose import dies at import time still closes: discovery reads syntax only."""
    pytest_lab.write("tests/conftest.py", "raise SystemExit('conftest ran')\n")
    pytest_lab.commit("a conftest nothing may run")
    closure = closure_of(pytest_lab, "tests/jobs/test_sample.py::test_runs[2]")
    assert "tests/conftest.py" in closure.files
    assert "pytest.ini" in closure.files


def test_a_dynamic_plugins_value_is_refused_rather_than_guessed(pytest_lab: Lab) -> None:
    """A `pytest_plugins` needing execution names nothing a dispatch can see, so it is refused."""
    pytest_lab.write("tests/conftest.py", "pytest_plugins = tuple(available())\n")
    pytest_lab.commit("plugins behind a call")
    with pytest.raises(MissionError, match="literal"):
        closure_of(pytest_lab, "tests/jobs/test_sample.py::test_runs[2]")


def test_a_literal_plugins_list_joins_the_closure(pytest_lab: Lab) -> None:
    """A module a conftest names in `pytest_plugins` ships, though nothing imports it."""
    pytest_lab.write("tests/plugins.py", "PLUGIN = 1\n")
    pytest_lab.write("tests/conftest.py", "pytest_plugins = ['tests.plugins']\n")
    pytest_lab.commit("a literal plugin")
    closure = closure_of(pytest_lab, "tests/jobs/test_sample.py::test_runs[2]")
    assert "tests/plugins.py" in closure.files


def test_an_unshipped_test_module_is_refused_through_the_rewrite_hook(pytest_lab: Lab) -> None:
    """The hook answers test_-named names from wherever the environment points; judged, a
    module the closure never carried is refused before it can execute, mirror or not."""
    done = sealed(pytest_lab, "tests/jobs/test_sample.py::test_imports_a_stray_module")
    assert done.returncode != 0
    assert "first-party code outside this job's closure" in done.stdout + done.stderr


def test_assertion_introspection_survives_the_judged_hook(pytest_lab: Lab) -> None:
    """Rewriting is untouched: a failing assert still reports the values it compared."""
    done = sealed(pytest_lab, "tests/jobs/test_sample.py::test_fails")
    assert done.returncode == 1
    assert "assert 41 == 42" in done.stdout


def test_a_plugin_argument_naming_a_test_module_is_refused(pytest_lab: Lab) -> None:
    """Early plugin loading obeys the same origin boundary as later test collection."""
    done = sealed(pytest_lab, "tests/jobs/test_sample.py::test_runs[2]", "-p", "tests.test_stray")
    assert done.returncode != 0
    assert "first-party code outside this job's closure" in done.stderr


@pytest.mark.parametrize("source", ["config", "addopts", "plugins", "conftest"])
def test_early_imports_cannot_bypass_the_boundary(
    pytest_lab: Lab, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Config, environment, and initial conftests must not execute unshipped test modules."""
    if source == "config":
        pytest_lab.write("pytest.ini", "[pytest]\naddopts = -p tests.test_stray\n")
    elif source == "addopts":
        monkeypatch.setenv("PYTEST_ADDOPTS", "-p tests.test_stray")
    elif source == "plugins":
        monkeypatch.setenv("PYTEST_PLUGINS", "tests.test_stray")
    else:
        pytest_lab.write(
            "tests/conftest.py",
            "import importlib\nimportlib.import_module('tests.test_' + 'stray')\n",
        )
    if source in {"config", "conftest"}:
        pytest_lab.commit("early import route")
    done = sealed(pytest_lab, "tests/jobs/test_sample.py::test_runs[2]")
    assert done.returncode != 0
    assert "first-party code outside this job's closure" in done.stdout + done.stderr


def test_the_declaration_survives_a_parametrized_node_id(pytest_lab: Lab) -> None:
    """`test_declared[1]` is the declaration on `test_declared`, wherever it sits."""
    closure = closure_of(pytest_lab, "tests/jobs/test_declared.py::test_declared[1]")
    assert closure.needs == ("data/corpus",)
    assert closure.fetch == "evidence"
    assert "templates/t.j2" in closure.files


def test_the_declaration_survives_a_class_node_id(pytest_lab: Lab) -> None:
    """`Class::test_method[1]` reaches the decorated method inside the class body."""
    closure = closure_of(pytest_lab, "tests/jobs/test_declared.py::TestCases::test_method[1]")
    assert closure.needs == ("data/other",)
