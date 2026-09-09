from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.jobs.closure import Closure, Module, Walker, _absolute
from mainboard.jobs.target import Target

from ..support import Lab

# The extension tests add a third distribution beside the lab's two, so nothing about the
# existing fixtures moves under it.
_EXT = "packages/ext/src"
_ENVIRONMENT = Lab.ENVIRONMENT

# What the lab's job ships: its node in full, the sibling package it imports, both distributions
# whole, the resource it declared and the manifest. Nothing from the other campaign, nothing
# from the ignored data directory.
_SHIPPED = (
    "mainboard.toml",
    "packages/core/src/core/__init__.py",
    "packages/core/src/core/data.txt",
    "packages/core/src/core/spare.py",
    "packages/core/src/core/util.py",
    "packages/sub/src/sub/__init__.py",
    "packages/sub/src/sub/thing.py",
    "research/camp/experiments/__init__.py",
    "research/camp/experiments/helper/__init__.py",
    "research/camp/experiments/helper/tools.py",
    "research/camp/experiments/node/__init__.py",
    "research/camp/experiments/node/node.md",
    "research/camp/experiments/node/run.py",
    "research/camp/registry.toml",
)


def closure_of(lab: Lab, spelling: str = Lab.JOB, **overrides: tuple[str, ...]) -> Closure:
    """The closure of `spelling` over the lab's two distributions, closed over an empty env."""
    target = Target.spelled([spelling], lab.root)
    assert target is not None
    return Closure.of(
        target,
        root=lab.root,
        distributions=Lab.DISTRIBUTIONS,
        environment=lab.root / _ENVIRONMENT,
        **overrides,
    )


def test_the_closure_is_the_node_what_it_imports_and_what_it_declared_and_nothing_else(
    lab: Lab,
) -> None:
    closure = closure_of(lab)
    assert closure.files == _SHIPPED
    assert closure.roots == ("research/camp", *Lab.DISTRIBUTIONS)
    assert closure.first_party == ("core", "experiments", "sub")
    assert closure.needs == ("data/corpus",)
    assert closure.fetch == "research/camp/experiments/node/evidence"
    assert closure.owner is not None and Path(closure.owner.path) == lab.root
    # An ignored file inside the node never ships; a module reached by import does, ignored or
    # not, since the job runs it.
    lab.write("research/camp/experiments/node/__pycache__/run.cpython-314.pyc", "")
    lab.write("research/camp/experiments/helper/x_generated.py", "X = 1\n")
    lab.write(
        "research/camp/experiments/node/run.py",
        "from ..helper import x_generated\napp = 1\n",
    )
    regenerated = closure_of(lab).files
    assert "research/camp/experiments/helper/x_generated.py" in regenerated
    assert not any("__pycache__" in file for file in regenerated)


def test_the_walk_follows_every_import_form_and_the_ancestor_packages_they_stand_in(
    lab: Lab,
) -> None:
    """Relative, `from pkg import name`, dotted, dynamic-by-literal: each lands the same module."""
    walker = Walker(lab.root, home=Lab.HOME, distributions=Lab.DISTRIBUTIONS)
    run = Module(path=Lab.JOB, root=Lab.HOME)
    assert list(walker.imported(run)) == [
        "sub.thing",
        "cyclopts",
        "cyclopts.App",
        "mainboard.jobs",
        "mainboard.jobs.job",
        "experiments.helper.tools",
        "experiments.helper.tools.tool",
    ]
    assert [module.path for module in walker.resolve("experiments.helper.tools")] == [
        "research/camp/experiments/__init__.py",
        "research/camp/experiments/helper/__init__.py",
        "research/camp/experiments/helper/tools.py",
    ]
    assert walker.resolve("cyclopts") == []
    assert walker.resolve("experiments.helper.tools.tool") == []
    assert walker.resolve("sub.thing") == [
        Module(path="packages/sub/src/sub/__init__.py", root="packages/sub/src"),
        Module(path="packages/sub/src/sub/thing.py", root="packages/sub/src"),
    ]
    # A directory without `__init__.py` is a namespace portion: walked through, nothing shipped.
    lab.write("research/camp/plain/leaf.py", "LEAF = 1\n")
    assert walker.resolve("plain.leaf") == [
        Module(path="research/camp/plain/leaf.py", root=Lab.HOME)
    ]
    assert walker.resolve("plain") == []
    assert walker.resolve("plain.missing") == []
    lab.write(
        "research/camp/experiments/node/run.py",
        "from importlib import import_module\n"
        'import_module("core.spare")\n'
        '__import__("experiments.helper")\n'
        'import_module(".relative", package=__name__)\n'
        "import_module(name)\n"
        "from . import __init__ as me\n"
        "from .. import helper\n"
        "from ...too import far\n",
    )
    assert set(walker.imported(run)) == {
        "importlib",
        "importlib.import_module",
        "core.spare",
        "experiments.helper",
        "experiments.node",
        "experiments.node.__init__",
        "experiments",
    }


@pytest.mark.parametrize(
    ("package", "level", "name", "expected"),
    [
        ("a.b.c", 0, "x.y", "x.y"),
        ("a.b.c", 1, "d", "a.b.d"),
        ("a.b.c", 1, "", "a.b"),
        ("a.b.c", 2, "d", "a.d"),
        ("a.b.c", 3, "d", None),
        ("a", 1, "", None),
        ("a", 2, "b", None),
    ],
)
def test_a_relative_import_is_spelled_out_against_the_importing_module(
    package: str, level: int, name: str, expected: str | None
) -> None:
    assert _absolute(package, level=level, name=name) == expected


def test_nested_package_initializer_reaches_its_sibling_and_parent_imports(lab: Lab) -> None:
    """A real census refactor exposed imports being resolved one package too high."""
    lab.write("research/camp/experiments/node/run.py", "from .census import measure\napp = 1\n")
    lab.write(
        "research/camp/experiments/node/census/__init__.py",
        "from ...helper.tools import tool\nfrom .record import measure\n",
    )
    lab.write("research/camp/experiments/node/census/record.py", "measure = 1\n")
    walker = Walker(lab.root, home=Lab.HOME, distributions=Lab.DISTRIBUTIONS)
    reached = {module.path for module in walker.reach("research/camp/experiments/node/run.py")}
    assert "research/camp/experiments/helper/tools.py" in reached
    assert "research/camp/experiments/node/census/record.py" in reached


def test_a_name_two_roots_both_hold_is_refused_rather_than_settled_by_order(lab: Lab) -> None:
    """Two campaigns each keep an `experiments` package; a job must name one."""
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    with pytest.raises(MissionError, match="`experiments` is a package under more than one"):
        Closure.of(
            target,
            root=lab.root,
            distributions=(*Lab.DISTRIBUTIONS, "research/other"),
            environment=lab.root / _ENVIRONMENT,
        )


def test_the_job_file_itself_reaches_its_home_root_even_when_nothing_else_is_imported(
    lab: Lab,
) -> None:
    lab.write("research/camp/experiments/node/run.py", "def main() -> None:\n    pass\n")
    closure = closure_of(lab)
    assert closure.roots == ("research/camp",)
    assert closure.first_party == ("core", "experiments", "sub")
    assert closure.files == (
        "mainboard.toml",
        "research/camp/experiments/node/__init__.py",
        "research/camp/experiments/node/node.md",
        "research/camp/experiments/node/run.py",
    )


def test_an_import_root_the_workspace_does_not_hold_defines_nothing_and_ships_nothing(
    lab: Lab,
) -> None:
    """A manifest can name a path dependency that is not checked out here; it is no root."""
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=(*Lab.DISTRIBUTIONS, "packages/absent/src"),
        environment=lab.root / _ENVIRONMENT,
    )
    assert closure.first_party == ("core", "experiments", "sub")
    assert closure.roots == ("research/camp", *Lab.DISTRIBUTIONS)


def test_a_declared_resource_pins_a_file_or_a_whole_kept_directory(lab: Lab) -> None:
    lab.write("research/camp/templates/a.j2", "a\n")
    lab.write("research/camp/templates/b.j2", "b\n")
    lab.write("research/camp/templates/c_generated.py", "ignored\n")
    lab.write(
        "research/camp/experiments/node/run.py",
        'from mainboard.jobs import job\n\n@job(resources=("research/camp/templates",))\n'
        "def main() -> None:\n    pass\n",
    )
    files = closure_of(lab).files
    assert "research/camp/templates/a.j2" in files and "research/camp/templates/b.j2" in files
    assert "research/camp/templates/c_generated.py" not in files


@pytest.mark.parametrize(
    ("resource", "complaint"),
    [
        ("/etc/passwd", "workspace-relative"),
        ("../elsewhere", "workspace-relative"),
        ("research/camp/nothing.txt", "not in the workspace"),
    ],
)
def test_a_resource_that_cannot_be_pinned_is_refused_by_name(
    lab: Lab, resource: str, complaint: str
) -> None:
    lab.write(
        "research/camp/experiments/node/run.py",
        f'from mainboard.jobs import job\n\n@job(resources=("{resource}",))\n'
        "def main() -> None:\n    pass\n",
    )
    with pytest.raises(MissionError, match=complaint):
        closure_of(lab)


def test_needs_join_the_declared_ones_and_may_never_sit_over_shipped_code(lab: Lab) -> None:
    closure = closure_of(lab, needs=("data/more", "data/corpus"))
    assert closure.needs == ("data/corpus", "data/more")
    with pytest.raises(MissionError, match="would sit over shipped code"):
        closure_of(lab, needs=("research/camp",))
    with pytest.raises(MissionError, match="workspace-relative"):
        closure_of(lab, needs=("../data",))
    with pytest.raises(MissionError, match="workspace-relative"):
        closure_of(lab, needs=("/data",))


def test_a_node_under_no_repository_cannot_say_what_it_keeps(lab: Lab, tmp_path: Path) -> None:
    loose = tmp_path / "loose"
    (loose / "node").mkdir(parents=True)
    (loose / "node" / "run.py").write_text("def main() -> None:\n    pass\n", encoding="utf-8")
    (loose / "mainboard.toml").write_text("", encoding="utf-8")
    target = Target.spelled(["node/run.py"], loose)
    assert target is not None
    with pytest.raises(MissionError, match="under no git repository"):
        Closure.of(target, root=loose, distributions=(), environment=loose / _ENVIRONMENT)


def _closure_with_ext(lab: Lab, *, environment: Path) -> Closure:
    """The lab's job, rewritten to import a third distribution, `ext`, closed over all three."""
    lab.write("packages/ext/src/ext/__init__.py", "")
    lab.write(
        "research/camp/experiments/node/run.py", "import ext\n\n\ndef main() -> None:\n    pass\n"
    )
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    return Closure.of(
        target,
        root=lab.root,
        distributions=(*Lab.DISTRIBUTIONS, _EXT),
        environment=environment,
    )


def test_a_compiled_extension_built_inside_its_package_ships_beside_it_marked_built(
    lab: Lab,
) -> None:
    """The target env's RECORD names the `.so`; it sits in the tree, so it ships beside it."""
    lab.write("packages/ext/.gitignore", "*.so\n")
    environment = lab.compiled(
        "camp-ext",
        lab.root / f"{_EXT}/ext/_native.cpython-314-x86_64-linux-gnu.so",
        imports="ext",
    )
    closure = _closure_with_ext(lab, environment=environment)
    shipped = "packages/ext/src/ext/_native.cpython-314-x86_64-linux-gnu.so"
    assert shipped in closure.files
    assert closure.built == (shipped,)
    assert closure.deferred == ()
    assert _EXT in closure.roots


def test_a_compiled_extension_only_in_the_environment_defers_the_whole_package(lab: Lab) -> None:
    """Where cutoken's own `_native` was actually found (2026-09-07): outside the tree entirely."""
    environment = lab.compiled(
        "camp-ext",
        lab.root / f"{_ENVIRONMENT}/lib/python3.14/site-packages/ext/"
        "_native.cpython-314-x86_64-linux-gnu.so",
    )
    closure = _closure_with_ext(lab, environment=environment)
    assert not any(file.startswith(f"{_EXT}/") for file in closure.files)
    assert closure.built == ()
    assert closure.deferred == ("ext",)
    assert _EXT not in closure.roots


def test_the_extension_answer_comes_from_the_target_environment_not_this_interpreter(
    lab: Lab,
) -> None:
    """The read is of the compiled env's dist-infos, never of the interpreter asking.

    The process dispatching a job is a uv tool whose own site-packages holds none of the job's
    packages: it read its own metadata for a day, found no extension, deferred nothing, and the
    installed CLI died importing `cutoken.tokenization.data._native` (2026-09-07). So the
    distribution here is named unlike its import — no name-based read could find it — and it
    exists in the target environment alone, which this test's own interpreter has never heard
    of. A package the target environment holds no record of stays pure source, the answer the
    right reading gives when there is genuinely nothing compiled to know.
    """
    lab.write(
        "research/camp/experiments/node/run.py",
        "import ext\nimport sub.thing\n\n\ndef main() -> None:\n    pass\n",
    )
    lab.write("packages/ext/src/ext/__init__.py", "")
    environment = lab.compiled(
        "camp-ext-binary",
        lab.root / f"{_ENVIRONMENT}/lib/python3.14/site-packages/ext/"
        "_native.cpython-314-x86_64-linux-gnu.so",
    )
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=(*Lab.DISTRIBUTIONS, _EXT),
        environment=environment,
    )
    # `sub` is in no environment's records anywhere: nothing compiled to know, so it ships.
    assert "packages/sub/src/sub/__init__.py" in closure.files
    assert not any(file.startswith(f"{_EXT}/") for file in closure.files)
    assert closure.deferred == ("ext",)
