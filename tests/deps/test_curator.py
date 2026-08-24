from typing import TYPE_CHECKING

import pytest

from mainboard import Board, MissionError
from mainboard.deps import Change, Dependencies, Index, ManifestText

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from mainboard import Manifest

# What the fake lock reports before and after a solve, so a diff has one arrival, one departure
# and one move to report and nothing else.
_BEFORE = {"torch": "2.9.0", "numpy": "2.4.6", "leaving": "1.0"}
_AFTER = {"torch": "2.9.1", "numpy": "2.4.6", "tqdm": "4.70.0"}

# One requirement of every shape the addressing has to write: an explicit range, a replacement
# of a requirement already there, npm's own `@` separator, a scoped npm name whose leading `@`
# is part of the name, and an ecosystem the manifest has no table for yet.
_WRITES = [
    ("tqdm>=4.70.0, <5", "tqdm", "python", True, "[dev.python.deps]", "absent", ">=4.70.0, <5"),
    ("torch>=3.0", "torch", "python", False, "[python.deps]", ">=2.9", ">=3.0"),
    ("vitest@^3", "vitest", "nodejs", True, "[nodejs.dev]", "absent", "^3"),
    (
        "@puppeteer/browsers>=4",
        "@puppeteer/browsers",
        "nodejs",
        True,
        "[nodejs.dev]",
        ">=3, <4",
        ">=4",
    ),
    ("serde>=1", "serde", "rust", False, "[rust.deps]", "absent", ">=1"),
]


class FakePixi:
    """A pixi seam that answers with a scripted lock reading instead of running anything."""

    def __init__(self, readings: list[dict[str, str]]) -> None:
        self.readings = readings

    def locked(self, env: str) -> dict[str, str]:
        return self.readings.pop(0)


class FakeProvisioner:
    """A provisioner that records how it was asked to solve and never touches a real one."""

    calls: list[tuple[str, bool, bool]] = []

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest
        self.pixi = FakePixi([dict(_BEFORE), dict(_AFTER)])

    def provision(self, env: str, *, resolve: bool = False, refresh: bool = False) -> None:
        FakeProvisioner.calls.append((env, resolve, refresh))


@pytest.fixture
def solved(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, bool, bool]]]:
    """Stand a scripted provisioner in for the real one, yielding the solves it was asked for."""
    FakeProvisioner.calls = []
    monkeypatch.setattr("mainboard.deps.curator.Provisioner", FakeProvisioner)
    yield FakeProvisioner.calls


@pytest.fixture
def deps(root: Path) -> Dependencies:
    """The dependency surface over a workspace holding the fixture manifest."""
    return Board(root).deps()


@pytest.fixture
def publishes(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Make every index publish one release, without a registry anywhere in reach.

    Each implementation carries its own `latest`, so every one is stood in for rather than the
    abstract method they all override.
    """

    def install(version: str) -> None:
        for implementation in Index.implementations():
            monkeypatch.setattr(implementation, "latest", lambda self, name: version)

    return install


def constraint(deps: Dependencies, table: str, name: str) -> str:
    """What the manifest on disk now pins `name` to in `table`."""

    path = tuple(table.strip("[]").split("."))
    return ManifestText(deps.path.read_text(encoding="utf-8")).constraint(path, name)


def moved(changes: list[Change], name: str) -> Change:
    """The one reported change for `name`."""
    return next(change for change in changes if change.name == name)


@pytest.mark.parametrize(("spec", "name", "ecosystem", "dev", "where", "before", "after"), _WRITES)
def test_add_writes_the_requirement_it_was_given_where_its_neighbours_already_are(
    deps: Dependencies,
    spec: str,
    name: str,
    ecosystem: str,
    dev: bool,
    where: str,
    before: str,
    after: str,
) -> None:
    """A spec carrying its own range is written exactly as the caller wrote it."""
    changes = deps.add(spec, ecosystem=ecosystem, dev=dev, resolve=False)
    assert changes == [Change(name=name, where=where, before=before, after=after)]
    assert constraint(deps, where, name) == after


@pytest.mark.parametrize(
    ("spec", "name", "ecosystem", "env", "dev", "where"),
    [
        ("tqdm>=4", "tqdm", "python", "", False, "[python.deps]"),
        ("ray>=2", "ray", "python", "serving", False, "[envs.serving.python.deps]"),
        ("protozero>=1", "protozero", "conda", "", True, "[dev.deps]"),
    ],
)
def test_adding_a_requirement_and_removing_it_restores_the_manifest(
    deps: Dependencies,
    spec: str,
    name: str,
    ecosystem: str,
    env: str,
    dev: bool,
    where: str,
) -> None:
    """The two verbs are one inverse pair, and dropping never asks where it was written."""
    before = deps.path.read_text(encoding="utf-8")
    deps.add(spec, ecosystem=ecosystem, env=env, dev=dev, resolve=False)
    dropped = moved(deps.remove(name, resolve=False), name)
    assert dropped.where == where
    assert dropped.before == spec.removeprefix(name)
    assert dropped.after == "absent"
    assert deps.path.read_text(encoding="utf-8") == before


def test_add_pins_a_bare_name_to_what_the_index_publishes(
    deps: Dependencies, publishes: Callable[[str], None]
) -> None:
    """A name with no range is looked up rather than left unconstrained."""
    publishes("4.70.0")
    changes = deps.add("tqdm", ecosystem="python", resolve=False)
    assert moved(changes, "tqdm").after == ">=4.70.0, <5"


def test_add_refuses_an_environment_the_manifest_never_declared(deps: Dependencies) -> None:
    """The refusal comes from the schema, roster and all, before anything is written."""
    with pytest.raises(MissionError, match="no environment 'ghost'"):
        deps.add("tqdm", env="ghost", resolve=False)


@pytest.mark.parametrize(
    ("ecosystem", "match"),
    [
        ("", r"nothing declares 'ghost'. Searched .*\[deps\]"),
        ("go", "Searched no table"),
    ],
)
def test_remove_refuses_a_name_nothing_declares_and_names_what_it_searched(
    deps: Dependencies, ecosystem: str, match: str
) -> None:
    """The refusal is actionable because it says where it already looked."""
    with pytest.raises(MissionError, match=match):
        deps.remove("ghost", ecosystem=ecosystem, resolve=False)


def test_a_name_declared_in_more_than_one_table_has_to_be_named(deps: Dependencies) -> None:
    """Guessing which one was meant is how the wrong requirement gets dropped silently."""
    deps.add("torch>=1", ecosystem="python", env="serving", resolve=False)
    with pytest.raises(MissionError, match=r"declared in .*Name one with --lang"):
        deps.remove("torch", resolve=False)
    changes = deps.remove("torch", ecosystem="python", env="serving", resolve=False)
    assert moved(changes, "torch").where == "[envs.serving.python.deps]"


def test_upgrade_moves_one_constraint_to_the_newest_release(
    deps: Dependencies, publishes: Callable[[str], None], solved: list[tuple[str, bool, bool]]
) -> None:
    """Named, the manifest itself moves, which is the only way past a declared ceiling."""
    publishes("3.1.0")
    changes = deps.upgrade("torch", ecosystem="python")
    assert moved(changes, "torch").before == ">=2.9"
    assert moved(changes, "torch").after == ">=3.1.0, <4"
    assert solved == [("default", True, False)]


def test_a_bare_upgrade_leaves_the_manifest_alone_and_refreshes_the_lock(
    deps: Dependencies, solved: list[tuple[str, bool, bool]]
) -> None:
    """Nothing was declared differently, so only the lock had anywhere to move."""
    before = deps.path.read_text(encoding="utf-8")
    changes = deps.upgrade()
    assert deps.path.read_text(encoding="utf-8") == before
    assert solved == [("default", True, True)]
    assert {change.where for change in changes} == {"pixi.lock"}


def test_a_solve_reports_every_pin_it_moved_in_the_environment_it_was_aimed_at(
    deps: Dependencies, solved: list[tuple[str, bool, bool]]
) -> None:
    """Adding one requirement and learning it dragged others is exactly what this reports."""
    changes = deps.add("ray>=2", ecosystem="python", env="serving")
    locked = {
        change.name: (change.before, change.after)
        for change in changes
        if change.where == "pixi.lock"
    }
    assert locked == {
        "torch": ("2.9.0", "2.9.1"),
        "tqdm": ("absent", "4.70.0"),
        "leaving": ("1.0", "absent"),
    }
    assert "numpy" not in locked
    assert solved == [("serving", True, False)]


def test_registries_report_only_what_the_manifest_actually_configures(
    deps: Dependencies,
) -> None:
    """The workspace channels and a declared Python mirror, and nothing invented for the rest."""
    assert deps.registries("conda") == ("rapidsai", "conda-forge")
    assert deps.registries("python") == ("https://mirror.internal/simple",)
    assert deps.registries("nodejs") == ()
    assert deps.registries("go") == ()
