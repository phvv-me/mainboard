from typing import TYPE_CHECKING

import pytest

from mainboard import Board, MissionError
from mainboard.deps import Dependencies, Index, ManifestText

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from mainboard import Manifest
    from mainboard.deps import Change

# What the fake lock reports before and after a solve, so a diff has one arrival, one departure
# and one move to report and nothing else.
_BEFORE = {"torch": "2.9.0", "numpy": "2.4.6", "leaving": "1.0"}
_AFTER = {"torch": "2.9.1", "numpy": "2.4.6", "tqdm": "4.70.0"}


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


def test_add_writes_the_constraint_it_was_given(deps: Dependencies) -> None:
    """A spec carrying its own range is written exactly as the caller wrote it."""
    changes = deps.add("tqdm>=4.70.0, <5", ecosystem="python", dev=True, resolve=False)
    assert moved(changes, "tqdm").model_dump() == {
        "name": "tqdm",
        "where": "[dev.python.deps]",
        "before": "absent",
        "after": ">=4.70.0, <5",
    }
    assert constraint(deps, "[dev.python.deps]", "tqdm") == ">=4.70.0, <5"


def test_add_pins_a_bare_name_to_what_the_index_publishes(
    deps: Dependencies, publishes: Callable[[str], None]
) -> None:
    """A name with no range is looked up rather than left unconstrained."""
    publishes("4.70.0")
    changes = deps.add("tqdm", ecosystem="python", resolve=False)
    assert moved(changes, "tqdm").after == ">=4.70.0, <5"


def test_add_reports_the_constraint_it_replaced(deps: Dependencies) -> None:
    """Overwriting an existing requirement shows both sides, which is the point of the verb."""
    changes = deps.add("torch>=3.0", ecosystem="python", resolve=False)
    assert moved(changes, "torch").before == ">=2.9"
    assert moved(changes, "torch").after == ">=3.0"


def test_add_lands_where_the_manifest_already_writes_that_kind_of_requirement(
    deps: Dependencies,
) -> None:
    """`[nodejs.dev]` is already there, so a dev entry joins it rather than starting a rival."""
    changes = deps.add("vitest@^3", ecosystem="nodejs", dev=True, resolve=False)
    assert moved(changes, "vitest").where == "[nodejs.dev]"


def test_add_refuses_an_environment_the_manifest_never_declared(deps: Dependencies) -> None:
    """The refusal comes from the schema, roster and all, before anything is written."""
    with pytest.raises(MissionError, match="no environment 'ghost'"):
        deps.add("tqdm", env="ghost", resolve=False)


def test_remove_finds_the_table_without_being_told_which(deps: Dependencies) -> None:
    """Dropping a requirement never asks the caller to remember where it was written."""
    changes = deps.remove("vite", resolve=False)
    assert moved(changes, "vite").model_dump() == {
        "name": "vite",
        "where": "[envs.serving.nodejs.dev]",
        "before": ">=7",
        "after": "absent",
    }


def test_remove_refuses_a_name_nothing_declares_and_names_what_it_searched(
    deps: Dependencies,
) -> None:
    """The refusal is actionable because it says where it already looked."""
    with pytest.raises(MissionError, match=r"nothing declares 'ghost'. Searched .*\[deps\]"):
        deps.remove("ghost", resolve=False)


def test_remove_refuses_a_name_declared_in_more_than_one_table(deps: Dependencies) -> None:
    """Guessing which one was meant is how the wrong requirement gets dropped silently."""
    deps.add("torch>=1", ecosystem="python", env="serving", resolve=False)
    with pytest.raises(MissionError, match=r"declared in .*Name one with --lang"):
        deps.remove("torch", resolve=False)


def test_a_narrowing_flag_tells_two_declarations_apart(deps: Dependencies) -> None:
    """Naming the environment is what makes an ambiguous name unambiguous again."""
    deps.add("torch>=1", ecosystem="python", env="serving", resolve=False)
    changes = deps.remove("torch", ecosystem="python", env="serving", resolve=False)
    assert moved(changes, "torch").where == "[envs.serving.python.deps]"


def test_a_search_narrowed_to_nothing_says_so(deps: Dependencies) -> None:
    """A flag combination reaching no declared table reports no table rather than an empty list."""
    with pytest.raises(MissionError, match="Searched no table"):
        deps.remove("ghost", ecosystem="go", resolve=False)


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


def test_a_solve_reports_every_pin_it_moved_and_nothing_it_left(
    deps: Dependencies, solved: list[tuple[str, bool, bool]]
) -> None:
    """Adding one requirement and learning it dragged others is exactly what this reports."""
    changes = deps.add("tqdm>=4", ecosystem="python", dev=True)
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


def test_an_edit_targeting_an_environment_resolves_that_environment(
    deps: Dependencies, solved: list[tuple[str, bool, bool]]
) -> None:
    """A requirement declared for one environment is solved against that environment."""
    deps.add("ray>=2", ecosystem="python", env="serving")
    assert solved == [("serving", True, False)]


def test_registries_report_only_what_the_manifest_actually_configures(
    deps: Dependencies,
) -> None:
    """The workspace channels and a declared Python mirror, and nothing invented for the rest."""
    assert deps.registries("conda") == ("rapidsai", "conda-forge")
    assert deps.registries("python") == ("https://mirror.internal/simple",)
    assert deps.registries("nodejs") == ()
    assert deps.registries("go") == ()
