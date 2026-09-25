from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import JsonValue, ValidationError

from mainboard import MissionError
from mainboard.ci import FAMILIES, Definition, Family, Package, family_of

from .conftest import declare, step

_FAMILY = st.sampled_from(FAMILIES)


@given(
    supported=st.sets(_FAMILY, min_size=1),
    reserved=st.lists(st.sets(_FAMILY), min_size=1, max_size=4),
    family=_FAMILY,
)
def test_a_family_runs_every_step_meant_for_it_in_order_and_only_those(
    supported: set[Family], reserved: list[set[Family]], family: Family
) -> None:
    """A step with no `only` runs everywhere the package runs, one with `only` there alone."""
    steps = [
        {"name": f"s{index}", "run": "tool", "only": sorted(only & supported)}
        for index, only in enumerate(reserved)
    ]
    definition = Definition.model_validate({"os": sorted(supported), "steps": steps})
    if family not in supported:
        with pytest.raises(MissionError, match=f"not {family}"):
            definition.on(family)
        return
    expected = [s["name"] for s in steps if not s["only"] or family in s["only"]]
    assert [s.name for s in definition.on(family)] == expected


@pytest.mark.parametrize(
    ("table", "refusal"),
    [
        ({"steps": []}, "at least 1"),
        ({"steps": [{"name": "x", "run": "  "}]}, "needs a command"),
        ({"steps": [{"name": "x", "run": "echo 'open"}]}, "quotation"),
        ({"os": ["linux"], "steps": [{"name": "x", "run": "t", "only": ["win"]}]}, "outside os"),
        ({"steps": [{"name": "x", "run": "t", "timeout": 0}]}, "greater than 0"),
    ],
    ids=["no steps", "an empty command", "unclosed quoting", "a stray family", "no time at all"],
)
def test_a_gate_that_could_not_run_is_refused_when_it_loads(
    table: dict[str, JsonValue], refusal: str
) -> None:
    with pytest.raises(ValidationError, match=refusal):
        Definition.model_validate(table)


def test_the_package_is_the_nearest_pyproject_declaring_a_gate(tmp_path: Path) -> None:
    """A pyproject without the table (a workspace root, say) is walked past, not taken."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "root"\n', encoding="utf-8")
    package = declare(tmp_path / "pkg", step("lint", "ruff check ."))
    inner = package / "src" / "pkg"
    inner.mkdir(parents=True)
    found = Package.found(inner)
    assert found.root == package
    assert found.definition.os == FAMILIES
    assert found.definition.steps[0].argv == ("ruff", "check", ".")
    with pytest.raises(MissionError, match=r"declares \[tool.mainboard.ci\]"):
        Package.found(tmp_path)


@pytest.mark.parametrize(
    ("platform", "family"),
    [("win-64", "win"), ("osx-arm64", "osx"), ("linux-aarch64", "linux"), ("linux", "linux")],
)
def test_a_platform_names_its_family(platform: str, family: Family) -> None:
    assert family_of(platform) == family


def test_a_platform_ci_never_runs_on_is_refused() -> None:
    with pytest.raises(MissionError, match="no platform family"):
        family_of("freebsd-64")
