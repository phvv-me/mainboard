import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import Manifest, MissionError
from mainboard.manifest import Engine, Env, Header, HostProfile, Scope, Spec, Toolchain

from ..strategies import SPECS, WORDS


@pytest.mark.parametrize(
    ("declaration", "version", "path", "editable"),
    [
        (">=2.9", ">=2.9", False, False),
        ("*", "*", False, False),
        ({"path": "packages/lab", "editable": True}, "*", True, True),
        ({"path": "packages/lab"}, "*", True, False),
    ],
)
def test_a_spec_reads_a_bare_string_or_a_table_and_names_its_own_kind(
    declaration: str | dict[str, str | bool], version: str, *, path: bool, editable: bool
) -> None:
    """`torch = ">=2.9"` is shorthand, and a source requirement says so rather than pin a range."""
    spec = Spec.model_validate(declaration)
    assert spec.version == version
    assert spec.is_path is path
    assert spec.is_editable is editable


@given(low=SPECS, high=SPECS)
def test_layering_a_spec_prefers_the_overlay_unless_it_is_a_wildcard(low: str, high: str) -> None:
    """A wildcard says nothing, so the layer underneath keeps its say and extras merge on top."""
    base = Spec.model_validate({"version": low, "index": "a", "channel": "x"})
    over = Spec.model_validate({"version": high, "index": "b"})
    merged = over.merged(base)
    assert merged.version == (low if high == "*" else high)
    assert (merged.model_extra or {}) == {"index": "b", "channel": "x"}


def test_an_ecosystem_entry_must_be_a_table_and_not_a_version_string() -> None:
    """`python = "3.14"` is a requirement in `[deps]`, never an ecosystem of its own."""
    with pytest.raises(ValueError, match="table with a deps key"):
        Toolchain.model_validate("3.14")


def test_a_scope_discovers_its_ecosystem_tables_and_layers_them_over_a_base() -> None:
    """Conda deps, each ecosystem and the plain extras all merge, and nothing else is a chain."""
    base = Scope.model_validate(
        {
            "deps": {"python": ">=3.13", "pueue": "*"},
            "python": {"deps": {"torch": ">=2.8"}, "dev": {"pytest": "*"}, "manager": "uv"},
            "notes": {"freeform": "old"},
        }
    )
    over = Scope.model_validate(
        {
            "deps": {"python": ">=3.14"},
            "python": {"deps": {"vllm": "*"}, "dev": {"ruff": "*"}},
            "rust": {"deps": {"serde": "*"}},
        }
    )
    assert set(base.toolchains()) == {"python"}
    merged = over.merged(base)
    assert merged.deps["python"].version == ">=3.14"
    assert merged.deps["pueue"].version == "*"
    chains = merged.toolchains()
    assert set(chains) == {"python", "rust"}
    assert set(chains["python"].deps) == {"torch", "vllm"}
    assert set(chains["python"].all_deps()) == {"torch", "vllm", "pytest", "ruff"}
    assert (chains["python"].model_extra or {})["manager"] == "uv"
    assert (merged.model_extra or {})["notes"] == {"freeform": "old"}


@given(names=st.lists(WORDS, min_size=1, max_size=5, unique=True))
def test_a_scope_collects_every_local_path_requirement_across_its_ecosystems(
    names: list[str],
) -> None:
    """A path requirement is what a sync has to ship, wherever in the scope it was declared."""
    scope = Scope.model_validate(
        {
            "deps": {names[0]: {"path": f"packages/{names[0]}"}},
            "python": {"deps": {name: {"path": f"packages/{name}"} for name in names}},
        }
    )
    assert set(scope.path_deps()) == set(names)


def test_the_environment_roster_answers_by_name_and_refuses_a_stranger() -> None:
    """`default` is always there because pixi always has it, and everything else is declared."""
    manifest = Manifest(workspace=Header(name="lab"), envs={"serving": Env()})
    assert manifest.environment("serving") is manifest.envs["serving"]
    assert manifest.environment("default").deps == {}
    with pytest.raises(MissionError, match="declared environments"):
        manifest.environment("ghost")


@pytest.mark.parametrize(
    ("tables", "match"),
    [
        ({"envs": {"default": Env()}}, "reserved environment names"),
        ({"hosts": {"gold": HostProfile(container="ghost")}}, "names container 'ghost'"),
        ({"hosts": {"gold": HostProfile(env="ghost")}}, "names environment 'ghost'"),
        (
            {"engines": {"vserve": Engine(command="true", container="ghost")}},
            "names container 'ghost'",
        ),
        (
            {"engines": {"vserve": Engine(command="true", env="ghost")}},
            "names environment 'ghost'",
        ),
    ],
)
def test_a_manifest_refuses_a_name_that_points_at_no_table(
    tables: dict[str, dict[str, Env | HostProfile | Engine]], match: str
) -> None:
    """A host or engine naming a container or environment that is not there fails at load."""
    with pytest.raises(ValueError, match=match):
        Manifest(workspace=Header(name="lab"), **tables)
