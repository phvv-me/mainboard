import pytest
from hypothesis import given
from hypothesis import strategies as st
from mainboard import Manifest, MissionError
from mainboard.manifest import Env, Header, Scope, Spec, Toolchain

_names = st.text(st.characters(categories=("Ll",)), min_size=1, max_size=8)
_versions = st.sampled_from(["*", ">=1.0", ">=2.9,<3", "==0.4.2"])


def test_spec_accepts_bare_strings_and_tables() -> None:
    assert Spec.model_validate(">=2.9").version == ">=2.9"
    editable = Spec.model_validate({"path": "packages/lab", "editable": True})
    assert editable.is_path and editable.is_editable
    assert not Spec.model_validate("*").is_path
    assert not Spec.model_validate({"path": "x"}).is_editable


@given(low=_versions, high=_versions)
def test_spec_merge_prefers_the_overlay_version_unless_wildcard(low: str, high: str) -> None:
    merged = Spec.model_validate(high).merged(Spec.model_validate(low))
    assert merged.version == (low if high == "*" else high)


def test_spec_merge_layers_extras_key_by_key() -> None:
    base = Spec.model_validate({"version": ">=1", "index": "a", "channel": "x"})
    over = Spec.model_validate({"index": "b"})
    merged = over.merged(base)
    assert merged.version == ">=1"
    assert (merged.model_extra or {})["index"] == "b"
    assert (merged.model_extra or {})["channel"] == "x"


def test_toolchain_rejects_bare_strings() -> None:
    with pytest.raises(ValueError, match="table with a deps key"):
        Toolchain.model_validate("3.14")


def test_toolchain_merge_and_all_deps() -> None:
    base = Toolchain.model_validate(
        {"deps": {"torch": ">=2.8", "numpy": "*"}, "dev": {"pytest": "*"}, "manager": "uv"}
    )
    over = Toolchain.model_validate({"deps": {"torch": ">=2.9"}, "dev": {"ruff": "*"}})
    merged = over.merged(base)
    assert merged.deps["torch"].version == ">=2.9"
    assert merged.deps["numpy"].version == "*"
    assert set(merged.all_deps()) == {"torch", "numpy", "pytest", "ruff"}
    assert (merged.model_extra or {})["manager"] == "uv"


def test_scope_discovers_ecosystem_tables_and_ignores_plain_extras() -> None:
    scope = Scope.model_validate(
        {
            "deps": {"python": ">=3.14"},
            "python": {"deps": {"torch": "*"}},
            "notes": {"freeform": "table"},
        }
    )
    assert set(scope.toolchains()) == {"python"}


def test_scope_merge_combines_conda_ecosystems_and_extras() -> None:
    base = Scope.model_validate(
        {
            "deps": {"python": ">=3.13", "pueue": "*"},
            "python": {"deps": {"torch": ">=2.8"}},
            "notes": {"freeform": "old"},
        }
    )
    over = Scope.model_validate(
        {
            "deps": {"python": ">=3.14"},
            "python": {"deps": {"vllm": "*"}},
            "rust": {"deps": {"serde": "*"}},
        }
    )
    merged = over.merged(base)
    assert merged.deps["python"].version == ">=3.14"
    assert merged.deps["pueue"].version == "*"
    chains = merged.toolchains()
    assert set(chains) == {"python", "rust"}
    assert set(chains["python"].deps) == {"torch", "vllm"}
    assert (merged.model_extra or {})["notes"] == {"freeform": "old"}


@given(names=st.lists(_names, min_size=1, max_size=5, unique=True))
def test_scope_path_deps_collects_across_ecosystems(names: list[str]) -> None:
    scope = Scope.model_validate(
        {
            "deps": {names[0]: {"path": f"packages/{names[0]}"}},
            "python": {"deps": {name: {"path": f"packages/{name}"} for name in names}},
        }
    )
    assert set(scope.path_deps()) == set(names)


def test_manifest_environment_lookup_and_reserved_names() -> None:
    manifest = Manifest(workspace=Header(name="lab"), envs={"serving": Env()})
    assert manifest.environment("serving") is manifest.envs["serving"]
    assert manifest.environment("default").deps == {}
    with pytest.raises(MissionError, match="declared environments"):
        manifest.environment("ghost")
    with pytest.raises(ValueError, match="reserved environment names"):
        Manifest(workspace=Header(name="lab"), envs={"default": Env()})
