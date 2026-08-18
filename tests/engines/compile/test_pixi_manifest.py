import tomllib
from typing import TYPE_CHECKING

import pytest

from mainboard.engines.compile.pixi_manifest import (
    PixiManifest,
    dependency_tables,
    pypi_options,
    rerooted,
    spec_toml,
)
from mainboard.engines.compile.platforms import PlatformMatrix
from mainboard.manifest import Env, Scope, Spec

if TYPE_CHECKING:
    from collections.abc import Callable

    from mainboard.manifest import Manifest

_PROJECT = "mainboard"


def test_spec_toml_is_bare_when_nothing_else_was_declared() -> None:
    assert spec_toml(Spec(version=">=2.9")) == ">=2.9"


def test_spec_toml_drops_a_default_version_beside_extras() -> None:
    spec = Spec.model_validate({"path": "packages/lab-core", "editable": True})
    assert spec_toml(spec) == {"path": "packages/lab-core", "editable": True}


def test_spec_toml_keeps_an_explicit_version_beside_extras() -> None:
    spec = Spec.model_validate({"version": ">=1", "git": "https://example.com/repo"})
    assert spec_toml(spec) == {"version": ">=1", "git": "https://example.com/repo"}


def test_dependency_tables_is_empty_for_a_bare_scope() -> None:
    assert dependency_tables(Scope()) == {}


def test_dependency_tables_carries_conda_deps() -> None:
    scope = Scope.model_validate({"deps": {"ripgrep": "*"}})
    assert dependency_tables(scope) == {"dependencies": {"ripgrep": "*"}}


def test_dependency_tables_adds_pypi_dependencies_from_the_python_toolchain() -> None:
    scope = Scope.model_validate({"python": {"deps": {"torch": ">=2.9"}}})
    assert dependency_tables(scope) == {"pypi-dependencies": {"torch": ">=2.9"}}


def test_dependency_tables_merges_dev_into_pypi_dependencies() -> None:
    scope = Scope.model_validate({"python": {"deps": {"torch": "*"}, "dev": {"ruff": "*"}}})
    assert dependency_tables(scope)["pypi-dependencies"] == {"torch": "*", "ruff": "*"}


def test_dependency_tables_omits_pypi_dependencies_for_an_empty_python_table() -> None:
    scope = Scope.model_validate({"python": {}})
    assert dependency_tables(scope) == {}


def test_dependency_tables_ignores_ecosystems_beyond_python() -> None:
    """Node/Rust/etc ride in the manifest untranslated, mainboard's deliberate simplification."""
    scope = Scope.model_validate({"nodejs": {"deps": {"prettier": "*"}}})
    assert dependency_tables(scope) == {}


def test_pypi_options_is_empty_without_a_python_table() -> None:
    assert pypi_options(Scope()) == {}


def test_pypi_options_is_empty_for_a_python_table_declaring_only_deps() -> None:
    assert pypi_options(Scope.model_validate({"python": {"deps": {"torch": "*"}}})) == {}


def test_pypi_options_forwards_the_solver_settings_pixi_defines() -> None:
    """The uv settings beside `[python.deps]` are what make an unsatisfiable set solve."""
    scope = Scope.model_validate(
        {
            "python": {
                "deps": {"torch": "*"},
                "index-strategy": "unsafe-best-match",
                "extra-index-urls": ["https://pypi.nvidia.com"],
                "no-build-isolation": ["fastsafetensors"],
                "prerelease-mode": "allow",
                "dependency-overrides": {"cuda-core": ">=1.1.1, <2"},
            }
        }
    )
    assert pypi_options(scope) == {
        "index-strategy": "unsafe-best-match",
        "extra-index-urls": ["https://pypi.nvidia.com"],
        "no-build-isolation": ["fastsafetensors"],
        "prerelease-mode": "allow",
        "dependency-overrides": {"cuda-core": ">=1.1.1, <2"},
    }


def test_pypi_options_leaves_out_what_pixi_never_reads() -> None:
    """A table beside the deps that configures something other than the solve stays put."""
    scope = Scope.model_validate(
        {"python": {"deps": {"torch": "*"}, "indexes": {"nvidia": "https://pypi.nvidia.com"}}}
    )
    assert pypi_options(scope) == {}


def test_pypi_options_emits_the_override_table_last() -> None:
    """TOML reads every key after a sub-table header as that sub-table's, so it goes last."""
    scope = Scope.model_validate(
        {
            "python": {
                "deps": {"torch": "*"},
                "dependency-overrides": {"cuda-core": ">=1.1.1"},
                "index-strategy": "unsafe-best-match",
            }
        }
    )
    assert list(pypi_options(scope))[-1] == "dependency-overrides"


def test_task_translates_a_bare_string_and_rebases_its_cwd() -> None:
    assert PixiManifest.task("pytest") == {"cmd": "pytest", "cwd": ".."}


def test_task_carries_its_own_environment_table() -> None:
    """pixi applies a task's `env` around the command, so a task needs no inlined assignment."""
    spec = {"run": "pytest", "env": {"PYTHONPATH": "research"}}
    assert PixiManifest.task(spec)["env"] == {"PYTHONPATH": "research"}


def test_task_renames_run_depends_and_dir() -> None:
    spec = {"run": "pytest", "depends": ["build"], "dir": "packages/lab-core"}
    assert PixiManifest.task(spec) == {
        "cmd": "pytest",
        "depends-on": ["build"],
        "cwd": "../packages/lab-core",
    }


def test_task_leaves_an_absolute_dir_untouched() -> None:
    assert PixiManifest.task({"run": "pytest", "dir": "/opt/project"})["cwd"] == "/opt/project"


def test_task_skips_the_cwd_rebase_for_a_command_less_aggregator() -> None:
    assert PixiManifest.task({"depends": ["build", "lint"]}) == {"depends-on": ["build", "lint"]}


def test_activation_table_is_empty_without_env_vars_or_dotenv(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\ndotenv = false\n')
    assert PixiManifest.activation_table(manifest) == {}


def test_activation_table_carries_env_vars_and_the_dotenv_loader(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n[env]\nFOO = "bar"\n')
    assert PixiManifest.activation_table(manifest) == {
        "env": {"FOO": "bar"},
        "scripts": ["dotenv.sh"],
    }


def test_activation_table_appends_the_workspace_scripts_after_the_dotenv_loader(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """A declared script is workspace-relative, so it is rerooted like any other location."""
    manifest = manifest_from(
        '[workspace]\nname = "w"\nscripts = ["scripts/activate.sh", "/opt/site.sh"]\n'
    )
    assert PixiManifest.activation_table(manifest)["scripts"] == [
        "dotenv.sh",
        "../scripts/activate.sh",
        "/opt/site.sh",
    ]


def test_activation_table_carries_declared_scripts_without_the_dotenv_loader(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from(
        '[workspace]\nname = "w"\ndotenv = false\nscripts = ["scripts/activate.sh"]\n'
    )
    assert PixiManifest.activation_table(manifest) == {"scripts": ["../scripts/activate.sh"]}


@pytest.mark.parametrize(
    ("declared", "compiled"),
    [("packages/lote", "../packages/lote"), ("/opt/lote", "/opt/lote"), ("", "..")],
)
def test_rerooted_shifts_only_a_workspace_relative_location(declared: str, compiled: str) -> None:
    """One rule for every declared location, since they all resolve from `.mainboard/`."""
    assert rerooted(declared) == compiled


def test_platform_array_renders_a_named_variant_as_an_inline_table() -> None:
    rendered = PixiManifest.platform_array(
        ["linux-64", {"name": "linux-64-serving", "platform": "linux-64", "cuda": "13.0"}]
    )
    assert list(rendered) == [
        "linux-64",
        {"name": "linux-64-serving", "platform": "linux-64", "cuda": "13.0"},
    ]


def test_feature_table_carries_channels_platforms_and_target() -> None:
    env = Env.model_validate(
        {
            "channels": ["nvidia"],
            "platforms": ["linux-64"],
            "on": {"linux": {"deps": {"cuda-toolkit": "*"}}},
        }
    )
    body = PixiManifest.feature_table(env)
    assert body["channels"] == ["nvidia"]
    assert body["platforms"] == ["linux-64"]
    assert body["target"] == {"linux": {"dependencies": {"cuda-toolkit": "*"}}}


def test_feature_table_carries_the_envs_own_solver_options() -> None:
    """An env declaring its own `[python]` settings keeps them, feature-scoped like its deps."""
    env = Env.model_validate(
        {"python": {"deps": {"vllm": "*"}, "index-strategy": "unsafe-best-match"}}
    )
    assert PixiManifest.feature_table(env)["pypi-options"] == {
        "index-strategy": "unsafe-best-match"
    }


def test_feature_table_is_minimal_for_a_plain_env() -> None:
    assert PixiManifest.feature_table(Env()) == {}


def test_declared_feature_includes_tasks_only_when_declared() -> None:
    empty_platforms = PlatformMatrix(workspace=[], environments={}, default=[])
    assert "tasks" not in PixiManifest.declared_feature("serving", Env(), empty_platforms)

    env = Env.model_validate({"tasks": {"serve": "python -m serve"}})
    feature = PixiManifest.declared_feature("serving", env, empty_platforms)
    assert feature["tasks"] == {"serve": {"cmd": "python -m serve", "cwd": ".."}}


def test_declared_feature_overrides_platforms_with_the_resolved_matrix() -> None:
    env = Env.model_validate({"platforms": ["linux-64"]})
    platforms = PlatformMatrix(
        workspace=[], environments={"serving": ["linux-64-serving"]}, default=[]
    )
    feature = PixiManifest.declared_feature("serving", env, platforms)
    assert feature["platforms"] == ["linux-64-serving"]


def test_features_is_empty_with_no_envs_dev_or_platform_floors(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    feature, environments = PixiManifest.features(
        manifest, PlatformMatrix(workspace=[], environments={}, default=[]), _PROJECT
    )
    assert feature == {}
    assert environments == {}


def test_features_adds_dev_and_a_default_environment(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n[dev.deps]\nruff = "*"\n')
    feature, environments = PixiManifest.features(
        manifest, PlatformMatrix(workspace=[], environments={}, default=[]), _PROJECT
    )
    assert feature["dev"] == {"dependencies": {"ruff": "*"}}
    assert environments["default"] == {"features": ["dev"]}


def test_features_adds_the_project_platforms_feature_when_routed(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\nplatforms = ["linux-64"]\n')
    platforms = PlatformMatrix(workspace=["linux-64"], environments={}, default=["linux-64"])
    feature, environments = PixiManifest.features(manifest, platforms, _PROJECT)
    assert feature[f"{_PROJECT}-platforms"] == {"platforms": ["linux-64"]}
    assert environments["default"] == {"features": [f"{_PROJECT}-platforms"]}


def test_features_marks_no_default_feature_for_a_no_default_env(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n[envs.isolated]\nno-default = true\n')
    _, environments = PixiManifest.features(
        manifest, PlatformMatrix(workspace=[], environments={}, default=[]), _PROJECT
    )
    assert environments["isolated"] == {"features": ["isolated"], "no-default-feature": True}


def test_from_manifest_reroots_a_local_path_dependency(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        [python.deps]
        lab-core = { path = "packages/lab-core", editable = true }
        """
    )
    compiled = PixiManifest.from_manifest(manifest, project_name=_PROJECT)
    assert compiled.pypi_dependencies["lab-core"] == {
        "path": "../packages/lab-core",
        "editable": True,
    }


def test_from_manifest_reroots_a_path_inside_a_dependency_override(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """An override is a dependency table like any other, so its local source shifts too."""
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        [python.deps]
        torch = ">=2.9"
        [python.dependency-overrides]
        sqlalchemy = { path = "packages/sqlalchemy" }
        cuda-core = ">=1.1.1, <2"
        """
    )
    compiled = PixiManifest.from_manifest(manifest, project_name=_PROJECT)
    assert compiled.pypi_options["dependency-overrides"] == {
        "sqlalchemy": {"path": "../packages/sqlalchemy"},
        "cuda-core": ">=1.1.1, <2",
    }


def test_from_manifest_leaves_an_absolute_path_dependency_untouched(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        [python.deps]
        vendored = { path = "/opt/vendored", editable = true }
        """
    )
    compiled = PixiManifest.from_manifest(manifest, project_name=_PROJECT)
    assert compiled.pypi_dependencies["vendored"]["path"] == "/opt/vendored"


def test_from_manifest_never_reroots_a_dependency_literally_named_path(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n[deps]\npath = "*"\n')
    compiled = PixiManifest.from_manifest(manifest, project_name=_PROJECT)
    assert compiled.dependencies["path"] == "*"


def test_from_manifest_compiles_target_platform_overlays(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        [on.linux.deps]
        libgcc = "*"
        """
    )
    compiled = PixiManifest.from_manifest(manifest, project_name=_PROJECT)
    assert compiled.target == {"linux": {"dependencies": {"libgcc": "*"}}}


def test_to_toml_renders_the_solver_options_as_a_table_pixi_can_read_back(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """Rendering is where the override sub-table could swallow the scalars written after it."""
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        platforms = ["linux-64"]
        [python.deps]
        torch = ">=2.9"
        [python]
        index-strategy = "unsafe-best-match"
        no-build-isolation = ["fastsafetensors"]
        [python.dependency-overrides]
        sqlalchemy = { path = "packages/sqlalchemy" }
        """
    )
    text = PixiManifest.from_manifest(manifest, project_name=_PROJECT).to_toml()
    assert tomllib.loads(text)["pypi-options"] == {
        "index-strategy": "unsafe-best-match",
        "no-build-isolation": ["fastsafetensors"],
        "dependency-overrides": {"sqlalchemy": {"path": "../packages/sqlalchemy"}},
    }


def test_to_toml_renders_hyphenated_tables_and_drops_defaults(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        platforms = ["linux-64"]
        [python.deps]
        torch = ">=2.9"
        """
    )
    text = PixiManifest.from_manifest(manifest, project_name=_PROJECT).to_toml()
    document = tomllib.loads(text)
    assert document["pypi-dependencies"] == {"torch": ">=2.9"}
    assert "dependencies" not in document
